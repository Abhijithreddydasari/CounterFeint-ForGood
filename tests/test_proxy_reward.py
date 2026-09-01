"""
Unit tests for the per-completion proxy reward used by GRPO.

The fixtures cover:
  * Format failure -> small negative.
  * Partial JSON -> partial credit (between -0.3 and -0.1).
  * Schema-valid completion -> consistent positive baseline.
  * Class-match / decision-match bonuses scale the right way.
  * Continuous components (confidence, conciseness, hash tiebreaker)
    produce reward variance.
  * The reward function works on completions GRPO never saw at
    rollout collection time.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from counterfeint.training.proxy_reward import (
    build_gold_lookup,
    make_proxy_reward_fn,
    proxy_reward_one,
)


_GOLD_NONE = {
    "action_type": None, "ad_id": None, "verdict": None,
    "investigation_target": None, "linked_ad_id": None,
}

# Hash tiebreaker adds a deterministic [0, 0.02] offset per completion.
_ABS = 0.03


def _verdict_completion(verdict: str = "reject", ad_id: str = "ad_001") -> str:
    return json.dumps({
        "action_type": "verdict",
        "ad_id": ad_id,
        "verdict": verdict,
        "confidence": 0.9,
        "rationale": "payment ring detected",
    })


def _investigate_completion(target: str = "payment_method", ad_id: str = "ad_001") -> str:
    return json.dumps({
        "action_type": "investigate",
        "ad_id": ad_id,
        "investigation_target": target,
        "rationale": "check payment trail",
    })


class TestSchemaValidity:
    def test_unparseable_completion_returns_negative(self) -> None:
        r = proxy_reward_one(
            "prompt about ad_001",
            "definitely not json",
            gold=_GOLD_NONE,
            gold_episode_score=0.0,
        )
        # Partial credit: -0.3 base (text exists but no JSON structure)
        assert r < 0.0

    def test_invalid_schema_returns_partial_credit(self) -> None:
        r = proxy_reward_one(
            "prompt about ad_001",
            json.dumps({"action_type": "make_coffee"}),
            gold=_GOLD_NONE,
            gold_episode_score=0.0,
        )
        # Partial credit: -0.3 + 0.05 (starts {) + 0.05 (has action_type) + 0.05 (ends })
        assert -0.2 < r < 0.0

    def test_valid_schema_baseline(self) -> None:
        r = proxy_reward_one(
            "prompt about ad_999",  # ad_001 NOT in prompt -> no coherence bonus
            _verdict_completion(),
            gold=_GOLD_NONE,
            gold_episode_score=0.0,
        )
        # 0.6 schema + 0.135 confidence(0.9) + 0.1 conciseness + ~hash
        assert r == pytest.approx(0.835, abs=_ABS)


class TestCoherenceBonus:
    def test_referenced_ad_id_in_prompt_gets_bonus(self) -> None:
        prompt = "Pending: ad_001, ad_002. Focus on ad_001."
        r = proxy_reward_one(
            prompt,
            _verdict_completion(ad_id="ad_001"),
            gold=_GOLD_NONE,
            gold_episode_score=0.0,
        )
        # 0.6 schema + 0.15 coherence + 0.135 confidence + 0.1 concise + ~hash
        assert r == pytest.approx(0.985, abs=_ABS)

    def test_referenced_linked_id_in_prompt_gets_bonus(self) -> None:
        prompt = "Pending: ad_001, ad_002, ad_003."
        completion = json.dumps({
            "action_type": "link_accounts",
            "ad_id": "ad_001",
            "linked_ad_id": "ad_003",
            "link_reason": "shared payment_id",
        })
        r = proxy_reward_one(
            prompt, completion, gold=_GOLD_NONE, gold_episode_score=0.0,
        )
        # 0.6 schema + 0.15 ad + 0.15 linked + 0.1 concise + ~hash
        assert r == pytest.approx(1.0, abs=_ABS)


class TestGoldClassMatch:
    def test_action_class_match_adds_class_bonus(self) -> None:
        gold = {
            **_GOLD_NONE,
            "action_type": "verdict",
            "verdict": "approve",
        }
        r = proxy_reward_one(
            "Pending: ad_001",
            _verdict_completion(verdict="reject"),
            gold=gold,
            gold_episode_score=0.0,
        )
        # 0.6 schema + 0.15 coherence + 0.2 class + 0.135 conf + 0.1 concise
        assert r == pytest.approx(1.185, abs=_ABS)

    def test_link_accounts_classified_with_verdicts(self) -> None:
        gold = {**_GOLD_NONE, "action_type": "link_accounts"}
        completion = json.dumps({
            "action_type": "verdict",
            "ad_id": "ad_001",
            "verdict": "approve",
            "confidence": 0.5,
            "rationale": "looks fine",
        })
        r = proxy_reward_one(
            "Pending: ad_001",
            completion,
            gold=gold,
            gold_episode_score=0.0,
        )
        # 0.6 + 0.15 + 0.2 class (both "verdict" class) + 0.075 conf + 0.1 concise
        assert r == pytest.approx(1.125, abs=_ABS)


class TestGoldDecisionMatch:
    def test_verdict_match_scales_with_recorded_quality(self) -> None:
        gold = {**_GOLD_NONE, "action_type": "verdict", "verdict": "reject"}
        r_high_quality = proxy_reward_one(
            "Pending: ad_001",
            _verdict_completion(verdict="reject"),
            gold=gold,
            gold_episode_score=1.0,
        )
        r_low_quality = proxy_reward_one(
            "Pending: ad_001",
            _verdict_completion(verdict="reject"),
            gold=gold,
            gold_episode_score=0.0,
        )
        # high: 0.6 + 0.15 + 0.2 + 0.6 decision + 0.135 conf + 0.1 concise
        assert r_high_quality == pytest.approx(1.785, abs=_ABS)
        assert r_low_quality == pytest.approx(1.185, abs=_ABS)
        assert r_high_quality > r_low_quality

    def test_target_match_scales_with_recorded_quality(self) -> None:
        gold = {
            **_GOLD_NONE,
            "action_type": "investigate",
            "investigation_target": "payment_method",
        }
        r = proxy_reward_one(
            "Pending: ad_001",
            _investigate_completion(target="payment_method"),
            gold=gold,
            gold_episode_score=0.5,
        )
        # 0.6 + 0.15 + 0.2 class + 0.25 target + 0.1 concise (no conf for investigate)
        assert r == pytest.approx(1.3, abs=_ABS)


class TestRewardFunctionIntegration:
    def test_reward_fn_handles_unseen_prompts_gracefully(self) -> None:
        gold_lookup = {
            "old prompt about ad_002": {
                "fields": {**_GOLD_NONE, "action_type": "verdict", "verdict": "reject"},
                "episode_score": 0.8,
            }
        }
        reward_fn = make_proxy_reward_fn(gold_lookup=gold_lookup)

        prompts = ["new unseen prompt about ad_001"]
        completions = [_verdict_completion(ad_id="ad_001")]
        rewards = reward_fn(prompts=prompts, completions=completions)

        assert len(rewards) == 1
        # 0.6 schema + 0.15 coherence + 0.135 conf + 0.1 concise (no gold)
        assert rewards[0] == pytest.approx(0.985, abs=_ABS)

    def test_trl_conversational_completion_is_unwrapped_not_partial_credit(self) -> None:
        """TRL 1.12 GRPO passes completions as chat dicts, not strings.

        The old ``" ".join(str(x) for x in val)`` path scored every
        completion at exactly -0.1 (partial-credit on the dict repr),
        so reward_std=0 and GRPO never updated. Unwrap must recover
        the assistant JSON and score it as a valid action.
        """
        gold_lookup: dict[str, Any] = {}
        reward_fn = make_proxy_reward_fn(gold_lookup=gold_lookup)
        json_text = _verdict_completion(ad_id="ad_001")
        conversational = [[{"role": "assistant", "content": json_text}]]
        rewards = reward_fn(
            prompts=["Pending: ad_001"],
            completions=conversational,
        )
        assert rewards[0] == pytest.approx(0.985, abs=_ABS)
        assert rewards[0] != pytest.approx(-0.1, abs=1e-6)

    def test_build_gold_lookup_extracts_action_class_from_repr(self) -> None:
        sample = SimpleNamespace(
            prompt="Pending: ad_001",
            completion=_verdict_completion(),
            terminal_grader_score=0.7,
            metadata={
                "action_repr": (
                    "AdReviewAction(action_type='verdict', ad_id='ad_001', "
                    "verdict='reject', confidence=0.93, rationale='...')"
                ),
                "action_class": "verdict",
            },
        )
        gold_lookup = build_gold_lookup([sample])
        gold = gold_lookup["Pending: ad_001"]
        assert gold["episode_score"] == pytest.approx(0.7)
        assert gold["fields"]["action_type"] == "verdict"
        assert gold["fields"]["verdict"] == "reject"
        assert gold["fields"]["ad_id"] == "ad_001"
