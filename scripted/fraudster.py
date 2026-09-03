"""
Scripted Fraudster policies (baselines + adversarial opponents).

All three policies consume a `FraudsterObservation` dict (as delivered over the
WebSocket) and return a `FraudsterAction` Pydantic instance.

    ScriptedFraudster    deterministic: propose 2 ads round 1, 1 ad thereafter,
                         commit_final once all proposals are used.
    ReactiveFraudster    adapts category + content based on which of its
                         prior proposals were rejected and which investigation
                         targets the Investigator kept pulling.
    GibberishFraudster   low-plausibility negative control: proposes random
                         gibberish ad copy (for sanity-checking plausibility
                         grader).
"""

from __future__ import annotations

import random
import string
from typing import Any, Dict, List, Optional

from ..models import FraudsterAction
from ._base import PolicyBase


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


_LEGIT_CAMOUFLAGE_CATEGORIES = (
    "ecommerce",
    "saas",
    "local_service",
    "education",
    "fitness",
)
_FRAUD_CATEGORIES = (
    "fake_giveaway",
    "fake_crypto",
    "fake_weight_loss",
    "gray_area_supplements",
    "fake_emergency",
)

_CAMOUFLAGE_ADS = [
    (
        "Boost your small business inventory tracking with CloudLedger — 14-day free trial.",
        "CloudLedger helps shops manage SKUs, stock levels, and vendor orders.",
        "saas",
        "SMB owners, US, interests: accounting / inventory",
    ),
    (
        "Upgrade your home gym with adjustable dumbbells, now $129 with free shipping.",
        "Space-saving dumbbells that replace a full rack, ships in 48h.",
        "fitness",
        "Adults 25-45, US, home-gym interest",
    ),
    (
        "Online Python fundamentals course with projects and a live mentor — enroll today.",
        "8-week beginner Python bootcamp, weekly office hours, job-ready portfolio.",
        "education",
        "Career-switchers 22-35",
    ),
    (
        "Downtown dog-walking service — $18 per walk, insured handlers, book online.",
        "Reliable neighborhood dog walking with GPS tracking and photo updates.",
        "local_service",
        "Urban dog owners, city centers",
    ),
    (
        "Refurbished iPhone 13 256GB — warranty included, $399 this week only.",
        "Certified refurbished phones with 12-month warranty and 30-day returns.",
        "ecommerce",
        "Adults 18-55, price-sensitive shoppers",
    ),
]

_FRAUD_ADS = [
    (
        "Free iPhone 15 — first 500 customers only, tap now!",
        "Giveaway ends at midnight, quick verify your shipping address.",
        "fake_giveaway",
        "Anyone, low-income targeting",
    ),
    (
        "Invest in MoonCoin — guaranteed 50x returns in 30 days, risk-free!",
        "Join our MoonCoin pre-sale, 10% bonus for first buyers.",
        "fake_crypto",
        "Men 18-35, crypto interest",
    ),
    (
        "Lose 20 pounds in 7 days with this one weird trick doctors hate!",
        "Revolutionary weight-loss pill that dissolves fat overnight.",
        "fake_weight_loss",
        "Women 25-55, weight-loss interest",
    ),
]


def _overt_fraud_allowed(observation: Dict[str, Any]) -> bool:
    allowed: List[str] = observation.get("allowed_categories") or []
    if not allowed:
        return True
    overt = set(_FRAUD_CATEGORIES) | {
        "miracle_cure", "clone_brand", "counterfeit_goods", "advance_fee",
        "network_crypto", "network_ecommerce", "network_fintech", "network_health",
    }
    return bool(set(allowed) & overt)


def _pick_surface_ad(rng: random.Random, observation: Dict[str, Any]):
    if _overt_fraud_allowed(observation):
        return rng.choice(_FRAUD_ADS)
    return rng.choice(_CAMOUFLAGE_ADS)


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _choose_allowed(
    preferred: str, observation: Dict[str, Any], *, fallback: str = "ecommerce"
) -> str:
    """Pick `preferred` if it's in the observation's allowed_categories, else fall back."""
    allowed: List[str] = observation.get("allowed_categories") or []
    if not allowed:
        return preferred
    if preferred in allowed:
        return preferred
    for cat in _LEGIT_CAMOUFLAGE_CATEGORIES:
        if cat in allowed:
            return cat
    if fallback in allowed:
        return fallback
    return allowed[0]


# ---------------------------------------------------------------------------
# ScriptedFraudster
# ---------------------------------------------------------------------------


class ScriptedFraudster(PolicyBase):
    """
    Deterministic fraudster. Round 1 proposes two ads (one camouflage, one
    overt fraud) and ends turn; subsequent rounds propose one more if budget
    allows; once proposals are exhausted (or on round N) it commits_final.

    Good for regression tests and as a baseline opponent.
    """

    def __init__(self, *, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._seed = seed
        self._turn_actions_taken = 0

    def reset(self) -> None:
        self._rng = random.Random(self._seed)
        self._turn_actions_taken = 0

    def act(self, observation: Dict[str, Any]) -> FraudsterAction:
        remaining = int(observation.get("proposals_remaining", 0))
        actions_left = int(observation.get("actions_left_this_turn", 1))
        round_number = int(observation.get("round_number", 1))

        if remaining <= 0:
            return FraudsterAction(
                action_type="commit_final",
                rationale="No proposals left; locking in.",
            )

        if actions_left <= 1:
            ad_copy, blurb, cat, targeting = self._pick_ad(round_number)
            cat = _choose_allowed(cat, observation)
            return FraudsterAction(
                action_type="propose_ad",
                ad_copy=_truncate(ad_copy, 2000),
                landing_page_blurb=_truncate(blurb, 2000),
                category=cat,
                targeting_summary=targeting,
                rationale=(
                    "ScriptedFraudster: final slot of the turn, proposing once "
                    "before yielding to Investigator."
                ),
            )

        ad_copy, blurb, cat, targeting = self._pick_ad(round_number)
        cat = _choose_allowed(cat, observation)
        return FraudsterAction(
            action_type="propose_ad",
            ad_copy=_truncate(ad_copy, 2000),
            landing_page_blurb=_truncate(blurb, 2000),
            category=cat,
            targeting_summary=targeting,
            rationale="ScriptedFraudster: executing canonical round plan.",
        )

    def _pick_ad(self, round_number: int):
        if round_number == 1:
            return self._rng.choice(_FRAUD_ADS)
        return self._rng.choice(_CAMOUFLAGE_ADS)


# ---------------------------------------------------------------------------
# ReactiveFraudster
# ---------------------------------------------------------------------------


class ReactiveFraudster(PolicyBase):
    """
    Reactive fraudster. Observes the Investigator's prior verdicts and
    investigation targets, then adapts:

      * If any of its own proposals were rejected, it switches to camouflage
        categories and invests in a stronger landing-page blurb.
      * If the Investigator keeps pulling `landing_page`, the Fraudster
        beefs up that field; if `advertiser_history`, it adds benign
        targeting context.
      * Commits_final early if the Investigator has been soft (high approval
        rate on prior proposals → try to lock them in before audit).
    """

    def __init__(self, *, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._seed = seed

    def reset(self) -> None:
        self._rng = random.Random(self._seed)

    def act(self, observation: Dict[str, Any]) -> FraudsterAction:
        remaining = int(observation.get("proposals_remaining", 0))
        actions_left = int(observation.get("actions_left_this_turn", 1))
        my_rejected = self._my_rejected_count(observation)
        my_approved = self._my_approved_count(observation)
        targets_emphasized = self._dominant_investigation_target(observation)

        if remaining <= 0:
            return FraudsterAction(
                action_type="commit_final",
                rationale=(
                    f"ReactiveFraudster: budget exhausted; {my_approved} of my "
                    f"proposals approved so far, locking in."
                ),
            )

        if my_approved >= 2 and self._rng.random() < 0.35:
            return FraudsterAction(
                action_type="commit_final",
                rationale=(
                    "ReactiveFraudster: Investigator is lenient; lock in current "
                    "slate before they harden."
                ),
            )

        mod = self._maybe_modify_pending(observation, targets_emphasized)
        if mod is not None:
            return mod

        if my_rejected >= 1:
            ad_copy, blurb, cat, targeting = self._rng.choice(_CAMOUFLAGE_ADS)
            rationale = (
                f"ReactiveFraudster: pivoting to camouflage ({cat}) after "
                f"{my_rejected} rejections; Investigator pulled "
                f"{targets_emphasized or 'no signal'}."
            )
        else:
            ad_copy, blurb, cat, targeting = _pick_surface_ad(self._rng, observation)
            rationale = (
                "ReactiveFraudster: Investigator hasn't rejected me yet; "
                "testing a borderline/fraud template."
            )

        if targets_emphasized == "landing_page":
            blurb = (
                "Trusted brand with 12,000+ reviews — verified customer "
                "testimonials, PCI-DSS compliant checkout, and a 30-day "
                "money-back guarantee. " + blurb
            )
        elif targets_emphasized == "advertiser_history":
            targeting = (
                targeting + "; advertiser active since 2018 with >3y domain age"
            )

        cat = _choose_allowed(cat, observation)
        if actions_left <= 1:
            rationale += " (final slot of this turn)"

        return FraudsterAction(
            action_type="propose_ad",
            ad_copy=_truncate(ad_copy, 2000),
            landing_page_blurb=_truncate(blurb, 2000),
            category=cat,
            targeting_summary=_truncate(targeting, 512),
            rationale=_truncate(rationale, 2000),
        )

    def _my_rejected_count(self, observation: Dict[str, Any]) -> int:
        return sum(
            1
            for v in observation.get("prior_verdicts", [])
            if v.get("was_my_proposal") and v.get("verdict") == "reject"
        )

    def _my_approved_count(self, observation: Dict[str, Any]) -> int:
        return sum(
            1
            for v in observation.get("prior_verdicts", [])
            if v.get("was_my_proposal") and v.get("verdict") == "approve"
        )

    def _dominant_investigation_target(
        self, observation: Dict[str, Any]
    ) -> Optional[str]:
        targets_used: Dict[str, List[str]] = observation.get(
            "investigation_targets_used", {}
        )
        counter: Dict[str, int] = {}
        for targets in targets_used.values():
            for t in targets:
                counter[t] = counter.get(t, 0) + 1
        if not counter:
            return None
        return max(counter, key=lambda k: counter[k])

    def _maybe_modify_pending(
        self, observation: Dict[str, Any], targets_emphasized: Optional[str]
    ) -> Optional[FraudsterAction]:
        if targets_emphasized is None:
            return None

        current_queue = observation.get("current_queue", [])
        for item in current_queue:
            if not item.get("is_my_proposal"):
                continue
            if item.get("status") not in ("pending", "investigating"):
                continue
            slot_index = item.get("slot_index")
            if slot_index is None:
                continue
            if self._rng.random() < 0.25:
                return FraudsterAction(
                    action_type="modify_pending_ad",
                    slot_index=int(slot_index),
                    new_landing_page_blurb=(
                        "Verified business with transparent pricing, real-time "
                        "order tracking, and 30-day returns. "
                    ),
                    rationale=(
                        "ReactiveFraudster: Investigator emphasizing "
                        f"{targets_emphasized}; strengthening pending slot "
                        f"{slot_index}."
                    ),
                )
        return None


# ---------------------------------------------------------------------------
# GibberishFraudster
# ---------------------------------------------------------------------------


class GibberishFraudster(PolicyBase):
    """
    Negative control: proposes nonsense ad copy to exercise the plausibility
    grader. The Auditor's Track-B score should collapse on this policy.
    """

    def __init__(self, *, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._seed = seed

    def reset(self) -> None:
        self._rng = random.Random(self._seed)

    def act(self, observation: Dict[str, Any]) -> FraudsterAction:
        remaining = int(observation.get("proposals_remaining", 0))
        if remaining <= 0:
            return FraudsterAction(
                action_type="commit_final",
                rationale="GibberishFraudster: budget gone.",
            )
        cat = _choose_allowed(
            self._rng.choice(_FRAUD_CATEGORIES + _LEGIT_CAMOUFLAGE_CATEGORIES),
            observation,
        )
        return FraudsterAction(
            action_type="propose_ad",
            ad_copy=self._random_gibberish(self._rng.randint(40, 120)),
            landing_page_blurb=self._random_gibberish(self._rng.randint(20, 80)),
            category=cat,
            targeting_summary="adults",
            rationale="GibberishFraudster: random bytes.",
        )

    def _random_gibberish(self, length: int) -> str:
        alphabet = string.ascii_lowercase + "     "  # include whitespace
        return "".join(self._rng.choice(alphabet) for _ in range(length))
