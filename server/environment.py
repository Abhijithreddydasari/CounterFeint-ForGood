"""
Core Ad Fraud Investigation Environment (Investigator role).

Implements the OpenEnv Environment interface for the Investigator agent:
reviewing a queue of ads, investigating them, and rendering verdicts under
a budget constraint.

In Round 1 this environment is used standalone (`AdFraudEnvironment` alias
preserves backwards compatibility).  In Round 2 it is driven by the
`RefereeEnvironment`, which pre-generates episodes and supplies a shared
`InvestigationToolRegistry` so Fraudster-proposed ads are reachable through
the same investigation code path.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..data.ad_generator import (
        TASK_CONFIGS,
        Ad,
        GeneratedEpisode,
        generate_episode,
    )
    from ..data.tool_registry import InvestigationToolRegistry
    from ..models import AdFraudState, AdReviewAction, AdReviewObservation
    from ..graders.base_grader import EpisodeRecord, LinkResult, VerdictResult, grade_episode
    from .evidence_ledger import build_evidence_ledger
except ImportError:
    from data.ad_generator import (
        TASK_CONFIGS,
        Ad,
        GeneratedEpisode,
        generate_episode,
    )
    from data.tool_registry import InvestigationToolRegistry
    from models import AdFraudState, AdReviewAction, AdReviewObservation
    from graders.base_grader import EpisodeRecord, LinkResult, VerdictResult, grade_episode
    from server.evidence_ledger import build_evidence_ledger

logger = logging.getLogger(__name__)

# Module-level store so the /grader endpoint can read the last score.
_last_grader_result: Dict[str, Any] = {}


def get_last_grader_result() -> Dict[str, Any]:
    return dict(_last_grader_result)


class InvestigatorEnvironment(
    Environment[AdReviewAction, AdReviewObservation, AdFraudState]
):
    """
    Ad fraud investigation environment (Investigator role).

    Each episode is a review session: the agent processes a queue of N ads
    within a limited action budget, choosing what to investigate and when
    to render verdicts. Unreviewed ads auto-approve at episode end.

    `reset()` accepts optional `episode` and `registry` kwargs so the
    Referee (or a test harness) can inject a pre-built `GeneratedEpisode`
    plus a shared `InvestigationToolRegistry`.  Without them, the
    environment generates its own synthetic episode and a fresh registry
    (the Round 1 behaviour).
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._state = AdFraudState(episode_id=str(uuid4()), step_count=0)
        self._episode: Optional[GeneratedEpisode] = None
        self._registry: Optional[InvestigationToolRegistry] = None
        self._verdicts: Dict[str, Dict[str, Any]] = {}
        self._links: List[Dict[str, Any]] = []
        self._investigations: Dict[str, List[str]] = {}
        # Total `investigate` attempts per ad — INCLUDING ones that the
        # env rejects (duplicate target, hit cap, etc.).

        self._investigation_attempts: Dict[str, int] = {}
        self._cumulative_reward: float = 0.0
        self._done = False
        self._last_feedback = ""
        self._focused_ad_id: Optional[str] = None
        self._queue_may_grow: bool = False

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> AdReviewObservation:
        task_id = kwargs.get("task_id", "task_1")
        if task_id not in TASK_CONFIGS:
            task_id = "task_1"

        injected_episode: Optional[GeneratedEpisode] = kwargs.get("episode")
        injected_registry: Optional[InvestigationToolRegistry] = kwargs.get("registry")
        self._queue_may_grow = bool(kwargs.get("queue_may_grow", False))

        if injected_episode is not None:
            self._episode = injected_episode
        else:
            effective_seed = (
                seed if seed is not None else hash(uuid4()) & 0xFFFFFFFF
            )
            self._episode = generate_episode(effective_seed, task_id)

        if injected_registry is not None:
            self._registry = injected_registry
        else:
            self._registry = InvestigationToolRegistry.from_episode(self._episode)

        config = self._episode.task_config
        self._state = AdFraudState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task_id,
            total_ads=config.queue_size,
            reviewed_count=0,
            remaining_budget=config.action_budget,
            verdicts={},
            grader_score=None,
        )
        self._verdicts = {}
        self._links = []
        self._investigations = {}
        self._investigation_attempts = {}
        self._cumulative_reward = 0.0
        self._done = False
        self._last_feedback = "Episode started. Review the ad queue and begin your investigation."
        self._focused_ad_id = self._episode.ads[0].ad_id if self._episode.ads else None

        return self._build_observation(reward=0.0, done=False)

    def step(
        self,
        action: AdReviewAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> AdReviewObservation:
        if self._done:
            return self._build_observation(
                reward=0.0, done=True,
                feedback_override="Episode is already complete. Call reset() to start a new episode.",
            )

        if self._episode is None:
            return self._build_observation(
                reward=0.0, done=False,
                feedback_override="Environment not initialized. Call reset() first.",
            )

        self._state.step_count += 1

        ad_ids = {a.ad_id for a in self._episode.ads}
        if action.ad_id not in ad_ids:
            self._last_feedback = f"Invalid ad_id '{action.ad_id}'. Valid IDs: {', '.join(sorted(ad_ids))}"
            return self._build_observation(reward=-0.05, done=False)

        if action.action_type == "investigate":
            reward = self._handle_investigate(action)
        elif action.action_type == "verdict":
            reward = self._handle_verdict(action)
        elif action.action_type == "link_accounts":
            reward = self._handle_link(action)
        else:
            self._last_feedback = f"Unknown action_type '{action.action_type}'."
            reward = -0.05

        self._cumulative_reward += reward

        done = self._check_done()
        if done and not self._done:
            end_reward = self._handle_episode_end()
            reward += end_reward
            self._cumulative_reward += end_reward
            self._done = True

        self._state.remaining_budget = max(0, self._state.remaining_budget)
        self._state.reviewed_count = len(self._verdicts)
        self._state.verdicts = {
            ad_id: v.get("verdict", "") for ad_id, v in self._verdicts.items()
        }

        return self._build_observation(reward=reward, done=self._done)

    @property
    def state(self) -> AdFraudState:
        return self._state

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    # Hard cap on TOTAL investigate attempts per ad — including the ones
    # the env rejects (duplicate target, ad already verdicted, etc.).
    
    _MAX_INVESTIGATION_ATTEMPTS_PER_AD = 5
    _MAX_INVESTIGATIONS_PER_AD = 3

    def _rotate_focus_off(self, current_ad_id: str) -> Optional[str]:
        """Move focus to the next pending ad that is NOT ``current_ad_id``.

        Returns the new focus ad_id, or ``None`` if there is no other
        pending ad. Used by ``_handle_investigate`` when the model is
        stuck looping on a single ad — rotating focus changes the
        ``Ad in Focus`` block in the next observation and breaks the
        prompt-anchoring effect that's keeping the 1.5B Investigator
        glued to one ad.
        """
        if self._episode is None:
            return None
        pending_others = [
            a.ad_id for a in self._episode.ads
            if a.ad_id not in self._verdicts and a.ad_id != current_ad_id
        ]
        if pending_others:
            self._focused_ad_id = pending_others[0]
            return pending_others[0]
        return None

    def _handle_investigate(self, action: AdReviewAction) -> float:
        ad_id = action.ad_id
        # Count EVERY attempt, including the ones we're about to reject —
        # this is the primary loop-breaking signal.
        attempts = self._investigation_attempts.get(ad_id, 0) + 1
        self._investigation_attempts[ad_id] = attempts

        if self._state.remaining_budget <= 0:
            self._last_feedback = "No budget remaining. You must render verdicts on remaining ads or end the episode."
            return -0.02

        if action.investigation_target is None:
            self._last_feedback = "investigation_target is required for action_type='investigate'."
            return -0.05

        if ad_id in self._verdicts:
            new_focus = self._rotate_focus_off(ad_id)
            self._last_feedback = (
                f"You already rendered a verdict on {ad_id}. "
                + (
                    f"Focus moved to {new_focus} — investigate or verdict that ad next."
                    if new_focus else
                    "All other ads are also verdicted; emit verdict actions for any remaining pending ads."
                )
            )
            return -0.02

        # Hard cap on TOTAL attempts (catches the duplicate-target spam
        # loop). Fires BEFORE the duplicate-target check so we hit the
        # cap first when the model is just spamming the same target.
        if attempts > self._MAX_INVESTIGATION_ATTEMPTS_PER_AD:
            new_focus = self._rotate_focus_off(ad_id)
            done_targets = self._investigations.get(ad_id, [])
            self._last_feedback = (
                f"REJECTED: {ad_id} has been probed {attempts} times — that's the cap. "
                f"STOP TRYING TO INVESTIGATE {ad_id}. "
                f"Issue a verdict on {ad_id} NOW (action_type='verdict', verdict in "
                "{approve, reject, escalate}). "
                + (
                    f"Successful pulls on {ad_id} so far: {', '.join(done_targets) or 'none'}. "
                    if done_targets else ""
                )
                + (
                    f"Focus moved to {new_focus}; investigate that ad if you'd rather "
                    "build evidence on a fresh ad first."
                    if new_focus else
                    "No other pending ads — verdict the remaining pending ads."
                )
            )
            return -0.10

        prev = self._investigations.setdefault(ad_id, [])
        target = action.investigation_target

        if target in prev:
            # Duplicate-target case. Don't spend budget. Surface the
            # un-pulled targets so the model has a concrete next pick.
            allowed = {
                "advertiser_history", "landing_page", "payment_method",
                "targeting_overlap", "campaign_structure", "policy_classifier",
            }
            remaining = sorted(allowed - set(prev))
            self._last_feedback = (
                f"You already investigated '{target}' for {ad_id} "
                f"(attempt {attempts}/{self._MAX_INVESTIGATION_ATTEMPTS_PER_AD}). "
                f"Either issue a verdict on {ad_id} now, OR pick a fresh target from "
                f"[{', '.join(remaining) if remaining else '(none left — verdict it)'}], "
                "OR investigate a different ad_id from the pending queue."
            )
            return -0.02

        # Sub-cap on successful pulls (tighter than the attempt cap).
        # Forces verdict after 3 successful investigations.
        if len(prev) >= self._MAX_INVESTIGATIONS_PER_AD:
            new_focus = self._rotate_focus_off(ad_id)
            self._last_feedback = (
                f"{ad_id} has reached the {self._MAX_INVESTIGATIONS_PER_AD}-investigation cap "
                f"(already pulled: {', '.join(prev)}). Issue a verdict on {ad_id} now "
                "(action_type='verdict' with verdict in {approve, reject, escalate})."
                + (f" Focus moved to {new_focus}." if new_focus else "")
            )
            return -0.05

        self._state.remaining_budget -= 1
        prev.append(target)
        # Note: focus is updated to the ad we just investigated only when
        # the Investigator hasn't already accumulated >=2 investigations
        # on it. Past 2 investigations on the same ad we keep focus on
        # the EXISTING focused ad (which may be a different one) so the
        # prompt doesn't re-anchor the model to an ad it should be
        # rendering a verdict on, NOT investigating further. This nudges
        # the policy toward "investigate twice → verdict → move on"
        # instead of looping investigations on a single ad.
        if len(prev) <= 2 or self._focused_ad_id is None:
            self._focused_ad_id = ad_id

        if self._registry is not None:
            findings = self._registry.lookup(ad_id, target)
        else:
            findings = self._episode.investigation_data.get(ad_id, {}).get(
                target, "No data available for this investigation type."
            )
        # Escalating per-investigate penalty so a runaway investigate
        # loop on a single ad gets progressively more expensive — pushes
        # the policy toward issuing a verdict once it has enough signal,
        # rather than burning steps re-checking the same ad.
        n_targets = len(prev)
        if n_targets <= 2:
            penalty = -0.02
        elif n_targets == 3:
            penalty = -0.05
        else:
            penalty = -0.10
        feedback_lines = [
            f"Investigation complete: {target} for {ad_id}.",
            f"--- Findings ---\n{findings}",
        ]
        if n_targets >= 2:
            feedback_lines.append(
                f"Note: you have now investigated {ad_id} {n_targets}x — "
                "issue a verdict on it instead of more investigations."
            )
        self._last_feedback = "\n".join(feedback_lines)
        return penalty

    def _handle_verdict(self, action: AdReviewAction) -> float:
        ad_id = action.ad_id

        if ad_id in self._verdicts:
            self._last_feedback = f"You already rendered a verdict on {ad_id}."
            return -0.02

        if action.verdict is None:
            self._last_feedback = "verdict field is required for action_type='verdict'."
            return -0.05

        confidence = action.confidence if action.confidence is not None else 0.5
        ad = self._get_ad(ad_id)
        ground_truth = ad.ground_truth_label if ad else "legit"
        severity = ad.severity if ad else 0.0

        self._verdicts[ad_id] = {
            "verdict": action.verdict,
            "confidence": confidence,
            "ground_truth": ground_truth,
        }

        reward = self._compute_verdict_reward(action.verdict, ground_truth, severity, confidence)

        pending = [a.ad_id for a in self._episode.ads if a.ad_id not in self._verdicts]
        self._last_feedback = (
            f"Verdict recorded for {ad_id}: {action.verdict} "
            f"(confidence: {confidence:.2f}). "
            f"{len(pending)} ad(s) remaining in queue."
        )

        if pending:
            self._focused_ad_id = pending[0]

        return reward

    def _handle_link(self, action: AdReviewAction) -> float:
        if action.linked_ad_id is None:
            self._last_feedback = "linked_ad_id is required for action_type='link_accounts'."
            return -0.05

        ad_ids = {a.ad_id for a in self._episode.ads}
        if action.linked_ad_id not in ad_ids:
            self._last_feedback = f"Invalid linked_ad_id '{action.linked_ad_id}'."
            return -0.05

        if action.ad_id == action.linked_ad_id:
            self._last_feedback = "Cannot link an ad to itself."
            return -0.05

        link_key = tuple(sorted([action.ad_id, action.linked_ad_id]))
        existing = {tuple(sorted([l["ad_id_1"], l["ad_id_2"]])) for l in self._links}
        if link_key in existing:
            self._last_feedback = f"Link between {action.ad_id} and {action.linked_ad_id} already recorded."
            return -0.02

        is_correct = self._check_link_correct(action.ad_id, action.linked_ad_id)

        self._links.append({
            "ad_id_1": action.ad_id,
            "ad_id_2": action.linked_ad_id,
            "reason": action.link_reason or "",
            "correct": is_correct,
        })

        self._last_feedback = (
            f"Network link recorded: {action.ad_id} <-> {action.linked_ad_id}. "
            f"Reason: {action.link_reason or 'not specified'}."
        )

        return 0.4 if is_correct else -0.25

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _compute_verdict_reward(
        self, verdict: str, ground_truth: str, severity: float, confidence: float
    ) -> float:
        if verdict == "reject" and ground_truth == "fraud":
            return 0.3 + 0.1 * severity
        elif verdict == "approve" and ground_truth == "legit":
            return 0.1
        elif verdict == "escalate" and ground_truth == "escalate":
            return 0.15
        elif verdict == "reject" and ground_truth == "legit":
            return -0.35
        elif verdict == "approve" and ground_truth == "fraud":
            return -0.5
        elif verdict == "escalate":
            return -0.05
        elif verdict == "approve" and ground_truth == "escalate":
            return -0.15
        elif verdict == "reject" and ground_truth == "escalate":
            return -0.1
        else:
            return -0.05

    def _handle_episode_end(self) -> float:
        """Apply end-of-episode adjustments for unreviewed ads, then delegate to graders."""
        unreviewed_fraud = 0
        for ad in self._episode.ads:
            if ad.ad_id not in self._verdicts:
                self._verdicts[ad.ad_id] = {
                    "verdict": "approve",
                    "confidence": 0.0,
                    "ground_truth": ad.ground_truth_label,
                    "auto_approved": True,
                }
                if ad.ground_truth_label == "fraud":
                    unreviewed_fraud += 1

        record = self._build_episode_record()
        grader_score = grade_episode(record)
        self._state.grader_score = grader_score

        reviewed_count = len([v for v in self._verdicts.values() if not v.get("auto_approved")])
        total_ads = len(self._episode.ads)
        total_correct = sum(
            1 for v in self._verdicts.values()
            if not v.get("auto_approved")
            and (
                (v["verdict"] == "reject" and v["ground_truth"] == "fraud")
                or (v["verdict"] == "approve" and v["ground_truth"] == "legit")
                or (v["verdict"] == "escalate" and v["ground_truth"] == "escalate")
            )
        )
        false_positives = sum(
            1 for v in self._verdicts.values()
            if not v.get("auto_approved")
            and v["verdict"] == "reject" and v["ground_truth"] == "legit"
        )
        false_negatives = sum(
            1 for v in self._verdicts.values()
            if not v.get("auto_approved")
            and v["verdict"] == "approve" and v["ground_truth"] == "fraud"
        )
        correct_links = sum(1 for l in self._links if l.get("correct"))
        incorrect_links = sum(1 for l in self._links if not l.get("correct"))

        global _last_grader_result
        _last_grader_result = {
            "task_id": self._state.task_id,
            "grader_score": grader_score,
            "episode_id": self._state.episode_id,
            "total_steps": self._state.step_count,
            "verdicts_rendered": reviewed_count,
            "correct_decisions": total_correct,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "auto_approved": total_ads - reviewed_count,
            "unreviewed_fraud": unreviewed_fraud,
            "network_links_correct": correct_links,
            "network_links_incorrect": incorrect_links,
        }

        feedback_lines = [
            f"Episode complete. Grader score: {grader_score:.3f}/1.000",
            f"Verdicts rendered: {reviewed_count}/{total_ads}",
            f"Correct decisions: {total_correct}/{reviewed_count}",
            f"False positives (legit rejected): {false_positives}",
            f"False negatives (fraud approved): {false_negatives}",
            f"Unreviewed ads auto-approved: {unreviewed_fraud}",
        ]
        if self._links:
            feedback_lines.append(
                f"Network links: {correct_links} correct, {incorrect_links} incorrect"
            )
        self._last_feedback = "\n".join(feedback_lines)

        return 0.0

    def _build_episode_record(self) -> EpisodeRecord:
        """Convert internal state into an EpisodeRecord for the grader."""
        verdict_results = []
        for ad in self._episode.ads:
            v = self._verdicts.get(ad.ad_id)
            if v:
                verdict_results.append(VerdictResult(
                    ad_id=ad.ad_id,
                    verdict=v["verdict"],
                    confidence=v.get("confidence", 0.5),
                    ground_truth=v["ground_truth"],
                    auto_approved=v.get("auto_approved", False),
                ))

        link_results = [
            LinkResult(ad_id_1=l["ad_id_1"], ad_id_2=l["ad_id_2"], correct=l["correct"])
            for l in self._links
        ]

        ads_metadata = [
            {"ad_id": ad.ad_id, "ground_truth": ad.ground_truth_label, "severity": ad.severity}
            for ad in self._episode.ads
        ]

        return EpisodeRecord(
            task_id=self._state.task_id,
            total_steps=self._state.step_count,
            action_budget=self._episode.task_config.action_budget,
            verdicts=verdict_results,
            links=link_results,
            ads_metadata=ads_metadata,
            n_fraud_rings=len(self._episode.fraud_rings),
            ring_sizes=[len(r.member_ad_ids) for r in self._episode.fraud_rings],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_done(self) -> bool:
        if self._episode is None:
            return True
        all_reviewed = all(
            ad.ad_id in self._verdicts for ad in self._episode.ads
        )
        steps_exhausted = self._state.step_count >= self._episode.task_config.action_budget
        return all_reviewed or steps_exhausted

    def _check_link_correct(self, ad_id_1: str, ad_id_2: str) -> bool:
        """Check if two ads share a fraud ring."""
        for ring in self._episode.fraud_rings:
            if ad_id_1 in ring.member_ad_ids and ad_id_2 in ring.member_ad_ids:
                return True
        return False

    def _get_ad(self, ad_id: str) -> Optional[Ad]:
        if self._episode is None:
            return None
        for ad in self._episode.ads:
            if ad.ad_id == ad_id:
                return ad
        return None

    def _build_observation(
        self,
        reward: float,
        done: bool,
        feedback_override: str | None = None,
    ) -> AdReviewObservation:
        feedback = feedback_override or self._last_feedback

        if self._episode is None:
            return AdReviewObservation(
                done=done,
                reward=reward,
                queue_summary="No episode loaded.",
                current_ad_info="",
                investigation_findings="",
                verdict_history_summary="",
                feedback=feedback,
                available_ads=[],
                queue_status={},
                queue_may_grow=self._queue_may_grow,
            )

        config = self._episode.task_config
        pending = [a for a in self._episode.ads if a.ad_id not in self._verdicts]
        reviewed = [a for a in self._episode.ads if a.ad_id in self._verdicts]

        steps_remaining = max(0, config.action_budget - self._state.step_count)

        # Surface the action-budget-vs-pending-ads pressure as a prominent
        # feedback banner. We use TWO triggers because they catch
        # different failure modes:
        #
        #   * `steps_remaining <= 2 * n_pending` — fires when you have
        #     less than ~2 actions per pending ad left. Catches the
        #     "looped on ad_001 for 20 steps and now there isn't time
        #     to verdict everyone" case the 1.5B Investigator falls into.
        #     This trigger doesn't depend on investigation budget — even
        #     with full investigation budget, if step budget is tight,
        #     verdicts MUST start now.
        #
        #   * `remaining_investigation_budget <= n_pending` — original
        #     trigger. Catches the case where investigation actions
        #     have actually been spent on real pulls, not on rejected
        #     duplicates.
        
        if not done and pending and feedback_override is None:
            n_pending = len(pending)
            budget = self._state.remaining_budget
            steps_left = steps_remaining
            steps_pressure = steps_left <= 2 * n_pending
            budget_pressure = budget <= n_pending
            if steps_pressure or budget_pressure:
                # Pick the tighter wording.
                if steps_pressure:
                    pressure_line = (
                        f"BUDGET PRESSURE: only {steps_left} step(s) left in this "
                        f"episode for {n_pending} pending ad(s). Stop investigating "
                        f"and START VERDICTING — issue verdict actions "
                        f"(action_type='verdict', verdict in approve/reject/escalate) "
                        f"on the pending ads using whatever evidence you have. "
                        f"Unverdict-ed ads auto-approve at audit and tank the score."
                    )
                else:
                    pressure_line = (
                        f"BUDGET PRESSURE: only {budget} investigation(s) left for "
                        f"{n_pending} pending ad(s). Stop investigating and START "
                        f"VERDICTING — issue verdict actions on the pending ads using "
                        f"whatever evidence you have (approve, reject, or escalate). "
                        f"Unverdict-ed ads auto-approve at audit time and tank the score."
                    )
                feedback = (
                    f"{pressure_line}\n\n{feedback}" if feedback else pressure_line
                )
        queue_summary = (
            f"Task: {config.name} ({config.difficulty})\n"
            f"Total ads: {config.queue_size} | "
            f"Reviewed: {len(reviewed)} | "
            f"Pending: {len(pending)} | "
            f"Steps remaining: {steps_remaining}/{config.action_budget} | "
            f"Investigation budget: {self._state.remaining_budget} | "
            f"Step: {self._state.step_count}"
        )

        current_ad_info = ""
        if self._focused_ad_id and not done:
            ad = self._get_ad(self._focused_ad_id)
            if ad and ad.ad_id not in self._verdicts:
                signals = ", ".join(ad.initial_risk_signals) if ad.initial_risk_signals else "None"
                investigated = self._investigations.get(ad.ad_id, [])
                attempts = self._investigation_attempts.get(ad.ad_id, 0)

                _all_targets = [
                    "advertiser_history", "landing_page", "payment_method",
                    "targeting_overlap", "campaign_structure", "policy_classifier",
                ]
                exhausted = [t for t in _all_targets if t in investigated]
                fresh = [t for t in _all_targets if t not in investigated]
                exhausted_line = (
                    f"ALREADY-EXHAUSTED targets for {ad.ad_id} (do NOT repeat): "
                    f"{', '.join(exhausted) if exhausted else 'none yet'}"
                )
                fresh_line = (
                    f"FRESH targets for {ad.ad_id} (you may pick one of these or verdict): "
                    f"{', '.join(fresh) if fresh else 'none — verdict this ad now'}"
                )

                # Contextual metadata visible before investigation
                profile = self._episode.advertiser_profiles.get(ad.ad_id)
                meta_lines = []
                if profile:
                    meta_lines.append(f"Advertiser country: {profile.country}")
                    meta_lines.append(f"Account age: {profile.account_age_days} days")
                    if profile.account_age_days < 30 and getattr(
                        self._episode.task_config, "stealth_level", 0
                    ) < 2:
                        meta_lines.append("Flag: New account (< 30 days)")
                context_meta = "\n".join(meta_lines)

                from ..data.meta_policy_taxonomy import lookup as _meta_lookup

                policy_entry = _meta_lookup(ad.category)
                meta_policy_line = (
                    f"Meta policy lens: {policy_entry.citation_id} — "
                    f"{policy_entry.section} > {policy_entry.subsection}"
                )

                # If the model has spammed this ad past N attempts,
                # surface that loud-and-clear at the top of the focus
                # block so the next observation pushes harder toward
                # verdict instead of silently re-anchoring.
                stuck_banner = ""
                if attempts >= 3:
                    stuck_banner = (
                        f"STUCK ON {ad.ad_id}: you have attempted {attempts} "
                        f"investigate actions on this ad. ISSUE A VERDICT NOW.\n"
                    )

                current_ad_info = (
                    f"{stuck_banner}"
                    f"=== Ad in Focus: {ad.ad_id} ===\n"
                    f"Category: {ad.category}\n"
                    f"{meta_policy_line}\n"
                    f"Ad copy: \"{ad.ad_copy}\"\n"
                    f"Targeting: {ad.targeting_summary}\n"
                    f"Initial risk signals: {signals}\n"
                    f"{context_meta}\n"
                    f"{exhausted_line}\n"
                    f"{fresh_line}"
                )

        investigation_findings = ""
        for ad_id, targets in self._investigations.items():
            for target in targets:
                if self._registry is not None:
                    finding = self._registry.lookup(ad_id, target)
                else:
                    finding = self._episode.investigation_data.get(ad_id, {}).get(target, "")
                if finding and not finding.startswith("No data") and not finding.startswith("Unknown"):
                    investigation_findings += f"\n[{ad_id} / {target}]\n{finding}\n"

        manual_verdicts = {
            ad_id: v for ad_id, v in self._verdicts.items()
            if not v.get("auto_approved")
        }
        if manual_verdicts:
            counts = {"approve": 0, "reject": 0, "escalate": 0}
            by_decision = {"approve": [], "reject": [], "escalate": []}
            for ad_id, v in manual_verdicts.items():
                counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
                by_decision[v["verdict"]].append(ad_id)
            summary_parts = [f"{c} {k}" for k, c in counts.items() if c > 0]
            verdict_lines = [
                f"Reviewed {len(manual_verdicts)} ad(s): {', '.join(summary_parts)}."
            ]
            for decision in ("reject", "approve", "escalate"):
                if by_decision[decision]:
                    verdict_lines.append(
                        f"  {decision}: {', '.join(by_decision[decision])}"
                    )
            verdict_history_summary = "\n".join(verdict_lines)
        else:
            verdict_history_summary = "No verdicts yet."

        available_ads = [a.ad_id for a in pending]

        queue_status = {
            "total_ads": config.queue_size,
            "reviewed": len(reviewed),
            "pending": len(pending),
            "investigation_budget": self._state.remaining_budget,
            "steps_remaining": steps_remaining,
            "step": self._state.step_count,
            "task_id": config.task_id,
        }

        evidence_ledger = self._build_evidence_ledger()
        queue_digest = self._build_queue_digest(pending)
        decided_ads = self._build_decided_ads()

        return AdReviewObservation(
            done=done,
            reward=reward,
            queue_summary=queue_summary,
            current_ad_info=current_ad_info,
            investigation_findings=investigation_findings.strip(),
            verdict_history_summary=verdict_history_summary,
            feedback=feedback,
            available_ads=available_ads,
            queue_status=queue_status,
            queue_may_grow=self._queue_may_grow,
            evidence_ledger=evidence_ledger,
            queue_digest=queue_digest,
            decided_ads=decided_ads,
        )

    # Curated columns surfaced in the no-investigation queue digest so
    # the Investigator has SOMETHING to triage on for every pending ad,
    # not just the focused one. Mix of:
    #
    #   * Discriminative (used for link_accounts decisions when shared
    #     across ads):  payment_type, registrar, domain
    #   * Decoy / non-discriminative (deliberately included so the
    #     policy must learn not to weight them):
    #     category, country, account_age_days
    #
    # Total of ~6 columns × 12 ads ≈ 720 chars worst case, well within
    # prompt budget. payment_id / advertiser_id / targeting_fingerprint
    # are NOT exposed here — those are the high-signal columns that
    # MUST require an explicit investigate to avoid trivialising the
    # task; they remain in the evidence_ledger only.
    _QUEUE_DIGEST_MAX_ADS = 12

    def _build_queue_digest(
        self, pending_ads: List[Ad]
    ) -> List[Dict[str, Any]]:
        if self._episode is None or not pending_ads:
            return []

        lp_map = getattr(self._episode, "landing_pages", {}) or {}
        profiles = getattr(self._episode, "advertiser_profiles", {}) or {}

        rows: List[Dict[str, Any]] = []
        for ad in pending_ads[: self._QUEUE_DIGEST_MAX_ADS]:
            row: Dict[str, Any] = {
                "ad_id": ad.ad_id,
                "category": ad.category,
            }
            profile = profiles.get(ad.ad_id)
            if profile is not None:
                row["country"] = profile.country
                row["account_age_days"] = profile.account_age_days
                row["payment_type"] = profile.payment_method_type
            lp = lp_map.get(ad.ad_id)
            if lp is not None:
                row["domain"] = getattr(lp, "domain", None)
                row["registrar"] = getattr(lp, "registrar", None)
            rows.append({k: v for k, v in row.items() if v is not None})
        return rows

    def _build_evidence_ledger(self) -> Dict[str, Dict[str, Any]]:
        """Assemble a per-ad structured evidence table for the Investigator.

        Delegates to :func:`build_evidence_ledger` so the Referee can reuse
        the exact same extraction logic (with a different ``ad_ids``
        selection) when building the Fraudster's ``my_proposal_signals``.
        """
        if self._episode is None:
            return {}
        candidate_ad_ids: set[str] = set(self._investigations.keys())
        if self._focused_ad_id:
            candidate_ad_ids.add(self._focused_ad_id)
        return build_evidence_ledger(
            episode=self._episode,
            registry=self._registry,
            ad_ids=candidate_ad_ids,
            investigations=self._investigations,
        )

    def _build_decided_ads(self) -> list[Dict[str, Any]]:
        """Build a per-decided-ad summary with verdict + key signals.

        Each entry carries the verdict, confidence, and a curated mix of
        discriminative + decoy parameters from the evidence ledger. This
        gives the Investigator structured memory of past decisions so it
        can detect cross-ad collisions for link_accounts.
        """
        if self._episode is None:
            return []

        decided_ad_ids = [
            ad_id for ad_id in self._verdicts
            if not self._verdicts[ad_id].get("auto_approved")
        ]
        if not decided_ad_ids:
            return []

        ledger = build_evidence_ledger(
            episode=self._episode,
            registry=self._registry,
            ad_ids=decided_ad_ids,
            investigations=self._investigations,
        )

        rows: list[Dict[str, Any]] = []
        for ad_id in decided_ad_ids:
            v = self._verdicts[ad_id]
            entry: Dict[str, Any] = {
                "ad_id": ad_id,
                "verdict": v.get("verdict", "?"),
                "confidence": v.get("confidence", 0.5),
            }
            signals = ledger.get(ad_id, {})
            entry.update(signals)
            rows.append(entry)
        return rows

    # ------------------------------------------------------------------
    # Referee integration hooks
    # ------------------------------------------------------------------

    def notify_queue_grew(self, new_ad_id: str) -> None:
        """
        Called by the Referee after `extend_episode_with_proposal` adds a
        new ad to the shared episode + registry.  Updates the Investigator's
        view of queue size and refocuses on the new ad if the Investigator
        is idle.
        """
        if self._episode is None:
            return
        self._state.total_ads = len(self._episode.ads)
        if self._focused_ad_id is None or self._focused_ad_id in self._verdicts:
            self._focused_ad_id = new_ad_id

    @property
    def episode(self) -> Optional[GeneratedEpisode]:
        """Read-only access to the loaded episode (used by the Referee)."""
        return self._episode

    @property
    def registry(self) -> Optional[InvestigationToolRegistry]:
        """Read-only access to the shared tool registry."""
        return self._registry

    @property
    def verdicts(self) -> Dict[str, Dict[str, Any]]:
        """Read-only snapshot of verdicts recorded so far (Referee/auditor)."""
        return dict(self._verdicts)

    def auto_approve_pending(self) -> int:
        """Materialize the benchmark's end-of-episode policy for pending ads."""
        if self._episode is None:
            return 0
        added = 0
        for ad in self._episode.ads:
            if ad.ad_id in self._verdicts:
                continue
            self._verdicts[ad.ad_id] = {
                "verdict": "approve",
                "confidence": 0.0,
                "ground_truth": ad.ground_truth_label,
                "auto_approved": True,
            }
            added += 1
        return added

    @property
    def investigations(self) -> Dict[str, List[str]]:
        """Read-only snapshot of investigation targets pulled per ad."""
        return {k: list(v) for k, v in self._investigations.items()}

    @property
    def links(self) -> List[Dict[str, Any]]:
        """Read-only snapshot of recorded network links."""
        return list(self._links)


# Backwards-compatible alias.  Round 1 code, tests, clients, and external
# integrations import `AdFraudEnvironment` directly; keeping the symbol
# means the rename is zero-breakage.
AdFraudEnvironment = InvestigatorEnvironment
