"""
RefereeEnvironment — the central multi-agent orchestrator for CounterFeint.

Owns a turn-based state machine with three roles:

  - Fraudster   proposes / modifies ads      (actions: propose_ad, modify_pending_ad, end_turn, commit_final)
  - Investigator reviews ads                  (actions: investigate, verdict, link_accounts)
  - Auditor     audits the trace post-hoc    (actions: flag_investigator, flag_fraudster, submit_audit_report)

All three WebSocket endpoints (`/ws/fraudster`, `/ws/investigator`, `/ws/auditor`)
share a single `RefereeEnvironment` instance per match, so state mutations
from one role are immediately visible to the others.

State machine:

    fraudster_turn  ─end_turn──────►  investigator_turn ─turn_cap/all_decided──► fraudster_turn  (next round)
          │                                   │
          ├─commit_final───►   audit_phase   ◄┘
          │                                   │
          └─action_cap──► investigator_turn   │                               max_rounds / budget / commit_final
                                              └──────── audit_phase → done ◄─────────────────

Phase 1 keeps the Auditor a no-op scaffold (flags accepted, report accepted, but
graders don't consume them yet).  Phase 2A/B/C plug in real audit logic.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Dict, List, Literal, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation

try:
    from ..data.ad_generator import (
        TASK_CONFIGS,
        Ad,
        GeneratedEpisode,
        generate_episode,
    )
    from ..data.episode_loader import extend_episode_with_proposal
    from ..data.tool_registry import INVESTIGATION_TARGETS, InvestigationToolRegistry
    from ..graders.auditor_track_a import (
        investigator_audit_score as track_a_score,
        run_track_a,
    )
    from ..graders.base_grader import (
        EpisodeRecord,
        LinkResult,
        VerdictResult,
        grade_episode,
    )
    from ..graders.multi_agent_rewards import (
        RewardInputs,
        compute_episode_rewards,
    )
    from ..graders.plausibility_score import compute_queue_plausibility
    from ..models import (
        AdFraudState,
        AdReviewAction,
        AdReviewObservation,
        AuditFlag,
        AuditorAction,
        AuditorObservation,
        AuditReport,
        FraudsterAction,
        FraudsterObservation,
        RefereeState,
    )
    from .environment import InvestigatorEnvironment
    from .evidence_ledger import build_evidence_ledger
except ImportError:
    from data.ad_generator import (
        TASK_CONFIGS,
        Ad,
        GeneratedEpisode,
        generate_episode,
    )
    from data.episode_loader import extend_episode_with_proposal
    from data.tool_registry import INVESTIGATION_TARGETS, InvestigationToolRegistry
    from graders.auditor_track_a import (
        investigator_audit_score as track_a_score,
        run_track_a,
    )
    from graders.base_grader import (
        EpisodeRecord,
        LinkResult,
        VerdictResult,
        grade_episode,
    )
    from graders.multi_agent_rewards import (
        RewardInputs,
        compute_episode_rewards,
    )
    from graders.plausibility_score import compute_queue_plausibility
    from models import (
        AdFraudState,
        AdReviewAction,
        AdReviewObservation,
        AuditFlag,
        AuditorAction,
        AuditorObservation,
        AuditReport,
        FraudsterAction,
        FraudsterObservation,
        RefereeState,
    )
    from server.environment import InvestigatorEnvironment
    from server.evidence_ledger import build_evidence_ledger


logger = logging.getLogger(__name__)

Phase = Literal["fraudster_turn", "investigator_turn", "audit_phase", "done"]
Role = Literal["fraudster", "investigator", "auditor"]

# Module-level grader result for parity with the Investigator env (/grader endpoint).
_last_grader_result: Dict[str, Any] = {}


def get_last_grader_result() -> Dict[str, Any]:
    return dict(_last_grader_result)


# Default categories the Fraudster can declare.  Combines plausible legit
# categories (so a sophisticated Fraudster can camouflage) with fraud
# templates (so it can propose obvious-fraud or borderline ads).
DEFAULT_ALLOWED_CATEGORIES: Tuple[str, ...] = (
    # Legit camouflage categories
    "ecommerce",
    "saas",
    "local_service",
    "education",
    "fitness",
    # Fraud / borderline templates
    "fake_giveaway",
    "counterfeit_goods",
    "miracle_cure",
    "advance_fee",
    "fake_crypto",
    "celebrity_endorsement_fraud",
    "clone_brand",
    "gray_area_supplements",
    "network_crypto",
    "network_ecommerce",
    "network_fintech",
    "network_health",
)


class RefereeEnvironment(Environment[Action, Observation, RefereeState]):
    """
    Multi-agent referee. Implements the OpenEnv `Environment` contract with
    a generic `Action`/`Observation` typing — each WebSocket route passes
    role-specific subclasses into `step()` via the `role` kwarg.

    Role-aware entry points (preferred):
      - `reset_match(seed, task_id, episode_id, **knobs)`
      - `step_as_fraudster(action)`
      - `step_as_investigator(action)`
      - `step_as_auditor(action)`
      - `build_<role>_observation()`
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    # Default knobs (overridable via reset kwargs).
    DEFAULT_MAX_ROUNDS = 4
    DEFAULT_MAX_PROPOSALS = 5
    # Per-turn action caps. Bumped from (3, 6) to (4, 10) so the
    # Investigator can comfortably investigate 2-3 ads per turn AND issue
    # verdicts in the same turn without being force-cut to the auditor
    # mid-thought (the previous (6) cap was triggering the
    # ``max_rounds`` short-circuit on the final round before the
    # Investigator could close out pending verdicts).
    DEFAULT_MAX_FRAUDSTER_ACTIONS_PER_TURN = 4
    DEFAULT_MAX_INVESTIGATOR_ACTIONS_PER_TURN = 10

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        super().__init__()
        self._match_id: str = str(uuid4())
        self._task_id: str = "task_1"
        self._rng = random.Random()

        self._investigator = InvestigatorEnvironment()
        self._episode: Optional[GeneratedEpisode] = None
        self._registry: Optional[InvestigationToolRegistry] = None

        self._phase: Phase = "fraudster_turn"
        self._round_number: int = 0
        self._max_rounds: int = self.DEFAULT_MAX_ROUNDS
        self._max_proposals: int = self.DEFAULT_MAX_PROPOSALS
        self._max_fraudster_actions_per_turn: int = (
            self.DEFAULT_MAX_FRAUDSTER_ACTIONS_PER_TURN
        )
        self._max_investigator_actions_per_turn: int = (
            self.DEFAULT_MAX_INVESTIGATOR_ACTIONS_PER_TURN
        )
        self._allowed_categories: List[str] = list(DEFAULT_ALLOWED_CATEGORIES)

        self._proposals_used: int = 0
        self._actions_this_turn: int = 0

        # Per-role logs (consumed by the Auditor).
        self._fraudster_log: List[Dict[str, Any]] = []
        self._investigator_log: List[Dict[str, Any]] = []
        self._audit_flags: List[AuditFlag] = []
        self._audit_report: Optional[AuditReport] = None

        self._fraudster_committed: bool = False
        self._done: bool = False
        self._end_reason: Optional[str] = None

        self._fraudster_reward_total: float = 0.0
        self._investigator_reward_total: float = 0.0
        self._auditor_reward_total: float = 0.0
        self._grader_score: Optional[float] = None
        self._per_ad_plausibility: Dict[str, float] = {}
        self._audit_ground_truth: Dict[str, int] = {}

        self._last_feedback: Dict[Role, str] = {
            "fraudster": "",
            "investigator": "",
            "auditor": "",
        }

        # Proposal slot_index -> ad_id map, so the Fraudster can modify its
        # own prior proposals without knowing the Referee's ad_id scheme.
        self._proposal_slot_to_ad_id: Dict[int, str] = {}
        # Set inside ``_fraudster_propose_ad`` on success, consumed (and
        # cleared) by ``_serialize_fraudster_action`` so the audit log entry
        # for a propose_ad always carries the resolved ``ad_id`` and slot
        # the env actually allocated, not just the LLM's raw payload (which
        # has no ad_id field for propose_ad).
        self._last_proposed_ad_id: Optional[str] = None
        self._last_proposed_slot: Optional[int] = None

    # ------------------------------------------------------------------
    # OpenEnv surface (generic)
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Generic reset. Returns the *Fraudster* observation because the
        Fraudster always goes first.  The role-specific endpoints can
        also call `build_<role>_observation()` directly.
        """
        self.reset_match(seed=seed, episode_id=episode_id, **kwargs)
        return self.build_fraudster_observation()

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Role-aware generic step. Expects `role` in kwargs, dispatches to
        the appropriate role-specific step method, and returns that role's
        observation.
        """
        role: Optional[Role] = kwargs.get("role")
        if role == "fraudster":
            return self.step_as_fraudster(action)  # type: ignore[arg-type]
        if role == "investigator":
            return self.step_as_investigator(action)  # type: ignore[arg-type]
        if role == "auditor":
            return self.step_as_auditor(action)  # type: ignore[arg-type]
        raise ValueError(
            "RefereeEnvironment.step(action, role=...) requires a role of "
            "'fraudster', 'investigator', or 'auditor'."
        )

    @property
    def state(self) -> RefereeState:
        inv_state = self._investigator.state
        return RefereeState(
            episode_id=self._match_id,
            step_count=(
                len(self._fraudster_log)
                + len(self._investigator_log)
                + len(self._audit_flags)
            ),
            task_id=self._task_id,
            phase=self._phase,
            round_number=self._round_number,
            max_rounds=self._max_rounds,
            action_budget=(
                self._episode.task_config.action_budget if self._episode else 0
            ),
            proposals_used=self._proposals_used,
            max_proposals=self._max_proposals,
            actions_this_turn=self._actions_this_turn,
            max_actions_per_turn=(
                self._max_fraudster_actions_per_turn
                if self._phase == "fraudster_turn"
                else self._max_investigator_actions_per_turn
            ),
            investigator_state=inv_state.model_dump() if inv_state else {},
            fraudster_proposals=list(self._fraudster_log),
            investigator_action_log=list(self._investigator_log),
            fraudster_committed=self._fraudster_committed,
            audit_report=(
                self._audit_report.model_dump() if self._audit_report else None
            ),
            fraudster_reward=self._fraudster_reward_total,
            investigator_reward=self._investigator_reward_total,
            auditor_reward=self._auditor_reward_total,
            grader_score=self._grader_score,
            end_reason=self._end_reason,
        )

    # ------------------------------------------------------------------
    # Match setup
    # ------------------------------------------------------------------

    def reset_match(
        self,
        *,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        max_rounds: Optional[int] = None,
        max_proposals: Optional[int] = None,
        max_fraudster_actions_per_turn: Optional[int] = None,
        max_investigator_actions_per_turn: Optional[int] = None,
        allowed_categories: Optional[List[str]] = None,
        episode: Optional[GeneratedEpisode] = None,
    ) -> None:
        """Initialize a fresh match. Sets phase to `fraudster_turn`, round 1."""
        self._match_id = episode_id or str(uuid4())
        self._task_id = task_id or "task_1"
        if self._task_id not in TASK_CONFIGS:
            self._task_id = "task_1"

        effective_seed = (
            seed if seed is not None else hash(uuid4()) & 0xFFFFFFFF
        )
        self._rng = random.Random(effective_seed)

        # Resolve each knob with precedence: explicit kwarg > TaskConfig curriculum > class default.
        task_cfg = TASK_CONFIGS[self._task_id]

        def _resolve(arg_value: Optional[int], cfg_attr: str, default: int) -> int:
            if arg_value is not None:
                return arg_value
            cfg_val = getattr(task_cfg, cfg_attr, None)
            return cfg_val if cfg_val is not None else default

        self._max_rounds = _resolve(max_rounds, "max_rounds", self.DEFAULT_MAX_ROUNDS)
        self._max_proposals = _resolve(
            max_proposals, "max_proposals", self.DEFAULT_MAX_PROPOSALS
        )
        self._max_fraudster_actions_per_turn = _resolve(
            max_fraudster_actions_per_turn,
            "max_fraudster_actions_per_turn",
            self.DEFAULT_MAX_FRAUDSTER_ACTIONS_PER_TURN,
        )
        self._max_investigator_actions_per_turn = _resolve(
            max_investigator_actions_per_turn,
            "max_investigator_actions_per_turn",
            self.DEFAULT_MAX_INVESTIGATOR_ACTIONS_PER_TURN,
        )

        cfg_categories = getattr(task_cfg, "allowed_fraud_categories", None)
        if allowed_categories is not None:
            self._allowed_categories = list(allowed_categories)
        elif cfg_categories:
            self._allowed_categories = list(cfg_categories)
        else:
            self._allowed_categories = list(DEFAULT_ALLOWED_CATEGORIES)

        if episode is not None:
            self._episode = episode
        else:
            self._episode = generate_episode(effective_seed, self._task_id)

        self._registry = InvestigationToolRegistry.from_episode(self._episode)

        self._investigator.reset(
            seed=effective_seed,
            episode_id=self._match_id,
            task_id=self._task_id,
            episode=self._episode,
            registry=self._registry,
            queue_may_grow=True,
        )

        self._phase = "fraudster_turn"
        self._round_number = 1
        self._proposals_used = 0
        self._actions_this_turn = 0
        self._fraudster_log = []
        self._investigator_log = []
        self._audit_flags = []
        self._audit_report = None
        self._fraudster_committed = False
        self._done = False
        self._end_reason = None
        self._fraudster_reward_total = 0.0
        self._investigator_reward_total = 0.0
        self._auditor_reward_total = 0.0
        self._grader_score = None
        self._per_ad_plausibility = {}
        self._audit_ground_truth = {}
        self._proposal_slot_to_ad_id = {}
        self._last_proposed_ad_id = None
        self._last_proposed_slot = None
        self._last_feedback = {
            "fraudster": (
                f"Match started. Round 1 of {self._max_rounds}. "
                f"You may propose up to {self._max_proposals} ads total, "
                f"{self._max_fraudster_actions_per_turn} actions per turn."
            ),
            "investigator": (
                "Waiting for Fraudster to finish their turn. The ad queue may "
                "grow during this episode as the Fraudster proposes new ads."
            ),
            "auditor": "Match in progress. Waiting for audit phase.",
        }

    # ------------------------------------------------------------------
    # Fraudster step handler
    # ------------------------------------------------------------------

    def step_as_fraudster(self, action: FraudsterAction) -> FraudsterObservation:
        self._guard_phase("fraudster_turn", role="fraudster")
        assert self._episode is not None and self._registry is not None

        reward = 0.0
        feedback_parts: List[str] = []
        action_type = action.action_type

        if action_type == "propose_ad":
            reward, msg = self._fraudster_propose_ad(action)
            feedback_parts.append(msg)
            self._actions_this_turn += 1

        elif action_type == "modify_pending_ad":
            reward, msg = self._fraudster_modify_pending_ad(action)
            feedback_parts.append(msg)
            self._actions_this_turn += 1

        elif action_type == "end_turn":
            feedback_parts.append("Fraudster ended turn. Control passes to Investigator.")
            self._transition(to="investigator_turn", note="fraudster end_turn")
            reward = 0.0

        elif action_type == "commit_final":
            feedback_parts.append(
                "Fraudster committed (no more proposals). Jumping to audit phase."
            )
            self._fraudster_committed = True
            self._end_reason = "commit_final"
            self._transition(to="audit_phase", note="fraudster commit_final")
            reward = 0.0

        else:
            feedback_parts.append(f"Unknown Fraudster action_type '{action_type}'.")
            reward = -0.05

        self._fraudster_reward_total += reward
        self._last_feedback["fraudster"] = " ".join(feedback_parts).strip()
        self._fraudster_log.append(self._serialize_fraudster_action(action, reward))

        # Auto-transition guards.
        if (
            self._phase == "fraudster_turn"
            and action_type in ("propose_ad", "modify_pending_ad")
            and self._actions_this_turn >= self._max_fraudster_actions_per_turn
        ):
            self._transition(to="investigator_turn", note="fraudster action cap")

        if (
            self._phase == "fraudster_turn"
            and action_type == "propose_ad"
            and self._proposals_used >= self._max_proposals
        ):
            self._last_feedback["fraudster"] += (
                " Proposal budget exhausted — control will pass to Investigator."
            )
            self._transition(to="investigator_turn", note="proposal budget exhausted")

        return self.build_fraudster_observation(reward=reward)

    def _fraudster_propose_ad(self, action: FraudsterAction) -> Tuple[float, str]:
        if self._proposals_used >= self._max_proposals:
            return -0.05, (
                f"Proposal budget exhausted ({self._proposals_used}/{self._max_proposals})."
            )
        if not action.ad_copy or not action.ad_copy.strip():
            return -0.05, "propose_ad requires non-empty `ad_copy`."
        if not action.category:
            return -0.05, "propose_ad requires `category`."
        if action.category not in self._allowed_categories:
            return -0.05, (
                f"category '{action.category}' not in allowed_categories. "
                f"Use one of: {', '.join(self._allowed_categories)}."
            )
        assert self._episode is not None and self._registry is not None

        proposal_seed = self._rng.randint(0, 2**31 - 1)
        ad = extend_episode_with_proposal(
            episode=self._episode,
            registry=self._registry,
            seed=proposal_seed,
            ad_copy=action.ad_copy,
            category=action.category,
            landing_page_blurb=action.landing_page_blurb,
            targeting_summary=action.targeting_summary,
        )
        slot_index = self._proposals_used
        self._proposal_slot_to_ad_id[slot_index] = ad.ad_id
        self._proposals_used += 1
        # Stash so ``_serialize_fraudster_action`` can attach the resolved
        # ``ad_id`` + ``slot_index`` to this propose_ad's audit log entry
        # (the FraudsterAction itself doesn't carry these — they're env-
        # allocated). Without this the auditor sees ``ad_id=None`` for
        # every propose_ad, which then poisons downstream Track B checks
        # (e.g. ``intrinsic_consistency_check`` cannot key flags onto an
        # ad and ``cross_ad_consistency_audit`` cannot dedupe by ad_id).
        self._last_proposed_ad_id = ad.ad_id
        self._last_proposed_slot = slot_index
        self._investigator.notify_queue_grew(ad.ad_id)

        feedback = (
            f"Proposal #{slot_index + 1} accepted: ad_id={ad.ad_id}, category={ad.category}. "
            f"Queue is now {len(self._episode.ads)} ads."
        )
        return 0.02, feedback

    def _fraudster_modify_pending_ad(self, action: FraudsterAction) -> Tuple[float, str]:
        if action.slot_index is None:
            return -0.05, "modify_pending_ad requires `slot_index`."
        slot = action.slot_index
        if slot not in self._proposal_slot_to_ad_id:
            return -0.05, f"Unknown slot_index {slot}. Propose an ad first."

        ad_id = self._proposal_slot_to_ad_id[slot]
        assert self._episode is not None and self._registry is not None

        # Locked once the Investigator has already rendered a verdict.
        already_decided = self._investigator.verdicts.get(ad_id, {}).get("verdict")
        if already_decided:
            return (
                -0.05,
                f"Cannot modify {ad_id}: Investigator already rendered verdict "
                f"'{already_decided}'.",
            )

        target_ad: Optional[Ad] = None
        for a in self._episode.ads:
            if a.ad_id == ad_id:
                target_ad = a
                break
        if target_ad is None:
            return -0.05, f"Internal error: ad {ad_id} not in episode."

        changes: List[str] = []
        if action.new_ad_copy is not None and action.new_ad_copy.strip():
            target_ad.ad_copy = action.new_ad_copy.strip()[:2000]
            changes.append("ad_copy")

        if action.new_landing_page_blurb is not None and action.new_landing_page_blurb.strip():
            lp = self._episode.landing_pages.get(ad_id)
            if lp is not None:
                from dataclasses import replace
                new_lp = replace(
                    lp, content_summary=action.new_landing_page_blurb.strip()[:2000]
                )
                self._episode.landing_pages[ad_id] = new_lp
                updated_text = new_lp.to_investigation_text()
                self._episode.investigation_data.setdefault(ad_id, {})["landing_page"] = updated_text
                self._registry.update_ad(ad_id, {"landing_page": updated_text})
                changes.append("landing_page")

        if not changes:
            return -0.02, "modify_pending_ad had nothing to change."
        return 0.01, f"Modified {ad_id} fields: {', '.join(changes)}."

    # ------------------------------------------------------------------
    # Investigator step handler
    # ------------------------------------------------------------------

    def step_as_investigator(self, action: AdReviewAction) -> AdReviewObservation:
        self._guard_phase("investigator_turn", role="investigator")
        assert self._episode is not None

        obs = self._investigator.step(action)
        reward = float(obs.reward or 0.0)
        self._investigator_reward_total += reward

        self._investigator_log.append(self._serialize_investigator_action(action, obs))
        self._actions_this_turn += 1
        self._last_feedback["investigator"] = obs.feedback or ""

        # Episode termination paths:
        #   1. Fraudster already committed AND all ads decided -> audit_phase.
        #   2. Max rounds reached AND no more proposals allowed -> audit_phase.
        #   3. Investigator budget exhausted (obs.done) -> audit_phase.
        #   4. Action cap for this turn hit -> fraudster_turn (next round, unless commit_final).

        all_decided = self._all_ads_decided()
        inv_done = bool(obs.done)

        if inv_done:
            self._end_reason = self._end_reason or "investigator_done"
            self._transition(to="audit_phase", note="investigator env signalled done")
            obs.done = False  # match isn't over until Auditor submits
            return obs

        if all_decided and (
            self._fraudster_committed
            or self._round_number >= self._max_rounds
            or self._proposals_used >= self._max_proposals
        ):
            self._end_reason = self._end_reason or "all_decided"
            self._transition(to="audit_phase", note="all ads decided")
            obs.done = False
            return obs

        if self._actions_this_turn >= self._max_investigator_actions_per_turn:
            if self._round_number >= self._max_rounds or self._fraudster_committed:
                self._end_reason = self._end_reason or "max_rounds"
                self._transition(to="audit_phase", note="max rounds reached")
                obs.done = False
            else:
                self._round_number += 1
                self._transition(to="fraudster_turn", note="investigator action cap")
                # One-line warning when the next investigator turn will be
                # the LAST one — gives a slow-to-verdict policy a clear
                # signal that pending ads will get auto-approved otherwise.
                if self._round_number == self._max_rounds:
                    self._last_feedback["investigator"] = (
                        "Final round next: pending ads not given an explicit "
                        "verdict will auto-approve at audit time."
                    )

        obs.done = self._phase == "done"
        return obs

    def _all_ads_decided(self) -> bool:
        if self._episode is None:
            return False
        verdicts = self._investigator.verdicts
        return all(a.ad_id in verdicts for a in self._episode.ads)

    # ------------------------------------------------------------------
    # Auditor step handler
    # ------------------------------------------------------------------

    def step_as_auditor(self, action: AuditorAction) -> AuditorObservation:
        self._guard_phase("audit_phase", role="auditor")

        feedback = ""
        if action.action_type == "flag_investigator":
            flag = AuditFlag(
                track="A",
                target_ad_id=action.target_ad_id,
                flag_type=action.flag_type or "unspecified",
                severity=action.severity if action.severity is not None else 0.5,
                note=action.note or "",
            )
            self._audit_flags.append(flag)
            feedback = (
                f"Track A flag recorded: {flag.flag_type} (severity={flag.severity:.2f})."
            )

        elif action.action_type == "flag_fraudster":
            flag = AuditFlag(
                track="B",
                target_ad_id=action.target_ad_id,
                flag_type=action.flag_type or "unspecified",
                severity=action.severity if action.severity is not None else 0.5,
                note=action.note or "",
            )
            self._audit_flags.append(flag)
            feedback = (
                f"Track B flag recorded: {flag.flag_type} (severity={flag.severity:.2f})."
            )

        elif action.action_type == "submit_audit_report":
            report_payload = action.audit_report or {}
            track_a_flags = [f for f in self._audit_flags if f.track == "A"]
            track_b_flags = [f for f in self._audit_flags if f.track == "B"]

            # Track A/B score *defaults* come from the real graders running
            # over the episode record — so even a dumb Auditor that submits an
            # empty report gets a principled score.  Caller-supplied values
            # override these (used by tests and LLM Auditors that compute
            # their own).
            default_a, default_b = self._compute_default_track_scores()
            investigator_score = float(
                report_payload.get("investigator_audit_score", default_a)
            )
            fraudster_score = float(
                report_payload.get("fraudster_plausibility_score", default_b)
            )
            investigator_score = min(1.0, max(0.0, investigator_score))
            fraudster_score = min(1.0, max(0.0, fraudster_score))

            self._audit_report = AuditReport(
                track_a_flags=track_a_flags,
                track_b_flags=track_b_flags,
                investigator_audit_score=investigator_score,
                fraudster_plausibility_score=fraudster_score,
                notes=str(report_payload.get("notes", "") or action.note or "")[:4000],
            )
            feedback = (
                "Audit report submitted. "
                f"Track A flags: {len(track_a_flags)}. "
                f"Track B flags: {len(track_b_flags)}. "
                f"investigator_audit_score={investigator_score:.2f}, "
                f"fraudster_plausibility_score={fraudster_score:.2f}."
            )
            self._finalize_audit()

        else:
            feedback = f"Unknown Auditor action_type '{action.action_type}'."

        self._last_feedback["auditor"] = feedback
        return self.build_auditor_observation(feedback=feedback)

    def _finalize_audit(self) -> None:
        """
        Compute grader score and per-role rewards using the multi-agent reward
        model (graders/multi_agent_rewards.py), close out the match, and
        transition to `done`.
        """
        if self._episode is None:
            return

        # The benchmark policy treats unresolved ads as automatic approvals.
        # Materialize these verdicts before grading so leakage and coverage do
        # not silently omit pending ads in the multi-agent path.
        self._investigator.auto_approve_pending()
        record = self._build_episode_record()
        self._grader_score = grade_episode(record)

        audit_report = self._audit_report or AuditReport(
            track_a_flags=[],
            track_b_flags=[],
            investigator_audit_score=1.0,
            fraudster_plausibility_score=1.0,
            notes="",
        )

        reward_inputs = RewardInputs(
            record=record,
            audit_report=audit_report,
            fraudster_proposal_log=list(self._fraudster_log),
            investigator_action_log=list(self._investigator_log),
            investigation_data_seen=(
                self._registry.to_dict() if self._registry else {}
            ),
            fraudster_ad_ids=list(self._proposal_slot_to_ad_id.values()),
        )
        rewards = compute_episode_rewards(reward_inputs)

        self._fraudster_reward_total = float(rewards["fraudster"])
        self._investigator_reward_total = float(rewards["investigator"])
        self._auditor_reward_total = float(rewards["auditor"])
        self._per_ad_plausibility = dict(rewards.get("per_ad_plausibility") or {})
        self._audit_ground_truth = dict(rewards.get("audit_ground_truth") or {})

        global _last_grader_result
        _last_grader_result = {
            "match_id": self._match_id,
            "task_id": self._task_id,
            "grader_score": self._grader_score,
            "phase": "done",
            "total_steps": (
                len(self._fraudster_log)
                + len(self._investigator_log)
                + len(self._audit_flags)
            ),
            "fraudster_reward": self._fraudster_reward_total,
            "investigator_reward": self._investigator_reward_total,
            "auditor_reward": self._auditor_reward_total,
            "per_ad_plausibility": self._per_ad_plausibility,
            "audit_ground_truth": self._audit_ground_truth,
            "proposals_used": self._proposals_used,
            "end_reason": self._end_reason,
            "audit_report": (
                self._audit_report.model_dump() if self._audit_report else None
            ),
        }

        self._transition(to="done", note="audit report submitted")
        self._done = True

    def _compute_default_track_scores(self) -> Tuple[float, float]:
        """
        Derive default investigator_audit_score and fraudster_plausibility_score
        from the Track A and Track B graders.  Used when the Auditor submits
        an empty report payload.
        """
        if self._episode is None:
            return 1.0, 1.0

        record = self._build_episode_record()
        investigation_data_seen = (
            self._registry.to_dict() if self._registry else {}
        )
        track_a_flags = run_track_a(
            record,
            investigator_actions=list(self._investigator_log),
            investigation_data_seen=investigation_data_seen,
        )
        investigator_score = track_a_score(track_a_flags)

        _per_ad, _flags, queue_plaus = compute_queue_plausibility(
            self._fraudster_log
        )
        # If the Fraudster never proposed anything, plausibility doesn't
        # apply — treat as 1.0 (no evidence the Fraudster was unrealistic).
        return investigator_score, queue_plaus if _per_ad else 1.0

    def _build_episode_record(self) -> EpisodeRecord:
        """Assemble an EpisodeRecord from Investigator's view, mirroring R1."""
        assert self._episode is not None
        verdicts = self._investigator.verdicts
        links = self._investigator.links
        inv_state: AdFraudState = self._investigator.state

        verdict_results = []
        for ad in self._episode.ads:
            v = verdicts.get(ad.ad_id)
            if v:
                verdict_results.append(
                    VerdictResult(
                        ad_id=ad.ad_id,
                        verdict=v["verdict"],
                        confidence=v.get("confidence", 0.5),
                        ground_truth=v["ground_truth"],
                        auto_approved=v.get("auto_approved", False),
                    )
                )

        link_results = [
            LinkResult(ad_id_1=l["ad_id_1"], ad_id_2=l["ad_id_2"], correct=l["correct"])
            for l in links
        ]

        ads_metadata = [
            {
                "ad_id": ad.ad_id,
                "ground_truth": ad.ground_truth_label,
                "severity": ad.severity,
            }
            for ad in self._episode.ads
        ]

        return EpisodeRecord(
            task_id=self._task_id,
            total_steps=inv_state.step_count,
            action_budget=self._episode.task_config.action_budget,
            verdicts=verdict_results,
            links=link_results,
            ads_metadata=ads_metadata,
            n_fraud_rings=len(self._episode.fraud_rings),
            ring_sizes=[len(r.member_ad_ids) for r in self._episode.fraud_rings],
        )

    # ------------------------------------------------------------------
    # Observation builders
    # ------------------------------------------------------------------

    def build_fraudster_observation(
        self, *, reward: float = 0.0
    ) -> FraudsterObservation:
        phase = self._phase
        done = phase == "done"

        if self._episode is None:
            return FraudsterObservation(
                done=done,
                reward=reward,
                feedback="No episode loaded. Call reset() first.",
                phase=phase,
            )

        current_queue = self._build_queue_summary()
        prior_verdicts = self._build_verdict_history()
        investigations = self._investigator.investigations

        rounds_remaining = max(0, self._max_rounds - self._round_number + 1)
        actions_left = max(
            0,
            self._max_fraudster_actions_per_turn - self._actions_this_turn,
        ) if phase == "fraudster_turn" else 0

        my_proposal_signals = self._build_my_proposal_signals()

        return FraudsterObservation(
            done=done,
            reward=reward,
            feedback=self._last_feedback["fraudster"],
            phase=phase,
            task_id=getattr(self._episode.task_config, "task_id", ""),
            round_number=self._round_number,
            rounds_remaining=rounds_remaining,
            proposals_used=self._proposals_used,
            proposals_remaining=max(0, self._max_proposals - self._proposals_used),
            actions_left_this_turn=actions_left,
            current_queue=current_queue,
            prior_verdicts=prior_verdicts,
            investigation_targets_used=investigations,
            allowed_categories=list(self._allowed_categories),
            my_proposal_signals=my_proposal_signals,
        )

    def _build_my_proposal_signals(self) -> Dict[str, Dict[str, Any]]:
        """Per-proposal structured signals for the Fraudster's own ads.

        For every Fraudster-proposed ad, expose the auto-assigned underlying
        signals (payment_id, registrar, domain, country, account_age_days,
        targeting_fingerprint) by reusing the same extraction logic the
        Investigator's evidence ledger uses.  We synthesise an
        "investigations" dict that pretends *all* targets were pulled — the
        Fraudster authored these ads, so it is allowed to know everything
        the env auto-assigned to them.  The Fraudster never sees signals
        for synthetic / non-self-proposed ads, only for its own slate.
        """
        if self._episode is None:
            return {}
        proposal_ad_ids = list(self._proposal_slot_to_ad_id.values())
        if not proposal_ad_ids:
            return {}

        full_targets = [
            "payment_method",
            "landing_page",
            "targeting_overlap",
            "advertiser_history",
        ]
        ledger = build_evidence_ledger(
            episode=self._episode,
            registry=self._registry,
            ad_ids=proposal_ad_ids,
            investigations={ad_id: full_targets for ad_id in proposal_ad_ids},
        )

        slot_by_ad_id = {
            ad_id: slot for slot, ad_id in self._proposal_slot_to_ad_id.items()
        }
        verdicts = self._investigator.verdicts
        for ad_id, entry in ledger.items():
            if ad_id in slot_by_ad_id:
                entry["slot_index"] = slot_by_ad_id[ad_id]
            v = verdicts.get(ad_id)
            entry["investigator_verdict"] = (
                v.get("verdict") if v else "pending"
            )
        return ledger

    def build_investigator_observation(self) -> AdReviewObservation:
        obs = self._investigator._build_observation(  # noqa: SLF001
            reward=0.0, done=(self._phase == "done")
        )
        obs.feedback = (
            self._last_feedback["investigator"] or obs.feedback
        )
        return obs

    def build_auditor_observation(
        self, *, feedback: str = ""
    ) -> AuditorObservation:
        phase = self._phase
        done = phase == "done"
        investigation_data_seen: Dict[str, Dict[str, str]] = {}
        if self._registry is not None:
            investigation_data_seen = self._registry.to_dict()

        record: Dict[str, Any] = {}
        if self._episode is not None:
            record = {
                "task_id": self._task_id,
                "round_number": self._round_number,
                "proposals_used": self._proposals_used,
                "end_reason": self._end_reason,
                "ads": [
                    {
                        "ad_id": ad.ad_id,
                        "ad_copy": ad.ad_copy,
                        "category": ad.category,
                        "ground_truth": ad.ground_truth_label,
                        "severity": ad.severity,
                        "fraud_type": ad.fraud_type,
                        "difficulty": ad.difficulty,
                        "is_fraudster_proposal": ad.ad_id
                        in self._proposal_slot_to_ad_id.values(),
                    }
                    for ad in self._episode.ads
                ],
                "verdicts": [
                    {"ad_id": ad_id, **v}
                    for ad_id, v in self._investigator.verdicts.items()
                ],
                "links": list(self._investigator.links),
                "grader_score": self._grader_score,
                "fraud_rings": [
                    {
                        "ring_id": ring.ring_id,
                        "topology": ring.topology,
                        "case_name": ring.case_name,
                        "provenance": ring.provenance,
                        "member_ad_ids": list(ring.member_ad_ids),
                        "shared_signal_types": list(ring.shared_signals.keys()),
                    }
                    for ring in self._episode.fraud_rings
                ],
            }

        return AuditorObservation(
            done=done,
            reward=self._auditor_reward_total,
            feedback=feedback or self._last_feedback["auditor"],
            phase=phase,
            full_episode_record=record,
            investigator_actions=list(self._investigator_log),
            fraudster_proposals=list(self._fraudster_log),
            investigation_data_seen=investigation_data_seen,
            pending_flags=[f.model_dump() for f in self._audit_flags],
        )

    # ------------------------------------------------------------------
    # State-machine helpers
    # ------------------------------------------------------------------

    def _guard_phase(self, expected: Phase, *, role: Role) -> None:
        if self._phase != expected:
            raise PermissionError(
                f"{role} cannot act during phase '{self._phase}' "
                f"(expected '{expected}')."
            )

    def _transition(self, *, to: Phase, note: str) -> None:
        if self._phase == to:
            return
        logger.debug("[referee] %s -> %s (%s)", self._phase, to, note)
        self._phase = to
        self._actions_this_turn = 0

    def _build_queue_summary(self) -> List[Dict[str, Any]]:
        assert self._episode is not None
        verdicts = self._investigator.verdicts
        proposal_ad_ids = set(self._proposal_slot_to_ad_id.values())
        slot_by_ad_id = {
            ad_id: slot for slot, ad_id in self._proposal_slot_to_ad_id.items()
        }

        out: List[Dict[str, Any]] = []
        for ad in self._episode.ads:
            v = verdicts.get(ad.ad_id)
            entry = {
                "ad_id": ad.ad_id,
                "ad_copy": ad.ad_copy,
                "category": ad.category,
                "status": (v["verdict"] if v else "pending"),
                "is_my_proposal": ad.ad_id in proposal_ad_ids,
            }
            if ad.ad_id in slot_by_ad_id:
                entry["slot_index"] = slot_by_ad_id[ad.ad_id]
            out.append(entry)
        return out

    def _build_verdict_history(self) -> List[Dict[str, Any]]:
        proposal_ad_ids = set(self._proposal_slot_to_ad_id.values())
        history: List[Dict[str, Any]] = []
        for entry in self._investigator_log:
            if entry.get("action_type") != "verdict":
                continue
            history.append(
                {
                    "ad_id": entry.get("ad_id"),
                    "verdict": entry.get("verdict"),
                    "confidence": entry.get("confidence"),
                    "rationale": entry.get("rationale"),
                    "was_my_proposal": entry.get("ad_id") in proposal_ad_ids,
                }
            )
        return history

    def _serialize_fraudster_action(
        self, action: FraudsterAction, reward: float
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "ts": time.time(),
            "phase": self._phase,
            "round_number": self._round_number,
            "action_type": action.action_type,
            "ad_copy": action.ad_copy,
            "category": action.category,
            "landing_page_blurb": action.landing_page_blurb,
            "targeting_summary": action.targeting_summary,
            "slot_index": action.slot_index,
            "new_ad_copy": action.new_ad_copy,
            "new_landing_page_blurb": action.new_landing_page_blurb,
            "rationale": action.rationale,
            "reward": reward,
            "ad_id": None,
        }

        # Enrich queue actions with the env-resolved ad context so the
        # auditor + downstream graders can key flags onto a real ad_id and
        # see the AD'S CURRENT STATE (not just the LLM's payload, which
        # for ``modify_pending_ad`` only carries the *delta* fields).
        if action.action_type == "propose_ad" and self._last_proposed_ad_id is not None:
            payload["ad_id"] = self._last_proposed_ad_id
            payload["slot_index"] = self._last_proposed_slot
            self._last_proposed_ad_id = None
            self._last_proposed_slot = None
        elif (
            action.action_type == "modify_pending_ad"
            and action.slot_index is not None
            and action.slot_index in self._proposal_slot_to_ad_id
        ):
            ad_id = self._proposal_slot_to_ad_id[action.slot_index]
            payload["ad_id"] = ad_id
            ad = self._find_episode_ad(ad_id)
            if ad is not None:
                # Always inject the ad's CURRENT state — the modify only
                # carries deltas, and post-modify the ad's authoritative
                # ``ad_copy`` / ``targeting_summary`` live on the
                # ``Ad`` object the env mutated in
                # ``_fraudster_modify_pending_ad``.
                payload.setdefault("category", ad.category)
                if not payload.get("ad_copy"):
                    payload["ad_copy"] = action.new_ad_copy or ad.ad_copy
                if not payload.get("targeting_summary"):
                    payload["targeting_summary"] = ad.targeting_summary
                if (
                    not payload.get("landing_page_blurb")
                    and self._episode is not None
                ):
                    lp = self._episode.landing_pages.get(ad_id)
                    if lp is not None:
                        payload["landing_page_blurb"] = (
                            action.new_landing_page_blurb
                            or lp.content_summary
                        )
        return payload

    def _find_episode_ad(self, ad_id: str) -> Optional[Ad]:
        if self._episode is None:
            return None
        for ad in self._episode.ads:
            if ad.ad_id == ad_id:
                return ad
        return None

    def _serialize_investigator_action(
        self, action: AdReviewAction, obs: AdReviewObservation
    ) -> Dict[str, Any]:
        return {
            "ts": time.time(),
            "phase": self._phase,
            "round_number": self._round_number,
            "action_type": action.action_type,
            "ad_id": action.ad_id,
            "investigation_target": action.investigation_target,
            "verdict": action.verdict,
            "confidence": action.confidence,
            "rationale": action.rationale,
            "linked_ad_id": action.linked_ad_id,
            "link_reason": action.link_reason,
            "reward": float(obs.reward or 0.0),
            "findings_excerpt": (obs.feedback or "")[:500],
        }

    # ------------------------------------------------------------------
    # Introspection helpers for the driver / clients
    # ------------------------------------------------------------------

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def done(self) -> bool:
        return self._done

    @property
    def match_id(self) -> str:
        return self._match_id

    @property
    def episode(self) -> Optional[GeneratedEpisode]:
        return self._episode

    @property
    def registry(self) -> Optional[InvestigationToolRegistry]:
        return self._registry

    @property
    def investigator(self) -> InvestigatorEnvironment:
        return self._investigator

    def grader_score(self) -> Optional[float]:
        return self._grader_score
