"""
Data models for the CounterFeint multi-agent FraudArena.

Round 1 contracts (`AdReviewAction`, `AdReviewObservation`, `AdFraudState`)
remain intact and re-exported as the *Investigator* role for backwards
compatibility.

Round 2 introduces three roles that share a single environment:

  - Fraudster:   `FraudsterAction`   / `FraudsterObservation`
  - Investigator: `InvestigatorAction` / `InvestigatorObservation` (aliases)
  - Auditor:    `AuditorAction`     / `AuditorObservation`

The `RefereeState` model exposes the global state machine
(fraudster_turn / investigator_turn / audit_phase / done) plus per-role
running rewards.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, ConfigDict, Field, model_validator


# =============================================================================
# Round 1 — Investigator role (kept verbatim for backwards compatibility)
# =============================================================================


class AdReviewAction(Action):
    """
    Action space for the ad fraud investigation agent.

    Three action types:
    - investigate: Spend budget to reveal information about an ad
    - verdict: Approve, reject, or escalate an ad
    - link_accounts: Flag two ads as part of the same fraud network
    """

    action_type: Literal["investigate", "verdict", "link_accounts"]
    ad_id: str = Field(..., description="Target ad identifier (e.g. 'ad_001')")

    investigation_target: Optional[
        Literal[
            "advertiser_history",
            "landing_page",
            "payment_method",
            "targeting_overlap",
            "campaign_structure",
            "policy_classifier",
        ]
    ] = Field(None, description="What to investigate (required for action_type='investigate')")

    verdict: Optional[Literal["approve", "reject", "escalate"]] = Field(
        None, description="Verdict decision (required for action_type='verdict')"
    )
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Agent's confidence in verdict (0.0-1.0)"
    )
    rationale: Optional[str] = Field(
        None,
        max_length=2000,
        description="Optional natural-language reason for the verdict (consumed by the Auditor)",
    )

    linked_ad_id: Optional[str] = Field(
        None, description="Other ad in suspected fraud ring (required for action_type='link_accounts')"
    )
    link_reason: Optional[str] = Field(
        None, description="Why the agent believes these ads are connected"
    )

    @model_validator(mode="after")
    def _validate_action_fields(self) -> "AdReviewAction":
        """Reject mismatched action/field combinations early.

        Without this, a small LLM that emits e.g.
        ``{"action_type": "investigate", "verdict": "approve"}`` passes
        Pydantic silently (every field is Optional) but the env quietly
        IGNORES the verdict because ``_handle_investigate`` doesn't read
        it — the model gets no feedback and re-emits the same broken
        action shape every step. Surfacing the mismatch as a hard error
        funnels these cases through the policy's deterministic fallback
        (which DOES make a clean choice) and eventually shows up in
        ``fallback_count``, which the training rollout exposes as a
        signal to improve.
        """
        if self.action_type == "investigate":
            if self.investigation_target is None:
                raise ValueError(
                    "investigation_target is required when "
                    "action_type='investigate'"
                )
            if self.verdict is not None:
                raise ValueError(
                    "verdict must be null when action_type='investigate' "
                    "— issue a separate verdict action."
                )
        elif self.action_type == "verdict":
            if self.verdict is None:
                raise ValueError(
                    "verdict is required when action_type='verdict'"
                )
            if self.investigation_target is not None:
                raise ValueError(
                    "investigation_target must be null when "
                    "action_type='verdict' — issue a separate investigate "
                    "action."
                )
        elif self.action_type == "link_accounts":
            if self.linked_ad_id is None:
                raise ValueError(
                    "linked_ad_id is required when action_type='link_accounts'"
                )
        return self


class AdReviewObservation(Observation):
    """
    Observation returned after each Investigator step.

    Text-heavy by design so LLM agents can reason about the content naturally.
    Structured data is in queue_status for programmatic access.
    """

    queue_summary: str = Field(
        default="", description="Natural language overview of the ad queue"
    )
    current_ad_info: str = Field(
        default="", description="Details of the ad currently in focus"
    )
    investigation_findings: str = Field(
        default="", description="Accumulated investigation results"
    )
    verdict_history_summary: str = Field(
        default="", description="Summary of verdicts rendered so far"
    )
    feedback: str = Field(
        default="", description="Natural language feedback on the last action taken"
    )
    available_ads: List[str] = Field(
        default_factory=list, description="Ad IDs still pending review"
    )
    queue_status: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured status: total_ads, reviewed, pending, budget, step",
    )
    queue_may_grow: bool = Field(
        default=False,
        description="True when running inside the Referee — Fraudster can still add ads",
    )
    evidence_ledger: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Per-ad structured evidence accumulated across investigations. "
            "Surface fields (category, country, account_age_days) are always "
            "present once an ad has been touched; investigation-only fields "
            "(payment_id, registrar, domain, targeting_fingerprint, "
            "advertiser_id) appear only after the corresponding "
            "`investigate` target has been pulled. Cross-ad collisions on a "
            "SUBSET of these fields indicate fraud rings — the policy must "
            "learn which fields are discriminative (payment_id collisions "
            "matter, country collisions usually don't)."
        ),
    )
    queue_digest: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "One-row-per-pending-ad summary surfaced WITHOUT requiring an "
            "investigation. Each row carries a curated subset of fields "
            "from the ad + advertiser_profile: a small set of "
            "potentially-discriminative columns (payment_type, registrar, "
            "domain) the Investigator can use as a pre-investigation "
            "ring-detection hint, plus a handful of decoy columns "
            "(category, country, account_age_days) that are intentionally "
            "non-discriminative so the policy must learn which collisions "
            "matter. Capped to ~12 ads to keep the prompt budget bounded."
        ),
    )
    decided_ads: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Per-decided-ad summary: verdict + confidence + a curated mix "
            "of discriminative (payment_id, registrar, domain, "
            "targeting_fingerprint) and decoy (category, country, "
            "account_age_days) signals from the evidence ledger. Gives "
            "the Investigator memory of past decisions for link_accounts."
        ),
    )


class AdFraudState(State):
    """
    Internal Investigator state exposed via the state() property.

    Inherits episode_id and step_count from State.
    Uses extra='allow' so custom fields are permitted.
    """

    task_id: str = ""
    total_ads: int = 0
    reviewed_count: int = 0
    remaining_budget: int = 0
    verdicts: Dict[str, Any] = Field(default_factory=dict)
    grader_score: Optional[float] = None


# Round 2 alias — same wire format, semantically scoped to the Investigator role
InvestigatorAction = AdReviewAction
InvestigatorObservation = AdReviewObservation
InvestigatorState = AdFraudState


# =============================================================================
# Round 2 — Fraudster role (turn-based, REACTIVE)
# =============================================================================


class FraudsterAction(Action):
    """
    Reactive turn-based action space for the Fraudster.

    Within a single Fraudster turn the agent may issue multiple actions
    (typically `propose_ad` and/or `modify_pending_ad`) before finishing
    the turn with `end_turn` (control flips to the Investigator) or
    `commit_final` (no more changes ever; episode fast-tracks to audit).

    Hard caps (configurable on the Referee):
      - max_proposals_per_episode  (default: 5)
      - max_actions_per_turn       (default: 3)
    """

    action_type: Literal["propose_ad", "modify_pending_ad", "end_turn", "commit_final"]

    # propose_ad
    ad_copy: Optional[str] = Field(
        None,
        max_length=2000,
        description="Surface text of the proposed ad (required for propose_ad)",
    )
    landing_page_blurb: Optional[str] = Field(
        None,
        max_length=2000,
        description="Optional landing-page summary the Fraudster wants the ad to advertise",
    )
    category: Optional[str] = Field(
        None,
        max_length=64,
        description="Self-declared ad category (must be one of the categories advertised in /tasks)",
    )
    targeting_summary: Optional[str] = Field(
        None,
        max_length=512,
        description="Audience the Fraudster claims to target (e.g. 'Adults 25-45, US, interests: investing')",
    )

    # modify_pending_ad
    slot_index: Optional[int] = Field(
        None,
        ge=0,
        description="Index into the Fraudster's own proposals list (0-based)",
    )
    new_ad_copy: Optional[str] = Field(
        None,
        max_length=2000,
        description="Replacement ad copy",
    )
    new_landing_page_blurb: Optional[str] = Field(
        None,
        max_length=2000,
        description="Replacement landing page blurb",
    )

    rationale: Optional[str] = Field(
        None,
        max_length=2000,
        description="Optional natural-language reason for this action (consumed by the Auditor)",
    )


class FraudsterObservation(Observation):
    """
    Reactive observation for the Fraudster.

    The Fraudster sees the Investigator's verdicts and which investigation
    targets the Investigator pulled, so it can adapt within the same episode
    (e.g. 'they keep checking landing_page → improve my landing page blurbs',
    or 'category=fake_crypto keeps getting rejected → try gray_area_supplements').
    """

    feedback: str = Field(default="", description="Free-form feedback on the last action")
    phase: Literal["fraudster_turn", "investigator_turn", "audit_phase", "done"] = Field(
        default="fraudster_turn", description="Global state-machine phase"
    )
    task_id: str = Field(
        default="",
        description=(
            "Currently-running task id (e.g. 'task_1', 'task_3_unseen'). "
            "Surfaced so the Fraudster can scale its stealth posture per "
            "task tier without the Referee having to mutate its system "
            "prompt: easy tiers want louder fraud cues so the Investigator "
            "can succeed; hard tiers want subtler fraud cues so the trained "
            "Investigator's evaluation gain is meaningful."
        ),
    )

    round_number: int = Field(default=0, ge=0, description="1-based round counter")
    rounds_remaining: int = Field(default=0, ge=0, description="Rounds left before audit_phase")
    proposals_used: int = Field(default=0, ge=0)
    proposals_remaining: int = Field(default=0, ge=0)
    actions_left_this_turn: int = Field(default=0, ge=0)

    current_queue: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Current ad queue: [{ad_id, ad_copy, category, status, "
            "is_my_proposal, slot_index?}]. status ∈ {pending, "
            "investigating, approved, rejected, escalated}."
        ),
    )
    prior_verdicts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "All verdicts rendered so far by the Investigator: "
            "[{ad_id, verdict, confidence, rationale, was_my_proposal}]"
        ),
    )
    investigation_targets_used: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Per-ad list of investigation_target names already pulled by the Investigator",
    )

    allowed_categories: List[str] = Field(
        default_factory=list,
        description="Whitelist of category strings the Fraudster may declare",
    )
    my_proposal_signals: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "For each Fraudster-proposed ad still on the queue, the "
            "auto-assigned underlying signals (payment_id, registrar, "
            "domain, country, account_age_days, targeting_fingerprint). "
            "These fields are NOT settable by `propose_ad` — the env "
            "samples them from the fraud-mode distribution. Surfacing "
            "them lets the Fraudster react via `modify_pending_ad` "
            "(e.g. soften the landing page on ad_004 because Investigator "
            "rejected ad_002 which shares its registrar) and reason about "
            "ring-style cross-ad collisions in its own slate."
        ),
    )


# =============================================================================
# Round 2 — Auditor role (post-hoc, dual-track)
# =============================================================================


class AuditorAction(Action):
    """
    Post-hoc audit actions.

    Track A audits the Investigator's *reasoning* (rationale coherence,
    citation, calibration, consistency, bias).  Track B audits the
    Fraudster's *output plausibility* (template diversity, parameter
    realism, market fit, etc.).  The Auditor accumulates flags and then
    submits a final report.
    """

    action_type: Literal[
        "flag_investigator",
        "flag_fraudster",
        "submit_audit_report",
    ]

    target_ad_id: Optional[str] = Field(
        None,
        description="Ad the flag applies to (required for flag_* actions)",
    )
    flag_type: Optional[str] = Field(
        None,
        max_length=64,
        description=(
            "Track A flag types: miscalibration, missing_citation, "
            "incoherent_rationale, inconsistency, bias. "
            "Track B flag types: gibberish, parameter_mismatch, "
            "template_repetition, market_implausible, branding_anomaly."
        ),
    )
    severity: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="0.0 = warning, 1.0 = critical"
    )
    note: Optional[str] = Field(
        None, max_length=2000, description="Free-form auditor note"
    )

    audit_report: Optional[Dict[str, Any]] = Field(
        None,
        description="Final report payload for action_type='submit_audit_report'",
    )


class AuditorObservation(Observation):
    """
    Post-hoc observation for the Auditor.

    Contains the full episode trace: every Fraudster proposal, every
    Investigator action+rationale, all verdicts, and the synthesized
    investigation data the Investigator saw.
    """

    feedback: str = Field(default="")
    phase: Literal["fraudster_turn", "investigator_turn", "audit_phase", "done"] = Field(
        default="audit_phase"
    )

    full_episode_record: Dict[str, Any] = Field(
        default_factory=dict,
        description="Serialized record of the entire episode",
    )
    investigator_actions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Ordered log of every Investigator action with rationales",
    )
    fraudster_proposals: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Ordered log of every Fraudster proposal/modification",
    )
    investigation_data_seen: Dict[str, Dict[str, str]] = Field(
        default_factory=dict,
        description="The actual findings text the Investigator pulled per (ad_id, target)",
    )
    pending_flags: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Flags accumulated so far in this audit",
    )


class AuditFlag(BaseModel):
    """One audit flag in either track."""

    model_config = ConfigDict(extra="forbid")

    track: Literal["A", "B"] = Field(..., description="A=Investigator audit, B=Fraudster plausibility")
    target_ad_id: Optional[str] = None
    flag_type: str
    severity: float = Field(default=0.5, ge=0.0, le=1.0)
    note: str = ""


class AuditReport(BaseModel):
    """Final audit report submitted at end of audit phase."""

    model_config = ConfigDict(extra="forbid")

    track_a_flags: List[AuditFlag] = Field(default_factory=list)
    track_b_flags: List[AuditFlag] = Field(default_factory=list)
    investigator_audit_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="1.0 = clean rationales/calibration, lower = miscalibrated/incoherent",
    )
    fraudster_plausibility_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="1.0 = realistic ads, lower = gibberish/template-collapse",
    )
    notes: str = Field(default="", max_length=4000)


# =============================================================================
# Round 2 — Referee state (global state machine)
# =============================================================================


class RefereeState(State):
    """
    Global state of the multi-agent FraudArena.

    Inherits episode_id and step_count from State.
    Uses extra='allow' so we can include rich nested dicts for the
    /state HTTP endpoint (judges + UI consume this).
    """

    task_id: str = ""
    phase: Literal["fraudster_turn", "investigator_turn", "audit_phase", "done"] = (
        "fraudster_turn"
    )

    round_number: int = Field(default=0, ge=0)
    max_rounds: int = Field(default=5, ge=1)

    proposals_used: int = Field(default=0, ge=0)
    max_proposals: int = Field(default=5, ge=0)

    action_budget: int = Field(
        default=0,
        ge=0,
        description="Investigator action budget for the loaded task",
    )

    actions_this_turn: int = Field(default=0, ge=0)
    max_actions_per_turn: int = Field(default=3, ge=1)

    investigator_state: Dict[str, Any] = Field(
        default_factory=dict,
        description="Snapshot of the inner InvestigatorEnvironment.state",
    )
    fraudster_proposals: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Ordered log of Fraudster proposals (and modifications)",
    )
    investigator_action_log: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Ordered log of every Investigator action",
    )
    fraudster_committed: bool = Field(
        default=False,
        description="True after Fraudster issued commit_final",
    )
    audit_report: Optional[Dict[str, Any]] = Field(
        default=None, description="Filled in once audit_phase completes"
    )

    fraudster_reward: float = 0.0
    investigator_reward: float = 0.0
    auditor_reward: float = 0.0
    grader_score: Optional[float] = None
    end_reason: Optional[str] = Field(
        default=None,
        description="One of: commit_final, all_decided, max_rounds, budget_exhausted",
    )
