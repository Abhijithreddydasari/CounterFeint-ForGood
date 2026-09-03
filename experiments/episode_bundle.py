"""
Collect and serialize full episode trajectories for corruption / ablation studies.

Runs in-process (no HTTP). Uses ScriptedInvestigator + ReactiveFraudster +
HeuristicAuditor by default so trajectories are reproducible on CPU.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from counterfeint.graders.base_grader import EpisodeRecord, LinkResult, VerdictResult
from counterfeint.scripted import HeuristicAuditor, ReactiveFraudster, ScriptedInvestigator
from counterfeint.server.referee import RefereeEnvironment


PolicyFactory = Callable[[], Any]


@dataclass
class EpisodeBundle:
    """Everything needed to re-run Track A/B audits and reward ablations."""

    task_id: str
    seed: int
    investigator_actions: List[Dict[str, Any]]
    fraudster_proposals: List[Dict[str, Any]]
    investigation_data_seen: Dict[str, Dict[str, str]]
    fraudster_ad_ids: List[str]
    episode_record: Dict[str, Any]
    corruption_label: Optional[str] = None
    expected_flags: List[str] = field(default_factory=list)
    expected_flag_keys: List[List[Optional[str]]] = field(default_factory=list)
    expected_llm_flags: List[str] = field(default_factory=list)
    side: Optional[str] = None
    exploit: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "investigator_actions": self.investigator_actions,
            "fraudster_proposals": self.fraudster_proposals,
            "investigation_data_seen": self.investigation_data_seen,
            "fraudster_ad_ids": self.fraudster_ad_ids,
            "episode_record": self.episode_record,
            "corruption_label": self.corruption_label,
            "expected_flags": list(self.expected_flags),
            "expected_flag_keys": list(self.expected_flag_keys),
            "expected_llm_flags": list(self.expected_llm_flags),
            "side": self.side,
            "exploit": self.exploit,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodeBundle":
        return cls(
            task_id=str(data["task_id"]),
            seed=int(data["seed"]),
            investigator_actions=list(data.get("investigator_actions") or []),
            fraudster_proposals=list(data.get("fraudster_proposals") or []),
            investigation_data_seen=dict(data.get("investigation_data_seen") or {}),
            fraudster_ad_ids=list(data.get("fraudster_ad_ids") or []),
            episode_record=dict(data.get("episode_record") or {}),
            corruption_label=data.get("corruption_label"),
            expected_flags=list(data.get("expected_flags") or []),
            expected_flag_keys=list(data.get("expected_flag_keys") or []),
            expected_llm_flags=list(data.get("expected_llm_flags") or []),
            side=data.get("side"),
            exploit=data.get("exploit"),
        )


def _dict_to_episode_record(payload: Dict[str, Any]) -> EpisodeRecord:
    verdicts = [
        VerdictResult(
            ad_id=str(v.get("ad_id", "")),
            verdict=str(v.get("verdict", "approve")),
            confidence=float(v.get("confidence", 0.5) or 0.5),
            ground_truth=str(v.get("ground_truth", "legit")),
            auto_approved=bool(v.get("auto_approved", False)),
        )
        for v in payload.get("verdicts") or []
        if v.get("verdict")
    ]
    links = [
        LinkResult(
            ad_id_1=str(l.get("ad_id_1", l.get("ad_a", ""))),
            ad_id_2=str(l.get("ad_id_2", l.get("ad_b", ""))),
            correct=bool(l.get("correct", False)),
        )
        for l in payload.get("links") or []
    ]
    ads_metadata = []
    for ad in payload.get("ads") or []:
        if not ad.get("ad_id"):
            continue
        ads_metadata.append(
            {
                "ad_id": ad.get("ad_id"),
                "ground_truth": ad.get("ground_truth", "legit"),
                "severity": float(ad.get("severity", 0.5) or 0.5),
                "fraud_type": ad.get("fraud_type", ""),
                "category": ad.get("category", ""),
                "country": ad.get("country", ""),
            }
        )
    return EpisodeRecord(
        task_id=str(payload.get("task_id", "")),
        total_steps=int(payload.get("total_steps", 0) or 0),
        action_budget=int(payload.get("action_budget", 0) or 0),
        verdicts=verdicts,
        links=links,
        ads_metadata=ads_metadata,
        n_fraud_rings=len(payload.get("fraud_rings") or []),
        ring_sizes=[
            len(r.get("member_ad_ids") or [])
            for r in (payload.get("fraud_rings") or [])
        ],
    )


def bundle_to_episode_record(bundle: EpisodeBundle) -> EpisodeRecord:
    return _dict_to_episode_record(bundle.episode_record)


def run_episode_bundle(
    *,
    task_id: str,
    seed: int,
    investigator_factory: Optional[PolicyFactory] = None,
    fraudster_factory: Optional[PolicyFactory] = None,
    auditor_factory: Optional[PolicyFactory] = None,
    max_steps: int = 300,
) -> EpisodeBundle:
    """Drive a full three-agent episode and return an auditable bundle."""
    investigator_factory = investigator_factory or (lambda: ScriptedInvestigator())
    fraudster_factory = fraudster_factory or (lambda: ReactiveFraudster(seed=seed))
    auditor_factory = auditor_factory or (lambda: HeuristicAuditor())

    env = RefereeEnvironment()
    env.reset_match(task_id=task_id, seed=seed)

    fraudster = fraudster_factory()
    investigator = investigator_factory()
    auditor = auditor_factory()

    role_handlers = {
        "fraudster_turn": (
            fraudster,
            env.build_fraudster_observation,
            env.step_as_fraudster,
        ),
        "investigator_turn": (
            investigator,
            env.build_investigator_observation,
            env.step_as_investigator,
        ),
        "audit_phase": (
            auditor,
            env.build_auditor_observation,
            env.step_as_auditor,
        ),
    }

    step_idx = 0
    while env.phase in role_handlers:
        policy, build_obs, step_fn = role_handlers[env.phase]
        role_name = env.phase.replace("_turn", "").replace("_phase", "")
        if role_name != "audit" and step_idx >= max_steps:
            break
        obs = build_obs()
        try:
            action = policy.act(obs.model_dump())
            step_fn(action)
        except Exception:
            break
        if role_name != "audit":
            step_idx += 1

    aud_obs = env.build_auditor_observation()
    record_payload = dict(aud_obs.full_episode_record or {})
    # EpisodeRecord expects investigator actions and the configured task
    # budget.  ``step_idx`` includes Fraudster turns, so it is not suitable
    # for the grader's investigation-cost and efficiency terms.
    record_payload["action_budget"] = int(
        env.episode.task_config.action_budget if env.episode else 0
    )
    record_payload["total_steps"] = len(aud_obs.investigator_actions or [])

    fraudster_ad_ids = list(getattr(env, "_proposal_slot_to_ad_id", {}).values())

    return EpisodeBundle(
        task_id=task_id,
        seed=seed,
        investigator_actions=list(aud_obs.investigator_actions or []),
        fraudster_proposals=list(aud_obs.fraudster_proposals or []),
        investigation_data_seen=dict(aud_obs.investigation_data_seen or {}),
        fraudster_ad_ids=fraudster_ad_ids,
        episode_record=record_payload,
    )


def save_bundle(bundle: EpisodeBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle.to_dict(), indent=2), encoding="utf-8")


def load_bundle(path: Path) -> EpisodeBundle:
    return EpisodeBundle.from_dict(json.loads(path.read_text(encoding="utf-8")))


__all__ = [
    "EpisodeBundle",
    "bundle_to_episode_record",
    "load_bundle",
    "run_episode_bundle",
    "save_bundle",
]
