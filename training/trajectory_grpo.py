"""On-policy, trajectory-level group-relative policy optimisation.

Each GRPO group starts from the same task/seed and frozen reactive Fraudster.
The Investigator samples a complete multi-turn episode, receives one terminal
environment reward, and that group-normalised advantage is applied to every
Investigator action token in the trajectory. No snapshot/gold-action proxy is
used in this path.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

from counterfeint.experiments.episode_bundle import (
    bundle_to_episode_record,
    run_episode_bundle,
)
from counterfeint.graders.auditor_pipeline import run_full_audit
from counterfeint.graders.base_grader import grade_episode
from counterfeint.graders.multi_agent_rewards import RewardInputs, compute_episode_rewards
from counterfeint.models import AuditReport
from counterfeint.scripted import HeuristicAuditor, ReactiveFraudster
from counterfeint.training.rollout import RecordingHFInvestigator


class RewardMode(str, Enum):
    ENVIRONMENT = "environment"
    GRADER_ONLY = "grader_only"
    PROXY = "proxy"  # retained only as an explicit ablation label


@dataclass
class TrajectoryTurn:
    prompt: str
    completion: str
    prompt_token_ids: List[int]
    completion_token_ids: List[int]
    fallback_used: bool = False

    def to_dict(self, *, include_text: bool = False) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "prompt_tokens": len(self.prompt_token_ids),
            "completion_tokens": len(self.completion_token_ids),
            "fallback_used": self.fallback_used,
        }
        if include_text:
            row.update(prompt=self.prompt, completion=self.completion)
        return row


@dataclass
class TrajectoryRecord:
    task_id: str
    seed: int
    group_id: str
    trajectory_idx: int
    grader_score: float
    investigator_reward: float
    raw_investigator_reward: float
    advantage: float = 0.0
    turns: List[TrajectoryTurn] = field(default_factory=list)
    fallback_count: int = 0
    call_count: int = 0
    n_fraudster_proposals: int = 0

    def to_dict(self, *, include_text: bool = False) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "group_id": self.group_id,
            "trajectory_idx": self.trajectory_idx,
            "grader_score": self.grader_score,
            "investigator_reward": self.investigator_reward,
            "raw_investigator_reward": self.raw_investigator_reward,
            "advantage": self.advantage,
            "n_turns": len(self.turns),
            "n_tokens": sum(len(t.completion_token_ids) for t in self.turns),
            "fallback_count": self.fallback_count,
            "call_count": self.call_count,
            "n_fraudster_proposals": self.n_fraudster_proposals,
            "turns": [t.to_dict(include_text=include_text) for t in self.turns],
        }


def compute_group_advantages(scores: Sequence[float]) -> List[float]:
    """Return zero-mean, unit-variance group-relative advantages."""
    if not scores:
        return []
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / len(scores)
    std = math.sqrt(variance)
    if std < 1e-8:
        return [0.0 for _ in scores]
    return [(score - mean) / std for score in scores]


def _trajectory_reward(*, bundle: Any, reward_mode: RewardMode) -> tuple[float, float]:
    record = bundle_to_episode_record(bundle)
    grader = float(grade_episode(record))
    if reward_mode == RewardMode.GRADER_ONLY:
        return grader, grader
    if reward_mode == RewardMode.PROXY:
        raise ValueError("Proxy reward is not a trajectory-level reward mode")

    audit = run_full_audit(
        record=record,
        investigator_action_log=bundle.investigator_actions,
        investigation_data_seen=bundle.investigation_data_seen,
        fraudster_proposal_log=bundle.fraudster_proposals,
    )
    report = AuditReport(
        track_a_flags=audit.track_a_flags,
        track_b_flags=audit.track_b_flags,
        investigator_audit_score=audit.investigator_audit_score,
        fraudster_plausibility_score=audit.fraudster_plausibility_score,
        notes="Deterministic full-audit reward for trajectory GRPO.",
    )
    rewards = compute_episode_rewards(
        RewardInputs(
            record=record,
            audit_report=report,
            fraudster_proposal_log=bundle.fraudster_proposals,
            investigator_action_log=bundle.investigator_actions,
            investigation_data_seen=bundle.investigation_data_seen,
            fraudster_ad_ids=bundle.fraudster_ad_ids,
            cache=audit.reward_cache,
        )
    )
    return grader, float(rewards["investigator"])


def _collect_one_trajectory(
    *,
    task_id: str,
    seed: int,
    group_id: str,
    trajectory_idx: int,
    investigator_factory: Callable[[], Any],
    reward_mode: RewardMode,
    fallback_penalty: float,
    max_steps: int,
) -> TrajectoryRecord:
    investigator = investigator_factory()
    recorder = RecordingHFInvestigator(investigator)
    recorder.reset()
    bundle = run_episode_bundle(
        task_id=task_id,
        seed=seed,
        investigator_factory=lambda: recorder,
        # Same frozen policy and RNG state within a group. It still observes
        # the Investigator and actively proposes/modifies ads in the arena.
        fraudster_factory=lambda: ReactiveFraudster(seed=seed),
        auditor_factory=lambda: HeuristicAuditor(),
        max_steps=max_steps,
    )
    grader, raw_reward = _trajectory_reward(bundle=bundle, reward_mode=reward_mode)

    turns: List[TrajectoryTurn] = []
    for row in recorder.step_records:
        prompt = row.get("prompt")
        completion = row.get("completion")
        prompt_ids = row.get("prompt_token_ids")
        completion_ids = row.get("completion_token_ids")
        if not isinstance(prompt, str) or not isinstance(completion, str):
            continue
        if not isinstance(prompt_ids, list) or not isinstance(completion_ids, list):
            continue
        if not prompt_ids or not completion_ids:
            continue
        turns.append(
            TrajectoryTurn(
                prompt=prompt,
                completion=completion,
                prompt_token_ids=[int(x) for x in prompt_ids],
                completion_token_ids=[int(x) for x in completion_ids],
                fallback_used=bool(row.get("fallback_used")),
            )
        )

    fallback_count = sum(int(turn.fallback_used) for turn in turns)
    proposals = sum(
        1
        for proposal in bundle.fraudster_proposals
        if proposal.get("action_type") == "propose_ad"
    )
    return TrajectoryRecord(
        task_id=task_id,
        seed=seed,
        group_id=group_id,
        trajectory_idx=trajectory_idx,
        grader_score=grader,
        investigator_reward=raw_reward - fallback_penalty * fallback_count,
        raw_investigator_reward=raw_reward,
        turns=turns,
        fallback_count=fallback_count,
        call_count=int(getattr(investigator, "call_count", len(turns)) or len(turns)),
        n_fraudster_proposals=proposals,
    )


def collect_trajectory_group(
    *,
    task_id: str,
    seed: int,
    group_size: int,
    investigator_factory: Callable[[], Any],
    reward_mode: RewardMode = RewardMode.ENVIRONMENT,
    fallback_penalty: float = 1.0,
    max_steps: int = 200,
) -> List[TrajectoryRecord]:
    """Sample complete trajectories from one identical initial state."""
    if group_size < 2:
        raise ValueError("GRPO group_size must be at least 2")
    group_id = f"{task_id}_{seed}"
    records: List[TrajectoryRecord] = []
    for idx in range(group_size):
        try:
            import torch

            rollout_seed = seed + (idx + 1) * 100_003
            torch.manual_seed(rollout_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(rollout_seed)
        except ImportError:  # pragma: no cover
            pass
        records.append(
            _collect_one_trajectory(
                task_id=task_id,
                seed=seed,
                group_id=group_id,
                trajectory_idx=idx,
                investigator_factory=investigator_factory,
                reward_mode=reward_mode,
                fallback_penalty=fallback_penalty,
                max_steps=max_steps,
            )
        )

    advantages = compute_group_advantages(
        [record.investigator_reward for record in records]
    )
    for record, advantage in zip(records, advantages):
        record.advantage = advantage
    return records


def _turn_mean_logprob(
    model: Any,
    turn: TrajectoryTurn,
    *,
    max_prompt_tokens: int,
) -> Any:
    """Differentiable mean log-probability of one sampled action."""
    import torch

    prompt_ids = turn.prompt_token_ids[-max_prompt_tokens:]
    completion_ids = turn.completion_token_ids
    if not prompt_ids or not completion_ids:
        raise ValueError("A trajectory turn must contain prompt and completion tokens")
    device = next(model.parameters()).device
    input_ids = torch.tensor(
        [prompt_ids + completion_ids], dtype=torch.long, device=device
    )
    outputs = model(input_ids=input_ids, use_cache=False)
    prompt_len = len(prompt_ids)
    n_completion = len(completion_ids)
    token_logits = outputs.logits[
        0, prompt_len - 1 : prompt_len - 1 + n_completion, :
    ]
    targets = input_ids[0, prompt_len : prompt_len + n_completion]
    return torch.nn.functional.log_softmax(token_logits.float(), dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1).mean()


def optimise_trajectory_group(
    *,
    model: Any,
    optimizer: Any,
    records: Sequence[TrajectoryRecord],
    clip_epsilon: float = 0.2,
    max_prompt_tokens: int = 2048,
    max_grad_norm: float = 1.0,
    update_epochs: int = 1,
) -> Dict[str, float]:
    """Apply a clipped group-relative update to every action in a trajectory."""
    import torch

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    usable = [record for record in records if record.turns and abs(record.advantage) > 1e-8]
    if not usable:
        return {"loss": 0.0, "grad_norm": 0.0, "updated_turns": 0.0}

    old_logps: Dict[tuple[int, int], Any] = {}
    model.eval()
    with torch.no_grad():
        for record_idx, record in enumerate(usable):
            for turn_idx, turn in enumerate(record.turns):
                old_logps[(record_idx, turn_idx)] = _turn_mean_logprob(
                    model, turn, max_prompt_tokens=max_prompt_tokens
                ).detach()

    losses: List[float] = []
    updated_turns = 0
    grad_norm_value = 0.0
    model.train()
    for _ in range(update_epochs):
        optimizer.zero_grad(set_to_none=True)
        for record_idx, record in enumerate(usable):
            turn_weight = 1.0 / (len(usable) * len(record.turns))
            advantage = torch.tensor(
                record.advantage,
                dtype=torch.float32,
                device=next(model.parameters()).device,
            )
            for turn_idx, turn in enumerate(record.turns):
                current_logp = _turn_mean_logprob(
                    model, turn, max_prompt_tokens=max_prompt_tokens
                )
                log_ratio = torch.clamp(
                    current_logp - old_logps[(record_idx, turn_idx)], -10.0, 10.0
                )
                ratio = torch.exp(log_ratio)
                unclipped = ratio * advantage
                clipped = torch.clamp(
                    ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon
                ) * advantage
                loss = -torch.minimum(unclipped, clipped) * turn_weight
                loss.backward()
                losses.append(float(loss.detach().cpu()))
                updated_turns += 1
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
        grad_norm_value = float(grad_norm.detach().cpu())
        optimizer.step()

    return {
        "loss": sum(losses),
        "grad_norm": grad_norm_value,
        "updated_turns": float(updated_turns),
    }


def run_trajectory_grpo_collection(
    *,
    seeds_by_task: Dict[str, List[int]],
    group_size: int = 4,
    investigator_factory: Callable[[], Any],
    reward_mode: RewardMode = RewardMode.ENVIRONMENT,
    output_dir: Path,
    fallback_penalty: float = 1.0,
    max_steps: int = 200,
) -> List[TrajectoryRecord]:
    """Collect and serialize trajectory groups without updating the policy."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_records: List[TrajectoryRecord] = []
    for task_id, seeds in seeds_by_task.items():
        for seed in seeds:
            all_records.extend(
                collect_trajectory_group(
                    task_id=task_id,
                    seed=seed,
                    group_size=group_size,
                    investigator_factory=investigator_factory,
                    reward_mode=reward_mode,
                    fallback_penalty=fallback_penalty,
                    max_steps=max_steps,
                )
            )
    (output_dir / "trajectory_groups.json").write_text(
        json.dumps([record.to_dict() for record in all_records], indent=2),
        encoding="utf-8",
    )
    return all_records


__all__ = [
    "RewardMode",
    "TrajectoryRecord",
    "TrajectoryTurn",
    "collect_trajectory_group",
    "compute_group_advantages",
    "optimise_trajectory_group",
    "run_trajectory_grpo_collection",
]
