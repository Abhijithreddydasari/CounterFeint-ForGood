"""
In-process held-out evaluation (no HTTP server).

Uses ScriptedInvestigator as the default baseline policy. Suitable for
reproducible CPU baselines before vLLM investigator eval.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    from ..eval_suite import EVAL_SEEDS, AggregatedMetrics, EpisodeMetrics
    from ..graders.auditor_pipeline import run_full_audit
    from ..graders.base_grader import grade_episode
    from ..scripted import ScriptedInvestigator
    from .episode_bundle import bundle_to_episode_record, run_episode_bundle
except ImportError:
    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from eval_suite import EVAL_SEEDS, AggregatedMetrics, EpisodeMetrics
    from experiments.episode_bundle import bundle_to_episode_record, run_episode_bundle
    from graders.auditor_pipeline import run_full_audit
    from graders.base_grader import grade_episode
    from scripted import ScriptedInvestigator


PolicyFactory = Callable[[], Any]


def _episode_metrics(
    *,
    tag: str,
    task_id: str,
    seed: int,
    investigator_factory: PolicyFactory,
) -> EpisodeMetrics:
    bundle = run_episode_bundle(
        task_id=task_id,
        seed=seed,
        investigator_factory=investigator_factory,
    )
    record = bundle_to_episode_record(bundle)
    grader_score = grade_episode(record)
    audit = run_full_audit(
        record=record,
        investigator_action_log=bundle.investigator_actions,
        investigation_data_seen=bundle.investigation_data_seen,
        fraudster_proposal_log=bundle.fraudster_proposals,
    )

    n_fraud = sum(
        1 for m in record.ads_metadata if m.get("ground_truth") == "fraud"
    )
    leaks = sum(
        1
        for v in record.verdicts
        if v.ground_truth == "fraud"
        and v.verdict in ("approve", "escalate")
        and not v.auto_approved
    ) + sum(
        1
        for v in record.verdicts
        if v.ground_truth == "fraud" and v.auto_approved
    )

    budget_used = record.total_steps
    budget_pct = (
        budget_used / record.action_budget if record.action_budget else 0.0
    )

    return EpisodeMetrics(
        tag=tag,
        task_id=task_id,
        seed=seed,
        grader_score=grader_score,
        track_a_score=audit.investigator_audit_score,
        track_b_score=audit.fraudster_plausibility_score,
        n_fraud_leaks=leaks,
        n_ground_truth_fraud=n_fraud,
        budget_used_pct=min(1.0, budget_pct),
        fallback_count=0,
        steps=record.total_steps,
        end_reason=None,
        rewards_by_role={},
    )


def _aggregate(tag: str, task_id: str, rows: List[EpisodeMetrics]) -> AggregatedMetrics:
    n = len(rows)
    if n == 0:
        return AggregatedMetrics(
            tag=tag,
            task_id=task_id,
            n_episodes=0,
            grader_score_mean=0.0,
            track_a_score_mean=0.0,
            n_fraud_leaks_mean=0.0,
            budget_used_pct_mean=0.0,
            fallback_count_total=0,
            errors=0,
        )
    return AggregatedMetrics(
        tag=tag,
        task_id=task_id,
        n_episodes=n,
        grader_score_mean=sum(r.grader_score for r in rows) / n,
        track_a_score_mean=sum(r.track_a_score for r in rows) / n,
        n_fraud_leaks_mean=sum(r.n_fraud_leaks for r in rows) / n,
        budget_used_pct_mean=sum(r.budget_used_pct for r in rows) / n,
        fallback_count_total=sum(r.fallback_count for r in rows),
        errors=sum(1 for r in rows if r.error),
    )


def run_in_process_eval(
    *,
    output_dir: Path,
    seeds: Optional[Dict[str, List[int]]] = None,
    tag: str = "scripted_baseline",
    investigator_factory: Optional[PolicyFactory] = None,
) -> Dict[str, Any]:
    seeds = seeds or EVAL_SEEDS
    investigator_factory = investigator_factory or (lambda: ScriptedInvestigator())
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[EpisodeMetrics] = []
    by_task: Dict[str, List[EpisodeMetrics]] = {}

    n_total = sum(len(s) for s in seeds.values())
    done = 0
    for task_id, task_seeds in seeds.items():
        task_rows: List[EpisodeMetrics] = []
        for seed in task_seeds:
            row = _episode_metrics(
                tag=tag,
                task_id=task_id,
                seed=seed,
                investigator_factory=investigator_factory,
            )
            task_rows.append(row)
            all_rows.append(row)
            done += 1
            print(
                f"[{done}/{n_total}] {task_id} seed={seed} "
                f"grader={row.grader_score:.3f} leaks={row.n_fraud_leaks}/"
                f"{row.n_ground_truth_fraud}",
                flush=True,
            )
        by_task[task_id] = task_rows

    aggregates = {
        task_id: asdict(_aggregate(tag, task_id, rows))
        for task_id, rows in by_task.items()
    }

    payload = {
        "tag": tag,
        "episodes": [asdict(r) for r in all_rows],
        "aggregates": aggregates,
    }
    (output_dir / "eval_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    lines = [
        f"# In-process eval: `{tag}`",
        "",
        "| Task | grader_score | track_a | track_b | fraud_leaks | budget_pct |",
        "|------|-------------:|--------:|--------:|------------:|-----------:|",
    ]
    for task_id, agg in aggregates.items():
        lines.append(
            f"| {task_id} | {agg['grader_score_mean']:.3f} | "
            f"{agg['track_a_score_mean']:.3f} | — | "
            f"{agg['n_fraud_leaks_mean']:.2f} | {agg['budget_used_pct_mean']:.2f} |"
        )
    (output_dir / "eval_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload
