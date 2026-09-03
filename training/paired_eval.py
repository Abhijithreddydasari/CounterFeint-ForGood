"""Analysis helpers for paired base-versus-adapter evaluation."""

from __future__ import annotations

import random
import statistics
from typing import Any, Dict, List, Sequence, Tuple


EpisodeRow = Dict[str, Any]
Pair = Tuple[EpisodeRow, EpisodeRow]


def _key(row: EpisodeRow) -> tuple[str, int]:
    return str(row["task_id"]), int(row["seed"])


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def pair_episode_rows(
    base_rows: Sequence[EpisodeRow], trained_rows: Sequence[EpisodeRow]
) -> List[Pair]:
    """Match conditions by task and environment seed, rejecting incomplete pairs."""
    base = {_key(row): row for row in base_rows}
    trained = {_key(row): row for row in trained_rows}
    if len(base) != len(base_rows) or len(trained) != len(trained_rows):
        raise ValueError("Duplicate task/seed rows in paired evaluation")
    if base.keys() != trained.keys():
        missing_trained = sorted(base.keys() - trained.keys())
        missing_base = sorted(trained.keys() - base.keys())
        raise ValueError(
            "Paired evaluation keys differ: "
            f"missing_trained={missing_trained}, missing_base={missing_base}"
        )
    return [(base[key], trained[key]) for key in sorted(base)]


def summarise_episode_rows(rows: Sequence[EpisodeRow]) -> Dict[str, float]:
    if not rows:
        return {
            "n_episodes": 0,
            "grader_mean": 0.0,
            "leak_rate": 0.0,
            "track_a_mean": 0.0,
            "fallback_rate": 0.0,
            "fallback_total": 0,
            "call_total": 0,
        }
    fraud = sum(int(row["n_ground_truth_fraud"]) for row in rows)
    leaks = sum(int(row["n_fraud_leaks"]) for row in rows)
    fallbacks = sum(int(row["fallback_count"]) for row in rows)
    calls = sum(int(row["call_count"]) for row in rows)
    return {
        "n_episodes": len(rows),
        "grader_mean": statistics.mean(float(row["grader_score"]) for row in rows),
        "leak_rate": _ratio(leaks, fraud),
        "track_a_mean": statistics.mean(float(row["track_a_score"]) for row in rows),
        "fallback_rate": _ratio(fallbacks, calls),
        "fallback_total": fallbacks,
        "call_total": calls,
    }


def _paired_metrics(pairs: Sequence[Pair]) -> Dict[str, float]:
    base_rows = [pair[0] for pair in pairs]
    trained_rows = [pair[1] for pair in pairs]
    base = summarise_episode_rows(base_rows)
    trained = summarise_episode_rows(trained_rows)
    return {
        "grader_delta": statistics.mean(
            float(trained_row["grader_score"]) - float(base_row["grader_score"])
            for base_row, trained_row in pairs
        ),
        "leak_rate_delta": trained["leak_rate"] - base["leak_rate"],
        "track_a_delta": statistics.mean(
            float(trained_row["track_a_score"]) - float(base_row["track_a_score"])
            for base_row, trained_row in pairs
        ),
        "fallback_rate_delta": trained["fallback_rate"] - base["fallback_rate"],
    }


def _percentile_interval(values: List[float], alpha: float) -> Dict[str, float]:
    values.sort()
    n = len(values)
    return {
        "lo": values[max(0, int(alpha / 2 * n))],
        "hi": values[min(n - 1, int((1 - alpha / 2) * n) - 1)],
    }


def paired_bootstrap_deltas(
    pairs: Sequence[Pair],
    *,
    n_boot: int = 5000,
    alpha: float = 0.05,
    random_seed: int = 17,
) -> Dict[str, Dict[str, Any]]:
    """Bootstrap paired episode deltas while preserving condition pairing."""
    if not pairs:
        raise ValueError("At least one episode pair is required")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    observed = _paired_metrics(pairs)
    samples: Dict[str, List[float]] = {metric: [] for metric in observed}
    rng = random.Random(random_seed)
    for _ in range(n_boot):
        draw = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        metrics = _paired_metrics(draw)
        for metric, value in metrics.items():
            samples[metric].append(value)

    result: Dict[str, Dict[str, Any]] = {}
    for metric, point in observed.items():
        interval = _percentile_interval(samples[metric], alpha)
        result[metric] = {
            "point": point,
            "ci95": interval,
            "excludes_zero": interval["lo"] > 0.0 or interval["hi"] < 0.0,
        }
    return result


def analyse_paired_conditions(
    base_rows: Sequence[EpisodeRow],
    trained_rows: Sequence[EpisodeRow],
    *,
    n_boot: int = 5000,
) -> Dict[str, Any]:
    pairs = pair_episode_rows(base_rows, trained_rows)
    task_ids = sorted({_key(base)[0] for base, _ in pairs})
    by_task: Dict[str, Any] = {}
    for task_id in task_ids:
        task_pairs = [pair for pair in pairs if _key(pair[0])[0] == task_id]
        by_task[task_id] = {
            "base": summarise_episode_rows([pair[0] for pair in task_pairs]),
            "trained": summarise_episode_rows([pair[1] for pair in task_pairs]),
            "paired_deltas": paired_bootstrap_deltas(
                task_pairs, n_boot=n_boot, random_seed=17
            ),
        }
    return {
        "n_pairs": len(pairs),
        "deterministic_decoding": True,
        "base": summarise_episode_rows([pair[0] for pair in pairs]),
        "trained": summarise_episode_rows([pair[1] for pair in pairs]),
        "paired_deltas": paired_bootstrap_deltas(pairs, n_boot=n_boot),
        "by_task": by_task,
        "pairs": [
            {
                "task_id": base["task_id"],
                "seed": base["seed"],
                "base": base,
                "trained": trained,
            }
            for base, trained in pairs
        ],
    }


__all__ = [
    "analyse_paired_conditions",
    "pair_episode_rows",
    "paired_bootstrap_deltas",
    "summarise_episode_rows",
]
