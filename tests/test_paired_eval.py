from __future__ import annotations

import pytest

from counterfeint.training.paired_eval import (
    analyse_paired_conditions,
    pair_episode_rows,
    paired_bootstrap_deltas,
)


def _row(
    task_id: str,
    seed: int,
    *,
    grader: float,
    leaks: int,
    fraud: int = 10,
    track_a: float = 0.0,
    fallbacks: int = 0,
    calls: int = 10,
) -> dict:
    return {
        "task_id": task_id,
        "seed": seed,
        "grader_score": grader,
        "n_fraud_leaks": leaks,
        "n_ground_truth_fraud": fraud,
        "track_a_score": track_a,
        "fallback_count": fallbacks,
        "call_count": calls,
    }


def test_pair_episode_rows_matches_by_task_and_seed() -> None:
    base = [
        _row("task_3", 2, grader=0.2, leaks=2),
        _row("task_2", 1, grader=0.1, leaks=1),
    ]
    trained = [
        _row("task_2", 1, grader=0.3, leaks=0),
        _row("task_3", 2, grader=0.4, leaks=1),
    ]
    pairs = pair_episode_rows(base, trained)
    assert [(pair[0]["task_id"], pair[0]["seed"]) for pair in pairs] == [
        ("task_2", 1),
        ("task_3", 2),
    ]


def test_pair_episode_rows_rejects_incomplete_pairing() -> None:
    with pytest.raises(ValueError, match="keys differ"):
        pair_episode_rows(
            [_row("task_2", 1, grader=0.1, leaks=1)],
            [_row("task_2", 2, grader=0.1, leaks=1)],
        )


def test_paired_bootstrap_reports_exact_constant_delta() -> None:
    base = [
        _row("task_2", seed, grader=0.4, leaks=4, track_a=0.1, fallbacks=2)
        for seed in range(4)
    ]
    trained = [
        _row("task_2", seed, grader=0.5, leaks=2, track_a=0.3, fallbacks=1)
        for seed in range(4)
    ]
    result = paired_bootstrap_deltas(
        pair_episode_rows(base, trained), n_boot=100
    )
    assert result["grader_delta"]["point"] == pytest.approx(0.1)
    assert result["grader_delta"]["ci95"] == pytest.approx(
        {"lo": 0.1, "hi": 0.1}
    )
    assert result["leak_rate_delta"]["point"] == pytest.approx(-0.2)
    assert result["track_a_delta"]["point"] == pytest.approx(0.2)
    assert result["fallback_rate_delta"]["point"] == pytest.approx(-0.1)


def test_analysis_contains_overall_and_task_level_results() -> None:
    base = [
        _row("task_2", 1, grader=0.4, leaks=4),
        _row("task_3_unseen", 2, grader=0.6, leaks=1),
    ]
    trained = [
        _row("task_2", 1, grader=0.5, leaks=3),
        _row("task_3_unseen", 2, grader=0.7, leaks=0),
    ]
    result = analyse_paired_conditions(base, trained, n_boot=50)
    assert result["n_pairs"] == 2
    assert result["deterministic_decoding"] is True
    assert set(result["by_task"]) == {"task_2", "task_3_unseen"}
    assert result["paired_deltas"]["grader_delta"]["point"] == pytest.approx(0.1)
