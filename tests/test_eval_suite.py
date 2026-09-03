"""Unit tests for counterfeint.eval_suite — parser and writer layers.

These tests intentionally stay below the network boundary: we exercise the
pure ``_parse_episode_metrics`` extraction helper and the JSON / markdown /
PNG writers against hand-crafted episode-result dicts so the test suite
runs without a live CounterFeint server.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from counterfeint.eval_suite import (
    EVAL_SEEDS,
    AggregatedMetrics,
    EpisodeMetrics,
    _aggregate_per_task,
    _parse_episode_metrics,
    _write_eval_json,
    _write_eval_plot,
    _write_eval_summary_md,
    summarize_real_world_holdout,
)


def _make_episode_result(
    *,
    task_id: str = "task_1",
    grader_score: float = 0.5,
    track_a: float = 0.9,
    track_b: float = 0.95,
    verdicts: dict | None = None,
    remaining_budget: int = 4,
    total_ads: int = 12,
    investigator_fallback: int = 0,
    steps: int = 30,
    end_reason: str | None = "audit_complete",
    error: str | None = None,
) -> dict:
    verdicts = verdicts if verdicts is not None else {}
    return {
        "task_id": task_id,
        "grader_score": grader_score,
        "steps": steps,
        "end_reason": end_reason,
        "rewards_by_role": {"investigator": 1.5, "fraudster": -0.5, "auditor": 0.0},
        "fallback_counts": {"investigator": investigator_fallback, "fraudster": 0},
        "final_state": {
            "audit_report": {
                "investigator_audit_score": track_a,
                "fraudster_plausibility_score": track_b,
            },
            "investigator_state": {
                "total_ads": total_ads,
                "remaining_budget": remaining_budget,
                "verdicts": verdicts,
            },
        },
        **({"error": error} if error is not None else {}),
    }


class TestEvalSeeds:
    # Per-task seed counts: 10 each on the training-tier tasks (task_1..3)
    # and 5 on the held-out generalisation task (task_3_unseen).  The
    # smaller count on the unseen task keeps eval wallclock from doubling
    # for what is purely a generalisation probe — see eval_suite.EVAL_SEEDS.
    EXPECTED_SEED_COUNTS = {
        "task_1": 10,
        "task_2": 10,
        "task_3": 10,
        "task_3_unseen": 5,
        "task_4": 5,
    }

    def test_expected_tasks_with_expected_seed_counts(self) -> None:
        assert set(EVAL_SEEDS.keys()) == set(self.EXPECTED_SEED_COUNTS)
        for task_id, expected in self.EXPECTED_SEED_COUNTS.items():
            seeds = EVAL_SEEDS[task_id]
            assert len(seeds) == expected, f"{task_id} has wrong seed count"
            assert len(set(seeds)) == expected, f"{task_id} has duplicate seeds"

    def test_seeds_disjoint_from_training_seed(self) -> None:
        all_seeds = {s for seeds in EVAL_SEEDS.values() for s in seeds}
        # Training baseline uses seed=42 and small self-play seeds; eval
        # seeds live in the 1000+ range so they never collide.
        assert 42 not in all_seeds
        assert all(s >= 1000 for s in all_seeds)

    def test_seed_ranges_disjoint_across_tasks(self) -> None:
        """Each task owns a distinct seed range so an eval failure can be
        traced to one task without ambiguity."""
        seen: dict = {}
        for task_id, seeds in EVAL_SEEDS.items():
            for s in seeds:
                assert s not in seen, f"seed {s} reused across {seen[s]} and {task_id}"
                seen[s] = task_id


class TestParseEpisodeMetrics:
    def test_parses_headline_fields(self) -> None:
        result = _make_episode_result()
        m = _parse_episode_metrics("before", "task_1", 1001, result)
        assert isinstance(m, EpisodeMetrics)
        assert m.tag == "before"
        assert m.task_id == "task_1"
        assert m.seed == 1001
        assert m.grader_score == pytest.approx(0.5)
        assert m.track_a_score == pytest.approx(0.9)
        assert m.track_b_score == pytest.approx(0.95)
        assert m.steps == 30
        assert m.end_reason == "audit_complete"
        assert m.rewards_by_role["investigator"] == 1.5

    def test_counts_fraud_leaks_and_ground_truth_totals(self) -> None:
        result = _make_episode_result(
            verdicts={
                "ad_1": {"verdict": "approve", "ground_truth": "fraud"},
                "ad_2": {"verdict": "reject", "ground_truth": "fraud"},
                "ad_3": {"verdict": "approve", "ground_truth": "legit"},
                "ad_4": {"verdict": "approve", "ground_truth": "fraud"},
                "ad_5": {"verdict": "escalate", "ground_truth": "escalate"},
            }
        )
        m = _parse_episode_metrics("x", "task_1", 1, result)
        assert m.n_ground_truth_fraud == 3
        assert m.n_fraud_leaks == 2  # ad_1 and ad_4

    def test_budget_used_pct_from_remaining_budget(self) -> None:
        result = _make_episode_result(total_ads=10, remaining_budget=3)
        m = _parse_episode_metrics("x", "task_1", 1, result)
        # 10 total ads, 3 left => 7/10 = 0.7 consumed
        assert m.budget_used_pct == pytest.approx(0.7)

    def test_budget_pct_clamps_to_unit_interval(self) -> None:
        # remaining_budget can exceed total_ads in degenerate cases — clamp.
        result = _make_episode_result(total_ads=5, remaining_budget=100)
        m = _parse_episode_metrics("x", "task_1", 1, result)
        assert 0.0 <= m.budget_used_pct <= 1.0

    def test_budget_pct_zero_when_no_ads(self) -> None:
        result = _make_episode_result(total_ads=0, remaining_budget=0)
        m = _parse_episode_metrics("x", "task_1", 1, result)
        assert m.budget_used_pct == 0.0

    def test_investigator_fallback_count_extracted(self) -> None:
        result = _make_episode_result(investigator_fallback=4)
        m = _parse_episode_metrics("x", "task_1", 1, result)
        assert m.fallback_count == 4

    def test_missing_audit_report_defaults_to_one(self) -> None:
        result = _make_episode_result()
        result["final_state"]["audit_report"] = {}
        m = _parse_episode_metrics("x", "task_1", 1, result)
        assert m.track_a_score == pytest.approx(1.0)
        assert m.track_b_score == pytest.approx(1.0)

    def test_error_round_trips(self) -> None:
        result = _make_episode_result(error="boom")
        m = _parse_episode_metrics("x", "task_1", 1, result)
        assert m.error == "boom"


class TestAggregation:
    def test_aggregates_only_valid_episodes(self) -> None:
        eps = [
            _parse_episode_metrics(
                "after", "task_1", 1, _make_episode_result(grader_score=0.8)
            ),
            _parse_episode_metrics(
                "after", "task_1", 2, _make_episode_result(grader_score=0.6)
            ),
            _parse_episode_metrics(
                "after",
                "task_1",
                3,
                _make_episode_result(grader_score=0.0, error="boom"),
            ),
        ]
        agg = _aggregate_per_task("after", "task_1", eps)
        assert isinstance(agg, AggregatedMetrics)
        assert agg.n_episodes == 2  # the errored one is excluded
        assert agg.errors == 1
        assert agg.grader_score_mean == pytest.approx(0.7)

    def test_all_errors_returns_zeroed_aggregate(self) -> None:
        eps = [
            _parse_episode_metrics(
                "x",
                "task_1",
                1,
                _make_episode_result(error="x", investigator_fallback=2),
            )
        ]
        agg = _aggregate_per_task("x", "task_1", eps)
        assert agg.n_episodes == 0
        assert agg.errors == 1
        assert agg.fallback_count_total == 2


class TestArtefactWriters:
    def _make_before_after(self, tmp_path: Path) -> tuple:
        before_eps = {
            "task_1": [
                _parse_episode_metrics(
                    "before",
                    "task_1",
                    seed,
                    _make_episode_result(grader_score=0.4, track_a=0.7),
                )
                for seed in EVAL_SEEDS["task_1"][:2]
            ]
        }
        after_eps = {
            "task_1": [
                _parse_episode_metrics(
                    "after",
                    "task_1",
                    seed,
                    _make_episode_result(grader_score=0.8, track_a=0.95),
                )
                for seed in EVAL_SEEDS["task_1"][:2]
            ]
        }
        before_agg = {"task_1": _aggregate_per_task("before", "task_1", before_eps["task_1"])}
        after_agg = {"task_1": _aggregate_per_task("after", "task_1", after_eps["task_1"])}
        return before_eps, after_eps, before_agg, after_agg

    def test_write_eval_json_roundtrips(self, tmp_path: Path) -> None:
        before_eps, after_eps, _, _ = self._make_before_after(tmp_path)
        out = tmp_path / "eval_results.json"
        _write_eval_json(before_eps, after_eps, "before", "after", out)

        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["schema"] == "counterfeint.eval_suite.v1"
        assert loaded["tags"] == {"before": "before", "after": "after"}
        assert len(loaded["before"]["task_1"]) == 2
        assert len(loaded["after"]["task_1"]) == 2

    def test_write_summary_md_mentions_delta(self, tmp_path: Path) -> None:
        _, _, before_agg, after_agg = self._make_before_after(tmp_path)
        out = tmp_path / "eval_summary.md"
        _write_eval_summary_md(before_agg, after_agg, "before", "after", out)

        text = out.read_text(encoding="utf-8")
        assert "before" in text
        assert "after" in text
        assert "grader_score" in text
        assert "track_a_score" in text
        # after > before, so we expect a "+" in the delta column.
        assert "+0.400" in text or "+0.4" in text

    def test_write_eval_plot_creates_png_or_stub(self, tmp_path: Path) -> None:
        _, _, before_agg, after_agg = self._make_before_after(tmp_path)
        out = tmp_path / "eval_plot.png"
        _write_eval_plot(before_agg, after_agg, "before", "after", out)

        # Either the PNG was written (matplotlib installed) or the .txt stub was.
        assert out.exists() or out.with_suffix(".txt").exists()

    def test_write_eval_json_includes_holdout_summary(self, tmp_path: Path) -> None:
        before_eps, after_eps, _, _ = self._make_before_after(tmp_path)
        out = tmp_path / "eval_results.json"
        holdout = {"n_ads_total": 15, "n_case_studies": 4}
        _write_eval_json(
            before_eps, after_eps, "before", "after", out, holdout_summary=holdout
        )
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["real_world_holdout"] == holdout


class TestRealWorldHoldoutSummary:
    def test_summary_reports_15_ads(self) -> None:
        s = summarize_real_world_holdout()
        assert s["n_ads_total"] == 15
        assert s["n_case_studies"] >= 3
        assert "Ghana DigitSol-style" in s["case_studies"]
        assert "Benin Digited-style" in s["case_studies"]
        assert "China-Russia-style hub" in s["case_studies"]
        assert sum(s["ads_per_case_study"].values()) == s["n_ads_total"]
