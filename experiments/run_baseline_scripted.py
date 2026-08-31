"""35-seed scripted baseline + integrity (Paper 2 CPU table)."""

from __future__ import annotations

from pathlib import Path

from counterfeint.eval_suite import EVAL_SEEDS
from counterfeint.experiments.integrity_metrics import run_integrity_suite
from counterfeint.experiments.run_eval import run_in_process_eval


def main() -> None:
    n = sum(len(s) for s in EVAL_SEEDS.values())
    print(f"=== Scripted baseline: {n} episodes on EVAL_SEEDS ===", flush=True)
    print({k: len(v) for k, v in EVAL_SEEDS.items()}, flush=True)

    eval_dir = Path("experiments/outputs/eval/baseline_scripted")
    payload = run_in_process_eval(output_dir=eval_dir)
    print(f"Wrote {eval_dir / 'eval_summary.md'}", flush=True)
    for tid, agg in payload["aggregates"].items():
        print(
            f"  {tid}: n={agg['n_episodes']} "
            f"grader={agg['grader_score_mean']:.3f} "
            f"leaks={agg['n_fraud_leaks_mean']:.2f}",
            flush=True,
        )

    print("=== Integrity suite on same seeds ===", flush=True)
    integ_dir = Path("experiments/outputs/integrity/baseline_scripted")
    rows = run_integrity_suite(seeds_by_task=EVAL_SEEDS, output_dir=integ_dir)
    leaks = sum(r.n_fraud_leaks for r in rows)
    fraud = sum(r.n_ground_truth_fraud for r in rows)
    rate = 100 * leaks / fraud if fraud else 0.0
    print(f"Wrote {integ_dir / 'integrity_metrics.json'}", flush=True)
    print(f"Total leaks {leaks}/{fraud} ({rate:.1f}%)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
