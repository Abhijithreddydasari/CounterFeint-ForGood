# CounterFeint — Frozen Claims Registry

Last updated: 2026-08-31. Every paper claim must trace to an artifact in `experiments/outputs/`. Verifier (Auditor) results live in companion work [anonymous ref] and are not re-reported here.

## Supported claims (with evidence)

| Claim | Evidence | Status |
|-------|----------|--------|
| Three-agent FraudArena with budgeted investigation | `server/referee.py`, `tests/test_three_agent_episode.py` | Implemented |
| Deterministic dual-track Auditor (Track A + Track B) | `graders/auditor_track_a.py`, `graders/auditor_track_b.py` | Implemented |
| Integrity metrics: leaks, ring links, policy citations | `experiments/integrity_metrics.py` | Implemented |
| CIB-inspired holdout loader (eval-only) | `data/real_world_loader.py`, `data/real_world_test_set.json` | Implemented; not yet wired into live episodes |
| Stage-1 split manifest | `experiments/outputs/data/dataset_manifest.json` | Generated (train/val/compositional/OOD); not yet evaluated as episodes |
| Scripted eval smoke (4 episodes) | `experiments/outputs/eval/eval_summary.md` | Exists; **not** the 35-seed `EVAL_SEEDS` sweep |

## Weak / pending (report, do not headline)

| Claim | Notes |
|-------|--------|
| Full 35-seed scripted baseline | Re-run `python -m counterfeint.experiments eval`; current file is a 4-episode smoke |
| Holdout ads inside episodes | Loader exists; integrity suite currently records metadata only |
| Compositional / OOD episode eval | Manifest exists; no policy has been scored on those splits |
| Trajectory-group GRPO on environment reward | Collector exists; not wired to `GRPOTrainer` |
| Policy-citation precision | Metric is loose (any `FSDP-*` counts) |

## Unsupported — do NOT claim

| Claim | Why |
|-------|-----|
| Trained 0.6B beats frozen 8B fraudster | Training uses `ReactiveFraudster`; 8B Ollama path 0/6 episodes |
| +0.18 / +0.35 grader improvement | Placeholder eval purged |
| GRPO materially improves fraud detection | Only +0.005 on 3 non-held-out episodes; trained row still `_pending_` |
| Reduces real ad fraud / production validated | Simulation only |
| Solved multi-agent oversight | Deterministic rules, one domain |
| Auditor corruption F1 / anti-gaming Δ | Companion paper; artifacts not in this repo |

## Artifact locations (this paper)

- Scripted baseline (full sweep still needed): `experiments/outputs/eval/`
- Integrity (partial seeds): `experiments/outputs/integrity/`
- Dataset splits: `experiments/outputs/data/`
- Training: `training/RESULTS.md` (cite only rows with a committed Source)
- Paper: `papers/trustworthy_ai/`
