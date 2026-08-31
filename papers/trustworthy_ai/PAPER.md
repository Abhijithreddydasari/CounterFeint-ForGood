# CounterFeint: Accountable Investigative Agents for Ad-Fraud and Information Integrity

**Target:** Trustworthy AI for Good @ NeurIPS 2026 (6-8 pages)

## Abstract

Ad-fraud review is a harm-sensitive sequential decision problem: false negatives deploy scams; false positives harm legitimate businesses. CounterFeint simulates budget-constrained investigation with coordinated fraud rings and policy-grounded rationales. We extend the verification benchmark with diverse synthetic splits (compositional + OOD holdouts), integrity metrics (fraud leaks, ring links, policy citations), and trajectory-group GRPO training on environment rewards. We report honest before/after results with confidence intervals and a real-world-inspired holdout vignette set.

## 1. Harm Model

FN > FP. Investigation under budget, not one-shot classification.

## 2. System (extends Paper 1)

Same three-agent FraudArena + dual-track Auditor. Verification benchmark in companion work [anonymous ref].

## 3. Diverse Synthetic Data

Two-stage generator: programmatic axis sampling (violation, industry, geography, topology, budget) + optional LLM surface realization. Splits: train / val / test-compositional / test-OOD / test-realworld.

**Artifacts:** `experiments/outputs/data/dataset_manifest.json`, `coverage_report.md`

## 4. Training

Trajectory-group GRPO on `investigator_reward` / `grade_episode` (not snapshot proxy). Frozen fraudster pool for training; live adversary for eval. Investigator: Qwen3-4B-Instruct-2507 + LoRA on A100 via vLLM batching.

## 5. Evaluation

- Held-out `EVAL_SEEDS` (35 episodes)
- Fraud leak rate, grader score, bootstrap 95% CI
- Ring link precision/recall by topology
- Policy citation accuracy (FSDP-* codes)
- Real-world holdout (`data/real_world_loader.py`, eval-only)

**Artifacts:** `experiments/outputs/eval/`, `experiments/outputs/integrity/`

## 6. Case Studies

Three episode traces: fraud leak, evidence-backed catch, ring uncovered.

## 7. Ethics & Limitations

Synthetic data; dual-use fraud generator; no production validation; human oversight required. Proxy-reward GRPO ablation included if environment reward transfer fails.

## Claims Policy

See `docs/CLAIMS.md`. Do not cite placeholder `eval_outputs/eval_summary.md`.
