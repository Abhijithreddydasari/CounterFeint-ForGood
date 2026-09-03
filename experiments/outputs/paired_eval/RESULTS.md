# Deterministic paired checkpoint evaluation

## Protocol

- Compare each base model with its trajectory-GRPO LoRA checkpoint.
- Use the same 24 fresh environment seeds for both conditions: 8 each from `task_2`, `task_3`, and `task_3_unseen`.
- Use greedy decoding to remove generation-sampling noise.
- Report trained-minus-base deltas with 95% paired bootstrap confidence intervals (5,000 resamples).

## Qwen3-0.6B

- Grader score: 0.432 -> 0.390; delta -0.042 (95% CI -0.090 to +0.007).
- Fraud leak rate: 37.09% -> 46.28%; delta +9.19 percentage points (95% CI +2.14 to +15.85).
- Tool-fallback rate: 4.55% -> 7.46%; delta +2.91 points (95% CI +0.59 to +5.29).
- On `task_3_unseen`, grader score fell by 0.053 (95% CI -0.095 to -0.009), fraud leakage rose by 12.84 points (95% CI +4.79 to +19.86), and fallback rate rose by 4.17 points (95% CI +0.83 to +7.92).

## Qwen3-8B

- Grader score: 0.701 -> 0.700; delta -0.0012 (95% CI -0.0037 to +0.00002).
- Fraud leak rate: 9.24% -> 9.24%; delta 0.00 points (95% CI 0.00 to 0.00).
- Tool-fallback rate remained 0/678 in both conditions.
- Leakage was 0% on `task_2` and `task_3`, but 22.37% on `task_3_unseen` in both conditions.

## Defensible paper claim

The compute-bounded GRPO pilot did not improve either model. It left the stronger model's greedy behavior effectively unchanged and caused measurable safety regressions in the weaker model, especially under the unseen condition. The environment therefore detects both scale-dependent capability gaps and training-induced safety regressions that aggregate reward alone can conceal. Treat this as a negative pilot result, not evidence that GRPO is generally harmful.
