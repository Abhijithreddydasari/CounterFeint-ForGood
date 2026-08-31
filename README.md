---
title: CounterFeint
emoji: "\U0001F575\uFE0F"
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
tags:
    - openenv
    - ad-fraud
    - reinforcement-learning
    - multi-agent
    - grpo
    - trust-and-safety
---

# CounterFeint — Investigative Agents for Ad-Fraud Harm Reduction

OpenEnv arena for **budget-constrained ad-fraud investigation**. False negatives put scams in front of people; false positives kill legitimate businesses. The environment scores investigators on **fraud leaks**, ring discovery, and policy-grounded rationales — not classification accuracy.

Verification benchmark in companion work [anonymous ref].

**[Deployed Env (HF Space)](https://huggingface.co/spaces/QuantumTransformer/CounterFeint)** ·
**[Training Notebook](training/official_hf_training.ipynb)** ·
**[GitHub](https://github.com/Abhijithreddydasari/CounterFeint-ForGood)**

See [`docs/CLAIMS.md`](docs/CLAIMS.md) for supported vs unsupported claims.

---

## Why this problem

Ad fraud costs the digital advertising industry over **$100 billion annually**. Platforms reject advertisers only at high confidence: a false positive cuts off a real business; a false negative deploys a scam to millions.

Catching sophisticated fraud is **investigation**, not classification. A reviewer starts with surface signals, chooses what to inspect under time and budget, and commits to a verdict when evidence is enough. CounterFeint simulates that loop: triage, investigate, decide. Unreviewed fraud auto-approves (a leak).

## FraudArena — three agents

| Role | What it does | Default policy |
|---|---|---|
| **Fraudster** | Proposes ads designed to evade detection | `ReactiveFraudster` (scripted); LLM for eval |
| **Investigator** | Reviews a queue, spends budget on signals, verdicts and links rings | Qwen3-4B-Instruct + LoRA (target); scripted baseline |
| **Auditor** | Grades investigator reasoning (Track A) and fraudster plausibility (Track B) | Deterministic `HeuristicAuditor` |

Training targets the Investigator. Default training adversary is scripted; LLM fraudsters are for robustness eval. See [`training/RESULTS.md`](training/RESULTS.md).

### Episode flow

```
reset(task_id, seed)
  |
  v
+-- Round 1..N (up to max_rounds) --------------------------+
|  FRAUDSTER: propose_ad / modify_pending_ad / end_turn     |
|  INVESTIGATOR: investigate / verdict / link_accounts      |
+------------------------------------------------------------+
  |
  v
AUDIT PHASE — Track A (reasoning) + Track B (plausibility)
  rewards computed; unreviewed fraud counts as a leak
```

### Three escalating tasks

| Task | Ads | Budget | Challenge |
|---|---:|---:|---|
| **Basic Triage** | 5 | 25 | Investigate → verdict loop |
| **Sophisticated Fraud** | 12 | 30 | Triage under budget (~2.5 actions/ad) |
| **Fraud Network Detection** | 20 | 35 | Coordinated rings (~1.75 actions/ad) |

Task 3 rings use clique / chain / hub-spoke topologies modeled on Meta CIB operations (Ghana DigitSol, Benin Digited, China–Russia hub).

## Harm-centric evaluation

- **Fraud leak rate** — ground-truth fraud that is approved, escalated, or auto-approved.
- **Ring-link precision/recall** — coordinated accounts, by topology.
- **Policy-citation** — `FSDP-*` codes in rationales.
- **Held-out splits** — `EVAL_SEEDS` (35 episodes) plus a CIB-inspired holdout (`data/real_world_loader.py`, eval-only).
- Asymmetric rewards: false negatives penalized 5× false positives.

```bash
python -m counterfeint.experiments eval
python -m counterfeint.experiments.integrity_metrics
```

## Quick start

```bash
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

```python
from counterfeint import AdFraudEnv, AdReviewAction

with AdFraudEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(seed=42, task_id="task_1")
    result = env.step(AdReviewAction(
        action_type="investigate",
        ad_id="ad_001",
        investigation_target="landing_page",
    ))
    result = env.step(AdReviewAction(
        action_type="verdict",
        ad_id="ad_001",
        verdict="reject",
        confidence=0.9,
    ))
```

**Train:** [`training/official_hf_training.ipynb`](training/official_hf_training.ipynb) on a T4. `MODE=smoke` (~5 min), `MODE=proper` (~3 hr). Investigator training is experimental; do not cite placeholder scores.

## Layout

```
server/           # FastAPI app, environment, referee
agents/           # LLM policies
scripted/         # Deterministic baselines
data/             # Ads, CIB rings, Meta policy taxonomy, holdout loader
graders/          # Task graders + auditor tracks
training/         # GRPO notebooks, proxy reward, trajectory-group GRPO
experiments/      # Held-out eval, integrity metrics
tests/
papers/trustworthy_ai/
```

## License

BSD 3-Clause
