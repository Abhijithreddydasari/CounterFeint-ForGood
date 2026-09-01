# CounterFeint - Training on Hugging Face

Step-by-step playbook for taking the Investigator from the current ~0.6 mean
`grader_score` baseline to a trained checkpoint with reward + loss curves and a
HF Hub release. All compute is sized for the **$30 HF Pro / Spaces credit**.

---

## TL;DR (the whole pipeline in 4 commands)

1. **Baseline eval** -> `baseline_eval.ipynb` on a T4 Space (~30 min, $0.20)
2. **Train** -> `official_hf_training.ipynb` on a T4 Space, `MODE = "proper"` (~3 hr, $1.20)
3. **Compare** -> `compare_runs.ipynb` locally (free, no GPU)
4. **Push** -> set `PUSH_TO_HUB = True` in the training notebook to ship the LoRA
  adapter + `eval_summary.json` to the Hub

That's one full bake-off run. You can afford ~20 of them inside the $30 budget.

---

## 0. What lives where

```
counterfeint/training/
├── baseline_eval.ipynb         # NEW   pre-training, multi-model bake-off
├── official_hf_training.ipynb  # main  GRPO training + post-training eval
├── compare_runs.ipynb          # NEW   aggregates baseline + trained runs into plots
├── proxy_reward.py             # deterministic reward function used during GRPO
├── rollout.py                  # in-process episode collector (no HTTP server)
├── smoke_official_hf.py        # quick local pipeline check (skip if you trust the notebooks)
└── TRAINING_GUIDE.md           # this file
```

After a baseline + training run, the directory tree looks like:

```
baseline_outputs/
├── qwen3-0.6b/baseline_results.json       # per-episode rows for that model
├── qwen2.5-1.5b/baseline_results.json
├── qwen3-1.7b/baseline_results.json
├── baseline_summary.json
└── baseline_comparison.png                # bar chart for the README

outputs/
└── counterfeint-investigator-qwen3-06b-grpo/  # one directory per training run
    ├── lora_adapter/                      # LoRA weights + tokenizer
    │   ├── adapter_config.json
    │   └── adapter_model.safetensors
    ├── eval_summary.json                  # before / after grader_score
    ├── log_history.json                   # raw TRL log (loss, reward, kl)
    ├── training_config.json               # exact config that produced this run
    ├── training_curves.png                # combined loss / reward / KL plot
    └── eval_plot.png                      # per-episode before / after bars

comparison_outputs/
├── before_after_grader.png                # headline plot
├── training_curves.png                    # multi-run overlay
└── comparison_table.csv
```

---

## 1. Pick your compute lane

You have **two** sensible options for running these notebooks. Both work.

### Lane A - HF Spaces with JupyterLab (uses HF credits directly)

Best when: you specifically want to spend the $30 HF credit, want artifacts
to live next to your Space, or want a persistent dev environment.

1. Go to [https://huggingface.co/new-space](https://huggingface.co/new-space).
2. Pick the **"JupyterLab"** Docker template (or "Notebooks").
3. Hardware: **T4 small** (`$0.40 / hr`). For multi-model ablations you can
  bump to **A10G small** (`$1.05 / hr`) to halve wall time.
4. Add a persistent disk (50 GB is plenty).
5. Once the Space is running, open the JupyterLab UI and either:
  - `git clone` your repo into `/data/`, or
  - upload the `counterfeint/` directory through the file browser.
6. Open `counterfeint/training/baseline_eval.ipynb` and run cell-by-cell.

**Cost reality:** T4 at $0.40/hr means a 30 min baseline + 3 hr proper training
run is ~**$1.40**. You can do ~20 such cycles inside $30.

### Lane B - Google Colab (free T4) + push artifacts to HF Hub

Best when: you want the cheapest path and don't care that the compute is
Google's; the $30 stays available for HF Inference Endpoints later (e.g. the
Llama 3.1 8B Fraudster for the demo video).

1. Open Colab ([https://colab.research.google.com/](https://colab.research.google.com/)).
2. `Runtime -> Change runtime type -> T4 GPU`.
3. Upload `baseline_eval.ipynb` (or open from GitHub via `File -> Open notebook`).
4. The first cell autodetects Colab and clones the repo for you.
5. Run cells. Push the `outputs/` and `baseline_outputs/` folders to your HF
  dataset repo at the end.

**Strong recommendation:** start in Colab to debug, then move to HF Spaces only
once you trust the pipeline end-to-end. This stretches the $30 further.

---

## 2. Run the BEFORE eval (baseline_eval.ipynb)

### What it does

Loads each base model in `MODELS = [...]`, runs **9 episodes** per model
(`task_1, task_2, task_3` x 3 held-out seeds), and writes:

- `baseline_outputs/<tag>/baseline_results.json`
- `baseline_outputs/baseline_summary.json`
- `baseline_outputs/baseline_comparison.png`

### How to run

1. Open `baseline_eval.ipynb` on your chosen GPU.
2. **Section 1** - run install cells. Restart the kernel if Colab asks.
3. **Section 1** - run `notebook_login()` and paste your HF token (READ scope
  is enough for base models). Skip if your token is already cached.
4. **Section 2** - edit `MODELS` if you want to drop a model. Default list:
  ```python
   MODELS = [
       ("Qwen/Qwen3-0.6B",            "qwen3-0.6b"),
       ("Qwen/Qwen2.5-1.5B-Instruct", "qwen2.5-1.5b"),
       ("Qwen/Qwen3-1.7B",            "qwen3-1.7b"),
   ]
  ```
5. Run all cells. Total wall time on T4: **~30 min** (3 models x ~10 min).
6. Inspect `baseline_outputs/baseline_comparison.png`. This is your "BEFORE"
  figure for the writeup.

### What the numbers should look like

From recent local runs (Qwen2.5-1.5B-Instruct with the in-process driver):


| Task    | Mean grader_score |
| ------- | ----------------- |
| task_1  | 0.84              |
| task_2  | 0.64              |
| task_3  | 0.32              |
| overall | 0.60              |


If your numbers differ by more than 0.1 on `task_1`, double-check the
in-process driver is healthy (no `[policy crash]` or `[env reject]` messages
in Section 4 output).

### (optional) Push baselines to the Hub

In Section 6, set:

```python
BASELINE_HUB_REPO_ID = "your-username/counterfeint-baselines"
```

then re-run that cell. Creates a public dataset repo with the JSON + PNG
artifacts.

---

## 3. Run the training (official_hf_training.ipynb)

### What it does

GRPO trains Qwen3-0.6B + LoRA on rollouts collected from your environment,
using `proxy_reward_fn` for fast deterministic per-completion scoring. Then
runs the same eval suite the baseline notebook used and saves a
before/after summary.

### How to run

1. Open `official_hf_training.ipynb` on the same GPU.
2. **Section 2** - pick a `MODE`:

  | MODE     | seeds | epochs | rollouts | wall time (T4) | use for                       |
  | -------- | ----- | ------ | -------- | -------------- | ----------------------------- |
  | `smoke`  | 2     | 1      | ~12      | ~10 min        | "does the pipeline build"     |
  | `demo`   | 6     | 1      | ~36      | ~40 min        | demo deck / video screen-grab |
  | `proper` | 12    | 2      | ~72      | ~3 hr          | the run that ships            |
  | `full`   | 24    | 3      | ~144     | ~6-8 hr        | "final main result" (A10G)    |

   Start with `proper`. If wall time matters, drop to `demo`.
3. Set `BASE_MODEL`. Defaults to `Qwen/Qwen3-0.6B`. To re-run with a different
  base model later, change this and the `TRAINED_TAG`.
4. Set `TRAINED_TAG` to something descriptive: e.g. `qwen3-0.6b-r16-proper`. Each
  run gets its own `outputs/<TRAINED_TAG>/` directory so they don't overwrite.
5. Set `PUSH_TO_HUB`:
  ```python
   PUSH_TO_HUB = True
   HUB_REPO_ID = "your-username/counterfeint-investigator"
  ```
6. Set `RUN_BEFORE_EVAL = True` for the FIRST run of any base model (so you
  get the matching "BEFORE" numbers for that run). For subsequent ablations
   on the SAME base model you can flip it to `False` to save ~10 min.
7. Run all cells. Watch the Section 5 (training) cell — TRL prints
  `loss`, `reward`, `kl` every `logging_steps`. Reward should creep up
   monotonically; if it's flat for the first 30 steps, see "Troubleshooting"
   below.

### Outputs

After the notebook finishes, `outputs/<TRAINED_TAG>/` contains everything you
need for the writeup:

- `eval_summary.json` - mean before/after grader_score (the headline number)
- `log_history.json`  - raw TRL log
- `training_curves.png` - combined loss / reward / KL plot
- `eval_plot.png`     - per-episode before/after bars
- `adapter_model.safetensors` - the trained LoRA adapter
- `training_config.json` - the exact config that produced this run

If `PUSH_TO_HUB = True`, all of these are mirrored to the HF Hub repo.

---

## 4. (optional) Run multiple training jobs for an ablation

Repeat Section 3 with different settings to populate `compare_runs.ipynb`:

```python
# run #1
BASE_MODEL  = "Qwen/Qwen3-0.6B"
TRAINED_TAG = "qwen3-0.6b-r16-proper"

# run #2 (bigger LoRA)
BASE_MODEL  = "Qwen/Qwen3-0.6B"
TRAINED_TAG = "qwen3-0.6b-r32-proper"
LORA_R, LORA_ALPHA = 32, 64

# run #3 (bigger base)
BASE_MODEL  = "Qwen/Qwen2.5-1.5B-Instruct"
TRAINED_TAG = "qwen2.5-1.5b-r16-proper"
```

Each run writes a separate `outputs/<TRAINED_TAG>/` directory, so you can collect
3-4 different ablations. Total budget: 3 runs x $1.20 = ~$3.60 on T4.

---

## 5. Aggregate everything (compare_runs.ipynb)

Runs **locally** (no GPU). Just `jupyter notebook compare_runs.ipynb` or
open it in Cursor. It auto-discovers:

- every `baseline_outputs/<tag>/baseline_results.json`
- every `outputs/<run_tag>/eval_summary.json`
- every `outputs/<run_tag>/log_history.json`

and produces:

- `comparison_outputs/before_after_grader.png` - the headline figure for your
README and slide deck
- `comparison_outputs/training_curves.png` - reward / loss / KL overlaid
across all runs
- `comparison_outputs/comparison_table.csv` - the table for the README

---

## 6. What to put in the README and submission

The hackathon submission asks for:

1. **A working training script** (Colab notebook) -> `official_hf_training.ipynb`
2. **Loss + reward plots from a real run** -> `outputs/<TRAINED_TAG>/training_curves.png`
  and `comparison_outputs/training_curves.png`
3. **Push your environment to a HF Space** -> already covered by the Space
  you set up in Step 1
4. **README that motivates the problem and shows results** ->
  `comparison_outputs/before_after_grader.png` is your hero figure

Suggested README skeleton:

```markdown
## Results (example layout — replace with `experiments/outputs/eval/` artifacts)

| Model              | Baseline | Trained | Delta |
|--------------------|---------:|--------:|------:|
| _pending_          |    —     |    —    |   —   |

Do **not** cite placeholder +0.18 deltas. See `docs/CLAIMS.md`.
```

---

## 7. Fraudster LLM choice (your question)

You're right that the Fraudster is **inference-only** — we never gradient
update the Fraudster, only the Investigator. So you have flexibility here:


| Option                            | Where it runs          | Pros                            | Cons                                     |
| --------------------------------- | ---------------------- | ------------------------------- | ---------------------------------------- |
| `ScriptedFraudster` (current)     | in-process, free       | deterministic, fast, free       | not a "real" LLM adversary               |
| `Llama-3.1-8B-Instruct` via HF IE | HF Inference Endpoints | strong, well-known model        | ~$0.10/1M input + $0.10/1M output tokens |
| `Qwen2.5-7B-Instruct` via HF IE   | HF Inference Endpoints | matches the Investigator family | similar cost to Llama 8B                 |
| `Llama-3.1-8B` via local Ollama   | your laptop            | free, private                   | slow on consumer GPU (~30s / proposal)   |


### My recommendation for **training rollouts**: keep `ScriptedFraudsterl`

Reasons:

1. **Determinism** - GRPO needs reproducible reward signal. An LLM Fraudster
  would inject sampling noise into the trajectory, which fights the proxy
   reward.
2. **Speed** - rollouts are the bottleneck. Scripted is ~50x faster than
  8B inference.
3. **Cost** - your $30 budget gets 6x more training time without LLM Fraudster
  in the rollout loop.

### My recommendation for the **demo / final eval**: Llama 3.1 8B Instruct via HF IE

For the demo video / final presentation eval, swap in a real LLM Fraudster so
your Investigator looks credible against a strong adversary. Steps:

1. In `replay_match.py`, set `--fraudster-backend openai` and point it at a
  HF Inference Endpoint serving `meta-llama/Meta-Llama-3.1-8B-Instruct`.
2. Run **3 demo episodes** (one per task) on `task_1 task_2 task_3` with a
  seed not in your eval set.
3. Capture the `replay_*.md` transcripts for the slide deck.
4. Total cost for ~3 episodes: well under $1.

For pure HF-native, use `Qwen/Qwen2.5-7B-Instruct` instead — same family as
the Investigator and slightly cheaper to host.

---

## 8. Troubleshooting

### "loss=0, kl=0, reward=-0.1, reward_std=0 for the whole run"

TRL ≥1.0 conversational GRPO passes completions as
`[[{"role": "assistant", "content": "<json>"}]]`, not raw strings. If the
reward function stringifies the chat dict, the dict repr still contains
`{`, `action_type`, `ad_id`, `}` so partial credit is **exactly -0.1**
on every sample → zero advantage → LoRA never moves. `proxy_reward.py`
must unwrap `completion[0]["content"]` before `json.loads`. Eval leak
rates can still look real (they score env JSON, not the proxy).

### "Reward is flat for the first 50 steps"

Usually means the Investigator's completions are not parsing as valid JSON, so
`proxy_reward_fn` returns the same penalty every step. Check:

1. Section 4 of the training notebook prints the JSON-parse rate of collected
  rollouts. If it's < 60%, the prompt template is wrong for this base model.
2. For Qwen3 models, make sure `enable_thinking=False` is set on
  `HFInvestigator`. Otherwise the model emits `<thinking>...</thinking>`
   before the JSON and parsing fails.

### "OOM during training"

T4 has 16 GB. With 4-bit + LoRA you should fit Qwen3-0.6B with
`batch_size=4` and `max_prompt_length=1024`. If you OOM:

1. Drop `per_device_train_batch_size` to 2.
2. Drop `max_prompt_length` to 768.
3. Switch base model to `Qwen3-0.6B` (not 1.7B).

### "GRPOConfig got an unexpected keyword argument 'max_prompt_length'"

You're on an older TRL. The notebook handles this dynamically (uses
`inspect.signature` to detect TRL's API), but if you're poking at the config
manually, set `tokenizer.model_max_length = 1024` instead.

### "UnicodeDecodeError on Windows"

Windows-only. Set `PYTHONUTF8=1` in the environment before running. Not an
issue on Spaces / Colab (both are Linux).

### "Hub push fails with 401"

Re-run `notebook_login()` in Section 1 with a token that has **WRITE** scope
(the baseline-only path can use READ).