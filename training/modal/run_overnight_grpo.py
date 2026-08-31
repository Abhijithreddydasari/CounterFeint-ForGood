"""
Overnight GRPO on Qwen3-4B-Instruct-2507 (proxy reward, frozen scripted fraudster).

Same loop as official_hf_training.ipynb, with three changes:
  * 4B instruct instead of 0.6B
  * before/after eval on the 35 EVAL_SEEDS with fraud leak rate
  * one Modal function: GPU dies when the function returns or hits timeout

Detach rules (read these):
  * Overnight uses .spawn() + --detach. Do NOT use .remote() overnight —
    closing the laptop cancels a waiting .remote() and kills the GPU job.
  * After spawn, do NOT run this file again, do NOT `modal serve`, do NOT
    `modal deploy`. A second launch or a live-reload replace stops the first run.
  * Local edits after spawn do not touch the running container (image is baked).
    They only matter if you start a new `modal run`.

From repo root (pip install modal && modal setup):

  modal run training/modal/run_overnight_grpo.py --mode smoke
  modal run --detach training/modal/run_overnight_grpo.py --mode proper

  modal app logs counterfeint-overnight-grpo
  modal app stop counterfeint-overnight-grpo

  modal volume get counterfeint-forgood /runs ./experiments/outputs/modal_grpo --force
"""

from __future__ import annotations

import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

try:
    import modal
except ImportError:
    modal = None  # type: ignore


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "pyproject.toml").exists() and (p / "training" / "rollout.py").exists():
            return p
    return Path("/root/counterfeint")


REPO_ROOT = _repo_root()

MODE_PRESETS: Dict[str, Dict[str, Any]] = {
    "smoke": {
        "train_seeds": [11],
        "task_3_seeds": [11],
        "epochs": 1,
        "eval_tasks": {"task_1": [1001]},
        "max_steps": 3,
    },
    "proper": {
        "train_seeds": list(range(11, 21)),
        "task_3_seeds": list(range(11, 14)),
        "epochs": 2,
        "eval_tasks": None,  # filled with EVAL_SEEDS at runtime
        "max_steps": None,
    },
}

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
TIMEOUT_SEC = 6 * 60 * 60


if modal is not None:
    app = modal.App("counterfeint-overnight-grpo")
    volume = modal.Volume.from_name("counterfeint-forgood", create_if_missing=True)

    image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.8.1-devel-ubuntu22.04",
            add_python="3.11",
        )
        .entrypoint([])
        .pip_install(
            "torch",
            extra_index_url="https://download.pytorch.org/whl/cu128",
        )
        .pip_install(
            "transformers>=4.51.0",
            "accelerate>=1.1.0",
            "datasets>=3.0.0",
            "peft>=0.13.0",
            "bitsandbytes>=0.44.0",
            "trl>=0.12.0",
            "safetensors>=0.4.5",
            "huggingface_hub>=0.26.0",
            "openenv-core[core]>=0.2.3",
            "fastapi>=0.115.0",
            "pydantic>=2.0.0",
            "faker==33.1.0",
            "networkx>=3.2",
            "openai>=1.0.0",
            "python-dotenv>=1.0.0",
            "requests>=2.31.0",
        )
        .env(
            {
                "PYTHONPATH": "/root",
                "CUDA_HOME": "/usr/local/cuda",
                "HF_HOME": "/vol/hf-cache",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "TRANSFORMERS_VERBOSITY": "error",
            }
        )
        .add_local_dir(
            str(REPO_ROOT),
            remote_path="/root/counterfeint",
            ignore=[
                "**/outputs/**",
                "**/experiments/outputs/**",
                "**/__pycache__/**",
                "**/.pytest_cache/**",
                "**/*.pyc",
                "**/.venv/**",
                "**/.git/**",
                "**/papers/**",
                "**/*.ipynb",
            ],
        )
    )


def _leak_rate(episodes: List[Dict[str, Any]]) -> float:
    leaks = sum(int(e["n_fraud_leaks"]) for e in episodes)
    fraud = sum(int(e["n_ground_truth_fraud"]) for e in episodes)
    return leaks / fraud if fraud else 0.0


def _bootstrap_ci(values: List[float], n_boot: int = 2000, alpha: float = 0.05) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0}
    import random

    rng = random.Random(7)
    means = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_i = int(alpha / 2 * n_boot)
    hi_i = int((1 - alpha / 2) * n_boot) - 1
    return {
        "mean": sum(values) / n,
        "lo": means[max(0, lo_i)],
        "hi": means[min(n_boot - 1, hi_i)],
    }


def _eval_investigator(
    hf_investigator: Any,
    *,
    seeds_by_task: Dict[str, List[int]],
    tag: str,
    output_dir: Path,
) -> Dict[str, Any]:
    from counterfeint.experiments.episode_bundle import (
        bundle_to_episode_record,
        run_episode_bundle,
    )
    from counterfeint.graders.auditor_pipeline import run_full_audit
    from counterfeint.graders.base_grader import grade_episode

    output_dir.mkdir(parents=True, exist_ok=True)
    episodes: List[Dict[str, Any]] = []
    n_total = sum(len(s) for s in seeds_by_task.values())
    done = 0

    def factory() -> Any:
        hf_investigator.reset()
        hf_investigator.model.eval()
        return hf_investigator

    for task_id, seeds in seeds_by_task.items():
        for seed in seeds:
            bundle = run_episode_bundle(
                task_id=task_id,
                seed=seed,
                investigator_factory=factory,
            )
            record = bundle_to_episode_record(bundle)
            grader = grade_episode(record)
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
                and (v.auto_approved or v.verdict in ("approve", "escalate"))
            )
            fallback = int(getattr(hf_investigator, "fallback_count", 0) or 0)
            calls = int(getattr(hf_investigator, "call_count", 0) or 0)
            row = {
                "tag": tag,
                "task_id": task_id,
                "seed": seed,
                "grader_score": grader,
                "track_a_score": audit.investigator_audit_score,
                "n_fraud_leaks": leaks,
                "n_ground_truth_fraud": n_fraud,
                "fallback_count": fallback,
                "call_count": calls,
                "steps": record.total_steps,
            }
            episodes.append(row)
            done += 1
            print(
                f"[{tag} {done}/{n_total}] {task_id} seed={seed} "
                f"grader={grader:.3f} leaks={leaks}/{n_fraud} "
                f"fallback={fallback}/{calls}",
                flush=True,
            )

    by_task: Dict[str, List[Dict[str, Any]]] = {}
    for e in episodes:
        by_task.setdefault(e["task_id"], []).append(e)
    aggregates = {}
    for task_id, rows in by_task.items():
        n = len(rows)
        aggregates[task_id] = {
            "n_episodes": n,
            "grader_score_mean": sum(r["grader_score"] for r in rows) / n,
            "n_fraud_leaks_mean": sum(r["n_fraud_leaks"] for r in rows) / n,
            "leak_rate": _leak_rate(rows),
            "fallback_total": sum(r["fallback_count"] for r in rows),
        }

    per_ep_leak = [
        (r["n_fraud_leaks"] / r["n_ground_truth_fraud"])
        if r["n_ground_truth_fraud"]
        else 0.0
        for r in episodes
    ]
    payload = {
        "tag": tag,
        "model": BASE_MODEL,
        "leak_rate_overall": _leak_rate(episodes),
        "leak_rate_bootstrap": _bootstrap_ci(per_ep_leak),
        "grader_mean": sum(e["grader_score"] for e in episodes) / max(1, len(episodes)),
        "fallback_total": sum(e["fallback_count"] for e in episodes),
        "call_total": sum(e["call_count"] for e in episodes),
        "aggregates": aggregates,
        "episodes": episodes,
    }
    (output_dir / "eval_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        f"# {tag}",
        "",
        f"Overall leak rate: {payload['leak_rate_overall']:.3f} "
        f"(bootstrap mean {payload['leak_rate_bootstrap']['mean']:.3f}, "
        f"95% CI {payload['leak_rate_bootstrap']['lo']:.3f}–"
        f"{payload['leak_rate_bootstrap']['hi']:.3f})",
        "",
        "| Task | n | grader | leaks/ep | leak rate | fallback |",
        "|------|--:|-------:|---------:|----------:|---------:|",
    ]
    for tid, agg in aggregates.items():
        lines.append(
            f"| {tid} | {agg['n_episodes']} | {agg['grader_score_mean']:.3f} | "
            f"{agg['n_fraud_leaks_mean']:.2f} | {agg['leak_rate']:.3f} | "
            f"{agg['fallback_total']} |"
        )
    (output_dir / "eval_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_dir / 'eval_summary.md'}", flush=True)
    return payload


def _run_training(mode: str, out_root: Path) -> Dict[str, Any]:
    import inspect

    import torch
    from peft import LoraConfig, get_peft_model
    from trl import GRPOConfig, GRPOTrainer

    from counterfeint.agents import HFInvestigator
    from counterfeint.agents.prompts import INVESTIGATOR_SYSTEM_PROMPT
    from counterfeint.eval_suite import EVAL_SEEDS
    from counterfeint.scripted import HeuristicAuditor, ReactiveFraudster
    from counterfeint.training import (
        build_gold_lookup,
        collect_dataset_in_process,
        make_proxy_reward_fn,
        samples_to_hf_dataset,
    )

    if mode not in MODE_PRESETS:
        raise ValueError(f"mode must be smoke|proper, got {mode!r}")
    preset = MODE_PRESETS[mode]
    eval_tasks = preset["eval_tasks"] or EVAL_SEEDS
    train_seeds_by_task = {
        "task_1": list(preset["train_seeds"]),
        "task_2": list(preset["train_seeds"]),
        "task_3": list(preset["task_3_seeds"]),
    }

    out_root.mkdir(parents=True, exist_ok=True)
    print(f"CUDA: {torch.cuda.is_available()} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}", flush=True)
    print(f"mode={mode} model={BASE_MODEL}", flush=True)

    t_load = time.perf_counter()
    hf = HFInvestigator.from_pretrained(
        BASE_MODEL,
        load_in_4bit=False,
        torch_dtype="bfloat16",
        max_new_tokens=128,
        temperature=0.3,
        do_sample=True,
        enable_thinking=False,
    )
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    hf.model = get_peft_model(hf.model, lora_cfg)
    if hasattr(hf.model, "enable_input_require_grads"):
        hf.model.enable_input_require_grads()
    hf.model.print_trainable_parameters()
    print(f"Loaded+LoRA in {time.perf_counter() - t_load:.1f}s", flush=True)

    probe = hf._call_chat(
        [
            {"role": "system", "content": "You output one line of JSON only."},
            {"role": "user", "content": 'Reply with {"ok": true}'},
        ]
    )
    print(f"Probe: {probe[:160]!r}", flush=True)

    print("=== BEFORE eval ===", flush=True)
    t0 = time.perf_counter()
    before = _eval_investigator(
        hf,
        seeds_by_task=eval_tasks,
        tag="before_grpo",
        output_dir=out_root / "eval_before",
    )
    print(f"BEFORE leak_rate={before['leak_rate_overall']:.3f} in {time.perf_counter()-t0:.0f}s", flush=True)

    print("=== Collect rollouts ===", flush=True)
    t0 = time.perf_counter()
    samples = collect_dataset_in_process(
        hf_investigator=hf,
        seeds_by_task=train_seeds_by_task,
        fraudster_factory=lambda: ReactiveFraudster(seed=42),
        auditor_factory=lambda: HeuristicAuditor(),
        max_steps=80,
        show_trace=False,
    )
    print(
        f"Collected {len(samples)} rows in {time.perf_counter()-t0:.0f}s "
        f"fallback={hf.fallback_count}/{hf.call_count}",
        flush=True,
    )
    if not samples:
        raise RuntimeError("No training rows — every step fell back to scripted.")

    clean = [
        s
        for s in samples
        if (s.completion or "").strip().startswith("{") and "action_type" in (s.completion or "")
    ]
    print(f"Kept {len(clean)}/{len(samples)} JSON rows", flush=True)
    samples = clean or samples
    rewards = [s.reward for s in samples]
    print(
        f"Collected-row reward mean={statistics.mean(rewards):+.4f} "
        f"std={statistics.pstdev(rewards):+.4f}",
        flush=True,
    )

    gold_lookup = build_gold_lookup(samples)
    proxy_fn = make_proxy_reward_fn(gold_lookup=gold_lookup)
    train_dataset = samples_to_hf_dataset(samples, system_prompt=INVESTIGATOR_SYSTEM_PROMPT)

    ckpt_dir = out_root / "trl_checkpoints"
    grpo_kwargs: Dict[str, Any] = dict(
        output_dir=str(ckpt_dir),
        learning_rate=3e-5,
        num_generations=4,
        beta=0.01,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_completion_length=256,
        num_train_epochs=int(preset["epochs"]),
        save_steps=50,
        logging_steps=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=7,
        remove_unused_columns=False,
    )
    if preset["max_steps"] is not None:
        grpo_kwargs["max_steps"] = int(preset["max_steps"])
    params = set(inspect.signature(GRPOConfig.__init__).parameters)
    if "max_prompt_length" in params:
        grpo_kwargs["max_prompt_length"] = 2048
    else:
        hf.tokenizer.model_max_length = 2048 + 256
    if "temperature" in params:
        grpo_kwargs["temperature"] = 0.9
    if "bf16" in params:
        grpo_kwargs["bf16"] = True

    trainer = GRPOTrainer(
        model=hf.model,
        args=GRPOConfig(**grpo_kwargs),
        train_dataset=train_dataset,
        reward_funcs=[proxy_fn],
        processing_class=hf.tokenizer,
    )
    if hasattr(trainer, "generation_config"):
        trainer.generation_config.temperature = 0.9
        trainer.generation_config.do_sample = True

    print("=== GRPO train ===", flush=True)
    t0 = time.perf_counter()
    train_result = trainer.train()
    train_sec = time.perf_counter() - t0
    print(f"train() finished in {train_sec:.0f}s metrics={getattr(train_result, 'metrics', {})}", flush=True)

    adapter_dir = out_root / "lora_adapter"
    hf.model.save_pretrained(str(adapter_dir))
    hf.tokenizer.save_pretrained(str(adapter_dir))
    log_path = out_root / "log_history.json"
    log_path.write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")

    print("=== AFTER eval ===", flush=True)
    t0 = time.perf_counter()
    after = _eval_investigator(
        hf,
        seeds_by_task=eval_tasks,
        tag="after_grpo",
        output_dir=out_root / "eval_after",
    )
    print(f"AFTER leak_rate={after['leak_rate_overall']:.3f} in {time.perf_counter()-t0:.0f}s", flush=True)

    summary = {
        "mode": mode,
        "base_model": BASE_MODEL,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "n_train_rows": len(samples),
        "train_seconds": round(train_sec, 1),
        "before": {
            "leak_rate": before["leak_rate_overall"],
            "leak_ci": before["leak_rate_bootstrap"],
            "grader_mean": before["grader_mean"],
            "fallback_total": before["fallback_total"],
        },
        "after": {
            "leak_rate": after["leak_rate_overall"],
            "leak_ci": after["leak_rate_bootstrap"],
            "grader_mean": after["grader_mean"],
            "fallback_total": after["fallback_total"],
        },
        "delta_leak_rate": after["leak_rate_overall"] - before["leak_rate_overall"],
        "delta_grader": after["grader_mean"] - before["grader_mean"],
        "output_dir": str(out_root),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if modal is not None:

    @app.function(
        image=image,
        gpu="A100-40GB",
        timeout=TIMEOUT_SEC,
        scaledown_window=30,
        max_containers=1,
        volumes={"/vol": volume},
    )
    def train_overnight(mode: str = "proper") -> Dict[str, Any]:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_root = Path("/vol/runs") / f"qwen3-4b-{mode}-{stamp}"
        print(f"Artifacts -> {out_root}", flush=True)
        try:
            summary = _run_training(mode, out_root)
        finally:
            volume.commit()
            print("Volume committed; function returning (GPU will stop).", flush=True)
        return summary

    @app.local_entrypoint()
    def main(mode: str = "proper") -> None:
        """smoke: block until done. proper: spawn and exit (use --detach)."""
        print(
            f"Launching GRPO mode={mode} on A100-40GB "
            f"(timeout {TIMEOUT_SEC // 3600}h)",
            flush=True,
        )
        if mode == "smoke":
            result = train_overnight.remote(mode=mode)
            print(json.dumps(result, indent=2), flush=True)
            print("Smoke finished. GPU released.", flush=True)
            return

        call = train_overnight.spawn(mode=mode)
        call_id = getattr(call, "object_id", None) or str(call)
        stamp_path = REPO_ROOT / "experiments" / "outputs" / "modal_grpo"
        stamp_path.mkdir(parents=True, exist_ok=True)
        (stamp_path / "last_call.json").write_text(
            json.dumps(
                {
                    "call_id": call_id,
                    "mode": mode,
                    "spawned_utc": datetime.now(timezone.utc).isoformat(),
                    "app": "counterfeint-overnight-grpo",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"SPAWNED call_id={call_id}", flush=True)
        print(f"Wrote {stamp_path / 'last_call.json'}", flush=True)
        print(
            "Leave this job alone until morning:\n"
            "  - do not run this script again\n"
            "  - do not `modal serve` or `modal deploy` this file\n"
            "  - local edits will not kill it unless you start a new run\n"
            "  modal app logs counterfeint-overnight-grpo",
            flush=True,
        )
