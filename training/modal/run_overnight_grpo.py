"""Overnight, full-episode GRPO for the CounterFeint Investigator.

Unlike the retired proxy runner, every sample here is a complete interactive
episode. A frozen ReactiveFraudster actively proposes/modifies ads, the
Investigator takes many sequential tool actions, and the terminal environment
reward is group-normalised and applied to every Investigator action token.

From the repository root:

  modal run training/modal/run_overnight_grpo.py --mode smoke --model both
  modal run --detach training/modal/run_overnight_grpo.py --mode proper --model both

  modal app logs counterfeint-trajectory-grpo
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
    for parent in [here.parent, *here.parents]:
        if (parent / "pyproject.toml").exists() and (
            parent / "training" / "trajectory_grpo.py"
        ).exists():
            return parent
    return Path("/root/counterfeint")


REPO_ROOT = _repo_root()
TIMEOUT_SEC = 8 * 60 * 60
MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "0.6b": {
        "base_model": "Qwen/Qwen3-0.6B",
        "slug": "qwen3-0.6b",
        "learning_rate": 2e-5,
        "gpu": "L4",
    },
    "8b": {
        "base_model": "Qwen/Qwen3-8B",
        "slug": "qwen3-8b",
        "learning_rate": 1e-5,
        "gpu": "A100-40GB",
    },
}

MODE_PRESETS: Dict[str, Dict[str, Any]] = {
    # Short learning validation: one Task-2 and one Task-3 GRPO group, plus
    # distinct held-out seeds before and after. Proper remains substantially
    # larger while smoke can demonstrate non-zero reward variance/gradients.
    "smoke": {
        "train_seeds": {
            "task_2": [21],
            "task_3": [31],
        },
        "group_size": 2,
        "eval_seeds": {
            "task_2": [2001],
            "task_3": [3001],
        },
        "update_epochs": 1,
    },
    # 20 complete training trajectories: one medium-horizon warm-up group and
    # four hard long-horizon network groups. Held-out task_3_unseen is eval-only.
    "proper": {
        "train_seeds": {
            "task_2": [21],
            "task_3": [31, 32, 33, 34],
        },
        "group_size": 4,
        "eval_seeds": {
            "task_2": [2001, 2002],
            "task_3": [3001, 3002, 3003],
            "task_3_unseen": [4001, 4002],
        },
        "update_epochs": 1,
    },
}


if modal is not None:
    app = modal.App("counterfeint-trajectory-grpo")
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
            "peft>=0.13.0",
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
    fraud = sum(int(row["n_ground_truth_fraud"]) for row in episodes)
    leaks = sum(int(row["n_fraud_leaks"]) for row in episodes)
    return leaks / fraud if fraud else 0.0


def _bootstrap_ci(
    values: List[float], n_boot: int = 2000, alpha: float = 0.05
) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0}
    import random

    rng = random.Random(7)
    means: List[float] = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return {
        "mean": sum(values) / len(values),
        "lo": means[int(alpha / 2 * n_boot)],
        "hi": means[int((1 - alpha / 2) * n_boot) - 1],
    }


def _eval_investigator(
    hf_investigator: Any,
    *,
    base_model: str,
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
    total = sum(len(seeds) for seeds in seeds_by_task.values())
    done = 0

    def investigator_factory() -> Any:
        hf_investigator.reset()
        hf_investigator.model.eval()
        return hf_investigator

    for task_id, seeds in seeds_by_task.items():
        for seed in seeds:
            bundle = run_episode_bundle(
                task_id=task_id,
                seed=seed,
                investigator_factory=investigator_factory,
            )
            record = bundle_to_episode_record(bundle)
            grader = float(grade_episode(record))
            audit = run_full_audit(
                record=record,
                investigator_action_log=bundle.investigator_actions,
                investigation_data_seen=bundle.investigation_data_seen,
                fraudster_proposal_log=bundle.fraudster_proposals,
            )
            n_fraud = sum(
                1 for meta in record.ads_metadata if meta.get("ground_truth") == "fraud"
            )
            leaks = sum(
                1
                for verdict in record.verdicts
                if verdict.ground_truth == "fraud"
                and (verdict.auto_approved or verdict.verdict in ("approve", "escalate"))
            )
            row = {
                "tag": tag,
                "task_id": task_id,
                "seed": seed,
                "grader_score": grader,
                "track_a_score": audit.investigator_audit_score,
                "n_fraud_leaks": leaks,
                "n_ground_truth_fraud": n_fraud,
                "fallback_count": int(hf_investigator.fallback_count),
                "call_count": int(hf_investigator.call_count),
                "steps": record.total_steps,
                "n_fraudster_proposals": sum(
                    1
                    for proposal in bundle.fraudster_proposals
                    if proposal.get("action_type") == "propose_ad"
                ),
            }
            episodes.append(row)
            done += 1
            print(
                f"[{tag} {done}/{total}] {task_id} seed={seed} "
                f"grader={grader:.3f} leaks={leaks}/{n_fraud} "
                f"fallback={row['fallback_count']}/{row['call_count']} "
                f"fraudster_ads={row['n_fraudster_proposals']}",
                flush=True,
            )

    per_episode_leak = [
        row["n_fraud_leaks"] / row["n_ground_truth_fraud"]
        if row["n_ground_truth_fraud"]
        else 0.0
        for row in episodes
    ]
    payload = {
        "tag": tag,
        "model": base_model,
        "n_episodes": len(episodes),
        "leak_rate_overall": _leak_rate(episodes),
        "leak_rate_bootstrap": _bootstrap_ci(per_episode_leak),
        "grader_mean": statistics.mean(row["grader_score"] for row in episodes),
        "track_a_mean": statistics.mean(row["track_a_score"] for row in episodes),
        "fallback_total": sum(row["fallback_count"] for row in episodes),
        "call_total": sum(row["call_count"] for row in episodes),
        "episodes": episodes,
    }
    (output_dir / "eval_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def _run_training(
    mode: str,
    out_root: Path,
    *,
    model_key: str,
) -> Dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model

    from counterfeint.agents import HFInvestigator
    from counterfeint.training.trajectory_grpo import (
        RewardMode,
        collect_trajectory_group,
        optimise_trajectory_group,
    )

    if mode not in MODE_PRESETS:
        raise ValueError(f"mode must be smoke|proper, got {mode!r}")
    if model_key not in MODEL_SPECS:
        raise ValueError(f"model must be 0.6b|8b, got {model_key!r}")
    preset = MODE_PRESETS[mode]
    model_spec = MODEL_SPECS[model_key]
    base_model = str(model_spec["base_model"])
    out_root.mkdir(parents=True, exist_ok=True)
    print(
        f"CUDA={torch.cuda.is_available()} "
        f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}",
        flush=True,
    )
    print(
        f"mode={mode} model={base_model} reward=full_environment "
        f"group_size={preset['group_size']}",
        flush=True,
    )

    load_started = time.perf_counter()
    hf = HFInvestigator.from_pretrained(
        base_model,
        load_in_4bit=False,
        torch_dtype="bfloat16",
        max_new_tokens=128,
        temperature=0.7,
        do_sample=True,
        enable_thinking=False,
    )
    hf.model = get_peft_model(
        hf.model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    if hasattr(hf.model, "enable_input_require_grads"):
        hf.model.enable_input_require_grads()
    hf.model.print_trainable_parameters()
    print(f"Loaded model+LoRA in {time.perf_counter()-load_started:.1f}s", flush=True)

    probe = hf._call_chat(
        [
            {"role": "system", "content": "Output JSON only."},
            {"role": "user", "content": 'Reply with {"action_type":"verdict"}'},
        ]
    )
    if not hf.last_prompt_token_ids or not hf.last_completion_token_ids:
        raise RuntimeError("HF Investigator did not expose rollout token IDs")
    print(
        f"Token capture probe: prompt={len(hf.last_prompt_token_ids)} "
        f"completion={len(hf.last_completion_token_ids)} text={probe[:100]!r}",
        flush=True,
    )

    print("=== HELD-OUT BEFORE ===", flush=True)
    before = _eval_investigator(
        hf,
        base_model=base_model,
        seeds_by_task=preset["eval_seeds"],
        tag="before_trajectory_grpo",
        output_dir=out_root / "eval_before",
    )

    optimizer = torch.optim.AdamW(
        [parameter for parameter in hf.model.parameters() if parameter.requires_grad],
        lr=float(model_spec["learning_rate"]),
        weight_decay=0.01,
    )
    all_trajectories: List[Any] = []
    update_log: List[Dict[str, Any]] = []
    train_started = time.perf_counter()
    group_number = 0
    n_groups = sum(len(seeds) for seeds in preset["train_seeds"].values())

    for task_id, seeds in preset["train_seeds"].items():
        for seed in seeds:
            group_number += 1
            print(
                f"=== GROUP {group_number}/{n_groups}: {task_id} seed={seed} "
                f"x{preset['group_size']} COMPLETE EPISODES ===",
                flush=True,
            )

            def investigator_factory() -> Any:
                hf.model.eval()
                return hf

            group = collect_trajectory_group(
                task_id=task_id,
                seed=seed,
                group_size=int(preset["group_size"]),
                investigator_factory=investigator_factory,
                reward_mode=RewardMode.ENVIRONMENT,
                fallback_penalty=1.0,
                max_steps=200,
            )
            if any(record.n_fraudster_proposals == 0 for record in group):
                raise RuntimeError("Frozen Fraudster failed to populate ads")
            if any(not record.turns for record in group):
                raise RuntimeError("A trajectory contained no trainable Investigator turns")

            for record in group:
                print(
                    f"  rollout={record.trajectory_idx} turns={len(record.turns)} "
                    f"fraudster_ads={record.n_fraudster_proposals} "
                    f"grader={record.grader_score:.3f} "
                    f"reward={record.investigator_reward:+.3f} "
                    f"adv={record.advantage:+.3f} "
                    f"fallback={record.fallback_count}/{record.call_count}",
                    flush=True,
                )

            update = optimise_trajectory_group(
                model=hf.model,
                optimizer=optimizer,
                records=group,
                clip_epsilon=0.2,
                max_prompt_tokens=2048,
                max_grad_norm=1.0,
                update_epochs=int(preset["update_epochs"]),
            )
            update.update(task_id=task_id, seed=seed, group_number=group_number)
            update_log.append(update)
            all_trajectories.extend(group)
            print(f"  update={json.dumps(update)}", flush=True)

            hf.model.save_pretrained(str(out_root / "lora_adapter"))
            hf.tokenizer.save_pretrained(str(out_root / "lora_adapter"))
            (out_root / "trajectory_groups.json").write_text(
                json.dumps(
                    [record.to_dict() for record in all_trajectories], indent=2
                ),
                encoding="utf-8",
            )
            (out_root / "update_log.json").write_text(
                json.dumps(update_log, indent=2), encoding="utf-8"
            )
            if modal is not None:
                volume.commit()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    train_seconds = time.perf_counter() - train_started
    print("=== HELD-OUT AFTER ===", flush=True)
    after = _eval_investigator(
        hf,
        base_model=base_model,
        seeds_by_task=preset["eval_seeds"],
        tag="after_trajectory_grpo",
        output_dir=out_root / "eval_after",
    )
    summary = {
        "mode": mode,
        "base_model": base_model,
        "model_key": model_key,
        "gpu_requested": model_spec["gpu"],
        "algorithm": "trajectory_group_relative_policy_optimisation",
        "reward_mode": "terminal_environment_reward",
        "frozen_fraudster": "ReactiveFraudster",
        "long_horizon": True,
        "group_size": preset["group_size"],
        "n_groups": n_groups,
        "n_training_trajectories": len(all_trajectories),
        "n_training_turns": sum(len(record.turns) for record in all_trajectories),
        "n_fraudster_proposals": sum(
            record.n_fraudster_proposals for record in all_trajectories
        ),
        "train_seconds": round(train_seconds, 1),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "before": {
            "grader_mean": before["grader_mean"],
            "leak_rate": before["leak_rate_overall"],
            "track_a_mean": before["track_a_mean"],
            "fallback_total": before["fallback_total"],
        },
        "after": {
            "grader_mean": after["grader_mean"],
            "leak_rate": after["leak_rate_overall"],
            "track_a_mean": after["track_a_mean"],
            "fallback_total": after["fallback_total"],
        },
        "delta_grader": after["grader_mean"] - before["grader_mean"],
        "delta_leak_rate": after["leak_rate_overall"] - before["leak_rate_overall"],
        "updates": update_log,
        "output_dir": str(out_root),
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if modal is not None:

    def _remote_train(model_key: str, mode: str) -> Dict[str, Any]:
        if model_key not in MODEL_SPECS:
            raise ValueError(f"Unknown model {model_key!r}; choose from {sorted(MODEL_SPECS)}")
        spec = MODEL_SPECS[model_key]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_root = Path("/vol/runs") / f"{spec['slug']}-trajectory-{mode}-{stamp}"
        print(f"Artifacts -> {out_root}", flush=True)
        try:
            return _run_training(mode, out_root, model_key=model_key)
        finally:
            volume.commit()
            print("Volume committed; GPU function ending.", flush=True)

    @app.function(
        image=image,
        gpu="L4",
        timeout=TIMEOUT_SEC,
        scaledown_window=30,
        max_containers=1,
        volumes={"/vol": volume},
    )
    def train_qwen_06b(mode: str = "proper") -> Dict[str, Any]:
        return _remote_train("0.6b", mode)

    @app.function(
        image=image,
        gpu="A100-40GB",
        timeout=TIMEOUT_SEC,
        scaledown_window=30,
        max_containers=1,
        volumes={"/vol": volume},
    )
    def train_qwen_8b(mode: str = "proper") -> Dict[str, Any]:
        return _remote_train("8b", mode)

    @app.local_entrypoint()
    def main(mode: str = "proper", model: str = "both") -> None:
        if mode not in MODE_PRESETS:
            raise ValueError(f"Unknown mode {mode!r}; choose from {sorted(MODE_PRESETS)}")
        functions = {"0.6b": train_qwen_06b, "8b": train_qwen_8b}
        if model == "both":
            model_keys = list(functions)
        elif model in functions:
            model_keys = [model]
        else:
            raise ValueError("--model must be one of: 0.6b, 8b, both")

        calls: Dict[str, Any] = {}
        call_metadata: Dict[str, Any] = {}
        for model_key in model_keys:
            spec = MODEL_SPECS[model_key]
            print(
                f"Spawning {spec['base_model']} mode={mode} on {spec['gpu']} "
                f"(hard timeout {TIMEOUT_SEC // 3600}h)",
                flush=True,
            )
            call = functions[model_key].spawn(mode=mode)
            calls[model_key] = call
            call_metadata[model_key] = {
                "call_id": getattr(call, "object_id", None) or str(call),
                "base_model": spec["base_model"],
                "gpu": spec["gpu"],
            }

        if mode == "smoke":
            results = {model_key: call.get() for model_key, call in calls.items()}
            print(json.dumps(results, indent=2), flush=True)
            return

        stamp_path = REPO_ROOT / "experiments" / "outputs" / "modal_grpo"
        stamp_path.mkdir(parents=True, exist_ok=True)
        (stamp_path / "last_calls.json").write_text(
            json.dumps(
                {
                    "calls": call_metadata,
                    "mode": mode,
                    "spawned_utc": datetime.now(timezone.utc).isoformat(),
                    "app": "counterfeint-trajectory-grpo",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"SPAWNED {json.dumps(call_metadata)}", flush=True)
        print("Monitor: modal app logs counterfeint-trajectory-grpo", flush=True)
