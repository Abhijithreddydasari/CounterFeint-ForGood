"""Paired deterministic evaluation of base and trajectory-GRPO checkpoints.

From the repository root:

  modal run --detach training/modal/run_paired_eval.py --model both
  modal app logs counterfeint-paired-eval

The base and trained policy see the same 24 fresh environment seeds. Greedy
decoding removes generation sampling noise; paired bootstrap intervals measure
the checkpoint delta rather than unrelated before/after variation.
"""

from __future__ import annotations

import json
import random
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
            parent / "training" / "paired_eval.py"
        ).exists():
            return parent
    return Path("/root/counterfeint")


REPO_ROOT = _repo_root()
TIMEOUT_SEC = 4 * 60 * 60
EVAL_SEEDS: Dict[str, List[int]] = {
    "task_2": list(range(5101, 5109)),
    "task_3": list(range(6101, 6109)),
    "task_3_unseen": list(range(7101, 7109)),
}
MODEL_SPECS: Dict[str, Dict[str, str]] = {
    "0.6b": {
        "base_model": "Qwen/Qwen3-0.6B",
        "slug": "qwen3-0.6b",
        "gpu": "L4",
        "adapter_path": (
            "/vol/runs/qwen3-0.6b-trajectory-proper-20260901T220521Z/"
            "lora_adapter"
        ),
    },
    "8b": {
        "base_model": "Qwen/Qwen3-8B",
        "slug": "qwen3-8b",
        "gpu": "A100-40GB",
        "adapter_path": (
            "/vol/runs/qwen3-8b-trajectory-proper-20260901T220514Z/"
            "lora_adapter"
        ),
    },
}


if modal is not None:
    app = modal.App("counterfeint-paired-eval")
    volume = modal.Volume.from_name("counterfeint-forgood", create_if_missing=False)
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


def _load_policy(base_model: str, *, adapter_path: str | None = None) -> Any:
    from counterfeint.agents import HFInvestigator

    return HFInvestigator.from_pretrained(
        base_model,
        load_in_4bit=False,
        torch_dtype="bfloat16",
        lora_path=adapter_path,
        max_new_tokens=128,
        temperature=1.0,
        do_sample=False,
        enable_thinking=False,
    )


def _evaluate_condition(
    policy: Any,
    *,
    condition: str,
    seeds_by_task: Dict[str, List[int]],
) -> List[Dict[str, Any]]:
    import torch

    from counterfeint.experiments.episode_bundle import (
        bundle_to_episode_record,
        run_episode_bundle,
    )
    from counterfeint.graders.auditor_pipeline import run_full_audit
    from counterfeint.graders.base_grader import grade_episode
    from counterfeint.scripted import HeuristicAuditor, ReactiveFraudster

    rows: List[Dict[str, Any]] = []
    total = sum(len(seeds) for seeds in seeds_by_task.values())
    completed = 0
    policy.model.eval()
    for task_id, seeds in seeds_by_task.items():
        for seed in seeds:
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            policy.reset()
            bundle = run_episode_bundle(
                task_id=task_id,
                seed=seed,
                investigator_factory=lambda: policy,
                fraudster_factory=lambda: ReactiveFraudster(seed=seed),
                auditor_factory=lambda: HeuristicAuditor(),
                max_steps=300,
            )
            record = bundle_to_episode_record(bundle)
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
                "condition": condition,
                "task_id": task_id,
                "seed": seed,
                "grader_score": float(grade_episode(record)),
                "track_a_score": float(audit.investigator_audit_score),
                "n_fraud_leaks": leaks,
                "n_ground_truth_fraud": n_fraud,
                "fallback_count": int(policy.fallback_count),
                "call_count": int(policy.call_count),
                "steps": record.total_steps,
                "n_fraudster_proposals": sum(
                    1
                    for proposal in bundle.fraudster_proposals
                    if proposal.get("action_type") == "propose_ad"
                ),
            }
            rows.append(row)
            completed += 1
            print(
                f"[{condition} {completed}/{total}] {task_id} seed={seed} "
                f"grader={row['grader_score']:.3f} leaks={leaks}/{n_fraud} "
                f"fallback={row['fallback_count']}/{row['call_count']}",
                flush=True,
            )
    return rows


def _run_paired_eval(model_key: str) -> Dict[str, Any]:
    import torch

    from counterfeint.training.paired_eval import analyse_paired_conditions

    if model_key not in MODEL_SPECS:
        raise ValueError(f"Unknown model {model_key!r}")
    spec = MODEL_SPECS[model_key]
    adapter_path = Path(spec["adapter_path"])
    if not adapter_path.exists():
        raise FileNotFoundError(f"Trained adapter not found: {adapter_path}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = Path("/vol/runs") / f"{spec['slug']}-paired-eval-{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)
    print(
        f"model={spec['base_model']} gpu={torch.cuda.get_device_name(0)} "
        f"pairs={sum(len(v) for v in EVAL_SEEDS.values())} output={out_root}",
        flush=True,
    )

    print("=== DETERMINISTIC BASE ===", flush=True)
    base_policy = _load_policy(spec["base_model"])
    base_rows = _evaluate_condition(
        base_policy, condition="base", seeds_by_task=EVAL_SEEDS
    )
    del base_policy
    torch.cuda.empty_cache()

    print("=== DETERMINISTIC TRAINED ===", flush=True)
    trained_policy = _load_policy(
        spec["base_model"], adapter_path=str(adapter_path)
    )
    trained_rows = _evaluate_condition(
        trained_policy, condition="trained", seeds_by_task=EVAL_SEEDS
    )

    analysis = analyse_paired_conditions(base_rows, trained_rows, n_boot=5000)
    payload = {
        "model_key": model_key,
        "base_model": spec["base_model"],
        "adapter_path": str(adapter_path),
        "decoding": {
            "strategy": "greedy",
            "do_sample": False,
            "max_new_tokens": 128,
            "enable_thinking": False,
        },
        "seeds": EVAL_SEEDS,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        **analysis,
    }
    (out_root / "paired_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in payload.items() if k != "pairs"}, indent=2))
    return {**payload, "output_dir": str(out_root)}


if modal is not None:

    @app.function(
        image=image,
        gpu="L4",
        timeout=TIMEOUT_SEC,
        scaledown_window=30,
        max_containers=1,
        volumes={"/vol": volume},
    )
    def evaluate_qwen_06b() -> Dict[str, Any]:
        try:
            return _run_paired_eval("0.6b")
        finally:
            volume.commit()

    @app.function(
        image=image,
        gpu="A100-40GB",
        timeout=TIMEOUT_SEC,
        scaledown_window=30,
        max_containers=1,
        volumes={"/vol": volume},
    )
    def evaluate_qwen_8b() -> Dict[str, Any]:
        try:
            return _run_paired_eval("8b")
        finally:
            volume.commit()

    @app.local_entrypoint()
    def main(model: str = "both") -> None:
        functions = {"0.6b": evaluate_qwen_06b, "8b": evaluate_qwen_8b}
        if model == "both":
            model_keys = list(functions)
        elif model in functions:
            model_keys = [model]
        else:
            raise ValueError("--model must be one of: 0.6b, 8b, both")

        metadata: Dict[str, Any] = {}
        for model_key in model_keys:
            call = functions[model_key].spawn()
            metadata[model_key] = {
                "call_id": getattr(call, "object_id", None) or str(call),
                "base_model": MODEL_SPECS[model_key]["base_model"],
                "gpu": MODEL_SPECS[model_key]["gpu"],
            }
            print(f"SPAWNED {model_key}: {metadata[model_key]['call_id']}", flush=True)

        output = REPO_ROOT / "experiments" / "outputs" / "paired_eval"
        output.mkdir(parents=True, exist_ok=True)
        (output / "last_calls.json").write_text(
            json.dumps(
                {
                    "calls": metadata,
                    "spawned_utc": datetime.now(timezone.utc).isoformat(),
                    "app": "counterfeint-paired-eval",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print("Monitor: modal app logs counterfeint-paired-eval", flush=True)
