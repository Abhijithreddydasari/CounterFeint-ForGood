"""Generate publication figures for the CounterFeint workshop paper."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = {
    "0.6B": ROOT / "experiments/outputs/paired_eval/qwen3-0.6b-paired-eval-20260902T053925Z/paired_results.json",
    "8B": ROOT / "experiments/outputs/paired_eval/qwen3-8b-paired-eval-20260902T053926Z/paired_results.json",
}

NAVY = "#19324D"
BLUE = "#3377B5"
TEAL = "#2A9D8F"
ORANGE = "#E07A3F"
RED = "#C64545"
LIGHT = "#F4F7FA"
GRAY = "#5D6873"


def _save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(HERE / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(HERE / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _box(ax, xy, width, height, title, lines, color, *, title_color="white"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.25, edgecolor=color, facecolor="white", zorder=2,
    )
    ax.add_patch(patch)
    header = FancyBboxPatch(
        (x, y + height - 0.075), width, 0.075,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=0, facecolor=color, zorder=3,
    )
    ax.add_patch(header)
    ax.text(x + width / 2, y + height - 0.037, title, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=title_color, zorder=4)
    for i, line in enumerate(lines):
        ax.text(x + 0.018, y + height - 0.112 - 0.043 * i, line,
                ha="left", va="top", fontsize=8.4, color=NAVY, zorder=4)


def _arrow(ax, start, end, label=None, *, color=GRAY, curve=0.0):
    arrow = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=12,
        linewidth=1.25, color=color,
        connectionstyle=f"arc3,rad={curve}", zorder=1,
    )
    ax.add_patch(arrow)
    if label:
        x = (start[0] + end[0]) / 2
        y = (start[1] + end[1]) / 2 + (0.03 if curve >= 0 else -0.03)
        ax.text(x, y, label, fontsize=7.8, color=color, ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5), zorder=5)


def environment_overview() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, (0.04, 0.57), 0.25, 0.30, "Reactive fraudster",
         ["Observes public queue state", "Injects or revises proposals", "Seed-controlled during evaluation"], RED)
    _box(ax, (0.375, 0.48), 0.25, 0.39, "CounterFeint referee",
         ["Seeded queue and hidden truth", "Budgets and turn order", "Mutable episode state", "Auto-approval at termination"], NAVY)
    _box(ax, (0.71, 0.57), 0.25, 0.30, "Investigator agent",
         ["Chooses evidence tools", "Approves, rejects, or escalates", "Links suspected accounts"], BLUE)
    _box(ax, (0.375, 0.07), 0.25, 0.25, "Independent audit",
         ["Deterministic task grader", "Decision-process checks", "Safety and reliability metrics"], TEAL)

    _arrow(ax, (0.29, 0.72), (0.375, 0.72), "proposals")
    _arrow(ax, (0.625, 0.72), (0.71, 0.72), "queue state")
    _arrow(ax, (0.71, 0.63), (0.625, 0.63), "actions")
    _arrow(ax, (0.50, 0.48), (0.50, 0.32), "complete trajectory")
    _arrow(ax, (0.43, 0.32), (0.23, 0.57), "Track B", curve=-0.12)
    _arrow(ax, (0.57, 0.32), (0.83, 0.57), "Track A", curve=0.12)

    ax.text(0.50, 0.94, "Long-horizon moderation as a closed-loop safety problem",
            ha="center", va="center", fontsize=14, fontweight="bold", color=NAVY)
    ax.text(0.50, 0.015,
            "Seeded generation makes the world replayable; role separation keeps policy behavior distinct from grading.",
            ha="center", va="bottom", fontsize=8.6, color=GRAY)
    _save(fig, "environment_overview")


def paired_results() -> None:
    payload = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in RESULTS.items()}
    tasks = ["task_2", "task_3", "task_3_unseen"]
    labels = ["Task 2", "Task 3", "Held-out"]

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.55), gridspec_kw={"wspace": 0.36})
    x = np.arange(len(tasks))
    width = 0.34

    for model, color, offset in [("0.6B", ORANGE, -width / 2), ("8B", BLUE, width / 2)]:
        vals = [100 * payload[model]["by_task"][t]["base"]["leak_rate"] for t in tasks]
        axes[0].bar(x + offset, vals, width, label=model, color=color, edgecolor="white")
    axes[0].set_title("Base-model fraud leakage")
    axes[0].set_ylabel("Fraud leakage (%)")
    axes[0].set_xticks(x, labels)
    axes[0].legend(frameon=False, ncol=2, fontsize=8)

    for model, color, offset in [("0.6B", ORANGE, -width / 2), ("8B", BLUE, width / 2)]:
        vals = [payload[model]["by_task"][t]["base"]["grader_mean"] for t in tasks]
        axes[1].bar(x + offset, vals, width, label=model, color=color, edgecolor="white")
    axes[1].set_title("Base-model task utility")
    axes[1].set_ylabel("Normalized grader score")
    axes[1].axhline(
        1.0, color=GRAY, linewidth=1.1, linestyle=(0, (4, 2)),
        label="Maximum normalized score",
    )
    axes[1].text(
        len(tasks) - 0.05, 1.0, "maximum = 1.0", ha="right", va="bottom",
        fontsize=7.5, color=GRAY,
        bbox=dict(facecolor="white", edgecolor="none", pad=1.5),
    )
    axes[1].set_ylim(0, 1.08)
    axes[1].set_xticks(x, labels)

    metrics = ["grader_delta", "leak_rate_delta", "fallback_rate_delta"]
    metric_labels = ["Grader\nscore", "Fraud\nleakage", "Tool\nfallback"]
    scale = [100, 100, 100]
    y = np.arange(len(metrics))
    for idx, (model, color, marker, yoff) in enumerate([
        ("0.6B", ORANGE, "o", -0.10), ("8B", BLUE, "s", 0.10)
    ]):
        points, lo, hi = [], [], []
        for metric, factor in zip(metrics, scale):
            d = payload[model]["paired_deltas"][metric]
            points.append(factor * d["point"])
            lo.append(factor * (d["point"] - d["ci95"]["lo"]))
            hi.append(factor * (d["ci95"]["hi"] - d["point"]))
        axes[2].errorbar(points, y + yoff, xerr=[lo, hi], fmt=marker, ms=5,
                         capsize=3, color=color, label=model, linewidth=1.3)
    axes[2].axvline(0, color=GRAY, linewidth=0.9)
    axes[2].set_title("Post-training change")
    axes[2].set_yticks(y, metric_labels)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Trained minus base (points)")
    axes[2].legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18, linewidth=0.7)
        ax.tick_params(labelsize=8)

    fig.suptitle("CounterFeint separates task utility, safety, and execution reliability",
                 fontsize=13, fontweight="bold", color=NAVY, y=1.03)
    _save(fig, "paired_results")


if __name__ == "__main__":
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    environment_overview()
    paired_results()
