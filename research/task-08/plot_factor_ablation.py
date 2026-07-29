#!/usr/bin/env python3
"""Render the Task 08 factor-ablation summary from tracked screen results."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).with_name("figures") / "factor-ablation.svg"


def main():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["svg.hashsalt"] = "task-08-factor-ablation"
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5))
    blue, orange, red, gray = "#4472C4", "#ED7D31", "#C44E52", "#A5A5A5"

    passes = axes[0].bar(
        ["Expert\n8192 vmap", "256-shot\nchunks"],
        [0, 5],
        color=[red, blue],
    )
    for bar, text in zip(passes, ["0/5\nOOM", "5/5\nPASS"]):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            max(bar.get_height(), 0.05),
            text,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[0].set_ylim(0, 5.8)
    axes[0].set_ylabel("Passing canonical runs")
    axes[0].set_title("Necessary OOM-to-PASS factor")

    names = ["256 chunks", "512 chunks", "128 chunks", "RX fusion", "scan", "OMECo 1×1"]
    seconds = [44.028, 50.843, 55.182, 60.989, 67.401, 131.959]
    y = np.arange(len(names))
    bars = axes[1].barh(y, seconds, color=[blue, gray, gray, orange, orange, red])
    axes[1].set_yticks(y, names)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Full-workload screen (s)")
    axes[1].set_title("Batch-size and rejected variants")
    for bar, value in zip(bars, seconds):
        axes[1].text(
            value + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}s",
            va="center",
            fontsize=8,
        )

    categories = ["256-shot\nscreen", "Exact-observable\naudit"]
    base = [24.601, 11.032]
    split = [30.307, 75.652]
    x = np.arange(2)
    width = 0.36
    b1 = axes[2].bar(x - width / 2, base, width, label="Original entangler", color=blue)
    b2 = axes[2].bar(x + width / 2, split, width, label="Rank-2 split", color=red)
    axes[2].set_xticks(x, categories)
    axes[2].set_ylabel("Runtime (s)")
    axes[2].set_title("Native rank-2 split ablation")
    axes[2].legend(fontsize=8)
    for bars in (b1, b2):
        for bar in bars:
            axes[2].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{bar.get_height():.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle(
        "Task 08 factor attribution — bounded 256-shot batching is the only promoted factor",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", metadata={"Date": None})


if __name__ == "__main__":
    main()
