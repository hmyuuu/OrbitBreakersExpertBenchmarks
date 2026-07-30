"""Generate the four summary figures embedded in the repository README.

The values are transcribed from the retained benchmark and paired-comparison
reports linked by README.md.  Run from the repository root:

    MPLCONFIGDIR=/tmp/orbitbreakers-mpl python3 docs/figures/make_readme_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter


OUT = Path(__file__).resolve().parent
TASKS = [f"{i:02d}" for i in range(1, 13)]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "svg.fonttype": "none",
    }
)


def save(fig, stem):
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"{stem}.{suffix}", facecolor="white")
    plt.close(fig)


def fable_runtime_ratios():
    ratios = np.array(
        [2.14, 3.01, 2.54, 11.94, 1.19, 1.09, 1.28, 2.64, 3.63, 18.06, 0.74, 2.22]
    )
    colors = [
        "#2A9D8F" if value <= 1 else "#4C78A8" if value <= 4 else "#E76F51"
        for value in ratios
    ]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    bars = ax.bar(
        TASKS,
        ratios,
        width=0.72,
        color=colors,
        edgecolor="#222222",
        linewidth=0.45,
    )
    ax.axhline(1, color="#555555", linestyle=(0, (4, 3)), linewidth=0.9)
    ax.text(11.55, 1.06, "expert", ha="right", va="bottom", color="#555555", fontsize=8)
    for bar, value in zip(bars, ratios):
        ax.annotate(
            f"{value:.2f}×",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=7.3,
        )

    ax.set_yscale("log")
    ax.set_ylim(0.5, 32)
    ax.set_xlabel("ORBIT-Q task")
    ax.set_ylabel("Candidate / expert runtime (log scale)")
    ax.set_title("Fable 5 artifact efficiency; all 12 artifacts passed")
    ax.grid(axis="y", which="both", color="#D9D9D9", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[
            Patch(facecolor="#2A9D8F", label="faster than expert"),
            Patch(facecolor="#4C78A8", label="≤4× expert time"),
            Patch(facecolor="#E76F51", label=">4× expert time"),
        ],
        loc="upper left",
        frameon=False,
        ncol=3,
    )
    fig.tight_layout()
    save(fig, "fable5-runtime-ratios")


def high_ultra_outcomes():
    high = np.array([0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1])
    ultra = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    outcomes = np.vstack([high, ultra])

    fig = plt.figure(figsize=(7.2, 5.2))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.45], hspace=1.0)

    ax0 = fig.add_subplot(grid[0])
    cmap = ListedColormap(["#D95F59", "#3A9D6F"])
    ax0.imshow(outcomes, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax0.set_xticks(np.arange(12), TASKS)
    ax0.set_yticks([0, 1], ["High · 10/12", "Ultra · 11/12"])
    ax0.set_title("Adjudicated task outcomes")
    ax0.set_xticks(np.arange(-0.5, 12, 1), minor=True)
    ax0.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax0.grid(which="minor", color="white", linewidth=1.4)
    ax0.tick_params(which="minor", bottom=False, left=False)
    for row in range(2):
        for col in range(12):
            ax0.text(
                col,
                row,
                "✓" if outcomes[row, col] else "×",
                ha="center",
                va="center",
                color="white",
                fontsize=10,
                fontweight="bold",
            )
    # Task 05 high and Task 07 ultra were corrected after source-level adjudication.
    for row, col in ((0, 4), (1, 6)):
        ax0.add_patch(
            plt.Rectangle(
                (col - 0.47, row - 0.47),
                0.94,
                0.94,
                fill=False,
                edgecolor="#F4C95D",
                linewidth=2.3,
            )
        )
    ax0.legend(
        handles=[
            Patch(facecolor="#3A9D6F", label="valid"),
            Patch(facecolor="#D95F59", label="invalid"),
            Patch(facecolor="none", edgecolor="#F4C95D", linewidth=2, label="adjudicated"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=3,
        frameon=False,
    )

    ax1 = fig.add_subplot(grid[1])
    labels = ["Valid solutions", "Tokens (M)", "Cost (USD)", "Wall time (min)"]
    high_values = np.array([10.0, 26.071, 25.527, 197.70])
    ultra_values = np.array([11.0, 33.048, 30.019, 182.80])
    normalized = np.vstack([high_values / high_values, ultra_values / high_values])
    x = np.arange(len(labels))
    width = 0.34
    ax1.bar(x - width / 2, normalized[0], width, color="#4C78A8", label="High")
    ax1.bar(x + width / 2, normalized[1], width, color="#7A5195", label="Ultra")
    for column, (hv, uv) in enumerate(zip(high_values, ultra_values)):
        high_label = f"{hv:.0f}" if column == 0 else f"{hv:.1f}"
        ultra_label = f"{uv:.0f}" if column == 0 else f"{uv:.1f}"
        ax1.text(column - width / 2, 1.02, high_label, ha="center", va="bottom", fontsize=7.5)
        ax1.text(
            column + width / 2,
            normalized[1, column] + 0.02,
            ultra_label,
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
    ax1.axhline(1, color="#555555", linewidth=0.7)
    ax1.set_xticks(x, labels)
    ax1.set_ylabel("Relative to High")
    ax1.set_ylim(0, 1.42)
    ax1.set_title("Coverage and solver-side resource use (raw values above bars)")
    ax1.grid(axis="y", color="#E1E1E1", linewidth=0.55)
    ax1.set_axisbelow(True)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(frameon=False, ncol=2, loc="upper right")

    save(fig, "gpt56-high-vs-ultra-outcomes")


def expert_optimization_runtime_bars():
    expert = np.array(
        [
            60.651,
            4.495,
            4.101,
            14.742,
            115.708,
            41.426,
            140.076,
            126.675,
            33.504,
            18.931,
            168.362,
            9.083,
        ]
    )
    optimized = np.array(
        [
            6.360,
            4.031,
            0.925,
            5.672,
            60.114,
            27.537,
            3.071,
            123.188,
            8.767,
            3.869,
            114.968,
            2.321,
        ]
    )
    paired_speedup = np.array(
        [
            9.636,
            1.116,
            4.435,
            2.602,
            1.939,
            1.504,
            45.758,
            1.045,
            3.822,
            4.898,
            1.464,
            3.914,
        ]
    )
    x = np.arange(12)
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    expert_bars = ax.bar(
        x - width / 2,
        expert,
        width,
        color="#D55E00",
        edgecolor="#222222",
        linewidth=0.55,
        label="Human expert",
        zorder=3,
    )
    optimized_bars = ax.bar(
        x + width / 2,
        optimized,
        width,
        color="#0072B2",
        edgecolor="#222222",
        linewidth=0.55,
        label="AI–human optimized",
        zorder=3,
    )

    ax.set_yscale("log")
    ax.set_ylim(0.65, 315)
    ax.set_yticks([1, 3, 10, 30, 100, 300])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.minorticks_off()
    ax.set_xticks(x, TASKS)
    ax.set_xlabel("ORBIT-Q task")
    ax.set_ylabel("Mean evaluator runtime (s; log scale)")
    ax.set_title("Human expert and AI–human optimized runtime")
    ax.grid(axis="y", color="#DEDEDE", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper left")

    for index, (expert_time, optimized_time, speedup) in enumerate(
        zip(expert, optimized, paired_speedup)
    ):
        suffix = "†" if index == 6 else "*" if index == 7 else ""
        close_pair = max(expert_time, optimized_time) / min(
            expert_time, optimized_time
        ) < 1.15
        nudge = 0.04 if close_pair else 0.0
        expert_label_y = expert_time * (1.05 if close_pair else 1.07)
        optimized_label_y = optimized_time * (1.24 if close_pair else 1.07)
        ax.text(
            expert_bars[index].get_x() + expert_bars[index].get_width() / 2 - nudge,
            expert_label_y,
            f"{expert_time:.2f}" if expert_time < 10 else f"{expert_time:.1f}",
            ha="center",
            va="bottom",
            fontsize=6.1,
        )
        ax.text(
            optimized_bars[index].get_x()
            + optimized_bars[index].get_width() / 2
            + nudge,
            optimized_label_y,
            f"{optimized_time:.2f}" if optimized_time < 10 else f"{optimized_time:.1f}",
            ha="center",
            va="bottom",
            fontsize=6.1,
        )
        ax.text(
            index,
            max(expert_time, optimized_time) * (1.58 if close_pair else 1.40),
            f"{speedup:.2f}×{suffix}",
            ha="center",
            va="bottom",
            fontsize=6.7,
            fontweight="bold",
        )

    ax.text(
        0.0,
        -0.16,
        "† Task 07 is an exact challenge-design reduction.  * Task 08 is a "
        "feasibility result; its runtime interval includes no change.",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#555555",
        va="top",
    )
    fig.tight_layout()
    save(fig, "expert-optimization-runtime-log-bars")


def factor_ablation_overview():
    # Direct parent comparisons from the task-level reports. Values are
    # incremental speedups and must not be multiplied across rows.
    panels = [
        (
            "01",
            [
                ("TFIM MPO", 1.512, True, None),
                ("Local fusion", 1.562, True, None),
                ("Batched gates", 3.575, True, None),
                ("Global batching", 1.079, True, None),
            ],
        ),
        (
            "02",
            [
                ("Training scan", 1.036, True, None),
                ("Batched purity", 1.074, True, None),
                ("Packed params", 0.428, False, None),
                ("Gate fusion", 0.966, False, None),
            ],
        ),
        (
            "03",
            [
                ("Product reduction", 1.660, True, None),
                ("Local-map vmap", 2.063, True, None),
                ("Training scan", 1.149, True, None),
                ("Observable vmap", 1.093, True, None),
            ],
        ),
        (
            "04",
            [
                ("Probe vmap", 2.165, True, None),
                ("Kraus pairing", 1.053, True, None),
                ("RXX–Kraus fusion", 1.104, True, None),
                ("Training scan", 0.987, False, None),
            ],
        ),
        (
            "05",
            [
                ("OMECo 4×4", 1.420, True, None),
                ("Filter fusion", 1.198, False, None),
            ],
        ),
        (
            "06",
            [
                ("Native jaxode", 1.529, True, None),
                ("Euler fusion", 1.152, True, None),
                ("Automatic dt0", 1.001, False, None),
                ("Sparse BCOO", 0.295, False, None),
            ],
        ),
        (
            "07",
            [
                ("Exact reduction", 45.758, True, None),
                ("Training scan", 0.935, False, None),
                ("Dense fusion", 0.849, False, None),
                ("Local contractor", 1.044, True, None),
            ],
        ),
        (
            "08",
            [
                ("End to end", 1.045, False, None),
                ("256 vs 512 shots", 1.155, True, None),
                ("256 vs 128 shots", 1.253, True, None),
            ],
        ),
        (
            "09",
            [
                ("Compact cones", 3.822, True, None),
                ("Inner light cone", 34.22, True, ">34.2×"),
                ("Separate groups", 1.082, True, None),
                ("Manual fusion", 1.020, False, None),
            ],
        ),
        (
            "10",
            [
                ("Bounded MPS/MPO", 4.670, True, "≥4.67×"),
                ("Training scan", 0.986, False, None),
                ("Rotation fusion", 1.052, True, None),
            ],
        ),
        (
            "11",
            [
                ("Onsite vector", 81.215, True, None),
                ("Batched Pade", 1.695, True, None),
                ("Training scan", 1.158, True, None),
            ],
        ),
        (
            "12",
            [
                ("Batched Pade", 3.109, True, None),
                ("Cold training scan", 1.083, True, None),
                ("Pair fusion", 1.093, False, None),
            ],
        ),
    ]

    fig, axes = plt.subplots(4, 3, figsize=(10.5, 10.6))
    keep_color = "#0072B2"
    reject_color = "#D55E00"

    for ax, (task, factors) in zip(axes.flat, panels):
        names = [factor[0] for factor in factors]
        values = np.array([factor[1] for factor in factors])
        y = np.arange(len(factors))
        colors = [keep_color if factor[2] else reject_color for factor in factors]
        lower = min(0.82, float(values.min()) * 0.72)
        upper = max(1.28, float(values.max()) * 1.45)

        for row, (value, color) in enumerate(zip(values, colors)):
            left = min(1.0, value)
            width = abs(value - 1.0)
            ax.barh(
                row,
                width,
                left=left,
                height=0.56,
                color=color,
                alpha=0.88,
                edgecolor="#222222",
                linewidth=0.45,
                zorder=3,
            )
            ax.scatter(value, row, s=20, color=color, edgecolor="#222222", linewidth=0.4, zorder=4)
            label = factors[row][3] or f"{value:.3g}×"
            ax.text(
                upper / 1.025,
                row,
                label,
                ha="right",
                va="center",
                fontsize=6.6,
            )

        ax.set_xscale("log")
        ax.set_xlim(lower, upper)
        ax.axvline(1.0, color="#555555", linestyle=(0, (3, 2)), linewidth=0.8, zorder=2)
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_title(f"Task {task}", loc="left", fontweight="bold")
        ax.grid(axis="x", which="both", color="#E1E1E1", linewidth=0.5, zorder=0)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    fig.legend(
        handles=[
            Patch(facecolor=keep_color, edgecolor="#222222", label="retained"),
            Patch(facecolor=reject_color, edgecolor="#222222", label="rejected / unresolved"),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.965),
    )
    fig.suptitle(
        "Factor ablations for all twelve expert-optimization campaigns",
        y=0.995,
        fontsize=12,
    )
    fig.text(
        0.5,
        0.005,
        "Direct-parent speedup (log scale; panel scales vary). Values from "
        "different rows are not multiplicative. Task 11 onsite-vector and "
        "Pade values are isolated kernels.",
        ha="center",
        fontsize=7.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0.01, 0.025, 0.995, 0.92), h_pad=1.2, w_pad=1.2)
    save(fig, "factor-ablation-overview")


if __name__ == "__main__":
    fable_runtime_ratios()
    high_ultra_outcomes()
    expert_optimization_runtime_bars()
    factor_ablation_overview()
