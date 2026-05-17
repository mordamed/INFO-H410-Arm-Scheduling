
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (works inside Docker)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from ..core.instruction import Instruction, ShareType

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
COLORS = {
    "bayesian": "#4C72B0",   # blue
    "csp":      "#DD8452",   # orange
    "mdp":      "#55A868",   # green
}
METHOD_LABELS = {
    "bayesian": "Bayesian Network",
    "csp":      "CSP",
    "mdp":      "MDP (Q-Learning)",
}
SHARE_COLORS = {
    ShareType.SHARE_A: "#E24A33",   # red
    ShareType.SHARE_B: "#348ABD",   # blue
    ShareType.NEUTRAL: "#777777",   # grey
}
DPI = 150


# ---------------------------------------------------------------------------
# Helper: grouped bar chart
# ---------------------------------------------------------------------------

def _grouped_bar(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    methods: List[str],
    sizes: List[int],
    ylabel: str,
    title: str,
    log_scale: bool = False,
) -> None:
    x = np.arange(len(sizes))
    width = 0.25
    offsets = np.linspace(-(len(methods) - 1) / 2, (len(methods) - 1) / 2, len(methods))

    for offset, method in zip(offsets, methods):
        means, stds = [], []
        for n in sizes:
            sub = df[(df["method"] == method) & (df["n_instructions"] == n)]
            means.append(sub[metric].mean() if not sub.empty else 0)
            stds.append(sub[metric].std() if not sub.empty else 0)
        ax.bar(
            x + offset * width,
            means,
            width,
            yerr=stds,
            label=METHOD_LABELS.get(method, method),
            color=COLORS.get(method, "#999999"),
            capsize=4,
            alpha=0.9,
            error_kw={"linewidth": 1.2},
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"n={n}" for n in sizes], fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    if log_scale:
        ax.set_yscale("log")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Figure 1 — Total Cycles
# ---------------------------------------------------------------------------

def plot_cycles(
    df: pd.DataFrame,
    methods: List[str],
    sizes: List[int],
    output_dir: str = "experiments/results",
) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    _grouped_bar(ax, df, "total_cycles", methods, sizes,
                 ylabel="Total Cycles", title="Schedule Length by Method and Block Size")
    fig.tight_layout()
    path = Path(output_dir) / "fig1_cycles.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------------
# Figure 2 — Scheduling Time (log scale)
# ---------------------------------------------------------------------------

def plot_time(
    df: pd.DataFrame,
    methods: List[str],
    sizes: List[int],
    output_dir: str = "experiments/results",
) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    _grouped_bar(ax, df, "wall_time", methods, sizes,
                 ylabel="Time (seconds, log scale)", title="Scheduling Time by Method and Block Size",
                 log_scale=True)
    fig.tight_layout()
    path = Path(output_dir) / "fig2_time.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------------
# Figure 3 — NOPs inserted
# ---------------------------------------------------------------------------

def plot_nops(
    df: pd.DataFrame,
    methods: List[str],
    sizes: List[int],
    output_dir: str = "experiments/results",
) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    _grouped_bar(ax, df, "n_nops", methods, sizes,
                 ylabel="NOP Slots Inserted", title="Safety NOPs by Method and Block Size")
    fig.tight_layout()
    path = Path(output_dir) / "fig3_nops.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------------
# Figure 4 — MDP Learning Curve
# ---------------------------------------------------------------------------

def plot_learning_curve(
    rewards: List[float],
    window: int = 100,
    output_dir: str = "experiments/results",
) -> str:
    fig, ax = plt.subplots(figsize=(7, 3.5))

    episodes = list(range(1, len(rewards) + 1))
    ax.plot(episodes, rewards, alpha=0.3, color=COLORS["mdp"], linewidth=0.8)

    # Smoothed moving average
    if len(rewards) >= window:
        smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(
            range(window, len(rewards) + 1), smooth,
            color=COLORS["mdp"], linewidth=2.0,
            label=f"Moving avg (window={window})",
        )

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Total Reward", fontsize=11)
    ax.set_title("Q-Learning Training Curve", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = Path(output_dir) / "fig4_learning_curve.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------------
# Figure 5 — Gantt Diagram (instruction timeline)
# ---------------------------------------------------------------------------

def plot_gantt(
    schedule: List[Tuple[int, Optional[Instruction]]],
    instructions: List[Instruction],
    title: str = "Schedule Gantt Diagram",
    output_dir: str = "experiments/results",
    filename: str = "fig5_gantt.png",
) -> str:
    fig, ax = plt.subplots(figsize=(max(10, len(schedule) // 2), 5))

    y_labels = [f"[{instr.idx}] {instr.name}" for instr in instructions]
    instr_ypos = {instr.idx: i for i, instr in enumerate(instructions)}

    ax.set_ylim(-0.5, len(instructions) - 0.5)

    for cycle, instr in schedule:
        if instr is None:
            # Draw NOP as a thin grey bar spanning all rows
            ax.axvspan(cycle, cycle + 1, alpha=0.08, color="grey")
            ax.text(cycle + 0.5, len(instructions) - 0.2, "NOP",
                    ha="center", va="center", fontsize=6, color="grey")
            continue

        y = instr_ypos[instr.idx]
        color = SHARE_COLORS[instr.share_type]
        ax.barh(y, instr.latency, left=cycle, height=0.6, color=color, alpha=0.85,
                edgecolor="white", linewidth=0.5)
        ax.text(cycle + instr.latency / 2, y, instr.name,
                ha="center", va="center", fontsize=7, color="white", fontweight="bold")

    # Axes formatting
    ax.set_yticks(list(instr_ypos.values()))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("Pipeline Cycle", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Legend
    patches = [
        mpatches.Patch(color=SHARE_COLORS[ShareType.SHARE_A], label="Share A"),
        mpatches.Patch(color=SHARE_COLORS[ShareType.SHARE_B], label="Share B"),
        mpatches.Patch(color=SHARE_COLORS[ShareType.NEUTRAL], label="Neutral"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=9)

    fig.tight_layout()
    path = Path(output_dir) / filename
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------------
# Generate all figures in one call
# ---------------------------------------------------------------------------

def generate_all_figures(
    df: pd.DataFrame,
    rewards: Optional[List[float]] = None,
    example_schedule: Optional[Tuple] = None,
    output_dir: str = "experiments/results",
    verbose: bool = True,
) -> List[str]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    methods = list(df["method"].unique())
    sizes = sorted(df["n_instructions"].unique().tolist())

    paths: List[str] = []

    p = plot_cycles(df, methods, sizes, output_dir)
    paths.append(p)
    if verbose:
        print(f"  Saved: {p}")

    p = plot_time(df, methods, sizes, output_dir)
    paths.append(p)
    if verbose:
        print(f"  Saved: {p}")

    p = plot_nops(df, methods, sizes, output_dir)
    paths.append(p)
    if verbose:
        print(f"  Saved: {p}")

    if rewards:
        p = plot_learning_curve(rewards, output_dir=output_dir)
        paths.append(p)
        if verbose:
            print(f"  Saved: {p}")

    if example_schedule:
        schedule, instructions, title = example_schedule
        p = plot_gantt(schedule, instructions, title=title, output_dir=output_dir)
        paths.append(p)
        if verbose:
            print(f"  Saved: {p}")

    return paths
