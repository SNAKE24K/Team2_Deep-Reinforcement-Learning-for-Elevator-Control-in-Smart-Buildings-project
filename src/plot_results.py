"""
Plot learning curves and final-performance comparison from results/.

Generates:
    results/fig1_learning_curves.png
    results/fig2_final_performance.png
    results/fig3_cumulative_reward.png

Run after train.py --agent all has populated results/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "qlearning": "#3B82F6",
    "sarsa": "#EF4444",
    "dqn": "#10B981",
    "baseline": "#E76F51",
}
LABELS = {
    "qlearning": "Q-Learning (tabular)",
    "sarsa": "SARSA (tabular)",
    "dqn": "DQN (deep)",
    "baseline": "Nearest-Floor",
}


def smooth(x: np.ndarray, window: int = 40) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float64)
    for i in range(len(x)):
        s = max(0, i - window + 1)
        out[i] = float(np.mean(x[s : i + 1]))
    return out


def main(out_dir: str = "results"):
    p = Path(out_dir)

    # ---------- Figure 1: avg waiting time ---------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    for name in ["qlearning", "sarsa", "dqn"]:
        f = p / f"{name}_waits.npy"
        if not f.exists():
            print(f"[skip] {f} not found")
            continue
        waits = np.load(f)
        ax.plot(smooth(waits), color=COLORS[name], linewidth=2, label=LABELS[name])
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Avg. Waiting Time (steps/step)", fontsize=11)
    ax.set_title(
        "Average Waiting Time per Episode (40-episode moving average)",
        fontsize=12, color="#1A2B4A", pad=10,
    )
    ax.legend(loc="upper right", fontsize=10, frameon=False)
    ax.grid(True, axis="y", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(p / "fig1_learning_curves.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    # ---------- Figure 2: final performance bar chart ----------------------
    sf = p / "summary.json"
    if sf.exists():
        with open(sf) as fp:
            summary = json.load(fp)
        order = ["baseline", "qlearning", "sarsa", "dqn"]
        labels, values, colors = [], [], []
        for k in order:
            if k in summary:
                labels.append(LABELS[k])
                values.append(summary[k]["avg_wait"])
                colors.append(COLORS[k])

        fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
        bars = ax.bar(labels, values, color=colors, width=0.6, edgecolor="white", linewidth=1.5)
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + max(values) * 0.015,
                f"{v:.2f}",
                ha="center", va="bottom",
                fontsize=12, fontweight="bold", color="#1F2937",
            )
        ax.set_ylabel("Avg. Waiting Time per Step", fontsize=11)
        ax.set_title(
            "Final Performance: Greedy Policy, 100 Test Episodes",
            fontsize=12, color="#1A2B4A", pad=10,
        )
        ax.set_ylim(0, max(values) * 1.18)
        ax.grid(True, axis="y", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(
            p / "fig2_final_performance.png", dpi=150, bbox_inches="tight", facecolor="white"
        )
        plt.close()

    # ---------- Figure 3: cumulative reward --------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    for name in ["qlearning", "sarsa", "dqn"]:
        f = p / f"{name}_rewards.npy"
        if not f.exists():
            continue
        r = np.load(f)
        ax.plot(smooth(r), color=COLORS[name], linewidth=2, label=LABELS[name])
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Cumulative Reward", fontsize=11)
    ax.set_title(
        "Cumulative Reward per Episode (40-episode moving average)",
        fontsize=12, color="#1A2B4A", pad=10,
    )
    ax.legend(loc="lower right", fontsize=10, frameon=False)
    ax.grid(True, axis="y", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(p / "fig3_cumulative_reward.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Figures saved under {p.resolve()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot learning curves from results/")
    parser.add_argument("--out-dir", type=str, default="results",
                        help="Directory containing the .npy curves and summary.json")
    args = parser.parse_args()
    main(args.out_dir)
