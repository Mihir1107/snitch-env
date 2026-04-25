"""Generate hero chart from saved training log.

Usage:
    python training/plot_results.py --log data/training_log_smoketest.json --out figures/reward_curve.png

Note: data/training_log_smoketest.json is the 50-step CPU smoke-test log,
NOT the 400-step headline run. The headline run's curves come from the Colab
notebook's trainer_state.json and are visualized in figures/training_curves.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_reward_curve(log_path: str, out_path: str) -> None:
    data = json.loads(Path(log_path).read_text())
    train = data.get("train", [])
    eval_data = data.get("eval", [])

    if not train:
        raise ValueError(f"No train entries in {log_path}")

    steps = [e["step"] for e in train]
    rewards = [e["reward_mean"] for e in train]

    eval_steps = [e["step"] for e in eval_data]
    eval_rewards = [e["eval_reward_mean"] for e in eval_data]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

    ax.plot(steps, rewards, marker="o", linewidth=2.5, markersize=8,
            color="#2E86AB", label="Train reward")

    if eval_steps:
        ax.scatter(eval_steps, eval_rewards, marker="s", s=150,
                   color="#E63946", label="Eval reward (held-out)", zorder=5)

    ax.axhline(y=-0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5,
               label="Parse-error floor")
    ax.axhline(y=1.0, color="green", linestyle=":", linewidth=1, alpha=0.4,
               label="Correct-classification target")

    ax.set_xlabel("Training step", fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean reward", fontsize=13, fontweight="bold")
    ax.set_title("The Snitch — Overseer Learning Trajectory\n"
                 "Qwen2.5-0.5B, GRPO, 50 training steps (smoke test)",
                 fontsize=14, fontweight="bold", pad=15)

    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="lower right", fontsize=11, framealpha=0.95)
    ax.set_ylim(-1.1, 1.6)

    # Annotation: show the improvement
    if len(rewards) >= 2:
        first = rewards[0]
        last = rewards[-1]
        improvement = last - first
        ax.annotate(
            f"Δ = +{improvement:.3f}\n({first:.2f} → {last:.2f})",
            xy=(steps[-1], last),
            xytext=(steps[-1] - 15, last + 0.35),
            fontsize=11, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#2E86AB", lw=1.5),
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F1FAEE",
                      edgecolor="#2E86AB", linewidth=1.5),
        )

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved hero chart -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="data/training_log_smoketest.json")
    ap.add_argument("--out", default="figures/reward_curve.png")
    args = ap.parse_args()
    plot_reward_curve(args.log, args.out)


if __name__ == "__main__":
    main()