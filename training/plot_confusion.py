"""Plot confusion matrix heatmap from eval results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CLASSES = ["HONEST", "REWARD_HACKER", "LAZY", "DECEIVER"]
ALL_COLS = CLASSES + ["PARSE_ERROR"]


def plot(results_path: str, out_path: str) -> None:
    results = json.loads(Path(results_path).read_text())
    cm = results["confusion_matrix"]

    # Build matrix: rows=true, cols=predicted
    matrix = np.zeros((len(CLASSES), len(ALL_COLS)))
    for i, true_cls in enumerate(CLASSES):
        for j, pred_cls in enumerate(ALL_COLS):
            matrix[i, j] = cm[true_cls].get(pred_cls, 0)

    # Normalize by row (percentage within each true class)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=120)
    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(ALL_COLS)))
    ax.set_xticklabels(ALL_COLS, rotation=30, ha="right", fontsize=11)
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels(CLASSES, fontsize=11)

    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("True class", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Overseer Confusion Matrix (Held-Out Set)\n"
        f"Overall accuracy: {results['overall_accuracy']:.2%} "
        f"| Parse errors: {results['parse_error_rate']:.2%}",
        fontsize=13, fontweight="bold", pad=15,
    )

    # Annotate cells with both count and percent
    for i in range(len(CLASSES)):
        for j in range(len(ALL_COLS)):
            count = int(matrix[i, j])
            pct = normalized[i, j]
            if count > 0:
                color = "white" if pct > 0.5 else "black"
                ax.text(j, i, f"{count}\n({pct:.0%})",
                        ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Fraction of true class")
    plt.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved confusion matrix -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="figures/eval_results.json")
    ap.add_argument("--out", default="figures/confusion_matrix.png")
    args = ap.parse_args()
    plot(args.results, args.out)


if __name__ == "__main__":
    main()