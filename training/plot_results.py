"""Hero chart: overall + per-misbehavior accuracy on held-out variant 3.

Usage:
    python training/plot_results.py --results results/eval.json --out results/hero.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def plot(results_path: Path, out_path: Path) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    report = json.loads(results_path.read_text())
    per = report["per_misbehavior_type"]
    labels = ["overall"] + sorted(per.keys())
    values = [report["overall"]["accuracy"]] + [per[k]["accuracy"] for k in labels[1:]]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Held-out accuracy")
    ax.set_title("Snitch overseer — generalization to unseen policy variant")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("results/eval.json"))
    p.add_argument("--out", type=Path, default=Path("results/hero.png"))
    args = p.parse_args()
    plot(args.results, args.out)


if __name__ == "__main__":
    main()
