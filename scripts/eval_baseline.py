"""Random-verdict baseline over the held-out eval split.

Usage:
    python scripts/eval_baseline.py
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from env.snitch_env import SnitchEnv


def run(traces_path: Path, n_episodes: int, seed: int) -> None:
    env = SnitchEnv(traces_path=traces_path, mode="eval", seed=seed)
    rng = random.Random(seed)
    correct = 0
    confusion: dict[tuple[str, str], int] = {}

    for _ in range(n_episodes):
        env.reset()
        action = rng.choice(["benign", "misbehavior"])
        _, reward, _, info = env.step(action)
        if reward > 0:
            correct += 1
        key = (info["true_label"], action)
        confusion[key] = confusion.get(key, 0) + 1

    print(f"random-baseline accuracy over {n_episodes} held-out episodes: {correct / n_episodes:.3f}")
    print("confusion (true -> pred):")
    for (true, pred), count in sorted(confusion.items()):
        print(f"  {true:>12s} -> {pred:<12s}: {count}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--traces", type=Path, default=Path("data/eval_traces.jsonl"))
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args.traces, args.n, args.seed)


if __name__ == "__main__":
    main()
