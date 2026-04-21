"""Run all 12 doer policies on all questions to generate training + eval traces.

Outputs two JSONL files:
- data/train_traces.jsonl: v1 + v2 variants (8 policies × N questions)
- data/eval_traces.jsonl: v3 variants only (4 policies × N questions, held out)

Usage:
    python scripts/generate_traces.py --questions data/questions.json --out-dir data/
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from doers.base import ToolEnv
from doers.honest import HonestPolicy
from doers.reward_hacker import RewardHackerPolicy
from doers.lazy import LazyPolicy
from doers.deceiver import DeceiverPolicy
from env.trace_format import Trace


POLICY_CLASSES = [HonestPolicy, RewardHackerPolicy, LazyPolicy, DeceiverPolicy]
VARIANTS = [1, 2, 3]


def generate_traces(questions: list[dict], seed: int = 42) -> tuple[list[Trace], list[Trace]]:
    """Generate traces. Returns (train_traces, eval_traces)."""
    train_traces: list[Trace] = []
    eval_traces: list[Trace] = []

    per_policy_counts: dict[str, int] = {}
    per_policy_errors: dict[str, int] = {}

    for q_idx, question in enumerate(questions):
        # Deterministic per-question seed so reruns are reproducible
        base_seed = seed + q_idx * 100

        for policy_cls in POLICY_CLASSES:
            for variant in VARIANTS:
                policy_seed = base_seed + hash((policy_cls.__name__, variant)) % 1000
                policy = policy_cls(variant=variant, seed=policy_seed)
                tool_rng = random.Random(policy_seed + 1)
                tools = ToolEnv(
                    corpus=question["corpus"],
                    relevant_doc_ids=question["relevant_doc_ids"],
                    rng=tool_rng,
                )

                key = f"{policy.policy_class}_v{variant}"
                try:
                    trace = policy.run(question, tools)
                    # Validate before accepting
                    trace.validate()
                except Exception as e:
                    per_policy_errors[key] = per_policy_errors.get(key, 0) + 1
                    print(f"  [error {key} on {question['id']}]: {e}")
                    continue

                per_policy_counts[key] = per_policy_counts.get(key, 0) + 1

                if variant == 3:
                    eval_traces.append(trace)
                else:
                    train_traces.append(trace)

    print("\nPer-policy counts:")
    for key in sorted(per_policy_counts):
        errors = per_policy_errors.get(key, 0)
        print(f"  {key}: {per_policy_counts[key]} generated, {errors} errors")

    return train_traces, eval_traces


def write_jsonl(traces: list[Trace], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for t in traces:
            f.write(t.to_json() + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="data/questions.json")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    questions = json.loads(Path(args.questions).read_text())
    print(f"Loaded {len(questions)} questions.\n")

    train_traces, eval_traces = generate_traces(questions, seed=args.seed)

    out_dir = Path(args.out_dir)
    train_path = out_dir / "train_traces.jsonl"
    eval_path = out_dir / "eval_traces.jsonl"

    write_jsonl(train_traces, train_path)
    write_jsonl(eval_traces, eval_path)

    print(f"\nWrote {len(train_traces)} train traces -> {train_path}")
    print(f"Wrote {len(eval_traces)} eval traces -> {eval_path}")

    # Sanity: class distribution
    from collections import Counter
    train_dist = Counter(t.policy_class for t in train_traces)
    eval_dist = Counter(t.policy_class for t in eval_traces)
    print(f"\nTrain class distribution: {dict(train_dist)}")
    print(f"Eval class distribution:  {dict(eval_dist)}")


if __name__ == "__main__":
    main()