"""Run every (policy, variant) combination on every question to produce traces.

Writes train/eval splits:
    - train_traces.jsonl   (variants 1 and 2)
    - eval_traces.jsonl    (variant 3 — held out)

Usage:
    python scripts/generate_traces.py
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from doers import ALL_POLICIES
from doers.base import ToolEnv


DEFAULT_QUESTIONS = Path("data/questions.json")
DEFAULT_TRAIN_OUT = Path("data/train_traces.jsonl")
DEFAULT_EVAL_OUT = Path("data/eval_traces.jsonl")


def generate(
    questions_path: Path,
    train_out: Path,
    eval_out: Path,
    seed: int = 0,
) -> None:
    questions = json.loads(questions_path.read_text())
    rng = random.Random(seed)
    rng.shuffle(questions)

    train_out.parent.mkdir(parents=True, exist_ok=True)
    n_train = n_eval = 0
    with train_out.open("w") as f_train, eval_out.open("w") as f_eval:
        for q in questions:
            tools = ToolEnv(corpus=q["corpus"], relevant_doc_ids=q["relevant_doc_ids"])
            for policy_cls in ALL_POLICIES.values():
                for variant in (1, 2, 3):
                    policy = policy_cls(variant=variant)
                    trace = policy.run(q, tools)
                    line = trace.to_json() + "\n"
                    if variant == 3:
                        f_eval.write(line)
                        n_eval += 1
                    else:
                        f_train.write(line)
                        n_train += 1

    print(f"train: {n_train} traces -> {train_out}")
    print(f"eval:  {n_eval} traces -> {eval_out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    p.add_argument("--train-out", type=Path, default=DEFAULT_TRAIN_OUT)
    p.add_argument("--eval-out", type=Path, default=DEFAULT_EVAL_OUT)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    generate(args.questions, args.train_out, args.eval_out, args.seed)


if __name__ == "__main__":
    main()
