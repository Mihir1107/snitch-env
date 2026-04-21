"""Print 10 random questions in readable form for manual quality review.

Usage: python scripts/spot_check.py [--path data/questions.json] [--n 10]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/questions.json")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    questions = json.loads(Path(args.path).read_text())
    if args.seed is not None:
        random.seed(args.seed)
    sample = random.sample(questions, min(args.n, len(questions)))

    for q in sample:
        print("=" * 80)
        print(f"ID: {q['id']}  |  domain: {q['domain']}  |  answer_type: {q['answer_type']}  |  noise: {q['noise_tier']}")
        print(f"Q: {q['question']}")
        print(f"GROUND TRUTH: {q['ground_truth_answer']}")
        print(f"RELEVANT DOCS: {q['relevant_doc_ids']}")
        print(f"PLAUSIBLE WRONG: {q['plausible_wrong_answers']}")
        print(f"MIN STEPS: {q['required_min_steps']}")
        print(f"CORPUS ({len(q['corpus'])} docs):")
        for doc_id, text in q["corpus"].items():
            marker = " *" if doc_id in q["relevant_doc_ids"] else "  "
            print(f" {marker} {doc_id}: {text[:120]}{'...' if len(text) > 120 else ''}")
        print()


if __name__ == "__main__":
    main()