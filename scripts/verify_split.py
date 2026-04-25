"""Prove that train and held-out eval sets are disjoint on both axes.

Headline claim in the README: "the held-out eval is doubly out-of-distribution
because it varies on question IDs *and* policy variant simultaneously." This
script makes that claim trivially auditable.

Run from repo root:
    python scripts/verify_split.py

Exit code 0 = disjoint (good). Non-zero = overlap detected (bad).
A snapshot of the output lives in results/data_split_verification.txt so judges
can verify without running anything.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "train_traces.jsonl"
EVAL = ROOT / "data" / "eval_traces.jsonl"


def load_axes(path: Path) -> tuple[set[str], set[int], int]:
    qids: set[str] = set()
    variants: set[int] = set()
    n = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        qids.add(rec["question_id"])
        variants.add(rec["policy_variant"])
        n += 1
    return qids, variants, n


def main() -> int:
    train_q, train_v, n_train = load_axes(TRAIN)
    eval_q, eval_v, n_eval = load_axes(EVAL)

    qid_overlap = train_q & eval_q
    var_overlap = train_v & eval_v

    print(f"train_traces.jsonl: n={n_train}, "
          f"unique question IDs={len(train_q)}, variants={sorted(train_v)}")
    print(f"eval_traces.jsonl:  n={n_eval}, "
          f"unique question IDs={len(eval_q)}, variants={sorted(eval_v)}")
    print()
    print(f"question ID overlap (train ∩ eval): {len(qid_overlap)}")
    print(f"variant     overlap (train ∩ eval): {len(var_overlap)}")

    failed = False
    if qid_overlap:
        sample = sorted(qid_overlap)[:10]
        print(f"\nFAIL: question IDs overlap. First {len(sample)}: {sample}")
        failed = True
    if var_overlap:
        print(f"\nFAIL: policy variants overlap: {sorted(var_overlap)}")
        failed = True

    if failed:
        print("\nResult: OVERLAP DETECTED — held-out claim is invalid.")
        return 1

    print("\nResult: PASS — train and eval are disjoint on BOTH "
          "question IDs and policy variants (doubly OOD).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
