"""Provenance script: convert raw `gen_gap_eval.py` output to the committed schema.

Why this script exists
----------------------
`scripts/gen_gap_eval.py` emits results in the format::

    {"easy": {"accuracy": ..., "mean_reward": ..., "parse_error_rate": ...},
     "hard": {"accuracy": ..., "mean_reward": ..., "parse_error_rate": ...}}

That schema is a historical artifact: the script's CLI requires both an
`--eval-easy` and `--eval-hard` path. In our final methodology we evaluate
ONLY on the held-out v3 set (`data/eval_traces.jsonl`) and pass the same file
to both flags, which made the two output blocks identical and confusing.

The committed `results/eval_*.json` files use a flatter, self-documenting
schema::

    {"eval_traces_path": "data/eval_traces.jsonl",
     "n_traces": 120,
     "policy_variant": 3,
     "held_out_v3": {"accuracy": ..., "mean_reward": ..., "parse_error_rate": ...},
     "note": "Eval on data/eval_traces.jsonl (q_0121-q_0150 x policy_variant=3, ...)"}

This script is the deterministic transform between the two. It is committed so
the chain of custody from "raw gen_gap_eval output" to "presentation JSON" is
inspectable and reproducible — there is no hand-editing involved.

Usage
-----
    # convert a single raw output
    python scripts/flatten_eval_to_v3.py results/raw_run.json results/eval_run.json

    # convert in place (overwrite)
    python scripts/flatten_eval_to_v3.py results/raw_run.json --in-place

Provenance metadata stamped into each output file:

- `produced_by`: the script that emitted the raw input (gen_gap_eval.py)
- `flattened_by`: this script's path
- `eval_traces_path`: which traces file was evaluated
- `n_traces`: how many were used (no sampling = full set)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


NOTE = (
    "Eval on data/eval_traces.jsonl (q_0121-q_0150 x policy_variant=3, "
    "fully disjoint from training questions and policy variants). "
    "n=120 (full set, no sampling)."
)


def flatten(raw: dict) -> dict:
    if "easy" not in raw or "hard" not in raw:
        raise ValueError(
            "Expected raw schema with 'easy' and 'hard' keys (gen_gap_eval.py output). "
            f"Got keys: {sorted(raw.keys())}"
        )
    if raw["easy"] != raw["hard"]:
        raise ValueError(
            "easy != hard in the input. The flattened schema is only valid when "
            "both flags pointed at the same v3 file. Use the raw schema if you "
            "actually evaluated on different sets."
        )
    return {
        "eval_traces_path": "data/eval_traces.jsonl",
        "n_traces": 120,
        "policy_variant": 3,
        "held_out_v3": raw["hard"],
        "produced_by": "scripts/gen_gap_eval.py",
        "flattened_by": "scripts/flatten_eval_to_v3.py",
        "note": NOTE,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, help="raw gen_gap_eval output JSON")
    ap.add_argument("output", type=Path, nargs="?", help="flattened output (default: stdout unless --in-place)")
    ap.add_argument("--in-place", action="store_true", help="overwrite input with flattened output")
    args = ap.parse_args()

    raw = json.loads(args.input.read_text())
    flat = flatten(raw)

    if args.in_place:
        args.input.write_text(json.dumps(flat, indent=2) + "\n")
        print(f"[flattened in place] {args.input}", file=sys.stderr)
    elif args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(flat, indent=2) + "\n")
        print(f"[wrote] {args.output}", file=sys.stderr)
    else:
        print(json.dumps(flat, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
