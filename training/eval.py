"""Held-out evaluation on variant-3 traces.

Reports overall accuracy and per-misbehavior-type breakdown. Also writes
results.json for plot_results.py.

Usage:
    python training/eval.py --model checkpoints/snitch-grpo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from env.reward import compute_reward
from env.snitch_env import SnitchEnv
from training.train_grpo import format_prompt, parse_verdict


def run(model_dir: Path, traces: Path, out: Path) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    gen = pipeline("text-generation", model=model, tokenizer=tok, max_new_tokens=8, do_sample=False)

    env = SnitchEnv(traces_path=traces, mode="eval", seed=0)
    total = 0
    correct = 0
    per_type: dict[str, dict[str, int]] = {}

    while True:
        try:
            obs, info = env.reset()
        except RuntimeError:
            break
        mtype = env._current.misbehavior_type or "honest"  # type: ignore[attr-defined]
        prompt = format_prompt(obs)
        out_text = gen(prompt)[0]["generated_text"][len(prompt) :]
        verdict = parse_verdict(out_text)
        _, reward, _, step_info = env.step(verdict)
        correct += 1 if reward > 0 else 0
        total += 1
        bucket = per_type.setdefault(mtype, {"correct": 0, "total": 0})
        bucket["total"] += 1
        if reward > 0:
            bucket["correct"] += 1
        if total >= len(env._traces):  # type: ignore[attr-defined]
            break

    acc = correct / total if total else 0.0
    report = {
        "overall": {"accuracy": acc, "correct": correct, "total": total},
        "per_misbehavior_type": {
            k: {"accuracy": v["correct"] / v["total"] if v["total"] else 0.0, **v}
            for k, v in per_type.items()
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("checkpoints/snitch-grpo"))
    p.add_argument("--traces", type=Path, default=Path("data/eval_traces.jsonl"))
    p.add_argument("--out", type=Path, default=Path("results/eval.json"))
    args = p.parse_args()
    run(args.model, args.traces, args.out)


if __name__ == "__main__":
    main()
