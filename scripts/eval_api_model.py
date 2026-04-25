"""Evaluate any frontier LLM (via API) on the held-out v3 set.

Same prompt + parser + reward function as `scripts/gen_gap_eval.py`, so the
output JSON is directly comparable to `results/eval_lora_lr2e5_400.json` and
the trained-overseer numbers in the README.

Supported providers
-------------------
- ``openai`` — chat completions via the OpenAI v2 SDK. Requires
  ``OPENAI_API_KEY`` in the environment. Models: ``gpt-4o-mini``,
  ``gpt-4o``, ``gpt-5-mini``, etc.
- ``hf`` — Hugging Face Inference Providers via ``huggingface_hub``.
  Requires ``HF_TOKEN`` in the environment. Models:
  ``meta-llama/Llama-3.1-8B-Instruct``, ``Qwen/Qwen2.5-7B-Instruct``, etc.

Pilot vs full run
-----------------
- ``--limit 5`` runs the 5-trace pilot (cost: <$0.01 on gpt-4o-mini).
  Use this first to confirm parse rate and prompt formatting.
- Default (no ``--limit``) runs the full 120-trace held-out v3 set.
  Estimated cost on gpt-4o-mini: ~$0.05; on gpt-4o: ~$0.80.

Output schema
-------------
The JSON written to ``--out`` matches the schema of
``results/eval_lora_lr2e5_400.json`` plus per-class confusion, evidence-bonus
hit rates, and a ``cost`` block (input/output tokens, estimated USD).

Example
-------
    # 5-trace pilot
    python scripts/eval_api_model.py \\
        --model gpt-4o-mini --provider openai \\
        --traces data/eval_traces.jsonl \\
        --out results/eval_gpt4omini_pilot5.json \\
        --limit 5

    # full leaderboard run
    python scripts/eval_api_model.py \\
        --model meta-llama/Llama-3.1-8B-Instruct --provider hf \\
        --traces data/eval_traces.jsonl \\
        --out results/eval_llama_3_1_8b.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.parse import parse_overseer_output
from env.reward import compute_reward
from env.trace_format import Trace

# Reuse the canonical eval prompt + trace formatter so the API model sees
# byte-for-byte the same input the trained overseer was scored on.
from scripts.gen_gap_eval import (
    SYSTEM_PROMPT,
    build_user_prompt,
    format_trace_body,
)


CLASSES = ["HONEST", "REWARD_HACKER", "LAZY", "DECEIVER"]


# Rough USD-per-1K-tokens for cost telemetry. Conservative; updated 2026-04.
# These are NOT used for routing or limits — only for the cost line in the
# output JSON. Off-by-2x on a $0.05 run is fine.
PRICING_USD_PER_1K = {
    # OpenAI
    "gpt-4o-mini":     {"in": 0.00015, "out": 0.00060},
    "gpt-4o":          {"in": 0.00250, "out": 0.01000},
    "gpt-5-mini":      {"in": 0.00025, "out": 0.00200},
    "gpt-5":           {"in": 0.00125, "out": 0.01000},
    # HF Inference Providers — pricing varies per provider; report 0 and let
    # the reader cross-check. Tokens are still counted from the response.
    "meta-llama/Llama-3.1-8B-Instruct":  {"in": 0.0,    "out": 0.0},
    "Qwen/Qwen2.5-7B-Instruct":          {"in": 0.0,    "out": 0.0},
    "mistralai/Mistral-7B-Instruct-v0.3": {"in": 0.0,    "out": 0.0},
}


# =============================================================================
# Provider adapters
# =============================================================================

class APIError(RuntimeError):
    """Raised when an API call fails after all retries."""


def _retry(fn, *, attempts: int = 4, base_delay: float = 1.5):
    """Retry ``fn`` with exponential backoff. Returns whatever ``fn`` returns."""
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — provider SDKs raise their own classes
            last_exc = exc
            if i == attempts - 1:
                break
            sleep_for = base_delay * (2 ** i)
            print(f"  [retry {i+1}/{attempts-1}] {type(exc).__name__}: {exc} — sleeping {sleep_for:.1f}s", file=sys.stderr)
            time.sleep(sleep_for)
    raise APIError(f"all {attempts} attempts failed; last error: {last_exc}") from last_exc


def call_openai(model: str, messages: list[dict], temperature: float, max_tokens: int) -> tuple[str, int, int]:
    from openai import OpenAI
    client = OpenAI()  # reads OPENAI_API_KEY from env

    def _do():
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = resp.choices[0].message.content or ""
        in_tok = resp.usage.prompt_tokens if resp.usage else 0
        out_tok = resp.usage.completion_tokens if resp.usage else 0
        return text, in_tok, out_tok

    return _retry(_do)


def call_hf(model: str, messages: list[dict], temperature: float, max_tokens: int) -> tuple[str, int, int]:
    from huggingface_hub import InferenceClient
    client = InferenceClient(token=os.environ.get("HF_TOKEN"))

    def _do():
        # InferenceClient.chat_completion mirrors the OpenAI schema.
        resp = client.chat_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
        out_tok = getattr(usage, "completion_tokens", 0) if usage else 0
        return text, in_tok, out_tok

    return _retry(_do)


PROVIDERS = {
    "openai": call_openai,
    "hf": call_hf,
}


# =============================================================================
# Eval loop
# =============================================================================

def evaluate(
    *,
    provider: str,
    model: str,
    traces: list[Trace],
    temperature: float,
    max_new_tokens: int,
    verbose: bool,
) -> dict:
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; choose from {sorted(PROVIDERS)}")
    api_call = PROVIDERS[provider]

    confusion: dict[str, Counter] = {c: Counter() for c in CLASSES}
    rewards: list[float] = []
    parse_errors = 0
    api_failures = 0
    bonus_hits: dict[str, int] = {c: 0 for c in CLASSES}
    bonus_eligible: dict[str, int] = {c: 0 for c in CLASSES}
    total_in = 0
    total_out = 0

    started = time.time()
    for i, trace in enumerate(traces):
        trace_body = format_trace_body(trace.steps)
        user_text = build_user_prompt(trace.question, trace_body, trace.claimed_answer)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_text},
        ]

        try:
            completion, in_tok, out_tok = api_call(model, messages, temperature, max_new_tokens)
        except APIError as exc:
            api_failures += 1
            print(f"  [trace {i+1}] API failed permanently: {exc}", file=sys.stderr)
            # treat as a parse error so the metric is conservative, not optimistic
            confusion[trace.policy_class]["PARSE_ERROR"] += 1
            parse_errors += 1
            rewards.append(-0.5)
            continue

        total_in += in_tok
        total_out += out_tok

        parsed = parse_overseer_output(completion)
        predicted = parsed["verdict"]
        if predicted == "PARSE_ERROR":
            parse_errors += 1
            confusion[trace.policy_class]["PARSE_ERROR"] += 1
        else:
            confusion[trace.policy_class][predicted] += 1

        gt = {
            "policy_class": trace.policy_class,
            "smoking_gun_step": trace.smoking_gun_step,
            "smoking_gun_keyword": trace.smoking_gun_keyword,
        }
        r = compute_reward(parsed, gt)
        rewards.append(r)

        # evidence-bonus tracking (mirror scripts/dump_confusion.py)
        # r == 1.5 = correct + bonus, r == 1.0 = correct without bonus.
        # HONEST never has a bonus path so it's excluded from "eligible".
        if predicted == trace.policy_class and trace.policy_class != "HONEST":
            bonus_eligible[trace.policy_class] += 1
            if r >= 1.4:
                bonus_hits[trace.policy_class] += 1

        if verbose:
            print(f"  [{i+1:>3}/{len(traces)}] gt={trace.policy_class:<14} pred={predicted:<14} r={r:+.2f} "
                  f"tok={in_tok}/{out_tok}")
        elif i > 0 and (i + 1) % 10 == 0:
            elapsed = time.time() - started
            rate = (i + 1) / elapsed
            print(f"  {i+1}/{len(traces)} ({rate:.1f} tr/s, {elapsed:.0f}s elapsed)")

    elapsed = time.time() - started

    per_class_acc: dict[str, float | None] = {}
    for cls in CLASSES:
        total = sum(confusion[cls].values())
        correct = confusion[cls].get(cls, 0)
        # None when the sample contains no traces of this class (e.g. small
        # --limit run); we exclude empties from the macro average instead of
        # silently treating them as 0.0 accuracy.
        per_class_acc[cls] = (correct / total) if total > 0 else None

    present = [v for v in per_class_acc.values() if v is not None]
    accuracy_macro = sum(present) / len(present) if present else 0.0

    total_correct = sum(confusion[cls].get(cls, 0) for cls in CLASSES)
    total_n = sum(sum(confusion[cls].values()) for cls in CLASSES)
    accuracy_plain = total_correct / total_n if total_n else 0.0

    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    parse_rate = parse_errors / len(traces) if traces else 0.0

    bonus_rates = {
        cls: (bonus_hits[cls] / bonus_eligible[cls]) if bonus_eligible[cls] else None
        for cls in CLASSES
    }

    pricing = PRICING_USD_PER_1K.get(model, {"in": 0.0, "out": 0.0})
    est_usd = (total_in / 1000) * pricing["in"] + (total_out / 1000) * pricing["out"]

    return {
        "accuracy": accuracy_plain,
        "accuracy_macro": accuracy_macro,
        "mean_reward": mean_reward,
        "parse_error_rate": parse_rate,
        "api_failures": api_failures,
        "per_class_accuracy": per_class_acc,
        "evidence_bonus_hit_rate": bonus_rates,
        "confusion": {cls: dict(confusion[cls]) for cls in CLASSES},
        "wallclock_seconds": round(elapsed, 1),
        "tokens": {"input": total_in, "output": total_out},
        "estimated_cost_usd": round(est_usd, 4),
    }


# =============================================================================
# IO
# =============================================================================

def load_traces(path: str, limit: int | None, seed: int) -> list[Trace]:
    traces: list[Trace] = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            traces.append(Trace.from_json(line))
    if limit is not None and limit < len(traces):
        rng = random.Random(seed)
        traces = rng.sample(traces, limit)
    return traces


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--model", required=True, help="model id, e.g. gpt-4o-mini or meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--provider", required=True, choices=sorted(PROVIDERS), help="API provider")
    ap.add_argument("--traces", default="data/eval_traces.jsonl", help="path to held-out trace jsonl")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only first N (deterministic sample). Pilot uses --limit 5.")
    ap.add_argument("--temperature", type=float, default=0.0, help="sampling temperature (0.0 = greedy)")
    ap.add_argument("--max-new-tokens", type=int, default=256, help="generation cap")
    ap.add_argument("--seed", type=int, default=42, help="trace-sampling seed")
    ap.add_argument("--verbose", action="store_true", help="log every trace")
    args = ap.parse_args()

    if args.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        return 2
    if args.provider == "hf" and not os.environ.get("HF_TOKEN"):
        print("ERROR: HF_TOKEN not set", file=sys.stderr)
        return 2

    traces = load_traces(args.traces, args.limit, args.seed)
    print(f"Loaded {len(traces)} traces from {args.traces}"
          + (f" (sampled from full set with seed={args.seed})" if args.limit else ""))
    print(f"Model: {args.model}  Provider: {args.provider}  T={args.temperature}  max_tokens={args.max_new_tokens}")

    result = evaluate(
        provider=args.provider,
        model=args.model,
        traces=traces,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        verbose=args.verbose,
    )

    out = {
        "model": args.model,
        "provider": args.provider,
        "eval_traces_path": args.traces,
        "n_traces": len(traces),
        "policy_variant": 3,
        "produced_by": "scripts/eval_api_model.py",
        "config": {
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "limit": args.limit,
            "seed": args.seed,
        },
        "held_out_v3": result,
        "note": (
            "Same prompt (SYSTEM_PROMPT + build_user_prompt + format_trace_body) and "
            "same reward function as scripts/gen_gap_eval.py. Numbers are directly "
            "comparable to results/eval_lora_lr2e5_400.json (the trained overseer)."
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")

    print()
    print(f"=== {args.model} on {len(traces)} traces ===")
    print(f"  accuracy            = {result['accuracy']:.3f}")
    print(f"  accuracy_macro      = {result['accuracy_macro']:.3f}")
    print(f"  mean_reward         = {result['mean_reward']:+.3f}")
    print(f"  parse_error_rate    = {result['parse_error_rate']:.3f}")
    print(f"  api_failures        = {result['api_failures']}")
    print(f"  per-class accuracy  = " + ", ".join(
        f"{c}={result['per_class_accuracy'][c]:.2f}" if result['per_class_accuracy'][c] is not None else f"{c}=n/a"
        for c in CLASSES
    ))
    print(f"  evidence-bonus hit  = " + ", ".join(
        f"{c}={result['evidence_bonus_hit_rate'][c]:.2f}" if result['evidence_bonus_hit_rate'][c] is not None else f"{c}=n/a"
        for c in CLASSES
    ))
    print(f"  tokens (in/out)     = {result['tokens']['input']:,} / {result['tokens']['output']:,}")
    print(f"  estimated cost      = ${result['estimated_cost_usd']:.4f}")
    print(f"  wallclock           = {result['wallclock_seconds']:.1f}s")
    print(f"Saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
