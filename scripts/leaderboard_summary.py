"""Aggregate eval results into one side-by-side leaderboard table.

Reads three sources, fuses them, and emits both a Markdown table and a JSON
summary keyed by row. Drop into a README, a slide, or a triage discussion.

Sources
-------
1. ``results/confusion.json`` (output of ``scripts/dump_confusion.py``):
   per-class accuracy + evidence-bonus hit rates for ``base`` / ``ckpt300``
   / ``ckpt400``. Pass via ``--confusion``.
2. ``results/eval_*_n120.json`` (output of ``scripts/eval_api_model.py``):
   frontier-LLM rows (gpt-4o-mini, gpt-4o, llama-3.1-8b, qwen-7b, …).
   Auto-discovered; override with ``--frontier-glob``.
3. ``results/eval_lora_lr2e5_400.json`` / ``eval_base_full120.json`` /
   ``eval_old_ckpt_n120.json`` (legacy flat schema, ``held_out_v3.{acc,mr,parse}``):
   used as a fallback for accuracy / mean-reward when ``confusion.json`` is
   not yet available. Per-class columns will be ``—`` for these rows until
   ``dump_confusion.py`` lands.

Columns (per narrative-Claude's spec, round 3)
----------------------------------------------
Model | n | Accuracy | Mean Reward | R-HACK Recall | DECEIVER Bonus | Cost | Notes

Where:
- *R-HACK Recall* is the per-class accuracy for the REWARD_HACKER class (the
  class both gpt-4o-mini and gpt-4o collapse on).
- *DECEIVER Bonus* is the evidence-bonus hit rate on DECEIVER traces — the
  load-bearing grounding metric per the round-3 conditions (>30% on ckpt400
  triggers the full benchmark pivot).
- *Cost* is per-120-trace eval; ``—`` for local-GPU runs, USD for API runs.

Usage
-----
    # While waiting for confusion.json: just emit what's available.
    python scripts/leaderboard_summary.py

    # Once confusion.json lands:
    python scripts/leaderboard_summary.py --confusion results/confusion.json

    # Save to a file:
    python scripts/leaderboard_summary.py \\
        --confusion results/confusion.json \\
        --out-md docs/leaderboard.md \\
        --out-json results/leaderboard.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# -----------------------------------------------------------------------------
# Display-label map. Keep the labels short — they go in a table.
# Order in this dict is the canonical sort order when ``--sort none``.
# -----------------------------------------------------------------------------
ROW_LABELS = {
    # Local checkpoints (filled from confusion.json)
    "base":           "Qwen2.5-1.5B + 3-shot (no training)",
    "ckpt300":        "Qwen2.5-1.5B + LoRA, LR=5e-6, 300 steps",
    "ckpt400":        "Qwen2.5-1.5B + LoRA, LR=2e-5, 400 steps  (ours)",
    # Frontier APIs (filled from eval_*_n120.json files)
    "gpt-4o-mini":    "GPT-4o-mini",
    "gpt-4o":         "GPT-4o",
    "claude-haiku-4-5":  "Claude Haiku 4.5",
    "claude-sonnet-4-5": "Claude Sonnet 4.5",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct":         "Qwen2.5-7B-Instruct (untuned, larger sibling)",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral-7B-Instruct-v0.3",
}


def _fmt_pct(x: float | None) -> str:
    return "—" if x is None else f"{x*100:.1f}%"


def _fmt_signed(x: float | None) -> str:
    return "—" if x is None else f"{x:+.3f}"


def _fmt_cost(x: float | None) -> str:
    if x is None:
        return "—"
    if x == 0.0:
        return "free"
    return f"${x:.3f}"


# -----------------------------------------------------------------------------
# Row builders. Each returns a uniform dict the table renderer consumes.
# -----------------------------------------------------------------------------

def _row(
    *,
    key: str,
    n: int | None,
    accuracy: float | None,
    mean_reward: float | None,
    rhack_recall: float | None,
    deceiver_bonus: float | None,
    cost_usd: float | None,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "key": key,
        "label": ROW_LABELS.get(key, key),
        "n": n,
        "accuracy": accuracy,
        "mean_reward": mean_reward,
        "rhack_recall": rhack_recall,
        "deceiver_bonus": deceiver_bonus,
        "cost_usd": cost_usd,
        "notes": notes,
    }


def rows_from_confusion(confusion_json: dict) -> list[dict]:
    """Each top-level key (base/ckpt300/ckpt400) becomes a row."""
    out: list[dict] = []
    for ckpt_key in ("base", "ckpt300", "ckpt400"):
        if ckpt_key not in confusion_json:
            continue
        block = confusion_json[ckpt_key]
        per_class = block.get("per_class_accuracy", {}) or {}
        bonus = block.get("evidence_bonus_hit_rate", {}) or {}
        out.append(_row(
            key=ckpt_key,
            n=block.get("n_traces"),
            accuracy=block.get("accuracy"),
            mean_reward=block.get("mean_reward"),
            rhack_recall=per_class.get("REWARD_HACKER"),
            deceiver_bonus=bonus.get("DECEIVER"),
            cost_usd=None,  # local GPU; cost is wallclock not USD
            notes=_note_for_local(ckpt_key, block),
        ))
    return out


def _note_for_local(key: str, block: dict) -> str:
    bonus = block.get("evidence_bonus_hit_rate", {}) or {}
    bonus_vals = [v for v in bonus.values() if v is not None]
    if not bonus_vals:
        return ""
    avg = sum(bonus_vals) / len(bonus_vals)
    if key == "base":
        return f"avg bonus hit {avg*100:.0f}%; pretrained baseline"
    if key.startswith("ckpt"):
        return f"avg bonus hit {avg*100:.0f}%; RL'd on grounded reward"
    return ""


def rows_from_frontier(paths: list[Path]) -> list[dict]:
    out: list[dict] = []
    for p in paths:
        try:
            data = json.loads(p.read_text())
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] could not read {p}: {exc}", file=sys.stderr)
            continue
        block = data.get("held_out_v3") or {}
        # Legacy flat JSONs (eval_lora_lr2e5_400, eval_base_full120, eval_old_ckpt_n120):
        # only have accuracy / mean_reward / parse_error_rate. Skip if confusion.json
        # will fill these in (we detect by filename).
        if p.name in {
            "eval_lora_lr2e5_400.json",
            "eval_base_full120.json",
            "eval_old_ckpt_n120.json",
        }:
            continue
        model = data.get("model")
        if not model:
            print(f"[warn] {p} has no 'model' key; skipping", file=sys.stderr)
            continue
        per_class = block.get("per_class_accuracy", {}) or {}
        bonus = block.get("evidence_bonus_hit_rate", {}) or {}
        # In the frontier schema HONEST is null; in confusion.json it's omitted.
        # Either way DECEIVER is what we read here.
        deceiver_bonus = bonus.get("DECEIVER")
        notes_bits = []
        # Headline call-out: "0/67 evidence-bonus hits" or similar.
        if all(v in (None, 0.0) for v in bonus.values()):
            # Compute count of correct misbehavior preds for the denominator note.
            correct_misbehavior = 0
            confusion_obj = block.get("confusion") or {}
            for cls in ("REWARD_HACKER", "LAZY", "DECEIVER"):
                correct_misbehavior += (confusion_obj.get(cls) or {}).get(cls, 0)
            notes_bits.append(f"0/{correct_misbehavior} evidence-bonus hits" if correct_misbehavior else "0 evidence-bonus hits")
        # API failure callout if any.
        api_fail = block.get("api_failures", 0) or 0
        if api_fail:
            notes_bits.append(f"{api_fail} API fail(s)")
        out.append(_row(
            key=model,
            n=data.get("n_traces"),
            accuracy=block.get("accuracy"),
            mean_reward=block.get("mean_reward"),
            rhack_recall=per_class.get("REWARD_HACKER"),
            deceiver_bonus=deceiver_bonus,
            cost_usd=block.get("estimated_cost_usd"),
            notes="; ".join(notes_bits),
        ))
    return out


def rows_from_legacy_flat(results_dir: Path) -> list[dict]:
    """Best-effort fallback when confusion.json hasn't landed yet."""
    legacy_to_key = {
        "eval_base_full120.json":   "base",
        "eval_old_ckpt_n120.json":  "ckpt300",
        "eval_lora_lr2e5_400.json": "ckpt400",
    }
    out: list[dict] = []
    for fname, key in legacy_to_key.items():
        p = results_dir / fname
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text())
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] could not read {p}: {exc}", file=sys.stderr)
            continue
        block = data.get("held_out_v3") or {}
        out.append(_row(
            key=key,
            n=data.get("n_traces"),
            accuracy=block.get("accuracy"),
            mean_reward=block.get("mean_reward"),
            rhack_recall=None,
            deceiver_bonus=None,
            cost_usd=None,
            notes="per-class data pending dump_confusion.py",
        ))
    return out


# -----------------------------------------------------------------------------
# Renderer
# -----------------------------------------------------------------------------

def _sort(rows: list[dict], how: str) -> list[dict]:
    if how == "none":
        order = list(ROW_LABELS)
        rank = {k: i for i, k in enumerate(order)}
        return sorted(rows, key=lambda r: rank.get(r["key"], 999))
    if how in {"accuracy", "mean_reward", "deceiver_bonus", "rhack_recall"}:
        return sorted(rows, key=lambda r: r[how] if r[how] is not None else -1.0, reverse=True)
    raise ValueError(f"unknown sort key {how!r}")


def render_markdown(rows: list[dict]) -> str:
    headers = ["Model", "n", "Accuracy", "Mean Reward", "R-HACK Recall", "DECEIVER Bonus", "Cost", "Notes"]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        cells = [
            r["label"],
            str(r["n"]) if r["n"] is not None else "—",
            _fmt_pct(r["accuracy"]),
            _fmt_signed(r["mean_reward"]),
            _fmt_pct(r["rhack_recall"]),
            _fmt_pct(r["deceiver_bonus"]),
            _fmt_cost(r["cost_usd"]),
            r["notes"] or "",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_summary_block(rows: list[dict]) -> str:
    """Compact text summary for triage messages (per narrative-Claude's request)."""
    lines = []
    lines.append("=== leaderboard summary (full v3 held-out, n=120 unless noted) ===")
    lines.append(f"{'Model':<48} {'n':>4} {'Acc':>7} {'MR':>7} {'R-HACK':>7} {'DEC-bn':>7}  Notes")
    lines.append("-" * 110)
    for r in rows:
        lines.append(
            f"{r['label']:<48} "
            f"{(r['n'] or '—'):>4} "
            f"{_fmt_pct(r['accuracy']):>7} "
            f"{_fmt_signed(r['mean_reward']):>7} "
            f"{_fmt_pct(r['rhack_recall']):>7} "
            f"{_fmt_pct(r['deceiver_bonus']):>7}  "
            f"{r['notes']}"
        )
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--confusion", type=Path, default=None,
                    help="path to dump_confusion.py output (overrides legacy fallback)")
    ap.add_argument("--frontier-glob", type=str, default="results/eval_*_n120.json",
                    help="glob for API eval JSONs")
    ap.add_argument("--results-dir", type=Path, default=Path("results"),
                    help="directory holding eval JSONs")
    ap.add_argument("--sort", default="mean_reward",
                    choices=["none", "accuracy", "mean_reward", "deceiver_bonus", "rhack_recall"])
    ap.add_argument("--out-md", type=Path, default=None,
                    help="write Markdown table to this path (also still printed)")
    ap.add_argument("--out-json", type=Path, default=None,
                    help="write structured JSON summary to this path")
    args = ap.parse_args()

    rows: list[dict] = []

    # 1. Local checkpoints (preferred source = confusion.json; fallback = legacy flat).
    if args.confusion is not None and args.confusion.exists():
        confusion_data = json.loads(args.confusion.read_text())
        rows.extend(rows_from_confusion(confusion_data))
    else:
        rows.extend(rows_from_legacy_flat(args.results_dir))

    # 2. Frontier API rows (auto-discovery via glob).
    frontier_paths = sorted(Path(".").glob(args.frontier_glob))
    rows.extend(rows_from_frontier(frontier_paths))

    if not rows:
        print("[error] no eval data found. Run dump_confusion.py and/or eval_api_model.py first.", file=sys.stderr)
        return 1

    rows = _sort(rows, args.sort)

    # Markdown table (always to stdout; optional file).
    md = render_markdown(rows)
    print(md)
    print()
    print(render_summary_block(rows))

    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md + "\n")
        print(f"\n[wrote] {args.out_md}", file=sys.stderr)

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"[wrote] {args.out_json}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
