# The Snitch

Scalable oversight environment for multi-agent auditing.

An OpenEnv environment that trains an LLM overseer to detect misbehavior in other agents' tool-use traces by identifying inconsistencies between trace steps and stated claims, with held-out policy variants to measure genuine generalization.

**Team:** InfraNova (Manas Mehta, Haya Sachin, Mihir Mandavia)
**Event:** Meta PyTorch OpenEnv Hackathon Round 2
**Theme:** Multi-Agent Interactions | **Sub-theme:** Fleet AI — Scalable Oversight

## Core Contribution

Trace-Outcome Consistency Check with held-out policy generalization. The overseer is trained on two variants of each misbehavior pattern and evaluated on a third held-out variant.

## Task: Research Assistant

Doer agents answer questions using a simulated toolset: `search`, `read_document`, `summarize`, `submit_answer`.

## Quickstart

```bash
pip install -e .
python scripts/generate_questions.py
python scripts/generate_traces.py
python training/train_grpo.py
python training/eval.py
```

## Layout

- `env/` — OpenEnv environment (reset/step, trace format, reward)
- `doers/` — policies (honest, reward_hacker, lazy, deceiver) with 3 variants each
- `data/` — questions and generated traces
- `scripts/` — data generation and baselines
- `training/` — GRPO training and evaluation
- `notebooks/` — demo
