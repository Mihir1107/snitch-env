---
title: The Snitch
emoji: 🕵️
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
license: mit
---

---

# The Snitch

Scalable oversight environment for LLM agent fleets. The overseer audits tool-use traces from other agents and detects three distinct misbehavior patterns (reward hacking, laziness, deception). Held-out policy variants measure whether the overseer learns real inconsistency detection or just memorizes training patterns.

## Tasks

- **easy** — single-variant training distribution
- **medium** — mixed v1 + v2 variants, broader distribution
- **hard** — held-out v3 variants, tests generalization

## Endpoints

- `POST /reset` — start a new episode
- `POST /step` — submit overseer verdict
- `GET /state` — current episode metadata
- `GET /tasks` — list tasks and action schema
- `POST /grader` — score a completed episode
- `GET /baseline` — run random-verdict baseline
- `WS /ws` — primary WebSocket transport

EOF
