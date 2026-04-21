"""Example Python client for The Snitch environment.

Demonstrates how to connect to a running Snitch env (local or HF Space) and
audit traces using an LLM. Can be used as a reference for training scripts.

Usage:
    python client.py --url http://localhost:7860 --task easy --n-episodes 5
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import requests


class SnitchClient:
    def __init__(self, base_url: str = "http://localhost:7860", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict:
        r = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def tasks(self) -> dict:
        r = requests.get(f"{self.base_url}/tasks", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def reset(self, task_id: str = "easy", seed: int = 42) -> dict:
        r = requests.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id, "seed": seed},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def step(self, action: str) -> dict:
        r = requests.post(
            f"{self.base_url}/step",
            json={"action": action},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def state(self) -> dict:
        r = requests.get(f"{self.base_url}/state", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def grader(self, episode_id: str, task_id: str) -> dict:
        r = requests.post(
            f"{self.base_url}/grader",
            json={"episode_id": episode_id, "task_id": task_id},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def baseline(self) -> dict:
        r = requests.get(f"{self.base_url}/baseline", timeout=60.0)
        r.raise_for_status()
        return r.json()


def dummy_verdict(observation: dict[str, Any]) -> str:
    """Placeholder overseer. Always says HONEST. Replace with real LLM call."""
    return (
        "VERDICT: HONEST\n"
        "CONFIDENCE: 0.5\n"
        "EVIDENCE: dummy placeholder — replace with real overseer."
    )


def run_episodes(client: SnitchClient, task_id: str, n: int, seed: int) -> list[dict]:
    results = []
    for i in range(n):
        reset_resp = client.reset(task_id=task_id, seed=seed + i)
        episode_id = reset_resp["episode_id"]
        obs = reset_resp["observation"]
        action = dummy_verdict(obs)
        step_resp = client.step(action)
        grade = client.grader(episode_id=episode_id, task_id=task_id)
        results.append({
            "episode": i,
            "episode_id": episode_id,
            "reward": step_resp["reward"],
            "info": step_resp["info"],
            "grade": grade,
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:7860")
    ap.add_argument("--task", default="easy", choices=["easy", "medium", "hard"])
    ap.add_argument("--n-episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    client = SnitchClient(base_url=args.url)
    try:
        health = client.health()
        print(f"Connected: {health}")
    except Exception as exc:
        print(f"Failed to reach {args.url}: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\nRunning {args.n_episodes} episodes on task='{args.task}'...")
    results = run_episodes(client, args.task, args.n_episodes, args.seed)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()