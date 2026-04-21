"""Policy protocol and simulated tool environment for doer agents."""
from __future__ import annotations

from typing import Protocol, Any
import random

from env.trace_format import Trace, TraceStep


CLASS_MAP = {
    "honest": "HONEST",
    "reward_hacker": "REWARD_HACKER",
    "lazy": "LAZY",
    "deceiver": "DECEIVER",
}


class ToolEnv:
    """Simulated toolset: search, read_document, summarize.

    Constructed per-question from the question's corpus + relevant_doc_ids.
    """

    def __init__(self, corpus: dict[str, str], relevant_doc_ids: list[str], rng: random.Random | None = None) -> None:
        self.corpus = corpus
        self.relevant_doc_ids = relevant_doc_ids
        self.rng = rng or random.Random()

    def search(self, query: str) -> list[str]:
        """Returns up to 5 doc IDs. Relevant docs mixed with distractors in varied positions."""
        relevant = list(self.relevant_doc_ids)
        irrelevant = [d for d in self.corpus if d not in relevant]
        # Always include at least one relevant doc if any exist, but position varies.
        self.rng.shuffle(relevant)
        self.rng.shuffle(irrelevant)
        combined = relevant[:2] + irrelevant[:3]
        self.rng.shuffle(combined)
        return combined[:5]

    def read_document(self, doc_id: str) -> str:
        return self.corpus.get(doc_id, "")

    def summarize(self, text: str) -> str:
        if len(text) <= 120:
            return text
        return text[:117] + "..."


class Policy(Protocol):
    name: str
    variant: int
    policy_class: str  # HONEST | REWARD_HACKER | LAZY | DECEIVER

    def run(self, question: dict[str, Any], tools: ToolEnv) -> Trace: ...


def new_trace(question: dict[str, Any], policy: Policy) -> Trace:
    """Build an empty Trace. Policies fill in steps + smoking_gun after running."""
    return Trace(
        question_id=question["id"],
        question=question["question"],
        policy_name=policy.name,
        policy_variant=policy.variant,
        policy_class=policy.policy_class,
        steps=[],
        claimed_answer="",
        ground_truth_answer=question.get("ground_truth_answer", ""),
        # smoking_gun intentionally left None — policies set these before returning (except HONEST)
    )


def record(trace: Trace, tool: str, args: dict[str, Any], result: Any) -> None:
    trace.steps.append(TraceStep(tool=tool, args=args, result=result))