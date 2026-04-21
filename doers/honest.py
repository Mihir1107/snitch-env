"""Honest doer — reads relevant docs and returns the ground truth. 3 variants."""
from __future__ import annotations

from typing import Any
import random

from env.trace_format import Trace
from doers.base import Policy, ToolEnv, new_trace, record


class HonestPolicy:
    name = "honest"
    policy_class = "HONEST"

    def __init__(self, variant: int, seed: int | None = None) -> None:
        assert variant in (1, 2, 3)
        self.variant = variant
        self._rng = random.Random(seed)

    def run(self, question: dict[str, Any], tools: ToolEnv) -> Trace:
        t = new_trace(question, self)
        relevant = question["relevant_doc_ids"]

        # All variants search first. Phrasing varied per variant.
        if self.variant == 1:
            query = question["question"]
        elif self.variant == 2:
            query = f"{question['question']}"  # same but will differ if we vary lazy differently
        else:
            query = question["question"]
        hits = tools.search(query)
        record(t, "search", {"query": query}, hits)

        if self.variant == 1:
            # Read all relevant docs in given order.
            docs_to_read = list(relevant)
        elif self.variant == 2:
            # Read only those relevant docs that appeared in search hits; fallback to all relevant.
            found_in_hits = [d for d in hits if d in relevant]
            docs_to_read = found_in_hits if found_in_hits else list(relevant)
        else:
            # Variant 3: read first relevant, summarize it, then read the rest.
            docs_to_read = list(relevant)

        for idx, doc_id in enumerate(docs_to_read):
            content = tools.read_document(doc_id)
            record(t, "read_document", {"doc_id": doc_id}, content)
            if self.variant == 3 and idx == 0 and len(content) > 120:
                s = tools.summarize(content)
                record(t, "summarize", {"text": content}, s)

        answer = question["ground_truth_answer"]
        record(t, "submit_answer", {"answer": answer}, None)
        t.claimed_answer = answer
        # HONEST: no smoking gun
        return t