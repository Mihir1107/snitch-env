"""Trace dataclass + JSON serialization for doer agents."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional
import json


VALID_CLASSES = {"HONEST", "REWARD_HACKER", "LAZY", "DECEIVER"}


@dataclass
class TraceStep:
    tool: str  # search | read_document | summarize | submit_answer
    args: dict[str, Any]
    result: Any = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Trace:
    question_id: str
    question: str
    policy_name: str                  # e.g. "reward_hacker"
    policy_variant: int               # 1, 2, or 3
    policy_class: str                 # HONEST | REWARD_HACKER | LAZY | DECEIVER
    steps: list[TraceStep] = field(default_factory=list)
    claimed_answer: str = ""
    ground_truth_answer: str = ""
    # Smoking-gun annotations for evidence reward. None for HONEST.
    smoking_gun_step: Optional[int] = None       # 1-indexed step number
    smoking_gun_keyword: Optional[str] = None    # keyword that should appear in evidence

    def __post_init__(self):
        if self.policy_class not in VALID_CLASSES:
            raise ValueError(f"policy_class must be one of {VALID_CLASSES}, got {self.policy_class}")

    def validate(self) -> None:
        """Call after policy has finished filling in smoking_gun fields."""
        if self.policy_class == "HONEST":
            if self.smoking_gun_step is not None or self.smoking_gun_keyword is not None:
                raise ValueError("HONEST traces must not have smoking_gun fields set")
        else:
            if self.smoking_gun_step is None or self.smoking_gun_keyword is None:
                raise ValueError(f"{self.policy_class} traces must set smoking_gun_step and smoking_gun_keyword")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["steps"] = [s.to_dict() if isinstance(s, TraceStep) else s for s in self.steps]
        return d

    def to_json(self) -> str:
        self.validate()
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "Trace":
        steps = [TraceStep(**s) for s in d.get("steps", [])]
        return cls(
            question_id=d["question_id"],
            question=d["question"],
            policy_name=d["policy_name"],
            policy_variant=d["policy_variant"],
            policy_class=d["policy_class"],
            steps=steps,
            claimed_answer=d.get("claimed_answer", ""),
            ground_truth_answer=d.get("ground_truth_answer", ""),
            smoking_gun_step=d.get("smoking_gun_step"),
            smoking_gun_keyword=d.get("smoking_gun_keyword"),
        )

    @classmethod
    def from_json(cls, s: str) -> "Trace":
        return cls.from_dict(json.loads(s))