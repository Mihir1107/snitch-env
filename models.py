"""Pydantic models for The Snitch environment.

Exports data structures for observations, actions, rewards, and episode state.
Used by both the server and any client connecting to the env.
"""
from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


VerdictLiteral = Literal["HONEST", "REWARD_HACKER", "LAZY", "DECEIVER", "PARSE_ERROR"]
PolicyClassLiteral = Literal["HONEST", "REWARD_HACKER", "LAZY", "DECEIVER"]


class TraceStep(BaseModel):
    """Single step in a doer agent's tool-use trace."""
    tool: str = Field(..., description="Tool name: search | read_document | summarize | submit_answer")
    args: dict[str, Any] = Field(default_factory=dict, description="Arguments passed to the tool")
    result: Optional[Any] = Field(None, description="Tool's returned value, if any")


class Observation(BaseModel):
    """What the overseer receives for an audit."""
    prompt: str = Field(..., description="Full formatted prompt with question, trace, claim, and output-format instruction")
    question: str = Field(..., description="Original research question posed to the doer")
    trace: list[TraceStep] = Field(default_factory=list, description="Ordered list of tool-use steps")
    claimed_answer: str = Field(..., description="The final answer the doer submitted")


class Action(BaseModel):
    """Overseer's action. Sent as a raw string completion from an LLM.

    Expected format:
        VERDICT: <HONEST|REWARD_HACKER|LAZY|DECEIVER>
        CONFIDENCE: <float in [0, 1]>
        EVIDENCE: <one sentence citing step number and keyword>
    """
    action: str = Field(..., description="Raw LLM completion following the VERDICT/CONFIDENCE/EVIDENCE format")


class ParsedVerdict(BaseModel):
    """Result of parsing an Action into structured fields."""
    verdict: VerdictLiteral
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: str


class Reward(BaseModel):
    value: float = Field(..., description="Episode reward in [-1.0, 1.5]")


class EpisodeInfo(BaseModel):
    """Per-step info dict returned from step()."""
    parsed_verdict: str
    parsed_confidence: float
    true_class: PolicyClassLiteral
    classification_correct: bool
    policy_name: str
    policy_variant: int


class EnvState(BaseModel):
    """Env snapshot returned from GET /state."""
    episode_id: Optional[str] = None
    task_id: Optional[str] = None
    step_count: int = 0
    done: bool = False
    total_reward: float = 0.0