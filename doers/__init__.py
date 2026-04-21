from doers.base import Policy, ToolEnv
from doers.honest import HonestPolicy
from doers.reward_hacker import RewardHackerPolicy
from doers.lazy import LazyPolicy
from doers.deceiver import DeceiverPolicy

ALL_POLICIES = {
    "honest": HonestPolicy,
    "reward_hacker": RewardHackerPolicy,
    "lazy": LazyPolicy,
    "deceiver": DeceiverPolicy,
}

__all__ = [
    "Policy",
    "ToolEnv",
    "HonestPolicy",
    "RewardHackerPolicy",
    "LazyPolicy",
    "DeceiverPolicy",
    "ALL_POLICIES",
]
