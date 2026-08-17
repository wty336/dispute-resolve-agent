"""Domain contracts for the dispute resolution agent.

This package contains the public observation, hidden ground truth, terminal
decision schema, and centralized policy constants.  Ground truth fields are
never serialized into the public observation or agent messages.
"""
from .policies import (
    HARD_FAILURE_REWARD,
    MAX_CONSECUTIVE_ILLEGAL_ACTIONS,
    MAX_COMPENSATION_RATIO,
    MAX_INVESTIGATION_CALLS,
    MAX_ROUNDS,
    REPEAT_CALL_PENALTY,
    INVALID_ACTION_PENALTY,
    REWARD_MAX,
    REWARD_MIN,
)
from .schemas import (
    Decision,
    DisputeGroundTruth,
    DisputeObservation,
    Escalation,
    Evidence,
    Liability,
    TerminalDecision,
    ToolResult,
)

__all__ = [
    "Decision",
    "DisputeGroundTruth",
    "DisputeObservation",
    "Escalation",
    "Evidence",
    "Liability",
    "TerminalDecision",
    "ToolResult",
    "HARD_FAILURE_REWARD",
    "INVALID_ACTION_PENALTY",
    "MAX_CONSECUTIVE_ILLEGAL_ACTIONS",
    "MAX_COMPENSATION_RATIO",
    "MAX_INVESTIGATION_CALLS",
    "MAX_ROUNDS",
    "REPEAT_CALL_PENALTY",
    "REWARD_MAX",
    "REWARD_MIN",
]
