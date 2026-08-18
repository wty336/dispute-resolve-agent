"""纠纷判责 Agent 的领域契约。

本包包含公开观测、隐藏真值、终局决策 schema 和集中式策略常量。真值字段
绝不会被序列化到公开观测或 Agent 消息中。
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
