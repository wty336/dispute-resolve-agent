"""Centralized policy constants for the dispute resolution environment.

These values are intentionally shared by the runtime, reward engine, and
evaluation so business rules are not duplicated across modules.
"""

# Episode limits
MAX_ROUNDS = 5
MAX_INVESTIGATION_CALLS = 4
MAX_CONSECUTIVE_ILLEGAL_ACTIONS = 2

# Compensation guardrail: compensation cannot exceed this ratio of order amount.
MAX_COMPENSATION_RATIO = 1.2

# Reward constants
HARD_FAILURE_REWARD = -1.5
REWARD_MAX = 1.0
REWARD_MIN = -1.5
INVALID_ACTION_PENALTY = -0.2
REPEAT_CALL_PENALTY = -0.1
MAX_ACTION_PENALTY = -0.4

# Tool cost normalization denominator (sum of the four investigation tools).
MAX_TOOL_COST = 16.0

# Reward branch weights (kept here for auditable configuration).
DECIDE_WEIGHTS = {
    "liability": 0.45,
    "compensation": 0.25,
    "escalation": 0.15,
    "grounding": 0.15,
    "tool_cost": 0.10,
}
ESCALATE_WEIGHTS = {
    "escalation": 0.60,
    "grounding": 0.25,
    "escalation_quality": 0.15,
    "tool_cost": 0.10,
}
