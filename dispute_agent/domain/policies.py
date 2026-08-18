"""纠纷判责环境的集中式策略常量。

这些值由运行时、奖励引擎和评估共同使用，避免业务规则在多个模块重复定义。
"""

# Episode 限制
MAX_ROUNDS = 5
MAX_INVESTIGATION_CALLS = 4
MAX_CONSECUTIVE_ILLEGAL_ACTIONS = 2

# 赔付护栏：赔付金额不能超过订单金额的该比例。
MAX_COMPENSATION_RATIO = 1.2

# 奖励常量
HARD_FAILURE_REWARD = -1.5
REWARD_MAX = 1.0
REWARD_MIN = -1.5
INVALID_ACTION_PENALTY = -0.2
REPEAT_CALL_PENALTY = -0.1
MAX_ACTION_PENALTY = -0.4

# 工具成本归一化分母（四个调查工具成本之和）。
MAX_TOOL_COST = 16.0

# 奖励分支权重（集中保存在这里，便于审计配置）。
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
