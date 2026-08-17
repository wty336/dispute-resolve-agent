"""强化学习奖励函数。

奖励设计：
    reward = 判责匹配度 + 长期收益 / reward_scale

其中长期收益来自 `payoff.compute_outcome`，它已综合了买家留存价值、
商家留存价值、赔付成本、风险成本与口碑。这样 RL 优化的正是
“长期收益最大化”，同时加入判责匹配度作为密集的公平性信号。

模型输出解析失败时返回 `fail_reward`（默认 -1.0）。
"""
from __future__ import annotations

import re

from .models import AgentDecision, DisputeCase, Liability
from .payoff import PayoffConfig, compute_outcome, liability_match_score
from .prompting import parse_agent_response

LIABILITY_MAP = {
    "商家责任": Liability.MERCHANT,
    "买家责任": Liability.BUYER,
    "双方共担": Liability.SPLIT,
    "无法认定": Liability.NONE,
}

_CASE_ID_PATTERN = re.compile(r"案例编号[:：]\s*(C\d{6})")


def text_to_decision(text: str) -> AgentDecision | None:
    """把模型输出文本解析为 AgentDecision；解析失败返回 None。"""
    data = parse_agent_response(text)
    if not data:
        return None
    try:
        liability = LIABILITY_MAP.get(data.get("liability", ""), Liability.NONE)
        comp = float(data.get("compensation", 0.0))
        escalate = bool(data.get("escalate", False))
        reason = str(data.get("reason", ""))[:200]
        return AgentDecision(
            liability=liability,
            compensation=max(0.0, comp),
            escalate=escalate,
            reason=reason,
        )
    except (TypeError, ValueError):
        return None


def compute_reward(
    case: DisputeCase,
    decision_or_text: AgentDecision | str,
    config: PayoffConfig | None = None,
    reward_scale: float = 1000.0,
    fail_reward: float = -1.0,
    tool_cost: float = 0.0,
) -> float:
    """计算单条决策的 RL 奖励。

    Args:
        case: 纠纷案例（含 ground truth）。
        decision_or_text: AgentDecision 或模型输出 JSON 文本。
        config: 收益模型参数。
        reward_scale: 长期收益的缩放系数。
        fail_reward: 输出解析失败时的奖励。
        tool_cost: 该轨迹已产生的工具调用成本（RL 阶段从 extra_info 读入）。
    """
    if isinstance(decision_or_text, str):
        decision = text_to_decision(decision_or_text)
        if decision is None:
            return fail_reward
    else:
        decision = decision_or_text

    outcome = compute_outcome(case, decision, config)
    accuracy = liability_match_score(decision, case)
    return accuracy + (outcome.long_term_value - tool_cost) / reward_scale


class RewardEngine:
    """供 RL 训练框架调用的批式奖励引擎。

    训练数据中每条 prompt 需要包含 `案例编号：Cxxxxxx`，
    RewardEngine 会按 case_id 查表并计算奖励。

    用法示例（TRL GRPO）:
        engine = RewardEngine.from_cases(cases)
        reward_func = engine.batch_reward  # 签名为 (prompts, completions, **kwargs)
    """

    def __init__(self, case_store: dict[str, DisputeCase], config: PayoffConfig | None = None) -> None:
        self.case_store = case_store
        self.config = config or PayoffConfig()

    @classmethod
    def from_cases(cls, cases: list[DisputeCase], config: PayoffConfig | None = None) -> "RewardEngine":
        return cls({c.case_id: c for c in cases}, config=config)

    def extract_case_id(self, prompt: str) -> str | None:
        m = _CASE_ID_PATTERN.search(prompt)
        return m.group(1) if m else None

    def single_reward(self, prompt: str, completion: str) -> float:
        case_id = self.extract_case_id(prompt)
        if case_id is None or case_id not in self.case_store:
            return -1.0
        return compute_reward(self.case_store[case_id], completion, config=self.config)

    def batch_reward(self, prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        """批式奖励，适配 TRL GRPOTrainer 的 reward 函数签名。"""
        return [self.single_reward(p, c) for p, c in zip(prompts, completions)]
