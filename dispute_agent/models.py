"""数据模型定义。

所有数据类集中在这里，方便后续扩展为数据库表或 LLM prompt 结构。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Party(str, Enum):
    """纠纷参与方。"""

    BUYER = "买家"
    MERCHANT = "商家"


class ClaimType(str, Enum):
    """投诉类型。"""

    QUALITY_ISSUE = "质量问题"
    NOT_AS_DESCRIBED = "描述不符"
    DAMAGED = "破损/少件"
    NOT_RECEIVED = "未收到货"
    AFTERSALES = "售后纠纷"


class Liability(str, Enum):
    """责任判定结果。"""

    MERCHANT = "商家责任"
    BUYER = "买家责任"
    SPLIT = "双方共担"
    NONE = "无法认定"


class BuyerStrategy(str, Enum):
    """买家博弈策略（仿真 ground truth）。"""

    HONEST = "诚实"
    EXAGGERATE = "夸大损失"
    FRAUD = "虚假投诉"


class MerchantStrategy(str, Enum):
    """商家博弈策略（仿真 ground truth）。"""

    HONEST = "诚实"
    EVASIVE = "推卸责任"
    DENY = "虚假否认"


@dataclass
class ChatMessage:
    """聊天记录中的一条消息。"""

    role: str  # "buyer" / "merchant" / "platform"
    content: str


@dataclass
class Evidence:
    """举证材料。"""

    type: str  # 图片/视频/物流凭证/质检报告等
    description: str
    party: Party
    strength: float = 0.5  # 0~1，证明力（仿真中用于生成，真实 Agent 不可直接读取）
    verified: bool = False  # 平台是否已核实（仿真中用 ground truth 推断）


@dataclass
class DisputeCase:
    """一单纠纷的完整输入。

    注意：`true_*` 与 `*_strategy` 字段是仿真的 ground truth，
    只允许环境/评估器读取；平台 Agent 决策时不应读取这些字段。
    """

    case_id: str
    order_id: str
    buyer_id: str
    merchant_id: str
    item_name: str
    order_amount: float
    claim_type: ClaimType
    buyer_claim: str
    buyer_requested_amount: float
    merchant_response: str
    chat_log: list[ChatMessage] = field(default_factory=list)
    buyer_evidence: list[Evidence] = field(default_factory=list)
    merchant_evidence: list[Evidence] = field(default_factory=list)

    # ---- 仿真 ground truth（平台 Agent 不可见） ----
    true_liability: Liability = Liability.NONE
    true_buyer_loss: float = 0.0
    buyer_strategy: BuyerStrategy = BuyerStrategy.HONEST
    merchant_strategy: MerchantStrategy = MerchantStrategy.HONEST


@dataclass
class AgentDecision:
    """平台 Agent 的决策输出。"""

    liability: Liability
    compensation: float  # 赔付金额（元）
    escalate: bool  # 是否人工升级
    reason: str  # 判定依据
    tool_cost: float = 0.0  # 多步工具调用累计成本（元），单步 Agent 为 0

    def __post_init__(self) -> None:
        self.compensation = max(0.0, float(self.compensation))
        self.tool_cost = max(0.0, float(self.tool_cost))


@dataclass
class CaseOutcome:
    """单轮决策后的收益核算结果。"""

    buyer_satisfaction: float
    merchant_satisfaction: float
    buyer_repurchase_prob: float
    merchant_retention_prob: float
    immediate_cost: float
    risk_cost: float
    reputation_gain: float
    long_term_value: float
    notes: list[str] = field(default_factory=list)
