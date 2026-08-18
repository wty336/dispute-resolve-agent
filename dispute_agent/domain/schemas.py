"""公开与隐藏领域 schema。

公开观测与隐藏真值有意只共享 ``case_id``。向 ``DisputeObservation`` 添加新字段
时不得泄漏隐藏真值属性；``tests/leakage/test_hidden_state.py`` 中的泄漏测试
会强制检查这一点。
"""
from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class Liability(str, Enum):
    """终局决策和真值使用的责任结果。"""

    MERCHANT = "merchant"
    BUYER = "buyer"
    SPLIT = "split"
    NONE = "none"


class Evidence(BaseModel):
    """Agent 可见的一条证据。"""

    evidence_id: str
    type: str
    description: str
    source: Literal["buyer", "merchant", "logistics", "platform"]
    visible: bool = True


class ToolResult(BaseModel):
    """一次模拟工具调用的结果。

    ``payload`` 特意保持为普通字符串，确保工具不会把隐藏真值字段序列化到
    Agent 可见状态中。
    """

    tool_name: str
    arguments: dict[str, str]
    payload: str
    cached: bool = False
    cost: float = 0.0
    success: bool = True
    error_type: Literal["timeout", "missing", "backend_error"] | None = None


class DisputeObservation(BaseModel):
    """一个纠纷中 Agent 被允许看到的全部信息。"""

    case_id: str
    order_id: str
    buyer_id: str
    merchant_id: str
    item_name: str
    order_amount: float = Field(ge=0)
    claim_type: str
    buyer_claim: str
    buyer_requested_amount: float = Field(ge=0)
    merchant_response: str
    chat_log: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)


class DisputeGroundTruth(BaseModel):
    """仅供数据生成、模拟器、奖励和评估使用的隐藏事实。

    这些字段绝不能出现在 prompt、工具参数、工具 payload 或序列化的 episode
    状态中。
    """

    case_id: str
    true_liability: Liability
    true_loss: float = Field(ge=0)
    reasonable_compensation_range: tuple[float, float]
    buyer_strategy: str
    merchant_strategy: str
    should_escalate: bool
    tool_information_value: dict[str, float] = Field(default_factory=dict)
    risk_level: Literal["low", "medium", "high"] = "medium"
    logistics_state: Literal[
        "delivered_verified", "delivered_disputed", "lost", "damaged_in_transit"
    ] = "delivered_verified"
    evidence_authenticity: dict[str, Literal["authentic", "suspicious", "forged"]] = Field(
        default_factory=dict
    )
    tool_noise_rate: float = Field(default=0.08, ge=0, le=1)
    tool_timeout_rate: float = Field(default=0.02, ge=0, le=1)
    tool_missing_rate: float = Field(default=0.03, ge=0, le=1)


class Decision(BaseModel):
    action: Literal["decide"]
    liability: Liability
    compensation: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[str] = Field(min_length=1)
    reason: str = Field(min_length=1, max_length=500)


class Escalation(BaseModel):
    action: Literal["escalate"]
    liability: None = None
    compensation: None = None
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[str] = Field(min_length=1)
    reason: str = Field(min_length=1, max_length=500)


TerminalDecision = Annotated[Decision | Escalation, Field(discriminator="action")]
