"""Public and hidden domain schemas.

The public observation and the hidden ground truth intentionally share only
``case_id``.  Any new field added to ``DisputeObservation`` must not leak a
hidden ground-truth attribute; the leakage test in
``tests/leakage/test_hidden_state.py`` enforces this.
"""
from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class Liability(str, Enum):
    """Liability outcome used by terminal decisions and ground truth."""

    MERCHANT = "merchant"
    BUYER = "buyer"
    SPLIT = "split"
    NONE = "none"


class Evidence(BaseModel):
    """A piece of evidence visible to the agent."""

    evidence_id: str
    type: str
    description: str
    source: Literal["buyer", "merchant", "logistics", "platform"]
    visible: bool = True


class ToolResult(BaseModel):
    """Result of one simulated tool call.

    ``payload`` is deliberately a plain string so tools never serialize hidden
    ground-truth fields into agent-visible state.
    """

    tool_name: str
    arguments: dict[str, str]
    payload: str
    cached: bool = False
    cost: float = 0.0
    success: bool = True


class DisputeObservation(BaseModel):
    """Everything an agent is allowed to see for one dispute."""

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
    """Hidden facts used only by data generation, simulators, reward, and eval.

    These fields must never appear in prompts, tool arguments, tool payloads,
    or serialized episode state.
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
