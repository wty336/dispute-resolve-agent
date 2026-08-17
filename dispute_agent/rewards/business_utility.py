"""Offline business utility model.

This module is intentionally separate from the training reward.  It computes a
fixed, auditable long-term value formula for evaluation only and is never used
as a training reward.
"""
from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from dispute_agent.domain.schemas import Decision, Escalation, Liability


class BusinessUtilityConfig(BaseModel):
    buyer_ltv: float = Field(default=1000.0, ge=0)
    merchant_ltv: float = Field(default=2000.0, ge=0)
    buyer_repurchase_satisfied: float = Field(default=0.85, ge=0, le=1)
    buyer_repurchase_unsatisfied: float = Field(default=0.45, ge=0, le=1)
    merchant_retention_satisfied: float = Field(default=0.90, ge=0, le=1)
    merchant_retention_unsatisfied: float = Field(default=0.60, ge=0, le=1)
    manual_review_cost: float = Field(default=30.0, ge=0)
    review_delay_cost: float = Field(default=10.0, ge=0)
    risk_cost: float = Field(default=20.0, ge=0)
    reputation_gain: float = Field(default=10.0, ge=0)


@dataclass
class BusinessUtility:
    raw: float
    normalized: float = 0.0


def score_business_utility(
    case,
    decision: Decision | Escalation,
    config: BusinessUtilityConfig,
) -> BusinessUtility:
    """Score a terminal decision with the offline business utility formula."""
    if decision.action == "escalate":
        oracle_raw = _oracle_raw(case, config)
        raw = oracle_raw - config.manual_review_cost - config.review_delay_cost
        return BusinessUtility(raw=raw)

    assert isinstance(decision, Decision)
    raw = _decide_raw(case, decision, config)
    return BusinessUtility(raw=raw)


def _get_case_attr(case, name: str, default=None):
    if hasattr(case, name):
        return getattr(case, name)
    if hasattr(case, "ground_truth") and hasattr(case.ground_truth, name):
        return getattr(case.ground_truth, name)
    if hasattr(case, "observation") and hasattr(case.observation, name):
        return getattr(case.observation, name)
    return default


def _oracle_raw(case, config: BusinessUtilityConfig) -> float:
    true_liability = _get_case_attr(case, "true_liability", Liability.NONE)
    reasonable_range = _get_case_attr(case, "reasonable_compensation_range", (0.0, 0.0))
    low, high = reasonable_range
    true_loss = _get_case_attr(case, "true_loss", low)
    compensation = max(low, min(high, true_loss))
    order_amount = _get_case_attr(case, "order_amount", compensation)
    tool_cost = _get_case_attr(case, "cumulative_cost", 0.0) or 0.0

    buyer_prob = config.buyer_repurchase_satisfied if true_liability in (
        Liability.MERCHANT,
        Liability.SPLIT,
    ) and compensation >= low else config.buyer_repurchase_unsatisfied
    merchant_prob = config.merchant_retention_satisfied if true_liability in (
        Liability.BUYER,
        Liability.NONE,
    ) or compensation <= high else config.merchant_retention_unsatisfied

    return (
        config.buyer_ltv * buyer_prob
        + config.merchant_ltv * merchant_prob
        - compensation
        - tool_cost
        - config.risk_cost
        + config.reputation_gain
    )


def _decide_raw(case, decision: Decision, config: BusinessUtilityConfig) -> float:
    true_liability = _get_case_attr(case, "true_liability", Liability.NONE)
    reasonable_range = _get_case_attr(case, "reasonable_compensation_range", (0.0, 0.0))
    low, high = reasonable_range
    compensation = decision.compensation
    order_amount = _get_case_attr(case, "order_amount", compensation)
    tool_cost = _get_case_attr(case, "cumulative_cost", 0.0) or 0.0

    buyer_prob = config.buyer_repurchase_satisfied if true_liability in (
        Liability.MERCHANT,
        Liability.SPLIT,
    ) and compensation >= low else config.buyer_repurchase_unsatisfied
    merchant_prob = config.merchant_retention_satisfied if true_liability in (
        Liability.BUYER,
        Liability.NONE,
    ) or compensation <= high else config.merchant_retention_unsatisfied

    return (
        config.buyer_ltv * buyer_prob
        + config.merchant_ltv * merchant_prob
        - compensation
        - tool_cost
        - config.risk_cost
        + config.reputation_gain
    )
