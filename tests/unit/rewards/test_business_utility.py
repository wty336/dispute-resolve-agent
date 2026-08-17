from dataclasses import dataclass

import pytest

from dispute_agent.domain.schemas import Decision, Escalation, Liability
from dispute_agent.rewards.business_utility import (
    BusinessUtilityConfig,
    score_business_utility,
)


@dataclass
class Case:
    order_amount: float = 100.0
    true_liability: Liability = Liability.MERCHANT
    reasonable_compensation_range: tuple[float, float] = (40.0, 60.0)
    true_loss: float = 50.0
    cumulative_cost: float = 0.0


@pytest.fixture
def config() -> BusinessUtilityConfig:
    return BusinessUtilityConfig()


@pytest.fixture
def case() -> Case:
    return Case()


def make_escalation(_case):
    return Escalation(
        action="escalate",
        confidence=0.8,
        evidence_ids=["chat:1"],
        reason="证据冲突",
    )


def make_oracle_decision(_case):
    low, high = _case.reasonable_compensation_range
    compensation = max(low, min(high, _case.true_loss))
    return Decision(
        action="decide",
        liability=_case.true_liability,
        compensation=compensation,
        confidence=0.99,
        evidence_ids=["chat:1"],
        reason="oracle",
    )


def test_escalation_uses_oracle_outcome_and_charges_review_and_delay(config, case):
    utility = score_business_utility(case, make_escalation(case), config)
    direct_oracle = score_business_utility(case, make_oracle_decision(case), config)
    assert utility.raw == pytest.approx(
        direct_oracle.raw - config.manual_review_cost - config.review_delay_cost
    )
