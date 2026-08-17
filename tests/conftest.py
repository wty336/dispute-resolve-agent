"""Shared fixtures for unit and integration tests."""
import pytest

from dispute_agent.domain.schemas import (
    Decision,
    DisputeGroundTruth,
    DisputeObservation,
    Evidence,
)
from dispute_agent.environment import EpisodeState
from dispute_agent.tools import ToolRegistry


def _make_episode(
    case_id: str = "case-1",
    group_seed: int = 42,
    order_id: str = "o-1",
    buyer_id: str = "b-1",
    merchant_id: str = "m-1",
) -> EpisodeState:
    observation = DisputeObservation(
        case_id=case_id,
        order_id=order_id,
        buyer_id=buyer_id,
        merchant_id=merchant_id,
        item_name="测试商品",
        order_amount=100.0,
        claim_type="damaged",
        buyer_claim="收到商品时已破损，要求赔偿 89 元。",
        buyer_requested_amount=89.0,
        merchant_response="发货前已检查完好，可能是物流导致。",
        chat_log=["买家：商品破损。", "商家：发货前完好。"],
        evidence=[
            Evidence(
                evidence_id="chat:1",
                type="聊天记录",
                description="买家反馈破损",
                source="buyer",
                visible=True,
            )
        ],
    )
    ground_truth = DisputeGroundTruth(
        case_id=case_id,
        true_liability="merchant",
        true_loss=50.0,
        reasonable_compensation_range=(40.0, 60.0),
        buyer_strategy="honest",
        merchant_strategy="honest",
        should_escalate=False,
        tool_information_value={},
        risk_level="low",
    )
    registry = ToolRegistry(case_id=case_id, case_seed=group_seed)
    return EpisodeState(
        observation=observation,
        ground_truth=ground_truth,
        case_seed=group_seed,
        tool_registry=registry,
    )


@pytest.fixture
def make_episode():
    return _make_episode


@pytest.fixture
def episode() -> EpisodeState:
    return _make_episode()


@pytest.fixture
def tool_registry() -> ToolRegistry:
    return _make_episode().tool_registry


@pytest.fixture
def valid_decision() -> Decision:
    return Decision(
        action="decide",
        liability="merchant",
        compensation=50.0,
        confidence=0.9,
        evidence_ids=["chat:1", "logistics:o-1"],
        reason="物流异常且证据支持商家责任",
    )
