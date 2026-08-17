import pytest
from dispute_agent.domain.schemas import (
    Decision,
    DisputeGroundTruth,
    DisputeObservation,
    Escalation,
    Evidence,
)
from dispute_agent.environment import EpisodeState
from dispute_agent.rewards.engine import RewardEngine
from dispute_agent.tools import ToolRegistry


def _make_episode(
    decision,
    *,
    should_escalate=False,
    true_liability="merchant",
    compensation_range=(40.0, 60.0),
    risk_level="low",
    tool_plan=(),
    repeat_calls=0,
    illegal_calls=0,
):
    observation = DisputeObservation(
        case_id="reward-case",
        order_id="o-1",
        buyer_id="b-1",
        merchant_id="m-1",
        item_name="测试商品",
        order_amount=100.0,
        claim_type="damaged",
        buyer_claim="商品破损",
        buyer_requested_amount=89.0,
        merchant_response="发货前完好",
        chat_log=["买家：破损"],
        evidence=[Evidence(evidence_id="chat:1", type="聊天记录", description="买家反馈", source="buyer", visible=True)],
    )
    ground_truth = DisputeGroundTruth(
        case_id="reward-case",
        true_liability=true_liability,
        true_loss=50.0,
        reasonable_compensation_range=compensation_range,
        buyer_strategy="honest",
        merchant_strategy="honest",
        should_escalate=should_escalate,
        tool_information_value={},
        risk_level=risk_level,
    )
    episode = EpisodeState(
        observation=observation,
        ground_truth=ground_truth,
        case_seed=42,
        tool_registry=ToolRegistry("reward-case", 42),
    )
    for tool_name, args in tool_plan:
        episode.call(tool_name, args)
    episode.invalid_actions.repeat_calls = repeat_calls
    episode.invalid_actions.illegal_calls = illegal_calls
    if decision is not None:
        episode.submit(decision)
    return episode


@pytest.fixture
def decide_episode():
    decision = Decision(
        action="decide",
        liability="merchant",
        compensation=50.0,
        confidence=0.9,
        evidence_ids=["chat:1"],
        reason="证据支持商家责任",
    )
    return _make_episode(decision, tool_plan=[("check_logistics", {"order_id": "o-1"})])


@pytest.fixture
def unfinished_episode():
    return _make_episode(None)


@pytest.fixture
def reward_fixture():
    def build(name):
        if name == "correct_efficient":
            return _make_episode(
                Decision(action="decide", liability="merchant", compensation=50.0, confidence=0.9,
                         evidence_ids=["chat:1"], reason="证据充分"),
                tool_plan=[],
            )
        if name == "correct_redundant":
            return _make_episode(
                Decision(action="decide", liability="merchant", compensation=50.0, confidence=0.9,
                         evidence_ids=["chat:1", "logistics:o-1", "buyer_history:b-1", "merchant_history:m-1"],
                         reason="证据充分"),
                tool_plan=[
                    ("check_logistics", {"order_id": "o-1"}),
                    ("check_buyer_history", {"buyer_id": "b-1"}),
                    ("check_merchant_history", {"merchant_id": "m-1"}),
                ],
            )
        if name == "correct_overpay":
            return _make_episode(
                Decision(action="decide", liability="merchant", compensation=85.0, confidence=0.9,
                         evidence_ids=["chat:1"], reason="证据充分"),
                tool_plan=[],
            )
        if name == "reasonable_escalation":
            return _make_episode(
                Escalation(action="escalate", confidence=0.8, evidence_ids=["chat:1"],
                           reason="证据冲突且风险较高"),
                should_escalate=True,
                risk_level="high",
                tool_plan=[
                    ("check_logistics", {"order_id": "o-1"}),
                    ("check_buyer_history", {"buyer_id": "b-1"}),
                    ("check_merchant_history", {"merchant_id": "m-1"}),
                    ("verify_evidence", {"evidence_id": "ev-1"}),
                ],
            )
        if name == "wrong_liability":
            return _make_episode(
                Decision(action="decide", liability="buyer", compensation=50.0, confidence=0.9,
                         evidence_ids=["chat:1"], reason="错误判责"),
                tool_plan=[],
            )
        if name == "invalid_output":
            return _make_episode(None)
        raise KeyError(name)

    return build


def test_decide_reward_uses_approved_weights(decide_episode):
    result = RewardEngine().score(decide_episode)
    expected = (
        0.45 * result.components.liability
        + 0.25 * result.components.compensation
        + 0.15 * result.components.escalation
        + 0.15 * result.components.grounding
        - 0.10 * result.components.normalized_tool_cost
        + result.components.invalid_action_penalty
    )
    assert result.total == pytest.approx(max(-1.5, min(1.0, expected)))


def test_missing_terminal_decision_is_hard_failure(unfinished_episode):
    assert RewardEngine().score(unfinished_episode).total == -1.5


def test_handcrafted_reward_ranking(reward_fixture):
    names = [
        "correct_efficient", "correct_redundant", "correct_overpay",
        "reasonable_escalation", "wrong_liability", "invalid_output",
    ]
    scores = [RewardEngine().score(reward_fixture(name)).total for name in names]
    assert scores == sorted(scores, reverse=True)
