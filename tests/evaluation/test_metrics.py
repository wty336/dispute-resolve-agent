import pytest

from dispute_agent.evaluation.metrics import compute_metrics


@pytest.fixture
def fixed_predictions():
    return [
        {
            "true_liability": "merchant",
            "pred_liability": "merchant",
            "true_compensation": 100.0,
            "pred_compensation": 100.0,
            "escalation_true": False,
            "escalation_pred": False,
            "evidence_ids": ["e1"],
            "visible_evidence_ids": ["e1"],
            "tool_calls": 1,
            "tool_cost": 2.0,
            "necessary_tool": True,
            "used_necessary_tool": True,
            "episode_success": True,
        },
        {
            "true_liability": "buyer",
            "pred_liability": "buyer",
            "true_compensation": 50.0,
            "pred_compensation": 75.0,
            "escalation_true": False,
            "escalation_pred": False,
            "evidence_ids": ["e2"],
            "visible_evidence_ids": ["e2"],
            "tool_calls": 2,
            "tool_cost": 5.0,
            "necessary_tool": True,
            "used_necessary_tool": False,
            "episode_success": True,
        },
        {
            "true_liability": "buyer",
            "pred_liability": "split",
            "true_compensation": 60.0,
            "pred_compensation": 10.0,
            "escalation_true": False,
            "escalation_pred": False,
            "evidence_ids": ["e3"],
            "visible_evidence_ids": ["e3"],
            "tool_calls": 0,
            "tool_cost": 0.0,
            "necessary_tool": False,
            "used_necessary_tool": False,
            "episode_success": True,
        },
        {
            "true_liability": "split",
            "pred_liability": "split",
            "true_compensation": 50.0,
            "pred_compensation": 50.0,
            "escalation_true": True,
            "escalation_pred": True,
            "evidence_ids": ["e4"],
            "visible_evidence_ids": ["e4"],
            "tool_calls": 3,
            "tool_cost": 8.0,
            "necessary_tool": True,
            "used_necessary_tool": True,
            "episode_success": True,
        },
        {
            "true_liability": "split",
            "pred_liability": "buyer",
            "true_compensation": 60.0,
            "pred_compensation": 110.0,
            "escalation_true": False,
            "escalation_pred": False,
            "evidence_ids": ["e5"],
            "visible_evidence_ids": ["e5"],
            "tool_calls": 1,
            "tool_cost": 3.0,
            "necessary_tool": False,
            "used_necessary_tool": False,
            "episode_success": False,
        },
    ]


def test_metrics_cover_task_business_agent_and_safety(fixed_predictions):
    report = compute_metrics(fixed_predictions)
    assert report.liability.macro_f1 == pytest.approx(0.666666, rel=1e-5)
    assert report.business.compensation_mae == pytest.approx(25.0)
    assert 0 <= report.agent.necessary_tool_recall <= 1
    assert 0 <= report.safety.escalation_f1 <= 1
    assert report.safety.evidence_hallucination_rate == 0
