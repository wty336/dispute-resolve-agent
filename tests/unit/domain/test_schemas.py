import pytest
from pydantic import TypeAdapter, ValidationError
from dispute_agent.domain.schemas import Decision, Escalation, TerminalDecision


adapter = TypeAdapter(TerminalDecision)


def test_decide_requires_liability_and_compensation():
    value = adapter.validate_python({
        "action": "decide", "liability": "merchant", "compensation": 89.0,
        "confidence": 0.91, "evidence_ids": ["logistics:1"], "reason": "物流签收异常",
    })
    assert isinstance(value, Decision)


def test_escalate_rejects_liability_and_compensation():
    with pytest.raises(ValidationError):
        adapter.validate_python({
            "action": "escalate", "liability": "buyer", "compensation": 0,
            "confidence": 0.3, "evidence_ids": ["chat:2"], "reason": "证据冲突",
        })
