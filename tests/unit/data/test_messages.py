import pytest

from dispute_agent.data.generator import generate_fact_instances
from dispute_agent.data.renderer import render_sft_trace
from dispute_agent.domain.schemas import Decision


@pytest.fixture
def rendered_trace():
    instance = generate_fact_instances(seed=1, n=1)[0]
    decision = Decision(
        action="decide",
        liability="merchant",
        compensation=50.0,
        confidence=0.9,
        evidence_ids=["chat:0"],
        reason="证据支持",
    )
    return render_sft_trace(
        instance,
        tool_plan=[("check_logistics", {"order_id": instance.observation.order_id})],
        decision=decision,
    )


def test_sft_trace_uses_native_tool_protocol(rendered_trace):
    assistant_call = next(m for m in rendered_trace if m["role"] == "assistant" and m.get("tool_calls"))
    call_id = assistant_call["tool_calls"][0]["id"]
    tool_message = next(m for m in rendered_trace if m["role"] == "tool")
    assert tool_message["tool_call_id"] == call_id
    assert not any(m["role"] == "user" and "tool_result" in m.get("content", "") for m in rendered_trace)
