from collections import Counter
import inspect

import pytest

from dispute_agent.data import generator
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


def test_sft_profiles_match_the_approved_800_500_200_mix():
    assert hasattr(generator, "plan_sft_profiles")
    profiles = generator.plan_sft_profiles(1500, seed=20260817)
    assert Counter(profile.category for profile in profiles) == {
        "direct": 800,
        "multi_tool": 500,
        "edge_case": 200,
    }
    assert all(not profile.tool_names for profile in profiles if profile.category == "direct")
    assert all(1 <= len(profile.tool_names) <= 4 for profile in profiles if profile.category == "multi_tool")
    assert all(profile.edge_case for profile in profiles if profile.category == "edge_case")
    assert Counter(profile.edge_case for profile in profiles if profile.category == "edge_case") == {
        "manual_escalation": 50,
        "tool_failure": 50,
        "illegal_result_recovery": 50,
        "high_risk": 50,
    }


def test_invalid_tool_result_can_be_followed_by_an_alternative_tool():
    assert "tool_result_overrides" in inspect.signature(render_sft_trace).parameters

    instance = generate_fact_instances(seed=2, n=1)[0]
    decision = Decision(
        action="decide",
        liability="merchant",
        compensation=50.0,
        confidence=0.9,
        evidence_ids=["chat:0"],
        reason="改用其他工具后完成判断",
    )
    messages = render_sft_trace(
        instance,
        tool_plan=[
            ("verify_evidence", {"evidence_id": "chat:0"}),
            ("check_buyer_history", {"buyer_id": instance.observation.buyer_id}),
        ],
        tool_result_overrides={0: "工具返回格式非法：缺少可核验证据字段，结果不可采信"},
        decision=decision,
    )
    tool_messages = [message for message in messages if message["role"] == "tool"]
    recovery_call = [message for message in messages if message.get("tool_calls")][1]
    assert "格式非法" in tool_messages[0]["content"]
    assert "改用其他证据源" in recovery_call["content"]
    assert "买家" in tool_messages[1]["content"]
