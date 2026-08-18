from types import SimpleNamespace

import pytest

from dispute_agent.agent.runtime import DisputeRuntime
from dispute_agent.agent.tools import build_agent_tools


def test_phase_one_filters_tools_but_keeps_submit(episode):
    tools = build_agent_tools(
        episode,
        allowed_tools={
            "check_logistics",
            "check_buyer_history",
            "check_merchant_history",
        },
    )
    names = {tool.name for tool in tools}

    assert names == {
        "check_logistics",
        "check_buyer_history",
        "check_merchant_history",
        "submit_decision",
    }
    assert "verify_evidence" not in names


@pytest.mark.asyncio
async def test_runtime_passes_turn_and_token_budgets(monkeypatch, episode):
    captured = {}

    async def fake_run(agent, input, *, max_turns):
        captured["max_turns"] = max_turns
        captured["max_tokens"] = agent.model_settings.max_tokens
        episode.submit(
            __import__("dispute_agent.domain.schemas", fromlist=["Decision"]).Decision(
                action="decide",
                liability="merchant",
                compensation=50.0,
                confidence=0.8,
                evidence_ids=["chat:1"],
                reason="ok",
            )
        )

    monkeypatch.setattr("dispute_agent.agent.runtime.Runner.run", fake_run)
    monkeypatch.setattr("dispute_agent.agent.runtime.create_chat_model", lambda **_: object())
    monkeypatch.setattr(
        "dispute_agent.agent.runtime.Agent",
        lambda **kwargs: SimpleNamespace(model_settings=kwargs["model_settings"]),
    )
    runtime = DisputeRuntime(base_url="http://fake", api_key="EMPTY")

    await runtime.run(episode, max_rounds=3, max_tokens_per_round=384)

    assert captured == {"max_turns": 4, "max_tokens": 384}
