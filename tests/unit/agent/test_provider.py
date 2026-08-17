import pytest

from dispute_agent.agent.runtime import build_runtime


@pytest.mark.asyncio
async def test_runtime_executes_tool_then_terminal_decision(fake_model_server, episode):
    runtime = build_runtime(
        base_url=fake_model_server.url,
        api_key="test",
        http_client=fake_model_server.http_client,
    )
    result = await runtime.run(episode, enable_thinking=True)
    assert [c.name for c in episode.tool_calls] == ["check_logistics"]
    assert result.action == "decide"
    assert fake_model_server.requests_contain("role", "tool")
    assert not fake_model_server.requests_contain_text("true_liability")
