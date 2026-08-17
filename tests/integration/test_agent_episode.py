import pytest

from dispute_agent.agent.runtime import build_runtime


@pytest.mark.asyncio
async def test_runtime_stops_after_submit_decision(fake_model_server, episode):
    runtime = build_runtime(
        base_url=fake_model_server.url,
        api_key="test",
        http_client=fake_model_server.http_client,
    )
    result = await runtime.run(episode, enable_thinking=True)
    assert result.action == "decide"
    assert episode.done is True
    assert episode.end_reason == "submitted"
    assert fake_model_server._call_count == 2
