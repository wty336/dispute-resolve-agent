import pytest


def test_terminal_decision_closes_episode(episode, valid_decision):
    episode.submit(valid_decision)
    assert episode.done and episode.end_reason == "submitted"
    with pytest.raises(RuntimeError, match="already terminated"):
        episode.record_tool_call("check_logistics", {"order_id": "o-1"})
