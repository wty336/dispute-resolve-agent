def test_same_case_tool_and_arguments_return_same_result(tool_registry, episode):
    first = tool_registry.execute(episode, "check_logistics", {"order_id": "o-1"})
    second = tool_registry.execute(episode, "check_logistics", {"order_id": "o-1"})
    assert first.payload == second.payload
    assert second.cached is True
    assert episode.invalid_actions.repeat_calls == 1
    assert "true_liability" not in str(first.payload).lower()


def test_grpo_group_shares_tool_outcomes_but_not_mutable_state(make_episode):
    episodes = [make_episode(case_id="case-7", group_seed=42) for _ in range(4)]
    results = [e.call("verify_evidence", {"evidence_id": "ev-1"}) for e in episodes]
    assert len({r.model_dump_json() for r in results}) == 1
    episodes[0].invalid_actions.illegal_calls += 1
    assert episodes[1].invalid_actions.illegal_calls == 0
