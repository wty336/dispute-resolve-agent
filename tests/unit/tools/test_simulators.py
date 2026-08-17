from dispute_agent.domain.schemas import DisputeGroundTruth
from dispute_agent.tools.simulators import simulate_tool


def _ground_truth(**updates):
    values = {
        "case_id": "case-signal",
        "true_liability": "merchant",
        "true_loss": 50.0,
        "reasonable_compensation_range": (40.0, 60.0),
        "buyer_strategy": "honest",
        "merchant_strategy": "honest",
        "should_escalate": False,
        "tool_information_value": {},
        "risk_level": "low",
        "logistics_state": "damaged_in_transit",
        "evidence_authenticity": {"ev-1": "authentic"},
        "tool_noise_rate": 0.0,
        "tool_timeout_rate": 0.0,
        "tool_missing_rate": 0.0,
    }
    values.update(updates)
    return DisputeGroundTruth(**values)


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


def test_buyer_history_is_conditioned_on_hidden_strategy_without_leaking_label():
    honest = simulate_tool(
        "check_buyer_history",
        case_id="case-signal",
        arguments={"buyer_id": "b-1"},
        case_seed=7,
        ground_truth=_ground_truth(buyer_strategy="honest"),
    )
    risky = simulate_tool(
        "check_buyer_history",
        case_id="case-signal",
        arguments={"buyer_id": "b-1"},
        case_seed=7,
        ground_truth=_ground_truth(buyer_strategy="fraud"),
    )

    assert "历史信誉良好" in honest.payload
    assert "多起投诉被平台驳回" in risky.payload
    assert honest.payload != risky.payload
    for hidden_label in ("fraud", "buyer_strategy", "true_liability"):
        assert hidden_label not in risky.payload.lower()


def test_evidence_verification_reflects_authenticity_with_calibrated_noise_disabled():
    authentic = simulate_tool(
        "verify_evidence",
        case_id="case-evidence",
        arguments={"evidence_id": "ev-1"},
        case_seed=11,
        ground_truth=_ground_truth(evidence_authenticity={"ev-1": "authentic"}),
    )
    forged = simulate_tool(
        "verify_evidence",
        case_id="case-evidence",
        arguments={"evidence_id": "ev-1"},
        case_seed=11,
        ground_truth=_ground_truth(evidence_authenticity={"ev-1": "forged"}),
    )

    assert "核验通过" in authentic.payload
    assert "明显伪造痕迹" in forged.payload


def test_tool_timeout_is_deterministic_and_structured():
    truth = _ground_truth(tool_timeout_rate=1.0)
    results = [
        simulate_tool(
            "check_logistics",
            case_id="case-timeout",
            arguments={"order_id": "o-1"},
            case_seed=13,
            ground_truth=truth,
        )
        for _ in range(2)
    ]

    assert results[0] == results[1]
    assert results[0].success is False
    assert results[0].error_type == "timeout"
    assert "超时" in results[0].payload
