import json

from dispute_agent.data.generator import generate_fact_instances
from dispute_agent.data.renderer import render_sft_trace
from scripts.generate_data import _render_row
from dispute_agent.data.validators import (
    validate_observation,
    validate_trace_messages,
)


def test_generated_observations_and_traces_do_not_leak_hidden_fields():
    instances = generate_fact_instances(seed=20260817, n=5)
    for instance in instances:
        assert validate_observation(instance.observation) == []
        trace = render_sft_trace(
            instance,
            tool_plan=[("check_logistics", {"order_id": instance.observation.order_id})],
        )
        assert validate_trace_messages(trace) == []


def test_grpo_row_contains_structured_public_observation_only():
    instance = generate_fact_instances(seed=20260817, n=1)[0]
    instance.split = "grpo_train"

    row = _render_row(instance)

    assert row["observation"] == instance.observation.model_dump(mode="json")
    serialized = json.dumps(row["observation"], ensure_ascii=False)
    for hidden in (
        "true_liability",
        "true_loss",
        "reasonable_compensation_range",
        "should_escalate",
        "evidence_authenticity",
    ):
        assert hidden not in serialized


def test_sft_row_does_not_gain_an_unused_observation_column():
    instance = generate_fact_instances(seed=20260817, n=1)[0]
    instance.split = "sft_train"

    assert "observation" not in _render_row(instance)
