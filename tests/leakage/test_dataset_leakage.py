from dispute_agent.data.generator import generate_fact_instances
from dispute_agent.data.renderer import render_sft_trace
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
