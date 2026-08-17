import pytest

from dispute_agent.evaluation.runner import resolve_runs


@pytest.fixture
def resolved_runs():
    return resolve_runs(
        ["rule_based", "base", "sft-500", "sft-1000", "sft-1500", "sft-1500+grpo", "oracle"]
    )


def test_all_model_variants_share_runtime_contract(resolved_runs):
    fields = ("case_hash", "tool_registry_hash", "max_rounds", "tool_budget", "terminal_schema_hash")
    baseline = {field: getattr(resolved_runs[0], field) for field in fields}
    assert all({field: getattr(run, field) for field in fields} == baseline for run in resolved_runs[1:])
