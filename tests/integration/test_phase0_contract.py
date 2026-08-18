import pytest

from dispute_agent.training.phase0 import (
    GateResult,
    GateStatus,
    Phase0EvidenceError,
    Phase0Report,
    REQUIRED_GATES,
    build_fixture_report,
    evaluate_actual_evidence,
)
from dispute_agent.training.grpo_runtime import PACKAGE_EXPECTATIONS


def test_fixture_report_marks_every_execution_gate_not_run(tmp_path):
    report = build_fixture_report(output_dir=tmp_path)

    assert report.mode == "fixture"
    assert {gate.status for gate in report.gates} == {GateStatus.NOT_RUN}
    assert report.overall_status == GateStatus.NOT_RUN
    assert not report.ready_for_formal_training


def test_pass_requires_evidence_file(tmp_path):
    with pytest.raises(Phase0EvidenceError):
        GateResult(
            name="adapter_reloaded",
            status=GateStatus.PASSED,
            summary="missing evidence",
            evidence_path=tmp_path / "missing.json",
        )


def test_report_cannot_claim_ready_with_failed_or_not_run_gate(tmp_path):
    for status in (GateStatus.FAILED, GateStatus.NOT_RUN):
        gates = [
            GateResult(name=name, status=(status if index == 0 else GateStatus.NOT_RUN), summary="no")
            for index, name in enumerate(REQUIRED_GATES)
        ]
        report = Phase0Report(
            mode="actual",
            run_id="r",
            gates=gates,
            overall_status=status,
            ready_for_formal_training=True,
        )
        assert not report.ready_for_formal_training


def test_actual_evidence_passes_every_gate_only_with_update_and_reload(tmp_path):
    evidence_dir = tmp_path / "evidence"
    report = evaluate_actual_evidence(
        run_id="phase0-test",
        evidence_dir=evidence_dir,
        environment={
            "package_versions": PACKAGE_EXPECTATIONS,
            "cuda_available": True,
            "gpu_count": 2,
            "bf16_supported": [True, True],
            "torch_cuda_version": "12.8",
        },
        initial_adapter={"path": "sft", "weight_hashes": {"adapter_model.safetensors": "before"}},
        training_manifest={
            "status": "completed",
            "inputs": {"adapter": {"path": "sft"}},
        },
        summary={
            "completed_count": 1,
            "failure_count": 0,
            "model_span_count": 2,
            "object_span_count": 1,
            "tool_event_span_count": 1,
            "reward_count": 1,
            "reward_span_count": 1,
            "single_reward_rollout_count": 1,
            "multi_turn_rollout_count": 1,
            "tool_rollout_count": 1,
            "thinking_enabled_count": 1,
        },
        checkpoint={
            "global_step": 1,
            "adapter_dir": "checkpoint",
            "weight_hashes": {"adapter_model.safetensors": "after"},
        },
        reload_evidence={
            "status": "passed",
            "dtype": "bfloat16",
            "generated_token_count": 1,
            "weight_hashes": {"adapter_model.safetensors": "after"},
        },
    )

    assert report.ready_for_formal_training
    assert report.overall_status is GateStatus.PASSED
    assert all(gate.evidence_path and gate.evidence_path.is_file() for gate in report.gates)

    failed = evaluate_actual_evidence(
        run_id="phase0-unchanged",
        evidence_dir=tmp_path / "unchanged",
        environment={
            "package_versions": PACKAGE_EXPECTATIONS,
            "cuda_available": True,
            "gpu_count": 2,
            "bf16_supported": [True, True],
            "torch_cuda_version": "12.8",
        },
        initial_adapter={"path": "sft", "weight_hashes": {"adapter_model.safetensors": "same"}},
        training_manifest={"status": "completed", "inputs": {"adapter": {"path": "sft"}}},
        summary={
            "completed_count": 1, "failure_count": 0, "model_span_count": 2,
            "object_span_count": 1, "reward_count": 1, "reward_span_count": 1,
            "tool_event_span_count": 1,
            "single_reward_rollout_count": 1, "multi_turn_rollout_count": 1,
            "tool_rollout_count": 1, "thinking_enabled_count": 1,
        },
        checkpoint={
            "global_step": 1, "adapter_dir": "checkpoint",
            "weight_hashes": {"adapter_model.safetensors": "same"},
        },
        reload_evidence={
            "status": "passed", "dtype": "bfloat16", "generated_token_count": 1,
            "weight_hashes": {"adapter_model.safetensors": "same"},
        },
    )
    assert not failed.ready_for_formal_training
    assert next(g for g in failed.gates if g.name == "grpo_update_reload").status is GateStatus.FAILED
