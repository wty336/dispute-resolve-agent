import pytest

from dispute_agent.training.phase0 import (
    GateResult,
    GateStatus,
    Phase0EvidenceError,
    Phase0Report,
    REQUIRED_GATES,
    build_fixture_report,
)


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
