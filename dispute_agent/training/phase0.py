"""Evidence-backed Phase 0 gate model for the dual-4090 handoff."""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


REQUIRED_GATES = (
    "sdk_vllm_multiturn",
    "thinking_tool_compatibility",
    "trace_complete",
    "trl_adapter_loaded_by_verl",
    "grpo_update_reload",
    "dual_gpu_no_oom",
    "single_model_span_and_reward",
)


class Phase0EvidenceError(ValueError):
    """Raised when a report attempts to claim an unsupported gate."""


class GateStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_RUN = "not_run"


class GateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    status: GateStatus
    summary: str = ""
    evidence_path: Path | None = None

    def __init__(self, **data):
        if data.get("status") == GateStatus.PASSED:
            evidence_path = data.get("evidence_path")
            if evidence_path is None or not Path(evidence_path).is_file():
                raise Phase0EvidenceError(
                    f"passed gate requires an evidence file: {data.get('name', '<unknown>')}"
                )
        super().__init__(**data)

    @model_validator(mode="after")
    def require_evidence_for_pass(self) -> "GateResult":
        if self.status is GateStatus.PASSED:
            if self.evidence_path is None or not self.evidence_path.is_file():
                raise Phase0EvidenceError(f"passed gate requires an evidence file: {self.name}")
        return self


class Phase0Report(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["phase0-v2"] = "phase0-v2"
    mode: Literal["fixture", "actual"]
    run_id: str
    gates: list[GateResult]
    overall_status: GateStatus
    ready_for_formal_training: bool

    @model_validator(mode="after")
    def enforce_aggregate_status(self) -> "Phase0Report":
        names = {gate.name for gate in self.gates}
        if len(names) != len(self.gates) or not names.issubset(set(REQUIRED_GATES)):
            raise Phase0EvidenceError("phase0 report contains duplicate or unknown gate names")
        all_passed = names == set(REQUIRED_GATES) and all(
            gate.status is GateStatus.PASSED for gate in self.gates
        )
        if self.mode == "fixture":
            for gate in self.gates:
                if gate.status is not GateStatus.NOT_RUN:
                    raise Phase0EvidenceError("fixture execution gates must be not_run")
            self.overall_status = GateStatus.NOT_RUN
            self.ready_for_formal_training = False
            return self
        if not all_passed:
            self.ready_for_formal_training = False
            self.overall_status = (
                GateStatus.FAILED
                if any(gate.status is GateStatus.FAILED for gate in self.gates)
                else GateStatus.NOT_RUN
            )
        else:
            self.overall_status = GateStatus.PASSED
            self.ready_for_formal_training = True
        return self


def build_fixture_report(*, output_dir: str | Path, run_id: str = "local-fixture") -> Phase0Report:
    """Describe local preparation without pretending that GPU gates ran."""
    root = Path(output_dir)
    gates = [
        GateResult(
            name=name,
            status=GateStatus.NOT_RUN,
            summary="not executed in fixture mode",
            evidence_path=None,
        )
        for name in REQUIRED_GATES
    ]
    return Phase0Report(
        mode="fixture",
        run_id=run_id,
        gates=gates,
        overall_status=GateStatus.NOT_RUN,
        ready_for_formal_training=False,
    )
