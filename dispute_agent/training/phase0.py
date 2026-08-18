"""Evidence-backed Phase 0 gate model for the dual-4090 handoff."""
from __future__ import annotations

from enum import Enum
import json
from pathlib import Path
from typing import Any, Literal

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
EXPECTED_TORCH_CUDA = "12.8"


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


def _count(document: dict[str, Any], name: str) -> int:
    value = document.get(name, 0)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def package_matrix_matches(package_versions: object) -> bool:
    """Accept wheel-local suffixes while requiring every pinned base version."""
    from dispute_agent.training.grpo_runtime import PACKAGE_EXPECTATIONS

    return isinstance(package_versions, dict) and all(
        isinstance(package_versions.get(name), str)
        and package_versions[name].split("+", 1)[0] == version
        for name, version in PACKAGE_EXPECTATIONS.items()
    )


def evaluate_actual_evidence(
    *,
    run_id: str,
    evidence_dir: str | Path,
    environment: dict[str, Any],
    initial_adapter: dict[str, Any],
    training_manifest: dict[str, Any],
    summary: dict[str, Any],
    checkpoint: dict[str, Any],
    reload_evidence: dict[str, Any],
) -> Phase0Report:
    """Evaluate the real smoke run from independently inspectable artifacts."""
    from dispute_agent.training.grpo_runtime import PACKAGE_EXPECTATIONS

    evidence_root = Path(evidence_dir)
    evidence_root.mkdir(parents=True, exist_ok=True)
    package_versions = environment.get("package_versions", {})
    versions_ok = package_matrix_matches(package_versions)
    completed = training_manifest.get("status") == "completed"
    completed_count = _count(summary, "completed_count")
    no_failures = _count(summary, "failure_count") == 0
    input_adapter = training_manifest.get("inputs", {}).get("adapter", {})
    adapter_loaded = (
        completed
        and bool(initial_adapter.get("weight_hashes"))
        and isinstance(input_adapter, dict)
        and input_adapter.get("path") == initial_adapter.get("path")
    )
    initial_hashes = initial_adapter.get("weight_hashes", {})
    checkpoint_hashes = checkpoint.get("weight_hashes", {})
    reload_hashes = reload_evidence.get("weight_hashes", {})
    checkpoint_updated = (
        bool(initial_hashes)
        and bool(checkpoint_hashes)
        and initial_hashes != checkpoint_hashes
    )
    reload_ok = (
        reload_evidence.get("status") == "passed"
        and reload_evidence.get("dtype") == "bfloat16"
        and _count(reload_evidence, "generated_token_count") >= 1
        and checkpoint_hashes == reload_hashes
    )
    environment_ok = (
        environment.get("cuda_available") is True
        and _count(environment, "gpu_count") == 2
        and environment.get("bf16_supported") == [True, True]
        and environment.get("torch_cuda_version") == EXPECTED_TORCH_CUDA
    )

    checks: dict[str, tuple[bool, str, dict[str, Any]]] = {
        "sdk_vllm_multiturn": (
            versions_ok and completed and completed_count > 0
            and _count(summary, "multi_turn_rollout_count") > 0,
            "exact package matrix and multi-turn smoke rollout",
            {"package_versions": package_versions, "expected_versions": PACKAGE_EXPECTATIONS,
             "completed_count": completed_count,
             "multi_turn_rollout_count": _count(summary, "multi_turn_rollout_count")},
        ),
        "thinking_tool_compatibility": (
            completed_count > 0
            and _count(summary, "thinking_enabled_count") == completed_count
            and _count(summary, "tool_rollout_count") > 0,
            "thinking mode is active and at least one completed rollout uses tools",
            {"completed_count": completed_count,
             "thinking_enabled_count": _count(summary, "thinking_enabled_count"),
             "tool_rollout_count": _count(summary, "tool_rollout_count")},
        ),
        "trace_complete": (
            completed_count > 0
            and _count(summary, "model_span_count") >= completed_count
            and _count(summary, "object_span_count") >= completed_count
            and _count(summary, "tool_event_span_count") > 0
            and _count(summary, "reward_span_count") == completed_count,
            "model, tool-event, object, and final-reward spans are complete",
            {key: summary.get(key) for key in (
                "completed_count", "model_span_count", "object_span_count",
                "tool_event_span_count", "reward_span_count"
            )},
        ),
        "trl_adapter_loaded_by_verl": (
            adapter_loaded,
            "validated SFT adapter is recorded as the VERL training input",
            {"initial_adapter": initial_adapter, "manifest_input_adapter": input_adapter,
             "training_status": training_manifest.get("status")},
        ),
        "grpo_update_reload": (
            completed and _count(checkpoint, "global_step") >= 1
            and checkpoint_updated and reload_ok,
            "GRPO changed adapter weights and the BF16 checkpoint reloaded for inference",
            {"checkpoint": checkpoint, "initial_weight_hashes": initial_hashes,
             "reload": reload_evidence, "checkpoint_updated": checkpoint_updated},
        ),
        "dual_gpu_no_oom": (
            environment_ok and completed and no_failures,
            "two BF16-capable GPUs completed the smoke run without rollout failures",
            {"environment": environment, "training_status": training_manifest.get("status"),
             "failure_count": _count(summary, "failure_count")},
        ),
        "single_model_span_and_reward": (
            completed_count > 0
            and _count(summary, "reward_count") == completed_count
            and _count(summary, "reward_span_count") == completed_count
            and _count(summary, "single_reward_rollout_count") == completed_count
            and _count(summary, "model_span_count") >= completed_count,
            "every completed rollout has model spans and exactly one final reward",
            {key: summary.get(key) for key in (
                "completed_count", "reward_count", "reward_span_count",
                "single_reward_rollout_count", "model_span_count"
            )},
        ),
    }

    gates: list[GateResult] = []
    for name in REQUIRED_GATES:
        passed, summary_text, payload = checks[name]
        evidence_path = evidence_root / f"{name}.json"
        evidence_path.write_text(
            json.dumps({"passed": passed, "checks": payload}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        gates.append(GateResult(
            name=name,
            status=GateStatus.PASSED if passed else GateStatus.FAILED,
            summary=summary_text if passed else f"failed: {summary_text}",
            evidence_path=evidence_path,
        ))
    return Phase0Report(
        mode="actual",
        run_id=run_id,
        gates=gates,
        overall_status=GateStatus.PASSED,
        ready_for_formal_training=True,
    )
