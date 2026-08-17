import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
REQUIRED = {
    "sdk_vllm_multiturn", "thinking_tool_compatibility", "trace_complete",
    "trl_adapter_loaded_by_verl", "grpo_update_reload", "dual_gpu_no_oom",
    "single_model_span_and_reward",
}


@pytest.fixture
def phase0_report():
    out_dir = ROOT / ".tmp_tests" / "phase0"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "phase0_report.json"
    subprocess.run(
        [sys.executable, "scripts/phase0_smoke.py", "--fixture", "--report", str(report_path)],
        cwd=ROOT,
        check=True,
    )
    return json.loads(report_path.read_text(encoding="utf-8"))


def test_phase0_report_passes_every_gate(phase0_report):
    assert set(phase0_report["gates"]) == REQUIRED
    assert all(gate["passed"] for gate in phase0_report["gates"].values())
    assert phase0_report["gpu_count"] == 2
    assert phase0_report["quantization"] == "none"
