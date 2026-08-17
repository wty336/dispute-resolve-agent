#!/usr/bin/env python3
"""Run an honest local Phase 0 fixture or the real dual-4090 smoke gate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dispute_agent.training.grpo_config import load_grpo_config
from dispute_agent.training.grpo_runtime import validate_input_adapter
from dispute_agent.training.phase0 import (
    REQUIRED_GATES,
    GateResult,
    GateStatus,
    Phase0Report,
    build_fixture_report,
)


def _write_report(report: Phase0Report, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "phase0_report.json"
    path.write_text(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _actual_report(*, config_path: Path, data_dir: Path, output_root: Path, run_id: str) -> Phase0Report:
    run_dir = output_root / run_id
    evidence_dir = run_dir / "evidence"
    statuses = {name: (GateStatus.NOT_RUN, "not executed") for name in REQUIRED_GATES}
    adapter_validated = False

    try:
        config = load_grpo_config(config_path)
        validate_input_adapter(config.actor_rollout_ref.model.lora_adapter_path, config)
        adapter_validated = True
        statuses["trl_adapter_loaded_by_verl"] = (
            GateStatus.NOT_RUN,
            "adapter metadata validated; VERL load evidence is still required",
        )
    except Exception as exc:
        statuses["trl_adapter_loaded_by_verl"] = (GateStatus.FAILED, f"{type(exc).__name__}: {exc}")

    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_agentic_grpo.py"),
        "--config", str(config_path),
        "--data-dir", str(data_dir),
        "--output-root", str(output_root),
        "--run-id", run_id,
        "--profile", "smoke",
        "--curriculum-phase", "1",
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if adapter_validated:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        (evidence_dir / "adapter.json").write_text(json.dumps({"status": "validated"}) + "\n", encoding="utf-8")
    if completed.returncode == 0:
        statuses["sdk_vllm_multiturn"] = (GateStatus.PASSED, "smoke training exited successfully")
        statuses["grpo_update_reload"] = (GateStatus.NOT_RUN, "checkpoint verification still required")
    else:
        statuses["sdk_vllm_multiturn"] = (GateStatus.FAILED, completed.stderr[-1000:])

    gates = []
    for name in REQUIRED_GATES:
        status, summary = statuses[name]
        evidence = None
        if status is GateStatus.PASSED:
            evidence = evidence_dir / f"{name}.json"
            evidence.parent.mkdir(parents=True, exist_ok=True)
            evidence.write_text(json.dumps({"status": status.value, "summary": summary}) + "\n", encoding="utf-8")
        gates.append(GateResult(name=name, status=status, summary=summary, evidence_path=evidence))
    return Phase0Report(
        mode="actual",
        run_id=run_id,
        gates=gates,
        overall_status=GateStatus.NOT_RUN,
        ready_for_formal_training=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["fixture", "actual"], default="fixture")
    parser.add_argument("--config", default="configs/grpo.yaml", type=Path)
    parser.add_argument("--data-dir", default="data/processed", type=Path)
    parser.add_argument("--output-root", default="outputs/grpo", type=Path)
    parser.add_argument("--run-id", default=f"phase0-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    args = parser.parse_args()

    if args.mode == "fixture":
        report = build_fixture_report(output_dir=args.output_root / args.run_id, run_id=args.run_id)
        path = _write_report(report, args.output_root / args.run_id)
        print(f"Phase 0 fixture report written to {path}")
        return 0
    report = _actual_report(
        config_path=args.config,
        data_dir=args.data_dir,
        output_root=args.output_root,
        run_id=args.run_id,
    )
    path = _write_report(report, args.output_root / args.run_id)
    print(f"Phase 0 report written to {path} ({report.overall_status.value})")
    return 0 if report.ready_for_formal_training else 1


if __name__ == "__main__":
    raise SystemExit(main())
