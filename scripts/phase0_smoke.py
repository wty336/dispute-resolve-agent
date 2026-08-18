#!/usr/bin/env python3
"""运行真实的本地 Phase 0 fixture 或双 4090 smoke gate。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dispute_agent.training.grpo_config import load_grpo_config
from dispute_agent.training.grpo_runtime import PACKAGE_EXPECTATIONS, validate_input_adapter
from dispute_agent.training.phase0 import (
    EXPECTED_TORCH_CUDA,
    Phase0Report,
    build_fixture_report,
    evaluate_actual_evidence,
    package_matrix_matches,
)


def _write_report(report: Phase0Report, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "phase0_report.json"
    path.write_text(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _read_json(path: Path) -> dict:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return document if isinstance(document, dict) else {}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _environment_evidence() -> dict:
    versions: dict[str, str | None] = {}
    for package in PACKAGE_EXPECTATIONS:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    result: dict[str, object] = {
        "package_versions": versions,
        "cuda_available": False,
        "gpu_count": 0,
        "bf16_supported": [],
        "devices": [],
        "torch_cuda_version": None,
    }
    try:
        import torch

        result["cuda_available"] = torch.cuda.is_available()
        result["torch_cuda_version"] = torch.version.cuda
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        result["gpu_count"] = count
        bf16_supported: list[bool] = []
        devices: list[dict[str, object]] = []
        for index in range(count):
            with torch.cuda.device(index):
                bf16_supported.append(bool(torch.cuda.is_bf16_supported()))
            devices.append({
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            })
        result["bf16_supported"] = bf16_supported
        result["devices"] = devices
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _checkpoint_evidence(run_dir: Path) -> dict:
    checkpoint_root = run_dir / "checkpoints"
    steps: list[tuple[int, Path]] = []
    if checkpoint_root.is_dir():
        for path in checkpoint_root.glob("global_step_*"):
            try:
                step = int(path.name.removeprefix("global_step_"))
            except ValueError:
                continue
            if path.is_dir():
                steps.append((step, path))
    if not steps:
        return {"error": "no global_step_* checkpoint found"}
    step, step_dir = max(steps, key=lambda item: item[0])
    preferred = step_dir / "actor" / "huggingface"
    candidates = [preferred] if (preferred / "adapter_config.json").is_file() else []
    if not candidates:
        candidates = sorted({path.parent for path in step_dir.rglob("adapter_config.json")})
    candidates = [
        path for path in candidates
        if list(path.glob("*.safetensors"))
    ]
    if len(candidates) != 1:
        return {
            "global_step": step,
            "error": f"expected one PEFT adapter directory, found {len(candidates)}",
            "candidates": [str(path.resolve()) for path in candidates],
        }
    adapter_dir = candidates[0]
    weights = sorted(adapter_dir.glob("*.safetensors"))
    return {
        "global_step": step,
        "adapter_dir": str(adapter_dir.resolve()),
        "adapter_config_sha256": _sha256(adapter_dir / "adapter_config.json"),
        "weight_hashes": {path.name: _sha256(path) for path in weights},
    }


def _actual_report(*, config_path: Path, data_dir: Path, output_root: Path, run_id: str) -> Phase0Report:
    run_dir = output_root / run_id
    environment = _environment_evidence()
    config = None
    initial_adapter: dict = {}
    preflight_errors: list[str] = []
    try:
        config = load_grpo_config(config_path)
        initial_adapter = validate_input_adapter(
            config.actor_rollout_ref.model.lora_adapter_path, config
        )
    except Exception as exc:
        preflight_errors.append(f"adapter: {type(exc).__name__}: {exc}")
    if not package_matrix_matches(environment.get("package_versions")):
        preflight_errors.append("installed package versions do not match the exact training matrix")
    if environment.get("gpu_count") != 2 or environment.get("bf16_supported") != [True, True]:
        preflight_errors.append("actual mode requires exactly two BF16-capable CUDA devices")
    if environment.get("torch_cuda_version") != EXPECTED_TORCH_CUDA:
        preflight_errors.append(f"PyTorch must use CUDA {EXPECTED_TORCH_CUDA}")

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
    process_evidence: dict[str, object] = {"command": command, "returncode": None}
    if preflight_errors:
        process_evidence["preflight_errors"] = preflight_errors
    else:
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        process_evidence.update({
            "returncode": completed.returncode,
            "stdout_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
            "stderr_sha256": hashlib.sha256(completed.stderr.encode("utf-8")).hexdigest(),
            "stdout_chars": len(completed.stdout),
            "stderr_chars": len(completed.stderr),
        })

    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "training_process.json").write_text(
        json.dumps(process_evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    training_manifest = _read_json(run_dir / "run_manifest.json")
    if not training_manifest:
        training_manifest = {
            "status": "failed",
            "error": "; ".join(preflight_errors) or "training produced no run manifest",
        }
    summary = _read_json(run_dir / "metrics" / "summary.json")
    checkpoint = _checkpoint_evidence(run_dir)
    reload_path = evidence_dir / "checkpoint_reload.json"
    if config is not None and checkpoint.get("adapter_dir"):
        public_prompt = evidence_dir / "public_reload_prompt.txt"
        public_prompt.write_text(
            "你是电商纠纷判责 Agent。请依据公开工单信息调查，并通过工具提交最终处理结果。",
            encoding="utf-8",
        )
        verify_command = [
            sys.executable,
            str(ROOT / "scripts" / "verify_grpo_checkpoint.py"),
            "--base-model", config.actor_rollout_ref.model.path,
            "--adapter-dir", str(checkpoint["adapter_dir"]),
            "--public-prompt-file", str(public_prompt),
            "--evidence-out", str(reload_path),
        ]
        verification = subprocess.run(verify_command, cwd=ROOT, capture_output=True, text=True)
        process_evidence["verification_returncode"] = verification.returncode
        (evidence_dir / "training_process.json").write_text(
            json.dumps(process_evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    reload_evidence = _read_json(reload_path)
    if not reload_evidence:
        reload_evidence = {"status": "failed", "error": "checkpoint reload was not completed"}
    return evaluate_actual_evidence(
        run_id=run_id,
        evidence_dir=evidence_dir,
        environment=environment,
        initial_adapter=initial_adapter,
        training_manifest=training_manifest,
        summary=summary,
        checkpoint=checkpoint,
        reload_evidence=reload_evidence,
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
