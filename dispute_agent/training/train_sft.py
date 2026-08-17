"""Configuration and orchestration for TRL BF16 LoRA SFT."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from dispute_agent.training.sft_dataset import SFTDatasetBundle
from dispute_agent.training.sft_runtime import RealSFTBackend, TrainingRequest, build_trainer_spec


class LoraSettings(BaseModel):
    rank: int = Field(default=32, gt=0)
    alpha: int = Field(default=64, gt=0)
    dropout: float = Field(default=0.0, ge=0, lt=1)
    target_modules: str = "all-linear"


class SFTConfig(BaseModel):
    model: str = "Qwen/Qwen3-8B"
    data_dir: str = "data/generated"
    bf16: bool = True
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    lora: LoraSettings = Field(default_factory=LoraSettings)
    assistant_only_loss: bool = True
    packing: bool = False
    seed: int = 20260817
    data_seed: int = 20260817
    max_length: int = Field(default=2048, gt=0)
    per_device_train_batch_size: int = Field(default=1, gt=0)
    per_device_eval_batch_size: int = Field(default=1, gt=0)
    global_batch_size: int = Field(default=16, gt=0)
    num_train_epochs: float = Field(default=3.0, gt=0)
    learning_rate: float = Field(default=2e-4, gt=0)
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = Field(default=0.03, ge=0, lt=1)
    weight_decay: float = Field(default=0.0, ge=0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    gradient_checkpointing: bool = True
    attention_implementation: str = "sdpa"
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = Field(default=2, gt=0)
    logging_steps: int = Field(default=5, gt=0)
    output_root: str = "checkpoints/sft"

    @model_validator(mode="after")
    def validate_training_mode(self) -> "SFTConfig":
        if not self.bf16 or self.load_in_4bit or self.load_in_8bit:
            raise ValueError("SFT must use BF16 LoRA without 4/8-bit quantization")
        if not self.assistant_only_loss or self.packing:
            raise ValueError("SFT requires assistant-only loss with packing disabled")
        return self

    def gradient_accumulation_steps(self, world_size: int) -> int:
        if world_size < 1:
            raise ValueError("world_size must be positive")
        micro_batch = world_size * self.per_device_train_batch_size
        if self.global_batch_size % micro_batch:
            raise ValueError("global_batch_size must be divisible by world_size * per-device batch")
        return self.global_batch_size // micro_batch


def load_sft_config(path: str | Path = "configs/sft.yaml") -> SFTConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return SFTConfig(**data)


class RunError(RuntimeError):
    """Raised when a run would overwrite or mismatch auditable artifacts."""


REQUIRED_CHECKPOINT_FILES = (
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _resume_checkpoint(output_dir: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    checkpoints = list(output_dir.glob("checkpoint-*"))
    if value == "latest" and not checkpoints:
        raise RunError("no checkpoint exists for latest resume")
    if value == "latest":
        try:
            candidate = max(checkpoints, key=lambda path: int(path.name.split("-")[-1]))
        except ValueError as exc:
            raise RunError("checkpoint directory suffix must be an integer") from exc
    else:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = output_dir / candidate
    try:
        candidate.resolve().relative_to(output_dir.resolve())
    except (ValueError, FileNotFoundError) as exc:
        raise RunError("resume checkpoint must be inside the run output directory") from exc
    missing = [name for name in REQUIRED_CHECKPOINT_FILES if not (candidate / name).is_file()]
    has_single_rng = (candidate / "rng_state.pth").is_file()
    has_two_rank_rng = all(
        (candidate / f"rng_state_{rank}.pth").is_file() for rank in range(2)
    )
    if not has_single_rng and not has_two_rank_rng:
        missing.append("rng_state.pth or rng_state_{0,1}.pth")
    if missing:
        raise RunError(f"resume checkpoint is incomplete: {missing}")
    return candidate


def _fingerprint(
    config: SFTConfig, bundle: SFTDatasetBundle, train_size: int
) -> dict[str, Any]:
    return {
        "model": config.model,
        "train_size": train_size,
        "data_hashes": bundle.file_hashes,
        "manifest_sha256": bundle.manifest_sha256,
        "lora": config.lora.model_dump(),
        "max_length": config.max_length,
        "global_batch_size": config.global_batch_size,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.num_train_epochs,
    }


def run_sft_training(
    config: SFTConfig,
    bundle: SFTDatasetBundle,
    *,
    train_size: int,
    output_dir: str | Path,
    best_dir: str | Path,
    backend: Any = None,
    world_size: int = 1,
    rank: int = 0,
    max_steps: int | None = None,
    resume_from_checkpoint: str | None = None,
    git_commit: str = "unknown",
    environment: dict[str, Any] | None = None,
) -> Path:
    """Run or explicitly resume one auditable SFT job."""

    output = Path(output_dir)
    best = Path(best_dir)
    staging = output / "adapter-staging"
    runtime = backend or RealSFTBackend()
    existing_manifest = output / "run_manifest.json"
    resume = (
        _resume_checkpoint(output, resume_from_checkpoint)
        if resume_from_checkpoint
        else None
    )
    current_fingerprint = _fingerprint(config, bundle, train_size)

    if rank == 0:
        if resume is None and output.exists() and any(output.iterdir()):
            raise RunError("output directory is non-empty; use explicit resume")
        if resume is None and best.exists():
            raise RunError("best adapter directory already exists")
        if staging.exists():
            raise RunError("adapter staging directory exists; archive it before retrying")
        if resume is not None:
            if not existing_manifest.is_file():
                raise RunError("resume requires run_manifest.json")
            try:
                previous = json.loads(existing_manifest.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise RunError("resume manifest is not valid JSON") from exc
            if previous.get("fingerprint") != current_fingerprint:
                raise RunError("resume configuration or dataset does not match the original run")
            if best.exists():
                raise RunError("remove or archive the old best adapter before resume")

    spec = build_trainer_spec(config, output, world_size, max_steps)
    manifest: dict[str, Any] = {
        "status": "running",
        "started_at": _now(),
        "git_commit": git_commit,
        "fingerprint": current_fingerprint,
        "data": {
            "manifest_sha256": bundle.manifest_sha256,
            "file_hashes": bundle.file_hashes,
            "train_rows": len(bundle.train_rows),
            "val_rows": len(bundle.val_rows),
        },
        "training": {
            "global_batch_size": config.global_batch_size,
            "world_size": world_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps(world_size),
            "trainer_spec": asdict(spec),
        },
        "environment": environment or {},
        "resume_checkpoint": str(resume) if resume else None,
    }
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        _write_json(existing_manifest, manifest)
    runtime.barrier()

    request = TrainingRequest(config, bundle, output, staging, spec, resume)
    try:
        result = runtime.train(request)
        runtime.barrier()
        if rank == 0:
            missing = [
                name
                for name in ("adapter_config.json", "adapter_model.safetensors")
                if not (staging / name).is_file()
            ]
            if missing:
                raise RunError(f"best adapter is incomplete: {missing}")
            if any(not math.isfinite(float(value)) for value in result.metrics.values()):
                raise RunError("metrics contain NaN or Inf")
            _write_json(output / "metrics.json", result.metrics)
            staging.replace(best)
            manifest.update(
                status="complete",
                completed_at=_now(),
                best_checkpoint=result.best_checkpoint,
                metrics=result.metrics,
            )
            _write_json(existing_manifest, manifest)
    except Exception as exc:
        if rank == 0:
            manifest.update(
                status="failed",
                completed_at=_now(),
                error=f"{type(exc).__name__}: {exc}",
            )
            _write_json(existing_manifest, manifest)
        raise
    return best
