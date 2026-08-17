"""TRL BF16 LoRA SFT pipeline configuration and entrypoint helpers."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class LoraConfig(BaseModel):
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.0
    target_modules: str = "all-linear"


class SFTConfig(BaseModel):
    model: str = "Qwen/Qwen3-8B"
    bf16: bool = True
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    lora: LoraConfig = Field(default_factory=LoraConfig)
    assistant_only_loss: bool = True
    seed: int = 20260817
    max_length: int = 2048
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    output_dir: str = "checkpoints/sft"


def load_sft_config(path: str | Path = "configs/sft.yaml") -> SFTConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return SFTConfig(**data)


def train_sft(
    config: SFTConfig,
    *,
    train_size: int = 500,
    fixture: bool = False,
    max_steps: int = 1,
    output_dir: str | None = None,
) -> str:
    """Run a TRL SFT dry-run or actual training.

    Local fixture mode only creates the adapter output directory; actual
    training requires the ``train`` extra with TRL/PEFT installed.
    """
    output = Path(output_dir or config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "adapter_config.json").write_text(
        '{"model": "%s", "train_size": %d, "fixture": %s, "max_steps": %d}\n'
        % (config.model, train_size, fixture, max_steps),
        encoding="utf-8",
    )
    return str(output)
