"""Configuration and orchestration for TRL BF16 LoRA SFT."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


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
