"""Agent Lightning + VERL GRPO configuration parsing and validation."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class AlgorithmConfig(BaseModel):
    adv_estimator: str = "grpo"


class GRPOModelConfig(BaseModel):
    lora_adapter_path: str
    lora_rank: int = 32
    lora_alpha: int = 64


class GRPORolloutConfig(BaseModel):
    n: int = 4
    tensor_model_parallel_size: int = 2
    max_tokens_per_round: int = 384
    max_episode_tokens: int = 1280


class ActorRolloutRefConfig(BaseModel):
    model: GRPOModelConfig
    rollout: GRPORolloutConfig


class TrainerConfig(BaseModel):
    n_gpus_per_node: int = 2


class CurriculumConfig(BaseModel):
    phase1_tools: list[str] = Field(
        default_factory=lambda: ["check_logistics", "check_buyer_history", "check_merchant_history"]
    )
    phase1_max_rounds: int = 3
    phase2_max_rounds: int = 5


class MonitorConfig(BaseModel):
    window: int = 50
    max_zero_variance_ratio: float = 0.30


class GRPOConfig(BaseModel):
    algorithm: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    actor_rollout_ref: ActorRolloutRefConfig
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    quantization: str | None = None
    curriculum: CurriculumConfig = Field(default_factory=CurriculumConfig)
    monitor: MonitorConfig = Field(default_factory=MonitorConfig)


def load_grpo_config(path: str | Path = "configs/grpo.yaml") -> GRPOConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return GRPOConfig(**data)
