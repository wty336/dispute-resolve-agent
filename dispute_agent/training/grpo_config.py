"""严格的项目配置以及 Agent Lightning/verl 配置渲染。"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AlgorithmConfig(StrictModel):
    adv_estimator: Literal["grpo"] = "grpo"
    use_kl_in_reward: bool = False


class GRPOModelConfig(StrictModel):
    path: str = "Qwen/Qwen3-8B"
    lora_adapter_path: str
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    target_modules: str = "all-linear"
    enable_gradient_checkpointing: bool = True
    use_remove_padding: bool = True


class GRPORolloutConfig(StrictModel):
    name: Literal["vllm"] = "vllm"
    n: int = 4
    tensor_model_parallel_size: int = 2
    max_tokens_per_round: int = 384
    max_episode_tokens: int = 1280
    gpu_memory_utilization: float = Field(default=0.45, gt=0, lt=1)
    max_num_seqs: int = 8
    load_format: Literal["safetensors"] = "safetensors"
    tool_format: Literal["hermes"] = "hermes"


class ActorConfig(StrictModel):
    ppo_mini_batch_size: int = 8
    ppo_micro_batch_size_per_gpu: int = 1
    learning_rate: float = 1e-5
    use_kl_loss: bool = True
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"
    param_offload: bool = True
    optimizer_offload: bool = True


class ActorRolloutRefConfig(StrictModel):
    model: GRPOModelConfig
    rollout: GRPORolloutConfig = Field(default_factory=GRPORolloutConfig)
    actor: ActorConfig = Field(default_factory=ActorConfig)


class DataConfig(StrictModel):
    train_batch_size: int = 2
    max_prompt_length: int = 2048
    max_response_length: int = 384


class TrainerConfig(StrictModel):
    n_gpus_per_node: int = 2
    nnodes: int = 1
    total_epochs: int = 1
    save_freq: int = 1
    test_freq: int = 10
    val_before_train: bool = True
    logger: list[str] = Field(default_factory=lambda: ["console", "wandb"])


class AgentLightningConfig(StrictModel):
    n_runners: int = 1
    trajectory_max_prompt_length: int = 2048
    trajectory_max_response_length: int = 4096


class CurriculumConfig(StrictModel):
    phase1_tools: list[str] = Field(default_factory=lambda: [
        "check_logistics", "check_buyer_history", "check_merchant_history"
    ])
    phase1_max_rounds: int = 3
    phase2_max_rounds: int = 5


class MonitorConfig(StrictModel):
    window: int = 50
    max_zero_variance_ratio: float = 0.30


class GRPOConfig(StrictModel):
    algorithm: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    actor_rollout_ref: ActorRolloutRefConfig
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    agentlightning: AgentLightningConfig = Field(default_factory=AgentLightningConfig)
    curriculum: CurriculumConfig = Field(default_factory=CurriculumConfig)
    monitor: MonitorConfig = Field(default_factory=MonitorConfig)
    quantization: str | None = None

    @model_validator(mode="after")
    def validate_training_contract(self) -> "GRPOConfig":
        model = self.actor_rollout_ref.model
        rollout = self.actor_rollout_ref.rollout
        if self.quantization is not None or model.lora_rank != 32 or model.lora_dropout != 0:
            raise ValueError("GRPO must use BF16 LoRA r=32 without quantization")
        if model.lora_alpha != 64 or model.target_modules != "all-linear":
            raise ValueError("GRPO LoRA must use alpha=64 and all-linear targets")
        if rollout.n != 4:
            raise ValueError("Agentic GRPO requires rollout n=4")
        if rollout.tensor_model_parallel_size != 2 or self.trainer.n_gpus_per_node != 2:
            raise ValueError("formal topology requires exactly two GPUs and TP=2")
        return self

    def to_verl_config(self, *, profile: str, output_dir: str | Path) -> dict:
        if profile not in {"smoke", "formal"}:
            raise ValueError(f"unknown profile: {profile}")
        model = self.actor_rollout_ref.model
        rollout = self.actor_rollout_ref.rollout
        actor = self.actor_rollout_ref.actor
        trainer = self.trainer
        return {
            "algorithm": {
                "adv_estimator": self.algorithm.adv_estimator,
                "use_kl_in_reward": self.algorithm.use_kl_in_reward,
            },
            "data": {
                "train_batch_size": self.data.train_batch_size,
                "max_prompt_length": self.data.max_prompt_length,
                "max_response_length": self.data.max_response_length,
            },
            "actor_rollout_ref": {
                "model": {
                    "path": model.path,
                    "lora_adapter_path": model.lora_adapter_path,
                    "lora_rank": model.lora_rank,
                    "lora_alpha": model.lora_alpha,
                    "target_modules": model.target_modules,
                    "enable_gradient_checkpointing": model.enable_gradient_checkpointing,
                    "use_remove_padding": model.use_remove_padding,
                },
                "rollout": {
                    "name": rollout.name,
                    "n": rollout.n,
                    "tensor_model_parallel_size": rollout.tensor_model_parallel_size,
                    "gpu_memory_utilization": rollout.gpu_memory_utilization,
                    "max_num_seqs": rollout.max_num_seqs,
                    "load_format": rollout.load_format,
                    "multi_turn": {"format": rollout.tool_format},
                },
                "actor": {
                    "ppo_mini_batch_size": actor.ppo_mini_batch_size,
                    "ppo_micro_batch_size_per_gpu": actor.ppo_micro_batch_size_per_gpu,
                    "optim": {"lr": actor.learning_rate},
                    "use_kl_loss": actor.use_kl_loss,
                    "kl_loss_coef": actor.kl_loss_coef,
                    "kl_loss_type": actor.kl_loss_type,
                    "entropy_coeff": 0,
                    "fsdp_config": {
                        "param_offload": actor.param_offload,
                        "optimizer_offload": actor.optimizer_offload,
                    },
                },
                "ref": {
                    "log_prob_micro_batch_size_per_gpu": 1,
                    "fsdp_config": {"param_offload": True},
                },
            },
            "agentlightning": {
                "trace_aggregator": {
                    "level": "trajectory",
                    "trajectory_max_prompt_length": self.agentlightning.trajectory_max_prompt_length,
                    "trajectory_max_response_length": self.agentlightning.trajectory_max_response_length,
                }
            },
            "trainer": {
                "n_gpus_per_node": trainer.n_gpus_per_node,
                "nnodes": trainer.nnodes,
                "total_epochs": trainer.total_epochs,
                "save_freq": 1 if profile == "smoke" else trainer.save_freq,
                "test_freq": 1 if profile == "smoke" else trainer.test_freq,
                "val_before_train": trainer.val_before_train,
                "logger": trainer.logger,
                "project_name": "dispute-resolve-agent",
                "experiment_name": Path(output_dir).name,
                "default_local_dir": str(Path(output_dir) / "checkpoints"),
                "resume_mode": "disable",
                "critic_warmup": 0,
            },
        }


def load_grpo_config(path: str | Path = "configs/grpo.yaml") -> GRPOConfig:
    return GRPOConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
