"""TRL/PEFT runtime kept lazy so local contract checks stay lightweight."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dispute_agent.training.sft_data import PreflightReport, preflight_dataset
from dispute_agent.training.sft_dataset import SFTDatasetBundle


@dataclass(frozen=True)
class TrainerSpec:
    training_args: dict[str, Any]
    peft_args: dict[str, Any]


@dataclass(frozen=True)
class BackendResult:
    metrics: dict[str, float]
    best_checkpoint: str | None


@dataclass(frozen=True)
class TrainingRequest:
    config: Any
    bundle: SFTDatasetBundle
    output_dir: Path
    adapter_staging_dir: Path
    trainer_spec: TrainerSpec
    resume_checkpoint: Path | None


def build_trainer_spec(
    config: Any, output_dir: Path, world_size: int, max_steps: int | None
) -> TrainerSpec:
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    interval_strategy = "steps" if max_steps is not None else config.eval_strategy
    training_args = {
        "output_dir": str(output_dir),
        "bf16": True,
        "tf32": True,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps(world_size),
        "num_train_epochs": config.num_train_epochs,
        "max_steps": -1 if max_steps is None else max_steps,
        "learning_rate": config.learning_rate,
        "lr_scheduler_type": config.lr_scheduler_type,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "max_grad_norm": config.max_grad_norm,
        "gradient_checkpointing": config.gradient_checkpointing,
        "assistant_only_loss": True,
        "packing": False,
        "max_length": config.max_length,
        "eval_strategy": interval_strategy,
        "save_strategy": interval_strategy,
        "save_total_limit": config.save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "logging_steps": config.logging_steps,
        "report_to": "none",
        "seed": config.seed,
        "data_seed": config.data_seed,
        "ddp_find_unused_parameters": False,
        "model_init_kwargs": {
            "dtype": "bfloat16",
            "attn_implementation": config.attention_implementation,
            "use_cache": False,
        },
    }
    if max_steps is not None:
        training_args.update(eval_steps=max_steps, save_steps=max_steps)
    peft_args = {
        "r": config.lora.rank,
        "lora_alpha": config.lora.alpha,
        "lora_dropout": config.lora.dropout,
        "target_modules": config.lora.target_modules,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    return TrainerSpec(training_args=training_args, peft_args=peft_args)


def prepare_training_tokenizer(model_name: str):
    from transformers import AutoTokenizer
    from trl.chat_template_utils import get_training_chat_template

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    template = get_training_chat_template(processing_class=tokenizer) or tokenizer.chat_template
    if not template or "{% generation %}" not in template or "{% endgeneration %}" not in template:
        raise RuntimeError("Qwen training template lacks assistant generation markers")
    tokenizer.chat_template = "{% set enable_thinking = false %}\n" + template
    return tokenizer


class RealSFTBackend:
    """Two-process CUDA backend for formal BF16 LoRA training."""

    def __init__(self) -> None:
        from accelerate import PartialState

        self.state = PartialState()

    def barrier(self) -> None:
        self.state.wait_for_everyone()

    def preflight(self, config: Any, bundle: SFTDatasetBundle) -> PreflightReport:
        tokenizer = prepare_training_tokenizer(config.model)
        return preflight_dataset(
            bundle.train_rows + bundle.val_rows,
            tokenizer,
            config.max_length,
        )

    def train(self, request: TrainingRequest) -> BackendResult:
        import math

        from datasets import Dataset
        from peft import LoraConfig as PeftLoraConfig
        import torch
        from trl import SFTConfig as TRLSFTConfig, SFTTrainer

        self._validate_cuda(request.config)
        tokenizer = prepare_training_tokenizer(request.config.model)
        preflight_dataset(
            request.bundle.train_rows + request.bundle.val_rows,
            tokenizer,
            request.config.max_length,
        )
        training_args = dict(request.trainer_spec.training_args)
        training_args["model_init_kwargs"] = {
            **training_args["model_init_kwargs"],
            "dtype": torch.bfloat16,
        }
        trainer = SFTTrainer(
            model=request.config.model,
            args=TRLSFTConfig(**training_args),
            train_dataset=Dataset.from_list(request.bundle.train_rows),
            eval_dataset=Dataset.from_list(request.bundle.val_rows),
            processing_class=tokenizer,
            peft_config=PeftLoraConfig(**request.trainer_spec.peft_args),
        )
        train_result = trainer.train(
            resume_from_checkpoint=(
                str(request.resume_checkpoint) if request.resume_checkpoint else None
            )
        )
        metrics = {**train_result.metrics, **trainer.evaluate()}
        required = ("train_loss", "eval_loss")
        if any(
            name not in metrics or not math.isfinite(float(metrics[name]))
            for name in required
        ):
            raise RuntimeError("training produced missing or non-finite losses")
        if trainer.is_world_process_zero():
            trainer.save_state()
            trainer.save_model(str(request.adapter_staging_dir))
            tokenizer.save_pretrained(str(request.adapter_staging_dir))
        return BackendResult(
            metrics={
                key: float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            },
            best_checkpoint=trainer.state.best_model_checkpoint,
        )

    def _validate_cuda(self, config: Any) -> None:
        import torch

        if self.state.num_processes != 2:
            raise RuntimeError("formal SFT requires exactly two Accelerate processes")
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError("formal SFT requires two visible CUDA devices")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("visible CUDA device does not support BF16")
        if config.load_in_4bit or config.load_in_8bit:
            raise RuntimeError("quantized loading is forbidden for this SFT run")
