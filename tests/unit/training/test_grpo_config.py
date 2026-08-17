import pytest
from pydantic import ValidationError

from dispute_agent.training.grpo_config import GRPOConfig, load_grpo_config


@pytest.fixture
def load_config():
    return lambda: load_grpo_config("configs/grpo.yaml")


def test_grpo_continues_from_sft_adapter(load_config):
    cfg = load_config()
    assert cfg.algorithm.adv_estimator == "grpo"
    assert cfg.actor_rollout_ref.model.lora_adapter_path.endswith("sft-1500-best")
    assert cfg.actor_rollout_ref.model.lora_rank == 32
    assert cfg.actor_rollout_ref.model.lora_alpha == 64
    assert cfg.actor_rollout_ref.rollout.n == 4
    assert cfg.actor_rollout_ref.rollout.tensor_model_parallel_size == 2
    assert cfg.trainer.n_gpus_per_node == 2
    assert cfg.quantization is None


def test_grpo_config_resolves_stable_two_gpu_lora_contract():
    cfg = load_grpo_config("configs/grpo.yaml")
    resolved = cfg.to_verl_config(profile="smoke", output_dir="artifacts/grpo/test")

    assert cfg.quantization is None
    assert resolved["algorithm"]["adv_estimator"] == "grpo"
    assert resolved["actor_rollout_ref"]["model"] == {
        "path": "Qwen/Qwen3-8B",
        "lora_adapter_path": "checkpoints/sft/sft-1500-best",
        "lora_rank": 32,
        "lora_alpha": 64,
        "target_modules": "all-linear",
        "enable_gradient_checkpointing": True,
        "use_remove_padding": True,
    }
    assert resolved["actor_rollout_ref"]["rollout"]["n"] == 4
    assert resolved["actor_rollout_ref"]["rollout"]["tensor_model_parallel_size"] == 2
    assert resolved["actor_rollout_ref"]["rollout"]["load_format"] == "safetensors"
    assert resolved["actor_rollout_ref"]["actor"]["ppo_micro_batch_size_per_gpu"] == 1
    assert resolved["actor_rollout_ref"]["actor"]["use_kl_loss"] is True
    assert resolved["trainer"]["n_gpus_per_node"] == 2
    assert resolved["trainer"]["total_epochs"] == 1
    assert resolved["agentlightning"]["trace_aggregator"]["level"] == "trajectory"


def test_grpo_config_rejects_quantization_or_wrong_group_size():
    raw = load_grpo_config("configs/grpo.yaml").model_dump(mode="python")
    raw["quantization"] = "4bit"
    with pytest.raises(ValidationError, match="BF16 LoRA"):
        GRPOConfig.model_validate(raw)

    raw["quantization"] = None
    raw["actor_rollout_ref"]["rollout"]["n"] = 2
    with pytest.raises(ValidationError, match="n=4"):
        GRPOConfig.model_validate(raw)
