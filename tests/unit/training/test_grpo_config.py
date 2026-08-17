import pytest

from dispute_agent.training.grpo_config import load_grpo_config


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
