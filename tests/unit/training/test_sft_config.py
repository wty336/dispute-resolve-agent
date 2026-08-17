import pytest

from dispute_agent.training.train_sft import load_sft_config


@pytest.fixture
def load_config():
    return lambda: load_sft_config("configs/sft.yaml")


def test_sft_is_bf16_lora_not_qlora(load_config):
    cfg = load_config()
    assert cfg.model == "Qwen/Qwen3-8B"
    assert cfg.bf16 is True and cfg.load_in_4bit is False and cfg.load_in_8bit is False
    assert (cfg.lora.rank, cfg.lora.alpha, cfg.lora.dropout) == (32, 64, 0)
    assert cfg.lora.target_modules == "all-linear"
    assert cfg.assistant_only_loss is True
