from dispute_agent.training.train_sft import load_sft_config


def test_sft_config_is_non_quantized_bf16_lora_with_stable_global_batch():
    cfg = load_sft_config("configs/sft.yaml")

    assert cfg.model == "Qwen/Qwen3-8B"
    assert cfg.bf16 is True
    assert cfg.load_in_4bit is False and cfg.load_in_8bit is False
    assert (cfg.lora.rank, cfg.lora.alpha, cfg.lora.dropout) == (32, 64, 0)
    assert cfg.lora.target_modules == "all-linear"
    assert cfg.assistant_only_loss is True and cfg.packing is False
    assert cfg.global_batch_size == 16
    assert cfg.gradient_accumulation_steps(world_size=2) == 8
    assert cfg.gradient_accumulation_steps(world_size=1) == 16
    assert cfg.output_root == "checkpoints/sft"
