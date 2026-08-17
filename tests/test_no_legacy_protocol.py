from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_legacy_single_step_protocol_is_removed():
    forbidden_files = [
        "dispute_agent/platform_agent.py", "dispute_agent/data_generation.py",
        "dispute_agent/verl_reward.py", "dispute_agent/tools.py",
        "dispute_agent/environment.py", "scripts/train_sft_ms_swift.sh",
        "scripts/train_grpo_verl.sh", "scripts/train_rl.py",
        "scripts/prepare_verl_data.py", "scripts/evaluate_model.py", "main.py",
    ]
    assert not [path for path in forbidden_files if (ROOT / path).exists()]
    source = "\n".join(p.read_text("utf-8") for p in (ROOT / "dispute_agent").rglob("*.py"))
    assert "Qwen2.5-7B" not in source
    assert '"role": "user", "content": tool_result' not in source
