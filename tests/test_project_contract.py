from pathlib import Path
import tomllib


ROOT = Path(__file__).parents[1]


def test_python_and_training_contract_are_declared():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    assert project["project"]["requires-python"] == ">=3.11,<3.12"
    pins = (ROOT / "constraints/train.txt").read_text("utf-8")
    for pin in (
        "torch==2.8.0",
        "torchvision==0.23.0",
        "transformers==4.55.4",
        "peft==0.18.1",
        "accelerate==1.10.1",
        "flash-attn==2.8.3",
        "vllm==0.10.2",
        "verl==0.5.0",
        "agentlightning==0.3.0",
        "openai-agents==0.6.0",
    ):
        assert pin in pins

    train_extra = project["project"]["optional-dependencies"]["train"]
    for pin in (
        "torch==2.8.0",
        "torchvision==0.23.0",
        "transformers==4.55.4",
        "peft==0.18.1",
        "accelerate==1.10.1",
        "flash-attn==2.8.3",
        "vllm==0.10.2",
        "verl==0.5.0",
        "agentlightning==0.3.0",
        "openai-agents==0.6.0",
    ):
        assert pin in train_extra

    sft_pins = (ROOT / "constraints/sft.txt").read_text("utf-8")
    for pin in (
        "torch==2.8.0",
        "transformers==4.57.6",
        "trl==1.3.0",
        "peft==0.18.1",
        "accelerate==1.12.0",
        "datasets==4.7.0",
    ):
        assert pin in sft_pins

    sft_extra = project["project"]["optional-dependencies"]["sft"]
    assert any(item.startswith("trl") for item in sft_extra)
    assert not any(item.startswith(("vllm", "verl", "agentlightning")) for item in sft_extra)
