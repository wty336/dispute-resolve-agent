from pathlib import Path
import tomllib


ROOT = Path(__file__).parents[1]


def test_python_and_training_contract_are_declared():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    assert project["project"]["requires-python"] == ">=3.11,<3.12"
    pins = (ROOT / "constraints/train.txt").read_text("utf-8")
    for pin in (
        "torch==2.8.0",
        "vllm==0.10.2",
        "verl==0.5.0",
        "agentlightning==0.3.0",
        "openai-agents==0.6.0",
    ):
        assert pin in pins
