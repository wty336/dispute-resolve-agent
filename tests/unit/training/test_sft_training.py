import json
from pathlib import Path

import pytest

from dispute_agent.training.sft_dataset import SFTDatasetBundle
from dispute_agent.training.sft_runtime import BackendResult, build_trainer_spec
from dispute_agent.training.train_sft import RunError, load_sft_config, run_sft_training


class FakeBackend:
    def __init__(self):
        self.requests = []

    def barrier(self):
        return None

    def train(self, request):
        self.requests.append(request)
        request.adapter_staging_dir.mkdir(parents=True, exist_ok=False)
        (request.adapter_staging_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        (request.adapter_staging_dir / "adapter_model.safetensors").write_bytes(b"adapter")
        return BackendResult(
            metrics={"train_loss": 0.8, "eval_loss": 0.7, "train_runtime": 1.0},
            best_checkpoint=str(request.output_dir / "checkpoint-1"),
        )


def _bundle() -> SFTDatasetBundle:
    row = {
        "case_id": "case-1",
        "messages": [
            {"role": "user", "content": "case"},
            {"role": "assistant", "content": "final"},
        ],
        "tools": [],
    }
    return SFTDatasetBundle(
        train_rows=[row] * 500,
        val_rows=[row] * 150,
        file_hashes={"sft_train.jsonl": "train-hash", "sft_val.jsonl": "val-hash"},
        manifest_sha256="manifest-hash",
    )


def test_training_writes_complete_manifest_and_requires_explicit_matching_resume(tmp_path):
    cfg = load_sft_config("configs/sft.yaml")
    spec = build_trainer_spec(cfg, output_dir=tmp_path / "spec", world_size=2, max_steps=None)
    assert spec.training_args["gradient_accumulation_steps"] == 8
    assert spec.training_args["bf16"] is True
    assert spec.training_args["packing"] is False
    assert spec.peft_args == {
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.0,
        "target_modules": "all-linear",
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }

    backend = FakeBackend()
    output = tmp_path / "sft-500"
    best = tmp_path / "sft-500-best"
    result = run_sft_training(
        cfg,
        _bundle(),
        train_size=500,
        output_dir=output,
        best_dir=best,
        backend=backend,
        world_size=2,
        rank=0,
        git_commit="abc123",
        environment={"torch": "2.8.0", "gpu_names": ["RTX 4090", "RTX 4090"]},
    )
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert result == best
    assert manifest["status"] == "complete"
    assert manifest["data"]["manifest_sha256"] == "manifest-hash"
    assert manifest["training"]["global_batch_size"] == 16
    assert json.loads((output / "metrics.json").read_text(encoding="utf-8"))["eval_loss"] == 0.7

    with pytest.raises(RunError, match="non-empty"):
        run_sft_training(
            cfg,
            _bundle(),
            train_size=500,
            output_dir=output,
            best_dir=best,
            backend=backend,
            world_size=2,
            rank=0,
            git_commit="abc123",
            environment={},
        )

    manifest["status"] = "failed"
    (output / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    checkpoint = output / "checkpoint-1"
    checkpoint.mkdir()
    for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (checkpoint / name).write_bytes(b"state")
    best.rename(tmp_path / "old-best")
    resumed = run_sft_training(
        cfg,
        _bundle(),
        train_size=500,
        output_dir=output,
        best_dir=best,
        backend=backend,
        world_size=2,
        rank=0,
        resume_from_checkpoint="latest",
        git_commit="abc123",
        environment={},
    )
    assert resumed == best
    assert backend.requests[-1].resume_checkpoint == checkpoint
