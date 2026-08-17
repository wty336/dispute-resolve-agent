import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dispute_agent.training.grpo_runtime import GRPORunRequest, run_grpo_training


class FakeStore:
    def __init__(self):
        self.rollouts = []

    def query_rollouts(self):
        return self.rollouts

    def query_spans(self, rollout_id):
        return []


class FakeTrainer:
    def __init__(self, **kwargs):
        fake_agl.trainer_kwargs = kwargs
        self.store = kwargs["store"]

    def fit(self, agent, *, train_dataset, val_dataset):
        fake_agl.fit_train_dataset = train_dataset
        fake_agl.fit_val_dataset = val_dataset


class FakeAlgorithm:
    def __init__(self, config):
        fake_agl.verl_configs.append(config)


class FakeTracer:
    kind = "otel"


class FakeAdapter:
    kind = "llm_proxy_triplet"


class FakeStoreFactory(FakeStore):
    pass


fake_agl = SimpleNamespace(
    InMemoryLightningStore=FakeStoreFactory,
    VERL=FakeAlgorithm,
    Trainer=FakeTrainer,
    OtelTracer=FakeTracer,
    LlmProxyTraceToTriplet=FakeAdapter,
    rollout=lambda fn: fn,
    emit_object=lambda value: value,
    verl_configs=[],
)


def _write_data(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rows = {
        "grpo_train.jsonl": {
            "fact_instance_id": "fact-0",
            "case_id": "case-0",
            "split": "grpo_train",
            "observation": {
                "case_id": "case-0", "order_id": "o-0", "buyer_id": "b-0", "merchant_id": "m-0",
                "item_name": "测试商品", "order_amount": 100.0, "claim_type": "damaged",
                "buyer_claim": "商品破损", "buyer_requested_amount": 50.0, "merchant_response": "发货前完好",
                "chat_log": ["买家：商品破损"], "evidence": [{"evidence_id": "chat:0", "type": "聊天记录", "description": "反馈", "source": "buyer", "visible": True}],
            }, "messages": [], "metadata": {},
        },
        "grpo_val.jsonl": {
            "fact_instance_id": "fact-1", "case_id": "case-1", "split": "grpo_val",
            "observation": {
                "case_id": "case-1", "order_id": "o-1", "buyer_id": "b-1", "merchant_id": "m-1",
                "item_name": "测试商品", "order_amount": 100.0, "claim_type": "damaged",
                "buyer_claim": "商品破损", "buyer_requested_amount": 50.0, "merchant_response": "发货前完好",
                "chat_log": ["买家：商品破损"], "evidence": [{"evidence_id": "chat:1", "type": "聊天记录", "description": "反馈", "source": "buyer", "visible": True}],
            }, "messages": [], "metadata": {},
        },
    }
    hidden = {
        "grpo_train.ground_truth.jsonl": {"case_id": "case-0", "ground_truth": {"case_id": "case-0", "true_liability": "merchant", "true_loss": 50.0, "reasonable_compensation_range": [40, 60], "buyer_strategy": "honest", "merchant_strategy": "honest", "should_escalate": False}},
        "grpo_val.ground_truth.jsonl": {"case_id": "case-1", "ground_truth": {"case_id": "case-1", "true_liability": "merchant", "true_loss": 50.0, "reasonable_compensation_range": [40, 60], "buyer_strategy": "honest", "merchant_strategy": "honest", "should_escalate": False}},
    }
    import hashlib
    hashes = {}
    for name, row in {**rows, **hidden}.items():
        path = root / name
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps({"counts": {"grpo_train": 1, "grpo_val": 1}, "file_hashes": hashes}), encoding="utf-8")
    return root


def test_real_run_builds_verl_and_trainer(tmp_path, monkeypatch):
    fake_agl.verl_configs.clear()
    data_dir = _write_data(tmp_path / "data")
    result = run_grpo_training(
        GRPORunRequest(
            config_path=Path("configs/grpo.yaml"),
            data_dir=data_dir,
            output_root=tmp_path / "outputs",
            run_id="test-run",
            profile="smoke",
            curriculum_phase=1,
        ),
        agl_module=fake_agl,
        dry_run=False,
        validate_adapter=False,
    )

    assert fake_agl.verl_configs[0]["actor_rollout_ref"]["rollout"]["n"] == 4
    assert fake_agl.trainer_kwargs["n_runners"] == 1
    assert fake_agl.trainer_kwargs["tracer"].kind == "otel"
    assert fake_agl.trainer_kwargs["adapter"].kind == "llm_proxy_triplet"
    assert len(fake_agl.fit_train_dataset) == 1
    assert result.manifest_path.exists()


def test_dry_run_writes_plan_without_gpu_imports(tmp_path, monkeypatch):
    data_dir = _write_data(tmp_path / "data")
    original_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"agentlightning", "verl", "ray", "vllm"}:
            raise AssertionError(f"GPU package imported during dry-run: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    result = run_grpo_training(
        GRPORunRequest(Path("configs/grpo.yaml"), data_dir, tmp_path / "outputs", "dry", "smoke", 1),
        dry_run=True,
    )
    assert result.dry_run
    assert (result.run_dir / "resolved_config.yaml").exists()
    assert (result.run_dir / "verl_config.yaml").exists()


def test_max_steps_is_injected_into_verl_config(tmp_path):
    fake_agl.verl_configs.clear()
    data_dir = _write_data(tmp_path / "data")
    run_grpo_training(
        GRPORunRequest(Path("configs/grpo.yaml"), data_dir, tmp_path / "outputs", "steps", "smoke", 1, max_steps=25),
        agl_module=fake_agl,
        validate_adapter=False,
    )
    assert fake_agl.verl_configs[-1]["trainer"]["total_training_steps"] == 25


def test_invalid_input_adapter_writes_failed_manifest(tmp_path):
    data_dir = _write_data(tmp_path / "data")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "Qwen/Qwen3-8B",
        "r": 16,
        "lora_alpha": 64,
        "lora_dropout": 0.0,
    }), encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"fixture")
    (adapter / "run_manifest.json").write_text(json.dumps({
        "training": {"trainer_spec": {"peft_args": {"target_modules": "all-linear"}}}
    }), encoding="utf-8")
    output_root = tmp_path / "outputs"
    with pytest.raises(Exception, match="r mismatch"):
        run_grpo_training(
            GRPORunRequest(Path("configs/grpo.yaml"), data_dir, output_root, "invalid", "smoke", 1, input_adapter=adapter),
            agl_module=fake_agl,
        )
    manifest = json.loads((output_root / "invalid" / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
