# Agent Lightning 0.3 + verl 0.5 Agentic GRPO Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the placeholder Agentic GRPO path with a real Agent Lightning 0.3 + verl 0.5 BF16 LoRA training entrypoint that can be prepared locally and proven with one genuine update on two RTX 4090 GPUs.

**Architecture:** Keep the existing OpenAI Agents SDK runtime, deterministic environment, and `RewardEngine`. Add a manifest-verified GRPO episode source, a dependency-light rollout core wrapped lazily with `@agl.rollout`, and a training runtime that builds `agl.VERL`/`agl.Trainer`, records auditable artifacts, and refuses to mark unexecuted Phase 0 gates as passed.

**Tech Stack:** Python 3.11, Pydantic 2, OpenAI Agents SDK 0.6.0, Agent Lightning 0.3.0, verl 0.5.0, vLLM 0.10.2, PyTorch 2.8.0, PEFT LoRA, pytest.

---

## Scope and file map

| File | Responsibility |
|---|---|
| `scripts/generate_data.py` | Add structured public observations only to GRPO rows |
| `dispute_agent/training/grpo_dataset.py` | Verify manifests, build leakage-safe tasks, and create fresh Episodes |
| `dispute_agent/training/grpo_config.py` | Strict project config and conversion to Agent Lightning/verl config |
| `dispute_agent/agent/tools.py` | Filter investigation tools without changing tool semantics |
| `dispute_agent/agent/runtime.py` | Accept curriculum limits and derived per-turn token budget |
| `dispute_agent/training/lightning_agent.py` | Dependency-light rollout core and lazy real `@agl.rollout` binding |
| `dispute_agent/training/grpo_runtime.py` | Trainer creation, run manifests, store export, and resume guards |
| `scripts/train_agentic_grpo.py` | One dry-run/smoke/formal CLI |
| `dispute_agent/training/phase0.py` | Phase 0 statuses, evidence validation, and checkpoint checks |
| `scripts/phase0_smoke.py` | Local `not_run` report or real server Phase 0 orchestration |
| `configs/grpo.yaml` | Full two-GPU BF16 LoRA/GRPO configuration |
| `pyproject.toml` | Installable `train` extra aligned with the stable matrix |
| `constraints/train.txt` | Initial reproducible stable training matrix |
| `tests/...` | Five focused local contract areas; no model downloads |
| `README.md`, `docs/experiments.md`, `docs/resume-evidence-checklist.md` | Accurate commands and evidence claims |

Local work stops after Task 8. Task 9 runs only on Ubuntu with two RTX 4090 GPUs.

### Task 1: Make GRPO public rows reconstructable

**Files:**
- Modify: `scripts/generate_data.py`
- Modify: `tests/leakage/test_dataset_leakage.py`

- [ ] **Step 1: Write the failing public-row contract test**

Add this test to `tests/leakage/test_dataset_leakage.py`:

```python
import json

from scripts.generate_data import _render_row


def test_grpo_row_contains_structured_public_observation_only():
    instance = generate_fact_instances(seed=20260817, n=1)[0]
    instance.split = "grpo_train"

    row = _render_row(instance)

    assert row["observation"] == instance.observation.model_dump(mode="json")
    serialized = json.dumps(row["observation"], ensure_ascii=False)
    for hidden in (
        "true_liability",
        "true_loss",
        "reasonable_compensation_range",
        "should_escalate",
        "evidence_authenticity",
    ):
        assert hidden not in serialized


def test_sft_row_does_not_gain_an_unused_observation_column():
    instance = generate_fact_instances(seed=20260817, n=1)[0]
    instance.split = "sft_train"

    assert "observation" not in _render_row(instance)
```

- [ ] **Step 2: Run the two tests and confirm the GRPO assertion fails**

Run:

```powershell
python -m pytest tests/leakage/test_dataset_leakage.py -q
```

Expected: the GRPO test fails with `KeyError: 'observation'`; the existing leakage test and the SFT-row test pass.

- [ ] **Step 3: Add the structured field only for GRPO splits**

In `scripts/generate_data.py`, build the row first and conditionally add `observation`:

```python
def _render_row(instance: FactInstance, profile: SFTProfile | None = None) -> dict:
    if profile is None:
        profile = SFTProfile(category="environment_task")
    if profile.category in {"direct", "multi_tool"}:
        instance.ground_truth.tool_timeout_rate = 0.0
        instance.ground_truth.tool_missing_rate = 0.0
    _apply_edge_case(instance, profile)
    tool_plan = [(name, _tool_arguments(instance, name)) for name in profile.tool_names]
    tool_result_overrides = (
        {0: "工具返回格式非法：缺少可核验证据字段，结果不可采信"}
        if profile.edge_case == "illegal_result_recovery"
        else None
    )
    metadata = {
        **instance.metadata,
        "sft_category": profile.category if profile.category != "environment_task" else None,
        "edge_case": profile.edge_case,
        "tool_call_count": len(profile.tool_names),
        "recovered_from_invalid_result": profile.edge_case == "illegal_result_recovery",
    }
    row = {
        "fact_instance_id": instance.fact_instance_id,
        "case_id": instance.case_id,
        "split": instance.split,
        "ood_bucket": instance.ood_bucket,
        "messages": render_sft_trace(
            instance,
            tool_plan=tool_plan,
            tool_result_overrides=tool_result_overrides,
            decision=_make_decision(instance, profile.tool_names),
        ),
        "_ground_truth": instance.ground_truth.model_dump(mode="json"),
        "_fact_fingerprint": fact_fingerprint(instance),
        "metadata": metadata,
    }
    if instance.split in {"grpo_train", "grpo_val"}:
        row["observation"] = instance.observation.model_dump(mode="json")
    return row
```

- [ ] **Step 4: Run only the leakage file**

Run:

```powershell
python -m pytest tests/leakage/test_dataset_leakage.py -q
```

Expected: all tests in that file pass.

- [ ] **Step 5: Commit the data contract**

```bash
git add scripts/generate_data.py tests/leakage/test_dataset_leakage.py
git commit -m "feat: persist public observations for grpo"
```

### Task 2: Add the manifest-verified GRPO dataset and Episode factory

**Files:**
- Create: `dispute_agent/training/grpo_dataset.py`
- Create: `tests/unit/training/test_grpo_dataset.py`

- [ ] **Step 1: Write one combined integrity/isolation test**

Create `tests/unit/training/test_grpo_dataset.py`:

```python
import hashlib
import json
from pathlib import Path

import pytest

from dispute_agent.training.grpo_dataset import (
    GRPODatasetError,
    load_grpo_dataset,
)


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _observation(index: int) -> dict:
    return {
        "case_id": f"case-{index}",
        "order_id": f"o-{index}",
        "buyer_id": f"b-{index}",
        "merchant_id": f"m-{index}",
        "item_name": "测试商品",
        "order_amount": 100.0,
        "claim_type": "damaged",
        "buyer_claim": "商品破损",
        "buyer_requested_amount": 50.0,
        "merchant_response": "发货前完好",
        "chat_log": ["买家：商品破损"],
        "evidence": [{
            "evidence_id": f"chat:{index}",
            "type": "聊天记录",
            "description": "买家反馈破损",
            "source": "buyer",
            "visible": True,
        }],
    }


def _ground_truth(index: int) -> dict:
    return {
        "case_id": f"case-{index}",
        "true_liability": "merchant",
        "true_loss": 50.0,
        "reasonable_compensation_range": [40.0, 60.0],
        "buyer_strategy": "honest",
        "merchant_strategy": "honest",
        "should_escalate": False,
    }


def _dataset_dir(root: Path) -> Path:
    public_hashes = {}
    counts = {"grpo_train": 2, "grpo_val": 1}
    cursor = 0
    for split, count in counts.items():
        public = []
        hidden = []
        for index in range(cursor, cursor + count):
            public.append({
                "fact_instance_id": f"fact-{index}",
                "case_id": f"case-{index}",
                "split": split,
                "observation": _observation(index),
                "messages": [],
                "metadata": {},
            })
            hidden.append({"case_id": f"case-{index}", "ground_truth": _ground_truth(index)})
        cursor += count
        for suffix, rows in (("jsonl", public), ("ground_truth.jsonl", hidden)):
            name = f"{split}.{suffix}"
            public_hashes[name] = _write_jsonl(root / name, rows)
    manifest = {"counts": counts, "file_hashes": public_hashes}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_loader_builds_safe_tasks_and_fresh_episodes(tmp_path):
    bundle = load_grpo_dataset(_dataset_dir(tmp_path), profile="smoke", curriculum_phase=1)

    assert len(bundle.train_tasks) == 2
    assert len(bundle.val_tasks) == 1
    serialized = json.dumps(bundle.train_tasks, ensure_ascii=False)
    assert "ground_truth" not in serialized
    assert "true_liability" not in serialized
    assert bundle.train_tasks[0] == {
        "case_id": "case-0",
        "scenario_id": "fact-0",
        "curriculum_phase": 1,
    }

    first = bundle.episode_source.create("case-0")
    second = bundle.episode_source.create("case-0")
    first.round = 3
    assert second.round == 0
    assert first is not second

    duplicate = json.loads((tmp_path / "grpo_val.jsonl").read_text(encoding="utf-8").splitlines()[0])
    duplicate["case_id"] = "case-0"
    duplicate["observation"]["case_id"] = "case-0"
    _write_jsonl(tmp_path / "grpo_val.jsonl", [duplicate])
    hidden_duplicate = json.loads(
        (tmp_path / "grpo_val.ground_truth.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    hidden_duplicate["case_id"] = "case-0"
    hidden_duplicate["ground_truth"]["case_id"] = "case-0"
    _write_jsonl(tmp_path / "grpo_val.ground_truth.jsonl", [hidden_duplicate])
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    for name in ("grpo_val.jsonl", "grpo_val.ground_truth.jsonl"):
        manifest["file_hashes"][name] = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(GRPODatasetError, match="overlap"):
        load_grpo_dataset(tmp_path, profile="smoke", curriculum_phase=1)

    _dataset_dir(tmp_path)
    train_path = tmp_path / "grpo_train.jsonl"
    train_path.write_text(train_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(GRPODatasetError, match="hash mismatch"):
        load_grpo_dataset(tmp_path, profile="smoke", curriculum_phase=1)
```

- [ ] **Step 2: Run the test and confirm the module is missing**

Run:

```powershell
python -m pytest tests/unit/training/test_grpo_dataset.py -q
```

Expected: collection fails because `dispute_agent.training.grpo_dataset` does not exist.

- [ ] **Step 3: Implement the loader and immutable Episode source**

Create `dispute_agent/training/grpo_dataset.py` with these public types and functions:

```python
"""Manifest-verified GRPO tasks and isolated Episode reconstruction."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from dispute_agent.domain.schemas import DisputeGroundTruth, DisputeObservation
from dispute_agent.environment import EpisodeState


class GRPODatasetError(ValueError):
    """Raised when frozen GRPO data is incomplete, changed, or unsafe."""


@dataclass(frozen=True)
class EpisodeSource:
    observations: dict[str, DisputeObservation]
    ground_truth: dict[str, DisputeGroundTruth]

    def create(self, case_id: str) -> EpisodeState:
        try:
            observation = self.observations[case_id].model_copy(deep=True)
            truth = self.ground_truth[case_id].model_copy(deep=True)
        except KeyError as exc:
            raise GRPODatasetError(f"unknown GRPO case_id: {case_id}") from exc
        return EpisodeState(
            observation=observation,
            ground_truth=truth,
            case_seed=case_id,
        )


@dataclass(frozen=True)
class GRPODatasetBundle:
    train_tasks: list[dict[str, Any]]
    val_tasks: list[dict[str, Any]]
    episode_source: EpisodeSource
    file_hashes: dict[str, str]
    manifest_sha256: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GRPODatasetError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise GRPODatasetError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _load_split(root: Path, split: str, manifest: dict) -> tuple[list[dict], dict[str, DisputeObservation], dict[str, DisputeGroundTruth], dict[str, str]]:
    public_path = root / f"{split}.jsonl"
    hidden_path = root / f"{split}.ground_truth.jsonl"
    hashes = {}
    for path in (public_path, hidden_path):
        if not path.is_file():
            raise GRPODatasetError(f"required GRPO file is missing: {path}")
        actual = _sha256(path)
        expected = manifest.get("file_hashes", {}).get(path.name)
        if expected != actual:
            raise GRPODatasetError(f"hash mismatch for {path.name}")
        hashes[path.name] = actual

    public_rows = _read_jsonl(public_path)
    hidden_rows = _read_jsonl(hidden_path)
    expected_count = manifest.get("counts", {}).get(split)
    if expected_count != len(public_rows) or len(public_rows) != len(hidden_rows):
        raise GRPODatasetError(f"count mismatch for {split}")

    observations = {}
    truths = {}
    hidden_by_id = {row.get("case_id"): row.get("ground_truth") for row in hidden_rows}
    for row in public_rows:
        case_id = row.get("case_id")
        if not case_id or case_id in observations or case_id not in hidden_by_id:
            raise GRPODatasetError(f"invalid or unmatched case_id in {split}: {case_id!r}")
        observation = DisputeObservation.model_validate(row.get("observation"))
        truth = DisputeGroundTruth.model_validate(hidden_by_id[case_id])
        if observation.case_id != case_id or truth.case_id != case_id:
            raise GRPODatasetError(f"case_id mismatch in {split}: {case_id}")
        observations[case_id] = observation
        truths[case_id] = truth
    return public_rows, observations, truths, hashes


def _tasks(rows: list[dict], curriculum_phase: int) -> list[dict[str, Any]]:
    return [
        {
            "case_id": row["case_id"],
            "scenario_id": row["fact_instance_id"],
            "curriculum_phase": curriculum_phase,
        }
        for row in rows
    ]


def load_grpo_dataset(
    data_dir: str | Path,
    *,
    profile: str,
    curriculum_phase: int,
) -> GRPODatasetBundle:
    if profile not in {"smoke", "formal"}:
        raise GRPODatasetError(f"unknown profile: {profile}")
    if curriculum_phase not in {1, 2}:
        raise GRPODatasetError("curriculum_phase must be 1 or 2")

    root = Path(data_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise GRPODatasetError(f"required GRPO manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_rows, train_obs, train_truth, train_hashes = _load_split(root, "grpo_train", manifest)
    val_rows, val_obs, val_truth, val_hashes = _load_split(root, "grpo_val", manifest)

    overlap = set(train_obs).intersection(val_obs)
    if overlap:
        raise GRPODatasetError(f"train/validation case_id overlap: {sorted(overlap)[:3]}")
    scenario_ids = [row.get("fact_instance_id") for row in train_rows + val_rows]
    if any(not value for value in scenario_ids) or len(scenario_ids) != len(set(scenario_ids)):
        raise GRPODatasetError("missing or duplicate GRPO fact_instance_id")

    if profile == "formal" and (len(train_rows), len(val_rows)) != (700, 100):
        raise GRPODatasetError("formal GRPO requires exactly 700 train and 100 validation rows")
    if profile == "smoke":
        train_rows = train_rows[:2]
        val_rows = val_rows[: min(2, len(val_rows))]

    return GRPODatasetBundle(
        train_tasks=_tasks(train_rows, curriculum_phase),
        val_tasks=_tasks(val_rows, curriculum_phase),
        episode_source=EpisodeSource({**train_obs, **val_obs}, {**train_truth, **val_truth}),
        file_hashes={**train_hashes, **val_hashes},
        manifest_sha256=_sha256(manifest_path),
    )
```

- [ ] **Step 4: Run the dataset test**

Run:

```powershell
python -m pytest tests/unit/training/test_grpo_dataset.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the GRPO loader**

```bash
git add dispute_agent/training/grpo_dataset.py tests/unit/training/test_grpo_dataset.py
git commit -m "feat: load isolated grpo episodes"
```

### Task 3: Expand and validate the complete GRPO configuration

**Files:**
- Modify: `dispute_agent/training/grpo_config.py`
- Modify: `configs/grpo.yaml`
- Modify: `tests/unit/training/test_grpo_config.py`

- [ ] **Step 1: Replace the narrow test with resolved-config assertions**

Extend `tests/unit/training/test_grpo_config.py`:

```python
import pytest
from pydantic import ValidationError

from dispute_agent.training.grpo_config import GRPOConfig, load_grpo_config


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
```

- [ ] **Step 2: Run the config test and confirm missing fields/methods**

Run:

```powershell
python -m pytest tests/unit/training/test_grpo_config.py -q
```

Expected: FAIL because `to_verl_config` and the complete schema do not exist.

- [ ] **Step 3: Implement strict nested configuration models**

Replace `dispute_agent/training/grpo_config.py` with models for these exact sections:

```python
"""Strict project config and Agent Lightning/verl config rendering."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AlgorithmConfig(StrictModel):
    adv_estimator: Literal["grpo"] = "grpo"
    use_kl_in_reward: bool = False


class GRPOModelConfig(StrictModel):
    path: str = "Qwen/Qwen3-8B"
    lora_adapter_path: str
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    target_modules: str = "all-linear"
    enable_gradient_checkpointing: bool = True
    use_remove_padding: bool = True


class GRPORolloutConfig(StrictModel):
    name: Literal["vllm"] = "vllm"
    n: int = 4
    tensor_model_parallel_size: int = 2
    max_tokens_per_round: int = 384
    max_episode_tokens: int = 1280
    gpu_memory_utilization: float = Field(default=0.45, gt=0, lt=1)
    max_num_seqs: int = 8
    load_format: Literal["safetensors"] = "safetensors"
    tool_format: Literal["hermes"] = "hermes"


class ActorConfig(StrictModel):
    ppo_mini_batch_size: int = 8
    ppo_micro_batch_size_per_gpu: int = 1
    learning_rate: float = 1e-5
    use_kl_loss: bool = True
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"
    param_offload: bool = True
    optimizer_offload: bool = True


class ActorRolloutRefConfig(StrictModel):
    model: GRPOModelConfig
    rollout: GRPORolloutConfig = Field(default_factory=GRPORolloutConfig)
    actor: ActorConfig = Field(default_factory=ActorConfig)


class DataConfig(StrictModel):
    train_batch_size: int = 2
    max_prompt_length: int = 2048
    max_response_length: int = 384


class TrainerConfig(StrictModel):
    n_gpus_per_node: int = 2
    nnodes: int = 1
    total_epochs: int = 1
    save_freq: int = 1
    test_freq: int = 10
    val_before_train: bool = True
    logger: list[str] = Field(default_factory=lambda: ["console", "wandb"])


class AgentLightningConfig(StrictModel):
    n_runners: int = 1
    trajectory_max_prompt_length: int = 2048
    trajectory_max_response_length: int = 4096


class CurriculumConfig(StrictModel):
    phase1_tools: list[str] = Field(default_factory=lambda: [
        "check_logistics", "check_buyer_history", "check_merchant_history"
    ])
    phase1_max_rounds: int = 3
    phase2_max_rounds: int = 5


class MonitorConfig(StrictModel):
    window: int = 50
    max_zero_variance_ratio: float = 0.30


class GRPOConfig(StrictModel):
    algorithm: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    actor_rollout_ref: ActorRolloutRefConfig
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    agentlightning: AgentLightningConfig = Field(default_factory=AgentLightningConfig)
    curriculum: CurriculumConfig = Field(default_factory=CurriculumConfig)
    monitor: MonitorConfig = Field(default_factory=MonitorConfig)
    quantization: str | None = None

    @model_validator(mode="after")
    def validate_training_contract(self) -> "GRPOConfig":
        model = self.actor_rollout_ref.model
        rollout = self.actor_rollout_ref.rollout
        if self.quantization is not None or model.lora_rank != 32 or model.lora_dropout != 0:
            raise ValueError("GRPO must use BF16 LoRA r=32 without quantization")
        if model.lora_alpha != 64 or model.target_modules != "all-linear":
            raise ValueError("GRPO LoRA must use alpha=64 and all-linear targets")
        if rollout.n != 4:
            raise ValueError("Agentic GRPO requires rollout n=4")
        if rollout.tensor_model_parallel_size != 2 or self.trainer.n_gpus_per_node != 2:
            raise ValueError("formal topology requires exactly two GPUs and TP=2")
        return self

    def to_verl_config(self, *, profile: str, output_dir: str | Path) -> dict:
        if profile not in {"smoke", "formal"}:
            raise ValueError(f"unknown profile: {profile}")
        model = self.actor_rollout_ref.model
        rollout = self.actor_rollout_ref.rollout
        actor = self.actor_rollout_ref.actor
        trainer = self.trainer
        return {
            "algorithm": {
                "adv_estimator": self.algorithm.adv_estimator,
                "use_kl_in_reward": self.algorithm.use_kl_in_reward,
            },
            "data": {
                "train_batch_size": self.data.train_batch_size,
                "max_prompt_length": self.data.max_prompt_length,
                "max_response_length": self.data.max_response_length,
            },
            "actor_rollout_ref": {
                "model": {
                    "path": model.path,
                    "lora_adapter_path": model.lora_adapter_path,
                    "lora_rank": model.lora_rank,
                    "lora_alpha": model.lora_alpha,
                    "target_modules": model.target_modules,
                    "enable_gradient_checkpointing": model.enable_gradient_checkpointing,
                    "use_remove_padding": model.use_remove_padding,
                },
                "rollout": {
                    "name": rollout.name,
                    "n": rollout.n,
                    "tensor_model_parallel_size": rollout.tensor_model_parallel_size,
                    "gpu_memory_utilization": rollout.gpu_memory_utilization,
                    "max_num_seqs": rollout.max_num_seqs,
                    "load_format": rollout.load_format,
                    "multi_turn": {"format": rollout.tool_format},
                },
                "actor": {
                    "ppo_mini_batch_size": actor.ppo_mini_batch_size,
                    "ppo_micro_batch_size_per_gpu": actor.ppo_micro_batch_size_per_gpu,
                    "optim": {"lr": actor.learning_rate},
                    "use_kl_loss": actor.use_kl_loss,
                    "kl_loss_coef": actor.kl_loss_coef,
                    "kl_loss_type": actor.kl_loss_type,
                    "entropy_coeff": 0,
                    "fsdp_config": {
                        "param_offload": actor.param_offload,
                        "optimizer_offload": actor.optimizer_offload,
                    },
                },
                "ref": {
                    "log_prob_micro_batch_size_per_gpu": 1,
                    "fsdp_config": {"param_offload": True},
                },
            },
            "agentlightning": {
                "trace_aggregator": {
                    "level": "trajectory",
                    "trajectory_max_prompt_length": self.agentlightning.trajectory_max_prompt_length,
                    "trajectory_max_response_length": self.agentlightning.trajectory_max_response_length,
                }
            },
            "trainer": {
                "n_gpus_per_node": trainer.n_gpus_per_node,
                "nnodes": trainer.nnodes,
                "total_epochs": trainer.total_epochs,
                "save_freq": 1 if profile == "smoke" else trainer.save_freq,
                "test_freq": 1 if profile == "smoke" else trainer.test_freq,
                "val_before_train": trainer.val_before_train,
                "logger": trainer.logger,
                "project_name": "dispute-resolve-agent",
                "experiment_name": Path(output_dir).name,
                "default_local_dir": str(Path(output_dir) / "checkpoints"),
                "resume_mode": "disable",
                "critic_warmup": 0,
            },
        }


def load_grpo_config(path: str | Path = "configs/grpo.yaml") -> GRPOConfig:
    return GRPOConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
```

- [ ] **Step 4: Replace `configs/grpo.yaml` with all project fields**

```yaml
algorithm:
  adv_estimator: grpo
  use_kl_in_reward: false
data:
  train_batch_size: 2
  max_prompt_length: 2048
  max_response_length: 384
actor_rollout_ref:
  model:
    path: Qwen/Qwen3-8B
    lora_adapter_path: checkpoints/sft/sft-1500-best
    lora_rank: 32
    lora_alpha: 64
    lora_dropout: 0.0
    target_modules: all-linear
    enable_gradient_checkpointing: true
    use_remove_padding: true
  rollout:
    name: vllm
    n: 4
    tensor_model_parallel_size: 2
    max_tokens_per_round: 384
    max_episode_tokens: 1280
    gpu_memory_utilization: 0.45
    max_num_seqs: 8
    load_format: safetensors
    tool_format: hermes
  actor:
    ppo_mini_batch_size: 8
    ppo_micro_batch_size_per_gpu: 1
    learning_rate: 1.0e-5
    use_kl_loss: true
    kl_loss_coef: 0.001
    kl_loss_type: low_var_kl
    param_offload: true
    optimizer_offload: true
trainer:
  n_gpus_per_node: 2
  nnodes: 1
  total_epochs: 1
  save_freq: 25
  test_freq: 25
  val_before_train: true
  logger: [console, wandb]
agentlightning:
  n_runners: 1
  trajectory_max_prompt_length: 2048
  trajectory_max_response_length: 4096
quantization: null
curriculum:
  phase1_tools: [check_logistics, check_buyer_history, check_merchant_history]
  phase1_max_rounds: 3
  phase2_max_rounds: 5
monitor:
  window: 50
  max_zero_variance_ratio: 0.30
```

- [ ] **Step 5: Run the config tests**

Run:

```powershell
python -m pytest tests/unit/training/test_grpo_config.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit the resolved configuration contract**

```bash
git add dispute_agent/training/grpo_config.py configs/grpo.yaml tests/unit/training/test_grpo_config.py
git commit -m "feat: define real agentic grpo config"
```

### Task 4: Let the existing agent runtime enforce curriculum limits

**Files:**
- Modify: `dispute_agent/agent/tools.py`
- Modify: `dispute_agent/agent/runtime.py`
- Create: `tests/unit/agent/test_curriculum_runtime.py`

- [ ] **Step 1: Write one failing test for tool filtering and turn limits**

```python
from dispute_agent.agent.runtime import DisputeRuntime
from dispute_agent.agent.tools import build_agent_tools


def test_phase_one_filters_tools_but_keeps_submit(sample_episode):
    tools = build_agent_tools(
        sample_episode,
        allowed_tools={
            "check_logistics",
            "check_buyer_history",
            "check_merchant_history",
        },
    )
    names = {tool.name for tool in tools}

    assert names == {
        "check_logistics",
        "check_buyer_history",
        "check_merchant_history",
        "submit_decision",
    }
    assert "verify_evidence" not in names
```

Extend the same file with a monkeypatched `Runner.run` assertion that
`DisputeRuntime.run(..., max_rounds=3, max_tokens_per_round=384)` passes
`max_turns=4` and a `ModelSettings.max_tokens` value of `384`.

- [ ] **Step 2: Confirm the focused test fails**

Run:

```powershell
python -m pytest tests/unit/agent/test_curriculum_runtime.py -q
```

Expected: FAIL because neither argument exists yet.

- [ ] **Step 3: Filter only investigation tools in `tools.py`**

Change the signature to:

```python
def build_agent_tools(
    episode: EpisodeState,
    allowed_tools: set[str] | None = None,
) -> list[FunctionTool]:
```

Build the four investigation tools in a name-to-builder mapping, include only
the requested names when `allowed_tools` is not `None`, reject unknown names
with `ValueError`, and append `submit_decision` unconditionally. Do not change
the implementation, timeout behavior, or output schema of any tool.

- [ ] **Step 4: Add explicit runtime budgets in `runtime.py`**

Extend `DisputeRuntime.run` with:

```python
allowed_tools: set[str] | None = None,
max_rounds: int = MAX_ROUNDS,
max_tokens_per_round: int = 384,
```

Validate both numeric values are positive, pass the filtered tools to the
Agent, add `max_tokens=max_tokens_per_round` to `ModelSettings`, and call
`Runner.run(..., max_turns=max_rounds + 1)`. The extra turn is reserved for
`submit_decision`; it does not grant another investigation round.

- [ ] **Step 5: Run the single runtime contract file**

Run:

```powershell
python -m pytest tests/unit/agent/test_curriculum_runtime.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit the runtime boundary**

```bash
git add dispute_agent/agent/tools.py dispute_agent/agent/runtime.py tests/unit/agent/test_curriculum_runtime.py
git commit -m "feat: enforce grpo curriculum limits"
```

### Task 5: Replace the simulated Lightning rollout with a real lazy binding

**Files:**
- Replace: `dispute_agent/training/lightning_agent.py`
- Replace: `tests/integration/test_lightning_rollout.py`

- [ ] **Step 1: Write a dependency-light rollout test**

The test must not import Agent Lightning. Use a fake endpoint descriptor, a
fresh `EpisodeSource`, a fake runtime returning a valid `DecisionOutput`, and
an annotation collector:

```python
@pytest.mark.asyncio
async def test_rollout_reconstructs_fresh_episode_and_returns_one_reward(
    episode_source, fake_runtime_factory
):
    annotations: list[dict[str, object]] = []
    task = {
        "case_id": "case-001",
        "scenario_id": "fact-001",
        "curriculum_phase": 1,
    }
    llm = SimpleNamespace(
        endpoint="http://127.0.0.1:8000/v1",
        api_key=None,
        model="Qwen/Qwen3-8B",
    )

    first = await run_dispute_rollout(
        task,
        llm,
        episode_source=episode_source,
        runtime_factory=fake_runtime_factory,
        annotation_emitter=annotations.append,
    )
    second = await run_dispute_rollout(
        task,
        llm,
        episode_source=episode_source,
        runtime_factory=fake_runtime_factory,
        annotation_emitter=annotations.append,
    )

    assert isinstance(first, float)
    assert first == second
    assert fake_runtime_factory.episodes[0] is not fake_runtime_factory.episodes[1]
    assert all("ground_truth" not in json.dumps(item) for item in annotations)
```

Also assert phase 1 receives exactly the configured three investigation tools,
the per-turn budget is
`min(max_tokens_per_round, max_episode_tokens // max_rounds)`, and a missing
submission maps to the existing hard-failure reward. Let unrelated transport
and infrastructure exceptions propagate so a broken worker is never recorded
as a bad policy sample.

- [ ] **Step 2: Confirm the old placeholder contract fails**

Run:

```powershell
python -m pytest tests/integration/test_lightning_rollout.py -q
```

Expected: FAIL because `run_dispute_rollout` does not exist.

- [ ] **Step 3: Implement the pure rollout core**

Delete `LitDisputeAgent`, `LightningRollout`, and the simulated event types.
Implement:

```python
async def run_dispute_rollout(
    task: Mapping[str, object],
    llm: object,
    *,
    episode_source: EpisodeSource,
    config: GRPOConfig,
    runtime_factory: Callable[..., DisputeRuntime] = build_runtime,
    reward_engine: RewardEngine | None = None,
    annotation_emitter: Callable[[dict[str, object]], None] | None = None,
) -> float:
```

The function must:

1. Validate `case_id`, `scenario_id`, and curriculum phase.
2. Create a fresh episode through `EpisodeSource.create(case_id)`.
3. Build `DisputeRuntime` from `llm.endpoint`, `llm.api_key or "EMPTY"`, and
   `llm.model`.
4. Apply phase 1's tool allow-list/round limit or phase 2's full toolset/limit.
5. Derive the per-turn token cap from the episode cap.
6. Score once with the existing `RewardEngine` and return `float(total)`.
7. Emit only non-secret component metrics and identifiers as annotations.

Do **not** call `agentlightning.emit_reward`; the returned float is the sole
reward channel. Catch only the known missing-decision/turn-limit terminal
condition and score it as the existing hard failure.

- [ ] **Step 4: Add the real wrapper without making local imports heavy**

Add `build_lightning_agent(config_path, data_dir, profile)` which imports
`agentlightning as agl` inside the function, loads the verified episode source,
and decorates a small async entrypoint with `@agl.rollout`. The decorated
entrypoint delegates to `run_dispute_rollout` and sends annotations with
`agl.emit_object`. No top-level import may require Agent Lightning, verl, Ray,
CUDA, or vLLM.

- [ ] **Step 5: Run the rollout contract**

Run:

```powershell
python -m pytest tests/integration/test_lightning_rollout.py -q
```

Expected: PASS without model downloads or training dependencies.

- [ ] **Step 6: Commit the real rollout adapter**

```bash
git add dispute_agent/training/lightning_agent.py tests/integration/test_lightning_rollout.py
git commit -m "feat: bind dispute rollout to agent lightning"
```

### Task 6: Build the real Agent Lightning/verl training entrypoint

**Files:**
- Create: `dispute_agent/training/grpo_runtime.py`
- Replace: `scripts/train_agentic_grpo.py`
- Modify: `pyproject.toml`
- Modify: `constraints/train.txt`
- Modify: `tests/test_project_contract.py`
- Create: `tests/unit/training/test_grpo_runtime.py`

- [ ] **Step 1: Write one trainer-construction and dry-run test file**

Use a fake `agentlightning` module to record constructor arguments. Assert:

```python
def test_real_run_builds_verl_and_trainer(tmp_path, fake_agl, generated_grpo_data):
    result = run_grpo_training(
        GRPORunRequest(
            config_path=Path("configs/grpo.yaml"),
            data_dir=generated_grpo_data,
            output_root=tmp_path,
            run_id="test-run",
            profile="smoke",
            curriculum_phase=1,
        ),
        agl_module=fake_agl,
    )

    assert fake_agl.verl_configs[0]["actor_rollout_ref"]["rollout"]["n"] == 4
    assert fake_agl.trainer_kwargs["n_runners"] == 1
    assert fake_agl.trainer_kwargs["tracer"].kind == "otel"
    assert fake_agl.trainer_kwargs["adapter"].kind == "llm_proxy_triplet"
    assert len(fake_agl.fit_train_dataset) == 2
    assert result.manifest_path.exists()
```

Add a CLI dry-run test that monkeypatches imports to fail if
`agentlightning`, `verl`, `ray`, or `vllm` is imported. It must still write a
resolved plan and exit zero. In the existing fake-backend test, verify that
`input_adapter` overrides the YAML path and `max_steps=25` becomes
`trainer.total_training_steps=25`. Add a negative test for an input adapter whose
`adapter_config.json` is not `r=32`, `lora_alpha=64`, dropout `0.0`, or base
`Qwen/Qwen3-8B`; also reject a provenance manifest whose saved resolved config
did not request `target_modules="all-linear"`.

- [ ] **Step 2: Confirm the tests fail against the placeholder CLI**

Run:

```powershell
python -m pytest tests/unit/training/test_grpo_runtime.py -q
```

Expected: FAIL.

- [ ] **Step 3: Implement the run request, manifest, and adapter guard**

In `grpo_runtime.py`, add immutable `GRPORunRequest` and `GRPOTrainingResult`
dataclasses. The request includes config/data/output paths, run id, profile,
curriculum phase, optional input-adapter override, optional positive max steps,
and resume flag. `build_run_plan` must load/validate config and data, resolve
paths, and record package expectations without importing GPU packages.

`validate_input_adapter` must require `adapter_config.json`, compare its base
model, rank, alpha, and dropout to the project config, require at least one
`.safetensors` adapter weight, and record SHA-256 values. Because PEFT expands
`all-linear` into concrete module names when saving, verify that original value
from the nearest ancestor run manifest: the SFT trainer spec for an SFT input,
or the resolved GRPO config for a completed phase-1 checkpoint. Do not compare
the saved module list to the literal string. Reject missing provenance,
quantization fields, and mismatches before any GPU allocation.

Every non-dry run writes these files below `outputs/grpo/<run_id>/`:

```text
run_manifest.json       # running/completed/failed, inputs, hashes, timestamps
resolved_config.yaml    # exact project configuration
verl_config.yaml        # exact dictionary passed to agl.VERL
metrics/rollouts.jsonl  # sanitized rollout ids, reward and component metrics
metrics/summary.json    # aggregate reward and failure counts
```

Use atomic temp-file replacement when changing the run manifest. A `--resume`
request must point to the same config/data/adapter hashes; otherwise stop.

- [ ] **Step 4: Construct the actual stable Agent Lightning stack**

Inside `run_grpo_training`, import Agent Lightning lazily and construct exactly:

```python
store = agl.InMemoryLightningStore()
algorithm = agl.VERL(verl_config)
trainer = agl.Trainer(
    algorithm=algorithm,
    n_runners=config.agentlightning.n_runners,
    tracer=agl.OtelTracer(),
    adapter=agl.LlmProxyTraceToTriplet(),
    store=store,
)
trainer.fit(
    build_lightning_agent(config_path, data_dir, profile),
    train_dataset=bundle.train_tasks,
    val_dataset=bundle.val_tasks,
)
```

Before construction, inject
`trainer.default_local_dir=<run_dir>/checkpoints`. For the `smoke` profile only,
override `trainer.total_training_steps=1`, `save_freq=1`, and `test_freq=1` in
the emitted verl dictionary; an explicit positive `max_steps` overrides that
value for a bounded formal phase. This makes Phase 0 prove one optimizer update
and produce a reloadable checkpoint. The formal profile otherwise keeps the
YAML schedule.

After `fit`, query `store.query_rollouts()` and each rollout's
`store.query_spans(rollout.rollout_id)`. Export only rollout id, status, final
reward, non-secret annotations, span counts, and timestamps. Never export
prompts, hidden ground truth, tool secrets, or raw span attributes.

- [ ] **Step 5: Replace the single CLI**

Support:

```text
--config configs/grpo.yaml
--data-dir data/processed
--output-root outputs/grpo
--run-id <required outside dry-run>
--profile smoke|formal
--curriculum-phase 1|2
--input-adapter <optional LoRA path overriding YAML>
--max-steps <optional positive integer>
--dry-run
--resume
```

`--dry-run` prints and writes the plan but never validates a physical adapter,
imports GPU packages, initializes Ray, or contacts Weights & Biases. A real run
performs all guards and writes `failed` plus the exception class/message before
re-raising an error. The effective adapter path and `max_steps` are part of the
run fingerprint; `--resume` cannot change either.

- [ ] **Step 6: Freeze the initial server matrix and install extra**

Set `constraints/train.txt` to the approved baseline (one requirement per line):

```text
torch==2.8.0
torchvision==0.23.0
transformers==4.55.4
peft==0.18.1
accelerate==1.10.1
flash-attn==2.8.3
vllm==0.10.2
verl==0.5.0
agentlightning==0.3.0
openai-agents==0.6.0
```

This file is the initial Phase 0 matrix, not a claim that Windows can run it.
Do not add an automatic version fallback.

Align `pyproject.toml`'s `train` extra with these compatible ranges/exact pins,
and extend `tests/test_project_contract.py` to require every critical pin. The
server installation command is `python -m pip install -e ".[dev,train]" -c
constraints/train.txt`; do not introduce a second requirements file.

- [ ] **Step 7: Run the focused runtime tests**

Run:

```powershell
python -m pytest tests/test_project_contract.py tests/unit/training/test_grpo_runtime.py -q
```

Expected: PASS using the fake Agent Lightning module.

- [ ] **Step 8: Commit the training path**

```bash
git add dispute_agent/training/grpo_runtime.py scripts/train_agentic_grpo.py pyproject.toml constraints/train.txt tests/test_project_contract.py tests/unit/training/test_grpo_runtime.py
git commit -m "feat: add real agent lightning verl trainer"
```

### Task 7: Make Phase 0 an evidence-based gate, never a fixture success

**Files:**
- Create: `dispute_agent/training/phase0.py`
- Replace: `scripts/phase0_smoke.py`
- Create: `scripts/verify_grpo_checkpoint.py`
- Replace: `tests/integration/test_phase0_contract.py`

- [ ] **Step 1: Write the failing status/report contract**

```python
def test_fixture_report_marks_every_execution_gate_not_run(tmp_path):
    report = build_fixture_report(output_dir=tmp_path)

    assert report.mode == "fixture"
    assert {gate.status for gate in report.gates} == {GateStatus.NOT_RUN}
    assert report.overall_status == GateStatus.NOT_RUN
    assert not report.ready_for_formal_training


def test_pass_requires_evidence_file(tmp_path):
    with pytest.raises(Phase0EvidenceError):
        GateResult(
            name="adapter_reloaded",
            status=GateStatus.PASSED,
            evidence_path=tmp_path / "missing.json",
        )
```

Also test that a report containing `failed` or `not_run` cannot set
`ready_for_formal_training=True`.

- [ ] **Step 2: Confirm the current fixture lies about success**

Run:

```powershell
python -m pytest tests/integration/test_phase0_contract.py -q
```

Expected: FAIL because the current script marks fixture gates passed.

- [ ] **Step 3: Add typed gate results and report validation**

In `phase0.py`, define:

```python
class GateStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_RUN = "not_run"


class GateResult(BaseModel):
    name: str
    status: GateStatus
    summary: str
    evidence_path: Path | None = None


class Phase0Report(BaseModel):
    schema_version: Literal["phase0-v2"] = "phase0-v2"
    mode: Literal["fixture", "actual"]
    run_id: str
    gates: list[GateResult]
    overall_status: GateStatus
    ready_for_formal_training: bool
```

Validation rules:

- `passed` requires a real evidence file beneath the run directory.
- `ready_for_formal_training` is true only when all required gates passed.
- Fixture mode forces every execution gate and overall status to `not_run`.
- Reports contain paths, hashes, counts, versions, and metrics, never hidden
  labels, prompts, raw traces, API keys, or environment dumps.

- [ ] **Step 4: Implement the independent checkpoint verifier**

`verify_grpo_checkpoint.py` must run in a fresh process after training. It
accepts `--base-model`, `--adapter-dir`, `--public-prompt-file`, and
`--evidence-out`, then:

1. Validate `adapter_config.json` and adapter safetensors.
2. Load the base model with `torch_dtype=torch.bfloat16` and no 4/8-bit flags.
3. Load the updated LoRA with `PeftModel.from_pretrained` on one GPU.
4. Generate at least one token from the public prompt with deterministic
   decoding.
5. Write hashes, device/dtype, generated-token count, and success/failure to
   the evidence file; do not write prompt or generated text.

Exit nonzero on load or inference failure. A file-existence check alone must
never satisfy this gate.

- [ ] **Step 5: Replace `phase0_smoke.py` with two explicit modes**

CLI:

```text
--mode fixture|actual
--config configs/grpo.yaml
--data-dir data/processed
--output-root outputs/grpo
--run-id phase0-<timestamp>
```

Fixture mode performs only local schema/command preparation and writes all
execution gates as `not_run`.

Actual mode, on Ubuntu, must:

1. Record exact package versions and two visible CUDA devices.
2. Validate the initial SFT adapter and hash its weights.
3. Start `train_agentic_grpo.py --profile smoke --curriculum-phase 1` in a
   subprocess and require exit code zero.
4. Require `global_step_1/actor/huggingface` (or the single equivalent
   checkpoint reported by verl), a PEFT config, and adapter safetensors.
5. Require an updated weight hash different from the initial SFT hash.
6. Require exported evidence of model spans, tool spans, and exactly one final
   reward per completed rollout.
7. Run `verify_grpo_checkpoint.py` in a fresh subprocess on GPU 0.
8. Write `phase0_report.json` and return nonzero unless every gate passed.

If the package matrix, checkpoint layout, or trace schema differs, record the
observed evidence and fail. Do not silently adapt versions or manufacture a
pass.

- [ ] **Step 6: Run the local Phase 0 contract only**

Run:

```powershell
python -m pytest tests/integration/test_phase0_contract.py -q
python scripts/phase0_smoke.py --mode fixture --run-id local-fixture
```

Expected: tests PASS; the script exits zero for report creation but its report
has `overall_status: not_run` and `ready_for_formal_training: false`.

- [ ] **Step 7: Commit the honest gate**

```bash
git add dispute_agent/training/phase0.py scripts/phase0_smoke.py scripts/verify_grpo_checkpoint.py tests/integration/test_phase0_contract.py
git commit -m "feat: require evidence for grpo phase zero"
```

### Task 8: Document the handoff and run only the focused local verification

**Files:**
- Modify: `README.md`
- Modify: `docs/experiments.md`
- Modify: `docs/resume-evidence-checklist.md`

- [ ] **Step 1: Update the README commands and platform boundary**

Document these local commands:

```powershell
python scripts/generate_data.py --config configs/data.yaml --output-dir data/processed
python scripts/train_agentic_grpo.py --config configs/grpo.yaml --data-dir data/processed --profile smoke --curriculum-phase 1 --dry-run
python scripts/phase0_smoke.py --mode fixture --config configs/grpo.yaml --data-dir data/processed --run-id local-fixture
```

State plainly that Windows can verify data/config/core behavior but cannot
claim Agent Lightning/verl training success. Point to Task 9 for Ubuntu.

- [ ] **Step 2: Make the resume evidence language exact**

`docs/resume-evidence-checklist.md` must distinguish:

- implemented locally,
- Phase 0 passed on the two-4090 server,
- formal GRPO completed,
- measured result.

Do not permit wording such as "completed Agentic GRPO" until the corresponding
real report, checkpoint, and metrics exist. Record Agent Lightning 0.3.0,
OpenAI Agents SDK 0.6.0, verl 0.5.0, BF16 LoRA, and the real sample counts.

- [ ] **Step 3: Update experiment templates, not fabricated metrics**

Add table columns for run id, git commit, config/data/adapter hashes, curriculum
phase, optimizer updates, valid rollout rate, reward mean/std, component means,
tool-call mean, failure rate, checkpoint path/hash, and Phase 0 report. Leave
unexecuted values as `not_run`, never zero or passed.

- [ ] **Step 4: Run the intentionally small local verification set**

Run exactly:

```powershell
python -m pytest tests/test_project_contract.py tests/leakage/test_dataset_leakage.py tests/unit/training/test_grpo_dataset.py tests/unit/training/test_grpo_config.py tests/unit/agent/test_curriculum_runtime.py tests/integration/test_lightning_rollout.py tests/unit/training/test_grpo_runtime.py tests/integration/test_phase0_contract.py -q
python scripts/train_agentic_grpo.py --config configs/grpo.yaml --data-dir data/processed --profile smoke --curriculum-phase 1 --dry-run
python scripts/phase0_smoke.py --mode fixture --config configs/grpo.yaml --data-dir data/processed --run-id local-fixture
```

No full model download, CUDA initialization, broad regression suite, or formal
training is required on this computer. If a focused test exposes a shared-core
regression, then run only the nearest existing test file needed to diagnose it.

- [ ] **Step 5: Inspect tracked outputs and working tree**

Run:

```powershell
git status --short
git ls-files outputs checkpoints wandb
```

Expected: no model weights, generated run artifacts, W&B data, secrets, or
hidden sidecars are staged. Keep `.gitkeep` only where the project intentionally
tracks an otherwise empty output directory.

- [ ] **Step 6: Commit the local handoff**

```bash
git add README.md docs/experiments.md docs/resume-evidence-checklist.md
git commit -m "docs: add agentic grpo execution handoff"
```

At this boundary the code may be pushed, but the resume must still say the
training path is implemented and locally contract-tested, not server-proven.

### Task 9: Prove Phase 0, then run the two-stage formal curriculum on Ubuntu

**Files:**
- Generate: `outputs/grpo/<phase0-run>/phase0_report.json`
- Generate: `outputs/grpo/<phase0-run>/checkpoints/...`
- Modify after evidence exists: `docs/experiments.md`
- Modify after evidence exists: `docs/resume-evidence-checklist.md`

- [ ] **Step 1: Prepare a clean server environment at the exact commit**

```bash
git clone https://github.com/wty336/dispute-resolve-agent.git
cd dispute-resolve-agent
git checkout <reviewed-commit>
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,train]" -c constraints/train.txt
python -m pip check
```

Record `nvidia-smi`, driver/CUDA information, `pip freeze`, and the git commit
under the Phase 0 run directory. Do not edit the dependency matrix on the
server without committing the change and returning to Task 8 verification.

- [ ] **Step 2: Transfer or regenerate verified inputs**

Place the SFT LoRA adapter at the configured path. Generate the GRPO data from
the frozen config/seed or transfer the exact generated directory, then run the
dataset loader/dry-run to verify manifest and SHA-256 values:

```bash
python scripts/generate_data.py --config configs/data.yaml --output-dir data/processed
python scripts/train_agentic_grpo.py --config configs/grpo.yaml --data-dir data/processed --profile smoke --curriculum-phase 1 --dry-run
```

- [ ] **Step 3: Run Phase 0 and stop on any failed gate**

```bash
python scripts/phase0_smoke.py \
  --mode actual \
  --config configs/grpo.yaml \
  --data-dir data/processed \
  --output-root outputs/grpo \
  --run-id phase0-$(date +%Y%m%d-%H%M%S)
```

Required outcome: two GPUs detected, exact package matrix, valid SFT adapter,
one real optimizer update, changed adapter hash, model/tool trace evidence,
exactly one final reward per rollout, and successful BF16 LoRA reload plus
inference. Any other outcome is a failed Phase 0 to diagnose; do not start the
formal run.

- [ ] **Step 4: Run the bounded restricted-tool warm-up**

Phase 0 proves plumbing on two cases but is not the curriculum model. Run a
bounded 25-update phase 1 over the formal dataset so the policy actually learns
under the restricted tool/round budget:

```bash
python scripts/train_agentic_grpo.py \
  --config configs/grpo.yaml \
  --data-dir data/processed \
  --output-root outputs/grpo \
  --run-id grpo-phase1-$(date +%Y%m%d-%H%M%S) \
  --profile formal \
  --curriculum-phase 1 \
  --max-steps 25
```

The formal loader must report exactly 700 train and 100 validation cases before
initializing GPUs. Require the completed phase-1 manifest and reloadable actor
LoRA before continuing.

- [ ] **Step 5: Train phase 2 from the phase-1 LoRA**

Use the exact `actor/huggingface` adapter directory recorded by the completed
phase-1 manifest:

```bash
python scripts/train_agentic_grpo.py \
  --config configs/grpo.yaml \
  --data-dir data/processed \
  --output-root outputs/grpo \
  --run-id grpo-phase2-$(date +%Y%m%d-%H%M%S) \
  --profile formal \
  --curriculum-phase 2 \
  --input-adapter <phase1-run>/checkpoints/global_step_25/actor/huggingface
```

If verl reports a different final global step, use the checkpoint path written
to phase 1's manifest rather than guessing. This is a new lineage-linked run,
not `--resume`; reserve `--resume` for interruption of the same phase, run id,
configuration, input adapter, and hashes.

- [ ] **Step 6: Record measured evidence and run a narrow evaluation**

After completion, copy aggregate values from `metrics/summary.json`, link the
Phase 0/formal manifests and checkpoint hashes in `docs/experiments.md`, and run
the existing evaluation entrypoint against the final adapter and the frozen
test set. Add only measurements produced by those artifacts.

- [ ] **Step 7: Commit evidence metadata, never weights or private traces**

```bash
git add docs/experiments.md docs/resume-evidence-checklist.md
git commit -m "docs: record agentic grpo server evidence"
git push origin master
```

Keep checkpoints, raw spans, W&B directories, hidden labels, and API keys out
of Git. The final resume claim is allowed only after this commit's referenced
evidence can be explained end to end.
