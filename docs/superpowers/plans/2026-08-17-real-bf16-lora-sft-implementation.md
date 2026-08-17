# Real BF16 LoRA SFT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the local SFT stub with a reproducible Qwen3-8B BF16 LoRA training path that supports one 500/1000/1500 run at a time, strict public-data and assistant-mask gates, safe resume, and auditable artifacts.

**Architecture:** Keep dataset integrity, tokenizer preflight, TRL/PEFT integration, and run orchestration in separate modules. The local machine exercises three high-risk seams with lightweight fakes; the server alone loads the real Qwen3 tokenizer/model and runs two-GPU training.

**Tech Stack:** Python 3.11, Pydantic 2, Hugging Face Datasets, Transformers 4.57.6, TRL 0.28.0, PEFT 0.18.1, Accelerate 1.12.0, PyTorch 2.8.0 BF16, pytest.

---

## Scope and test budget

Design source: `docs/superpowers/specs/2026-08-17-real-bf16-lora-sft-design.md`.

This plan adds only three new test files and extends one existing configuration test:

1. `tests/unit/training/test_sft_dataset.py` — one comprehensive dataset integrity test;
2. `tests/unit/training/test_sft_preflight.py` — one valid contract test plus one parametrized rejection test;
3. `tests/unit/training/test_sft_training.py` — one end-to-end orchestration test with a fake backend;
4. `tests/unit/training/test_sft_config.py` — extend the existing test instead of creating another file.

Do not add tests for CLI help text, trivial property access, logging wording, or third-party TRL internals. Real tokenizer behavior and CUDA memory are server smoke gates, not local mocks.

## File map

| File | Responsibility |
| --- | --- |
| `constraints/sft.txt` | Exact initial SFT-only dependency matrix |
| `pyproject.toml` | Declare the separate `sft` optional dependency group |
| `configs/sft.yaml` | All model, LoRA, optimizer, batching, checkpoint, and data defaults |
| `dispute_agent/training/sft_dataset.py` | Read public JSONL, verify manifest hashes/counts/protocol, attach tool schemas |
| `dispute_agent/training/sft_data.py` | Strict Qwen chat-template encoding and preflight report; no approximate fallback |
| `dispute_agent/training/sft_runtime.py` | Lazy imports, non-thinking tokenizer preparation, TRL/PEFT backend |
| `dispute_agent/training/train_sft.py` | Pydantic config, output safety, resume validation, run manifest lifecycle |
| `scripts/train_sft.py` | Fixture, preflight, and formal-training CLI modes |
| `README.md` | Accurate local/server commands and output description |
| `docs/experiments.md` | Record only the fixture checks actually executed locally |
| `docs/resume-evidence-checklist.md` | Point resume claims to manifests, metrics, and adapters |

---

### Task 1: Lock the SFT-only environment and expand the configuration contract

**Files:**
- Create: `constraints/sft.txt`
- Modify: `pyproject.toml`
- Modify: `configs/sft.yaml`
- Modify: `dispute_agent/training/train_sft.py`
- Modify: `tests/unit/training/test_sft_config.py`
- Modify: `tests/test_project_contract.py`

- [ ] **Step 1: Extend the existing configuration tests first**

Replace `tests/unit/training/test_sft_config.py` with:

```python
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
```

Extend `tests/test_project_contract.py` inside its existing test:

```python
    sft_pins = (ROOT / "constraints/sft.txt").read_text("utf-8")
    for pin in (
        "torch==2.8.0",
        "transformers==4.57.6",
        "trl==0.28.0",
        "peft==0.18.1",
        "accelerate==1.12.0",
    ):
        assert pin in sft_pins

    sft_extra = project["project"]["optional-dependencies"]["sft"]
    assert any(item.startswith("trl") for item in sft_extra)
    assert not any(item.startswith(("vllm", "verl", "agentlightning")) for item in sft_extra)
```

- [ ] **Step 2: Run the focused tests and verify the expected failures**

Run:

```powershell
..\..\.venv\Scripts\python.exe -m pytest tests/unit/training/test_sft_config.py tests/test_project_contract.py -q -p no:cacheprovider
```

Expected: FAIL because `global_batch_size`, `output_root`, `gradient_accumulation_steps()` and `constraints/sft.txt` do not exist.

- [ ] **Step 3: Add the SFT-only dependency group and exact initial matrix**

Add this optional dependency group to `pyproject.toml` without changing the existing `train` group:

```toml
sft = [
  "torch>=2.8,<2.9",
  "transformers>=4.57,<4.58",
  "trl>=0.28,<0.29",
  "peft>=0.18,<0.19",
  "accelerate>=1.12,<1.13",
]
```

Create `constraints/sft.txt`:

```text
torch==2.8.0
transformers==4.57.6
trl==0.28.0
peft==0.18.1
accelerate==1.12.0
datasets==4.4.1
tokenizers==0.22.2
```

This is the initial SFT matrix. The server smoke records the fully resolved environment; the file must not be described as GRPO-compatible before Phase 0.

- [ ] **Step 4: Replace `configs/sft.yaml` with the complete resolved defaults**

```yaml
model: Qwen/Qwen3-8B
data_dir: data/generated
bf16: true
load_in_4bit: false
load_in_8bit: false
lora:
  rank: 32
  alpha: 64
  dropout: 0.0
  target_modules: all-linear
assistant_only_loss: true
packing: false
seed: 20260817
data_seed: 20260817
max_length: 2048
per_device_train_batch_size: 1
per_device_eval_batch_size: 1
global_batch_size: 16
num_train_epochs: 3
learning_rate: 0.0002
lr_scheduler_type: cosine
warmup_ratio: 0.03
weight_decay: 0.0
max_grad_norm: 1.0
gradient_checkpointing: true
attention_implementation: sdpa
eval_strategy: epoch
save_strategy: epoch
save_total_limit: 2
logging_steps: 5
output_root: checkpoints/sft
```

- [ ] **Step 5: Replace the configuration models at the top of `dispute_agent/training/train_sft.py`**

Keep `load_sft_config`, remove the current file-writing `train_sft` stub, and define:

```python
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class LoraSettings(BaseModel):
    rank: int = Field(default=32, gt=0)
    alpha: int = Field(default=64, gt=0)
    dropout: float = Field(default=0.0, ge=0, lt=1)
    target_modules: str = "all-linear"


class SFTConfig(BaseModel):
    model: str = "Qwen/Qwen3-8B"
    data_dir: str = "data/generated"
    bf16: bool = True
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    lora: LoraSettings = Field(default_factory=LoraSettings)
    assistant_only_loss: bool = True
    packing: bool = False
    seed: int = 20260817
    data_seed: int = 20260817
    max_length: int = Field(default=2048, gt=0)
    per_device_train_batch_size: int = Field(default=1, gt=0)
    per_device_eval_batch_size: int = Field(default=1, gt=0)
    global_batch_size: int = Field(default=16, gt=0)
    num_train_epochs: float = Field(default=3.0, gt=0)
    learning_rate: float = Field(default=2e-4, gt=0)
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = Field(default=0.03, ge=0, lt=1)
    weight_decay: float = Field(default=0.0, ge=0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    gradient_checkpointing: bool = True
    attention_implementation: str = "sdpa"
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = Field(default=2, gt=0)
    logging_steps: int = Field(default=5, gt=0)
    output_root: str = "checkpoints/sft"

    @model_validator(mode="after")
    def validate_training_mode(self) -> "SFTConfig":
        if not self.bf16 or self.load_in_4bit or self.load_in_8bit:
            raise ValueError("SFT must use BF16 LoRA without 4/8-bit quantization")
        if not self.assistant_only_loss or self.packing:
            raise ValueError("SFT requires assistant-only loss with packing disabled")
        return self

    def gradient_accumulation_steps(self, world_size: int) -> int:
        micro_batch = world_size * self.per_device_train_batch_size
        if world_size < 1 or self.global_batch_size % micro_batch:
            raise ValueError("global_batch_size must be divisible by world_size * per-device batch")
        return self.global_batch_size // micro_batch


def load_sft_config(path: str | Path = "configs/sft.yaml") -> SFTConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return SFTConfig(**data)
```

- [ ] **Step 6: Run focused tests and commit**

Run:

```powershell
..\..\.venv\Scripts\python.exe -m pytest tests/unit/training/test_sft_config.py tests/test_project_contract.py -q -p no:cacheprovider
```

Expected: PASS.

```powershell
git add constraints/sft.txt pyproject.toml configs/sft.yaml dispute_agent/training/train_sft.py tests/unit/training/test_sft_config.py tests/test_project_contract.py
git commit -m "chore: define reproducible sft training contract"
```

---

### Task 2: Load only verified public SFT rows and attach tool schemas

**Files:**
- Create: `dispute_agent/training/sft_dataset.py`
- Create: `tests/unit/training/test_sft_dataset.py`

- [ ] **Step 1: Write one comprehensive dataset integrity test**

Create `tests/unit/training/test_sft_dataset.py`:

```python
import hashlib
import json
from pathlib import Path

import pytest

from dispute_agent.training.sft_dataset import DatasetIntegrityError, load_sft_dataset


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _row(index: int) -> dict:
    return {
        "fact_instance_id": f"fact-{index:06d}",
        "case_id": f"case-{index:06d}",
        "split": "sft_train",
        "ood_bucket": None,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": f"case {index}"},
            {"role": "assistant", "content": '{"action":"escalate"}'},
        ],
        "metadata": {"sft_category": "direct"},
    }


def _dataset_dir(tmp_path: Path) -> Path:
    train_rows = [_row(index) for index in range(1500)]
    val_rows = [{**_row(2000 + index), "split": "sft_val"} for index in range(150)]
    train_hash = _write_jsonl(tmp_path / "sft_train.jsonl", train_rows)
    val_hash = _write_jsonl(tmp_path / "sft_val.jsonl", val_rows)
    manifest = {
        "counts": {"sft_train": 1500, "sft_val": 150},
        "file_hashes": {
            "sft_train.jsonl": train_hash,
            "sft_val.jsonl": val_hash,
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_loader_builds_nested_public_subsets_and_rejects_hash_drift(tmp_path):
    data_dir = _dataset_dir(tmp_path)
    bundle_500 = load_sft_dataset(data_dir, 500)
    bundle_1000 = load_sft_dataset(data_dir, 1000)
    bundle_1500 = load_sft_dataset(data_dir, 1500)

    ids_500 = [row["case_id"] for row in bundle_500.train_rows]
    ids_1000 = [row["case_id"] for row in bundle_1000.train_rows]
    ids_1500 = [row["case_id"] for row in bundle_1500.train_rows]
    assert ids_500 == ids_1000[:500] == ids_1500[:500]
    assert ids_1000 == ids_1500[:1000]
    assert len(bundle_1500.val_rows) == 150
    assert all(len(row["tools"]) == 4 for row in bundle_500.train_rows)
    assert "ground_truth" not in json.dumps(bundle_500.train_rows)

    with (data_dir / "sft_train.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"tampered":true}\n')
    with pytest.raises(DatasetIntegrityError, match="hash"):
        load_sft_dataset(data_dir, 500)

    data_dir = _dataset_dir(tmp_path)
    train_path = data_dir / "sft_train.jsonl"
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["ground_truth"] = {"true_liability": "merchant"}
    replacement_hash = _write_jsonl(train_path, rows)
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["file_hashes"]["sft_train.jsonl"] = replacement_hash
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DatasetIntegrityError, match="hidden field"):
        load_sft_dataset(data_dir, 500)
```

- [ ] **Step 2: Run the test and verify import failure**

Run:

```powershell
..\..\.venv\Scripts\python.exe -m pytest tests/unit/training/test_sft_dataset.py -q -p no:cacheprovider
```

Expected: FAIL because `dispute_agent.training.sft_dataset` does not exist.

- [ ] **Step 3: Create the dataset loader with one clear public contract**

Create `dispute_agent/training/sft_dataset.py` with these public types and constants:

```python
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from dispute_agent.data.validators import validate_trace_messages
from dispute_agent.domain.schemas import DisputeGroundTruth


class DatasetIntegrityError(ValueError):
    pass


HIDDEN_KEYS = (set(DisputeGroundTruth.model_fields) - {"case_id"}) | {"ground_truth"}


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "check_logistics",
            "description": "查询订单物流签收状态。",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_buyer_history",
            "description": "查询买家历史投诉记录。",
            "parameters": {
                "type": "object",
                "properties": {"buyer_id": {"type": "string"}},
                "required": ["buyer_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_merchant_history",
            "description": "查询商家历史纠纷记录。",
            "parameters": {
                "type": "object",
                "properties": {"merchant_id": {"type": "string"}},
                "required": ["merchant_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_evidence",
            "description": "核验一条证据的真实性。",
            "parameters": {
                "type": "object",
                "properties": {"evidence_id": {"type": "string"}},
                "required": ["evidence_id"],
                "additionalProperties": False,
            },
        },
    },
]


@dataclass(frozen=True)
class SFTDatasetBundle:
    train_rows: list[dict]
    val_rows: list[dict]
    file_hashes: dict[str, str]
    manifest_sha256: str
```

Add these implementation functions in the same file:

```python
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise DatasetIntegrityError(f"invalid JSON in {path.name}:{line_number}") from exc
    return rows


def _all_keys(value) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(_all_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(_all_keys(item))
    return keys


def _validate_public_row(row: dict, file_name: str, index: int) -> dict:
    leaked = sorted(_all_keys(row) & HIDDEN_KEYS)
    if leaked or any(key.startswith("_") for key in row):
        raise DatasetIntegrityError(f"hidden field in {file_name} row {index}")
    errors = validate_trace_messages(row.get("messages", []))
    if errors:
        raise DatasetIntegrityError(f"invalid messages in {file_name} row {index}: {errors[0]}")
    return {**row, "tools": TOOL_SCHEMAS}


def load_sft_dataset(data_dir: str | Path, train_size: int | None) -> SFTDatasetBundle:
    root = Path(data_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise DatasetIntegrityError("manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = ("sft_train.jsonl", "sft_val.jsonl")
    actual_hashes: dict[str, str] = {}
    for name in names:
        path = root / name
        if not path.is_file():
            raise DatasetIntegrityError(f"{name} is missing")
        actual_hashes[name] = _sha256(path)
        if manifest.get("file_hashes", {}).get(name) != actual_hashes[name]:
            raise DatasetIntegrityError(f"hash mismatch for {name}")

    train_rows = _read_jsonl(root / names[0])
    val_rows = _read_jsonl(root / names[1])
    expected = manifest.get("counts", {})
    if len(train_rows) != expected.get("sft_train") or len(val_rows) != expected.get("sft_val"):
        raise DatasetIntegrityError("row count does not match manifest")
    if train_size is not None and train_size not in {500, 1000, 1500}:
        raise DatasetIntegrityError("train_size must be 500, 1000, or 1500")
    if train_size is not None and (len(train_rows), len(val_rows)) != (1500, 150):
        raise DatasetIntegrityError("formal SFT requires 1500 training rows and 150 validation rows")
    selected = train_rows if train_size is None else train_rows[:train_size]
    return SFTDatasetBundle(
        train_rows=[_validate_public_row(row, names[0], index) for index, row in enumerate(selected)],
        val_rows=[_validate_public_row(row, names[1], index) for index, row in enumerate(val_rows)],
        file_hashes=actual_hashes,
        manifest_sha256=_sha256(manifest_path),
    )
```

- [ ] **Step 4: Run the focused test and commit**

Run:

```powershell
..\..\.venv\Scripts\python.exe -m pytest tests/unit/training/test_sft_dataset.py -q -p no:cacheprovider
```

Expected: PASS.

```powershell
git add dispute_agent/training/sft_dataset.py tests/unit/training/test_sft_dataset.py
git commit -m "feat: load verified public sft datasets"
```

---

### Task 3: Replace approximate masking with a strict non-thinking preflight

**Files:**
- Rewrite: `dispute_agent/training/sft_data.py`
- Replace: `tests/unit/training/test_sft_data.py`
- Create: `tests/unit/training/test_sft_preflight.py`

- [ ] **Step 1: Keep one encoding test and add only the two high-risk preflight cases**

Replace `tests/unit/training/test_sft_data.py` with:

```python
from dispute_agent.training.sft_data import preprocess_example


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        assert kwargs["tools"]
        return {
            "input_ids": [10, 11, 12, 13, 14],
            "attention_mask": [1, 1, 1, 1, 1],
            "assistant_masks": [0, 0, 1, 0, 1],
        }


def test_preprocess_labels_only_assistant_tokens():
    encoded = preprocess_example(
        messages=[{"role": "user", "content": "case"}, {"role": "assistant", "content": "final"}],
        tools=[{"type": "function", "function": {"name": "check_logistics"}}],
        tokenizer=FakeTokenizer(),
        max_length=8,
    )
    assert encoded["labels"] == [-100, -100, 12, -100, 14]
```

Create `tests/unit/training/test_sft_preflight.py`:

```python
import pytest

from dispute_agent.training.sft_data import SFTPreflightError, preflight_dataset


class AuditTokenizer:
    texts = {
        1: "__SFT_SYSTEM_SENTINEL__",
        2: "__SFT_USER_SENTINEL__",
        3: "__SFT_ASSISTANT_SENTINEL__",
        4: "__SFT_TOOL_SENTINEL__",
        5: "__SFT_FINAL_SENTINEL__",
    }

    def __init__(self, mask=None, length=5):
        self.mask = mask or [0, 0, 1, 0, 1]
        self.length = length

    def apply_chat_template(self, messages, **kwargs):
        return {
            "input_ids": list(range(1, self.length + 1)),
            "attention_mask": [1] * self.length,
            "assistant_masks": (self.mask + [0] * self.length)[: self.length],
        }

    def decode(self, token_ids, **kwargs):
        return " ".join(self.texts.get(token_id, "ordinary") for token_id in token_ids)


ROW = {
    "case_id": "case-1",
    "messages": [{"role": "user", "content": "case"}, {"role": "assistant", "content": "final"}],
    "tools": [{"type": "function", "function": {"name": "check_logistics"}}],
}


def test_preflight_accepts_non_thinking_assistant_only_contract():
    report = preflight_dataset([ROW], AuditTokenizer(), max_length=8)
    assert report.checked_rows == 1
    assert report.max_token_length == 5
    assert report.supervised_tokens == 2


@pytest.mark.parametrize(
    ("tokenizer", "message"),
    [
        (AuditTokenizer(mask=[1, 0, 1, 0, 1]), "non-assistant"),
        (AuditTokenizer(length=9), "exceeds max_length"),
    ],
)
def test_preflight_rejects_role_leakage_and_overlength(tokenizer, message):
    with pytest.raises(SFTPreflightError, match=message):
        preflight_dataset([ROW], tokenizer, max_length=8)
```

- [ ] **Step 2: Run the tests and verify they fail against the approximate implementation**

Run:

```powershell
..\..\.venv\Scripts\python.exe -m pytest tests/unit/training/test_sft_data.py tests/unit/training/test_sft_preflight.py -q -p no:cacheprovider
```

Expected: FAIL because the current function has no `tools` parameter, truncates silently, and has no strict preflight.

- [ ] **Step 3: Rewrite `dispute_agent/training/sft_data.py` without a fallback mask**

```python
"""Strict non-thinking tokenization and assistant-only SFT preflight."""
from __future__ import annotations

from dataclasses import dataclass


class SFTPreflightError(ValueError):
    pass


@dataclass(frozen=True)
class PreflightReport:
    checked_rows: int
    max_token_length: int
    supervised_tokens: int


def preprocess_example(
    messages: list[dict],
    tools: list[dict],
    tokenizer,
    max_length: int,
) -> dict:
    encoded = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    input_ids = list(encoded["input_ids"])
    mask = encoded.get("assistant_masks", encoded.get("assistant_tokens_mask"))
    if mask is None or len(mask) != len(input_ids):
        raise SFTPreflightError("assistant mask is missing or has the wrong length")
    if len(input_ids) > max_length:
        raise SFTPreflightError(f"sample exceeds max_length: {len(input_ids)} > {max_length}")
    if not any(mask):
        raise SFTPreflightError("assistant mask is empty")
    labels = [token if bool(flag) else -100 for token, flag in zip(input_ids, mask, strict=True)]
    return {"input_ids": input_ids, "attention_mask": encoded["attention_mask"], "labels": labels}


def _audit_template(tokenizer, tools: list[dict], max_length: int) -> None:
    messages = [
        {"role": "system", "content": "__SFT_SYSTEM_SENTINEL__"},
        {"role": "user", "content": "__SFT_USER_SENTINEL__"},
        {
            "role": "assistant",
            "content": "__SFT_ASSISTANT_SENTINEL__",
            "tool_calls": [{
                "id": "call_0",
                "type": "function",
                "function": {"name": "check_logistics", "arguments": '{"order_id":"o-1"}'},
            }],
        },
        {"role": "tool", "tool_call_id": "call_0", "content": "__SFT_TOOL_SENTINEL__"},
        {"role": "assistant", "content": "__SFT_FINAL_SENTINEL__"},
    ]
    encoded = preprocess_example(messages, tools, tokenizer, max_length)
    supervised = [token for token, label in zip(encoded["input_ids"], encoded["labels"], strict=True) if label != -100]
    text = tokenizer.decode(supervised, skip_special_tokens=False)
    if "__SFT_SYSTEM_SENTINEL__" in text or "__SFT_USER_SENTINEL__" in text or "__SFT_TOOL_SENTINEL__" in text:
        raise SFTPreflightError("non-assistant tokens are supervised")
    if "__SFT_ASSISTANT_SENTINEL__" not in text or "__SFT_FINAL_SENTINEL__" not in text:
        raise SFTPreflightError("assistant tokens are not supervised")
    if "<think>" in text or "</think>" in text:
        raise SFTPreflightError("thinking tokens are supervised")


def preflight_dataset(rows: list[dict], tokenizer, max_length: int) -> PreflightReport:
    if not rows:
        raise SFTPreflightError("dataset is empty")
    _audit_template(tokenizer, rows[0]["tools"], max_length)
    max_seen = 0
    supervised = 0
    for row in rows:
        try:
            encoded = preprocess_example(row["messages"], row["tools"], tokenizer, max_length)
        except SFTPreflightError as exc:
            raise SFTPreflightError(f"case {row.get('case_id', '<missing>')}: {exc}") from exc
        max_seen = max(max_seen, len(encoded["input_ids"]))
        supervised += sum(label != -100 for label in encoded["labels"])
    return PreflightReport(len(rows), max_seen, supervised)
```

- [ ] **Step 4: Run the focused tests and commit**

Run:

```powershell
..\..\.venv\Scripts\python.exe -m pytest tests/unit/training/test_sft_data.py tests/unit/training/test_sft_preflight.py -q -p no:cacheprovider
```

Expected: PASS.

```powershell
git add dispute_agent/training/sft_data.py tests/unit/training/test_sft_data.py tests/unit/training/test_sft_preflight.py
git commit -m "feat: enforce strict sft template preflight"
```

---

### Task 4: Add the real TRL/PEFT backend and auditable run orchestration

**Files:**
- Create: `dispute_agent/training/sft_runtime.py`
- Complete: `dispute_agent/training/train_sft.py`
- Create: `tests/unit/training/test_sft_training.py`

- [ ] **Step 1: Write one fake-backend test covering configuration, artifacts, collision, and resume**

Create `tests/unit/training/test_sft_training.py`:

```python
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
        "messages": [{"role": "user", "content": "case"}, {"role": "assistant", "content": "final"}],
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
            cfg, _bundle(), train_size=500, output_dir=output, best_dir=best,
            backend=backend, world_size=2, rank=0, git_commit="abc123", environment={},
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
```

- [ ] **Step 2: Run the test and verify missing runtime/orchestration APIs**

Run:

```powershell
..\..\.venv\Scripts\python.exe -m pytest tests/unit/training/test_sft_training.py -q -p no:cacheprovider
```

Expected: FAIL because `sft_runtime.py`, `build_trainer_spec`, `RunError`, and `run_sft_training` do not exist.

- [ ] **Step 3: Create the pure Trainer specification and backend contracts**

Create `dispute_agent/training/sft_runtime.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dispute_agent.training.sft_data import PreflightReport, preflight_dataset
from dispute_agent.training.sft_dataset import SFTDatasetBundle


@dataclass(frozen=True)
class TrainerSpec:
    training_args: dict[str, Any]
    peft_args: dict[str, Any]


@dataclass(frozen=True)
class BackendResult:
    metrics: dict[str, float]
    best_checkpoint: str | None


@dataclass(frozen=True)
class TrainingRequest:
    config: Any
    bundle: SFTDatasetBundle
    output_dir: Path
    adapter_staging_dir: Path
    trainer_spec: TrainerSpec
    resume_checkpoint: Path | None


def build_trainer_spec(config, output_dir: Path, world_size: int, max_steps: int | None) -> TrainerSpec:
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    interval_strategy = "steps" if max_steps is not None else config.eval_strategy
    training_args = {
        "output_dir": str(output_dir),
        "bf16": True,
        "tf32": True,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps(world_size),
        "num_train_epochs": config.num_train_epochs,
        "max_steps": -1 if max_steps is None else max_steps,
        "learning_rate": config.learning_rate,
        "lr_scheduler_type": config.lr_scheduler_type,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "max_grad_norm": config.max_grad_norm,
        "gradient_checkpointing": config.gradient_checkpointing,
        "use_cache": False,
        "assistant_only_loss": True,
        "packing": False,
        "max_length": config.max_length,
        "eval_strategy": interval_strategy,
        "save_strategy": interval_strategy,
        "save_total_limit": config.save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "logging_steps": config.logging_steps,
        "report_to": "none",
        "seed": config.seed,
        "data_seed": config.data_seed,
        "ddp_find_unused_parameters": False,
        "model_init_kwargs": {"dtype": "bfloat16", "attn_implementation": config.attention_implementation},
    }
    if max_steps is not None:
        training_args.update(eval_steps=max_steps, save_steps=max_steps)
    peft_args = {
        "r": config.lora.rank,
        "lora_alpha": config.lora.alpha,
        "lora_dropout": config.lora.dropout,
        "target_modules": config.lora.target_modules,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    return TrainerSpec(training_args=training_args, peft_args=peft_args)
```

- [ ] **Step 4: Add the lazy real backend to `sft_runtime.py`**

Append:

```python
def prepare_training_tokenizer(model_name: str):
    from transformers import AutoTokenizer
    from trl.chat_template_utils import get_training_chat_template

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    template = get_training_chat_template(processing_class=tokenizer) or tokenizer.chat_template
    if not template or "{% generation %}" not in template or "{% endgeneration %}" not in template:
        raise RuntimeError("Qwen training template lacks assistant generation markers")
    tokenizer.chat_template = "{% set enable_thinking = false %}\n" + template
    return tokenizer


class RealSFTBackend:
    def __init__(self) -> None:
        from accelerate import PartialState

        self.state = PartialState()

    def barrier(self) -> None:
        self.state.wait_for_everyone()

    def preflight(self, config, bundle: SFTDatasetBundle) -> PreflightReport:
        tokenizer = prepare_training_tokenizer(config.model)
        return preflight_dataset(bundle.train_rows + bundle.val_rows, tokenizer, config.max_length)

    def train(self, request: TrainingRequest) -> BackendResult:
        import math

        from datasets import Dataset
        from peft import LoraConfig as PeftLoraConfig
        import torch
        from trl import SFTConfig as TRLSFTConfig, SFTTrainer

        self._validate_cuda(request.config)
        tokenizer = prepare_training_tokenizer(request.config.model)
        preflight_dataset(
            request.bundle.train_rows + request.bundle.val_rows,
            tokenizer,
            request.config.max_length,
        )
        training_args = dict(request.trainer_spec.training_args)
        training_args["model_init_kwargs"] = {
            **training_args["model_init_kwargs"],
            "dtype": torch.bfloat16,
        }
        trainer = SFTTrainer(
            model=request.config.model,
            args=TRLSFTConfig(**training_args),
            train_dataset=Dataset.from_list(request.bundle.train_rows),
            eval_dataset=Dataset.from_list(request.bundle.val_rows),
            processing_class=tokenizer,
            peft_config=PeftLoraConfig(**request.trainer_spec.peft_args),
        )
        train_result = trainer.train(
            resume_from_checkpoint=str(request.resume_checkpoint) if request.resume_checkpoint else None
        )
        metrics = {**train_result.metrics, **trainer.evaluate()}
        required = ("train_loss", "eval_loss")
        if any(name not in metrics or not math.isfinite(float(metrics[name])) for name in required):
            raise RuntimeError("training produced missing or non-finite losses")
        if trainer.is_world_process_zero():
            trainer.save_state()
            trainer.save_model(str(request.adapter_staging_dir))
            tokenizer.save_pretrained(str(request.adapter_staging_dir))
        return BackendResult(
            metrics={key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))},
            best_checkpoint=trainer.state.best_model_checkpoint,
        )

    def _validate_cuda(self, config) -> None:
        import torch

        if self.state.num_processes != 2:
            raise RuntimeError("formal SFT requires exactly two Accelerate processes")
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError("formal SFT requires CUDA")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("visible CUDA device does not support BF16")
        if config.load_in_4bit or config.load_in_8bit:
            raise RuntimeError("quantized loading is forbidden for this SFT run")
```

- [ ] **Step 5: Complete orchestration in `dispute_agent/training/train_sft.py`**

Below the configuration code, add the imports, public exception, and helpers:

```python
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from dispute_agent.training.sft_dataset import SFTDatasetBundle
from dispute_agent.training.sft_runtime import RealSFTBackend, TrainingRequest, build_trainer_spec


class RunError(RuntimeError):
    pass


REQUIRED_CHECKPOINT_FILES = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _resume_checkpoint(output_dir: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    checkpoints = list(output_dir.glob("checkpoint-*"))
    if value == "latest" and not checkpoints:
        raise RunError("no checkpoint exists for latest resume")
    candidate = max(checkpoints, key=lambda path: int(path.name.split("-")[-1])) if value == "latest" else Path(value)
    try:
        candidate.resolve().relative_to(output_dir.resolve())
    except (ValueError, FileNotFoundError) as exc:
        raise RunError("resume checkpoint must be inside the run output directory") from exc
    missing = [name for name in REQUIRED_CHECKPOINT_FILES if not (candidate / name).is_file()]
    if missing:
        raise RunError(f"resume checkpoint is incomplete: {missing}")
    return candidate


def _fingerprint(config: SFTConfig, bundle: SFTDatasetBundle, train_size: int) -> dict[str, Any]:
    return {
        "model": config.model,
        "train_size": train_size,
        "data_hashes": bundle.file_hashes,
        "manifest_sha256": bundle.manifest_sha256,
        "lora": config.lora.model_dump(),
        "max_length": config.max_length,
        "global_batch_size": config.global_batch_size,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.num_train_epochs,
    }
```

Then add the complete public orchestration function:

```python
def run_sft_training(
    config: SFTConfig,
    bundle: SFTDatasetBundle,
    *,
    train_size: int,
    output_dir: str | Path,
    best_dir: str | Path,
    backend=None,
    world_size: int = 1,
    rank: int = 0,
    max_steps: int | None = None,
    resume_from_checkpoint: str | None = None,
    git_commit: str = "unknown",
    environment: dict[str, Any] | None = None,
) -> Path:
    output = Path(output_dir)
    best = Path(best_dir)
    staging = output / "adapter-staging"
    runtime = backend or RealSFTBackend()
    existing_manifest = output / "run_manifest.json"
    resume = _resume_checkpoint(output, resume_from_checkpoint) if resume_from_checkpoint else None
    current_fingerprint = _fingerprint(config, bundle, train_size)

    if resume is None and output.exists() and any(output.iterdir()):
        raise RunError("output directory is non-empty; use explicit resume")
    if resume is None and best.exists():
        raise RunError("best adapter directory already exists")
    if staging.exists():
        raise RunError("adapter staging directory exists; archive it before retrying")
    if resume is not None:
        if not existing_manifest.is_file():
            raise RunError("resume requires run_manifest.json")
        previous = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if previous.get("fingerprint") != current_fingerprint:
            raise RunError("resume configuration or dataset does not match the original run")
        if best.exists():
            raise RunError("remove or archive the old best adapter before resume")

    spec = build_trainer_spec(config, output, world_size, max_steps)
    manifest = {
        "status": "running",
        "started_at": _now(),
        "git_commit": git_commit,
        "fingerprint": current_fingerprint,
        "data": {
            "manifest_sha256": bundle.manifest_sha256,
            "file_hashes": bundle.file_hashes,
            "train_rows": len(bundle.train_rows),
            "val_rows": len(bundle.val_rows),
        },
        "training": {
            "global_batch_size": config.global_batch_size,
            "world_size": world_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps(world_size),
            "trainer_spec": asdict(spec),
        },
        "environment": environment or {},
        "resume_checkpoint": str(resume) if resume else None,
    }
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        _write_json(existing_manifest, manifest)
    runtime.barrier()

    request = TrainingRequest(config, bundle, output, staging, spec, resume)
    try:
        result = runtime.train(request)
        runtime.barrier()
        if rank == 0:
            missing = [name for name in ("adapter_config.json", "adapter_model.safetensors") if not (staging / name).is_file()]
            if missing:
                raise RunError(f"best adapter is incomplete: {missing}")
            if any(not math.isfinite(float(value)) for value in result.metrics.values()):
                raise RunError("metrics contain NaN or Inf")
            _write_json(output / "metrics.json", result.metrics)
            staging.replace(best)
            manifest.update(
                status="complete",
                completed_at=_now(),
                best_checkpoint=result.best_checkpoint,
                metrics=result.metrics,
            )
            _write_json(existing_manifest, manifest)
    except Exception as exc:
        if rank == 0:
            manifest.update(status="failed", completed_at=_now(), error=f"{type(exc).__name__}: {exc}")
            _write_json(existing_manifest, manifest)
        raise
    return best
```

- [ ] **Step 6: Run the training orchestration test and commit**

Run:

```powershell
..\..\.venv\Scripts\python.exe -m pytest tests/unit/training/test_sft_training.py -q -p no:cacheprovider
```

Expected: PASS without importing TRL, PEFT, Transformers, Torch, or downloading a model.

```powershell
git add dispute_agent/training/sft_runtime.py dispute_agent/training/train_sft.py tests/unit/training/test_sft_training.py
git commit -m "feat: add real trl peft sft runtime"
```

---

### Task 5: Replace the CLI and document truthful local/server workflows

**Files:**
- Rewrite: `scripts/train_sft.py`
- Modify: `README.md`
- Modify: `docs/experiments.md`
- Modify: `docs/resume-evidence-checklist.md`

- [ ] **Step 1: Rewrite `scripts/train_sft.py` with three mutually exclusive modes**

Use this control flow; keep imports of the real backend lazy through `sft_runtime.py`:

```python
#!/usr/bin/env python3
"""Validate or run Qwen3-8B BF16 LoRA SFT."""
from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dispute_agent.training.sft_dataset import load_sft_dataset
from dispute_agent.training.sft_runtime import RealSFTBackend
from dispute_agent.training.train_sft import load_sft_config, run_sft_training


def _git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _environment() -> dict:
    packages = ("torch", "transformers", "trl", "peft", "accelerate")
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    gpu_names: list[str] = []
    gpu_memory_bytes: list[int] = []
    cuda_version = None
    try:
        import torch

        cuda_version = torch.version.cuda
        gpu_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        gpu_memory_bytes = [torch.cuda.get_device_properties(index).total_memory for index in range(torch.cuda.device_count())]
    except (ImportError, RuntimeError):
        pass
    return {
        "python": sys.version.split()[0],
        "packages": versions,
        "cuda": cuda_version,
        "gpu_names": gpu_names,
        "gpu_memory_bytes": gpu_memory_bytes,
        "argv": sys.argv,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--train-size", type=int, choices=[500, 1000, 1500], default=500)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fixture", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    config = load_sft_config(args.config)
    data_dir = args.data_dir or config.data_dir
    bundle = load_sft_dataset(data_dir, None if args.fixture else args.train_size)
    if args.fixture:
        print(f"SFT fixture passed: train={len(bundle.train_rows)} val={len(bundle.val_rows)}; no model loaded")
        return 0

    backend = RealSFTBackend()
    if args.preflight:
        report = backend.preflight(config, bundle)
        print(
            f"SFT preflight passed: rows={report.checked_rows} "
            f"max_tokens={report.max_token_length} supervised_tokens={report.supervised_tokens}"
        )
        return 0

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    output = Path(args.output_dir or Path(config.output_root) / f"sft-{args.train_size}")
    best_name = f"{output.name}-best" if args.output_dir else f"sft-{args.train_size}-best"
    best = output.parent / best_name
    result = run_sft_training(
        config,
        bundle,
        train_size=args.train_size,
        output_dir=output,
        best_dir=best,
        backend=backend,
        world_size=world_size,
        rank=rank,
        max_steps=args.max_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
        git_commit=_git_commit(),
        environment=_environment(),
    )
    if rank == 0:
        print(f"SFT training complete: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Update README commands and remove the old fake-training wording**

In `README.md`:

- replace `python scripts/train_sft.py --config configs/sft.yaml --fixture --max-steps 1` with:

```bash
python scripts/generate_data.py --seed 20260817 --fixture-size 24 --output artifacts/data-smoke
python scripts/train_sft.py --config configs/sft.yaml --data-dir artifacts/data-smoke --fixture
```

- add the SFT environment and server commands:

```bash
uv pip install -e ".[dev,sft]" -c constraints/sft.txt
python scripts/generate_data.py --seed 20260817 --output data/generated
python scripts/generate_data.py --freeze-test --output data/generated
python scripts/train_sft.py --train-size 500 --preflight
accelerate launch --num_processes 2 scripts/train_sft.py --train-size 500 --max-steps 2 --output-dir checkpoints/sft/smoke-500
accelerate launch --num_processes 2 scripts/train_sft.py --train-size 500
```

- state explicitly that `--fixture` validates config/data only, `--preflight` loads the tokenizer only, and only the Accelerate command trains model weights.

- [ ] **Step 3: Update evidence documents without inventing metrics**

In `docs/experiments.md`, change the local SFT entry to:

```markdown
- [x] SFT fixture 数据/配置检查：`python scripts/train_sft.py --config configs/sft.yaml --data-dir artifacts/data-smoke --fixture`；未加载模型，未产生训练指标。
```

Add these unchecked server items:

```markdown
- [ ] Qwen3 tokenizer non-thinking / assistant-mask preflight。
- [ ] 双 RTX 4090 两步 BF16 LoRA smoke 与 checkpoint 续训。
- [ ] SFT-500 / SFT-1000 / SFT-1500 正式训练。
```

In `docs/resume-evidence-checklist.md`, point the real SFT claim to:

```markdown
| TRL Qwen3-8B BF16 LoRA SFT 真实训练入口 | `dispute_agent/training/sft_runtime.py`, `configs/sft.yaml`, `constraints/sft.txt` |
| SFT 数据与运行可追溯性 | `run_manifest.json`, `metrics.json`, `sft-{size}-best/adapter_config.json` |
```

Keep all performance claims in the “待训练机完成后补充” section.

- [ ] **Step 4: Run the local fixture using fresh generated data**

Run:

```powershell
..\..\.venv\Scripts\python.exe scripts/generate_data.py --seed 20260817 --fixture-size 24 --output artifacts/sft-plan-smoke
..\..\.venv\Scripts\python.exe scripts/train_sft.py --config configs/sft.yaml --data-dir artifacts/sft-plan-smoke --fixture
```

Expected:

```text
Generated 24 rows into artifacts\sft-plan-smoke
SFT fixture passed: train=12 val=1; no model loaded
```

After recording the result, resolve and remove only the exact temporary directory `artifacts/sft-plan-smoke`; do not delete the `artifacts` root.

- [ ] **Step 5: Run focused tests and commit**

Run:

```powershell
..\..\.venv\Scripts\python.exe -m pytest tests/unit/training/test_sft_config.py tests/unit/training/test_sft_dataset.py tests/unit/training/test_sft_data.py tests/unit/training/test_sft_preflight.py tests/unit/training/test_sft_training.py -q -p no:cacheprovider
```

Expected: all focused SFT tests PASS.

```powershell
git add scripts/train_sft.py README.md docs/experiments.md docs/resume-evidence-checklist.md
git commit -m "docs: document real sft execution workflow"
```

---

### Task 6: Final local verification and server handoff

**Files:**
- Modify only if verification finds a defect: files already listed in Tasks 1–5
- Runtime only, do not commit: `artifacts/sft-final-smoke/`

- [ ] **Step 1: Run the complete local suite once**

Run:

```powershell
..\..\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Expected: all tests PASS. The existing Starlette deprecation warning is acceptable and unrelated to SFT.

- [ ] **Step 2: Run one final data/fixture smoke and clean only its exact output**

Run:

```powershell
..\..\.venv\Scripts\python.exe scripts/generate_data.py --seed 20260817 --fixture-size 24 --output artifacts/sft-final-smoke
..\..\.venv\Scripts\python.exe scripts/train_sft.py --config configs/sft.yaml --data-dir artifacts/sft-final-smoke --fixture
..\..\.venv\Scripts\python.exe scripts/generate_data.py --freeze-test --output artifacts/sft-final-smoke
```

Expected: generation, fixture, and first freeze snapshot all PASS. Resolve `artifacts/sft-final-smoke` to an absolute path inside this worktree before removing that exact directory.

- [ ] **Step 3: Check repository hygiene**

Run:

```powershell
git diff --check
git status --short
git ls-files | Select-String -Pattern '\.(safetensors|bin|pt|pth)$'
rg -n "TBD|TODO|FIXME" dispute_agent scripts configs README.md docs/experiments.md docs/resume-evidence-checklist.md
```

Expected: no whitespace errors, no committed model weights, no unresolved implementation markers, and only intentional source/document changes.

- [ ] **Step 4: Review the full branch diff against the design**

Run:

```powershell
git diff master...HEAD --stat
git diff master...HEAD -- configs/sft.yaml constraints/sft.txt dispute_agent/training scripts/train_sft.py
```

Verify all of the following manually:

- no code path reads `*.ground_truth.jsonl`;
- formal training uses BF16 and no quantization;
- `assistant_only_loss=True`, non-thinking template, `packing=False`;
- two GPUs yield gradient accumulation 8 and global batch 16;
- `--max-steps` defaults to full epoch training;
- non-empty output requires explicit matching resume;
- `sft-{size}-best` is created only after finite losses and complete adapter files;
- fixture/preflight output cannot be mistaken for real training completion.

- [ ] **Step 5: Commit any verification-only corrections, otherwise leave the branch clean**

If Step 1–4 required a correction, commit only those corrected files:

```powershell
git add <corrected-file-paths>
git commit -m "fix: close sft verification gaps"
```

If no correction was needed, do not create an empty commit.

- [ ] **Step 6: Execute the first server gate after copying the branch**

Run on Ubuntu 22.04 / Python 3.11 / two RTX 4090 GPUs:

```bash
uv pip install -e ".[dev,sft]" -c constraints/sft.txt
python scripts/generate_data.py --seed 20260817 --output data/generated
python scripts/generate_data.py --freeze-test --output data/generated
python scripts/train_sft.py --train-size 500 --preflight
accelerate launch --num_processes 2 scripts/train_sft.py --train-size 500 --max-steps 2 --output-dir checkpoints/sft/smoke-500
mv checkpoints/sft/smoke-500-best checkpoints/sft/smoke-500-step2-adapter
accelerate launch --num_processes 2 scripts/train_sft.py --train-size 500 --output-dir checkpoints/sft/smoke-500 --resume-from-checkpoint latest --max-steps 3
```

Expected:

- tokenizer preflight reports all 650 selected rows, a maximum length at or below 2048, and non-zero supervised tokens;
- both GPUs are visible and BF16 is supported;
- the two-step run writes a complete checkpoint and run manifest;
- the resumed run restores the checkpoint instead of restarting optimizer state;
- `checkpoints/sft/sft-500-best` is not claimed until a non-smoke formal run completes.

If dependency resolution, template patching, or memory fails, record the exact error and stop. Do not silently change to QLoRA, truncate traces, disable assistant-only loss, or report a completed SFT run.

---

## Commit sequence

1. `chore: define reproducible sft training contract`
2. `feat: load verified public sft datasets`
3. `feat: enforce strict sft template preflight`
4. `feat: add real trl peft sft runtime`
5. `docs: document real sft execution workflow`
6. Optional only after a discovered defect: `fix: close sft verification gaps`

## Completion boundary

Local implementation is ready for server validation when all local tests and fixture checks pass, the branch contains no generated data or model weights, and the CLI can distinguish fixture, tokenizer preflight, short formal training, and resumed training. The project may claim “implemented a real TRL BF16 LoRA training path” after code review; it may claim actual SFT training or quote metrics only after the server commands produce a valid adapter, manifest, and evaluation result.
