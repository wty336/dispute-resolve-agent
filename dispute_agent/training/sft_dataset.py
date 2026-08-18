"""经校验的公开 SFT trace 加载。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from dispute_agent.data.validators import validate_trace_messages
from dispute_agent.domain.schemas import DisputeGroundTruth


class DatasetIntegrityError(ValueError):
    """冻结的 SFT 数据不完整、被修改或不安全时抛出。"""


HIDDEN_KEYS = (set(DisputeGroundTruth.model_fields) - {"case_id"}) | {"ground_truth"}


def _tool_schema(name: str, description: str, argument: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {argument: {"type": "string"}},
                "required": [argument],
                "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS = [
    _tool_schema("check_logistics", "查询订单的公开物流记录。", "order_id"),
    _tool_schema("check_buyer_history", "查询买家的公开历史摘要。", "buyer_id"),
    _tool_schema("check_merchant_history", "查询商家的公开历史摘要。", "merchant_id"),
    _tool_schema("verify_evidence", "核验一项公开证据。", "evidence_id"),
]


@dataclass(frozen=True)
class SFTDatasetBundle:
    train_rows: list[dict[str, Any]]
    val_rows: list[dict[str, Any]]
    file_hashes: dict[str, str]
    manifest_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetIntegrityError(
                    f"invalid JSON at {path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise DatasetIntegrityError(
                    f"expected object at {path}:{line_number}, got {type(row).__name__}"
                )
            rows.append(row)
    return rows


def _all_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _all_keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _all_keys(item)}
    return set()


def _validate_public_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = _all_keys(row)
    leaked = sorted(keys & HIDDEN_KEYS)
    serialized = json.dumps(row, ensure_ascii=False).lower()
    embedded = sorted(key for key in HIDDEN_KEYS if key.lower() in serialized)
    private = sorted(key for key in keys if key.startswith("_"))
    if leaked or embedded or private:
        fields = sorted(set(leaked + embedded + private))
        raise DatasetIntegrityError(f"public SFT row contains hidden field(s): {fields}")

    messages = row.get("messages")
    if not isinstance(messages, list):
        raise DatasetIntegrityError("public SFT row must contain a messages list")
    errors = validate_trace_messages(messages)
    if errors:
        raise DatasetIntegrityError(f"invalid public SFT messages: {'; '.join(errors)}")
    return {**row, "tools": TOOL_SCHEMAS}


def load_sft_dataset(data_dir: str | Path, train_size: int | None) -> SFTDatasetBundle:
    """加载经清单校验的公开训练/验证数据。

    正式运行要求冻结的 1500/150 语料，并从中确定性选择 500、1000 或 1500
    条训练样本的前缀。``None`` 仅用于本地 fixture 检查，此时加载所有可用行。
    """

    root = Path(data_dir)
    manifest_path = root / "manifest.json"
    train_path = root / "sft_train.jsonl"
    val_path = root / "sft_val.jsonl"
    for path in (manifest_path, train_path, val_path):
        if not path.is_file():
            raise DatasetIntegrityError(f"required SFT data file is missing: {path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise DatasetIntegrityError(f"cannot read SFT manifest: {exc}") from exc

    declared_hashes = manifest.get("file_hashes", {})
    actual_hashes: dict[str, str] = {}
    for path in (train_path, val_path):
        expected = declared_hashes.get(path.name)
        actual = _sha256(path)
        actual_hashes[path.name] = actual
        if not expected or expected != actual:
            raise DatasetIntegrityError(
                f"hash mismatch for {path.name}: expected {expected!r}, got {actual!r}"
            )

    train_rows = _read_jsonl(train_path)
    val_rows = _read_jsonl(val_path)
    counts = manifest.get("counts", {})
    if counts.get("sft_train") != len(train_rows) or counts.get("sft_val") != len(val_rows):
        raise DatasetIntegrityError(
            "manifest counts do not match sft_train.jsonl and sft_val.jsonl"
        )

    allowed_sizes = {500, 1000, 1500}
    if train_size is not None:
        if train_size not in allowed_sizes:
            raise DatasetIntegrityError(
                f"formal train_size must be one of {sorted(allowed_sizes)}, got {train_size}"
            )
        if len(train_rows) != 1500 or len(val_rows) != 150:
            raise DatasetIntegrityError(
                "formal SFT runs require exactly 1500 train and 150 validation rows"
            )
        train_rows = train_rows[:train_size]

    return SFTDatasetBundle(
        train_rows=[_validate_public_row(row) for row in train_rows],
        val_rows=[_validate_public_row(row) for row in val_rows],
        file_hashes=actual_hashes,
        manifest_sha256=_sha256(manifest_path),
    )
