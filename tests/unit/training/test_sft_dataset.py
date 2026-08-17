import hashlib
import json
from pathlib import Path

import pytest

from dispute_agent.training.sft_dataset import DatasetIntegrityError, load_sft_dataset


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def test_loader_builds_nested_public_subsets_and_rejects_integrity_drift(tmp_path):
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
    rows[0]["messages"][1]["content"] += ' {"ground_truth":{"label":"merchant"}}'
    replacement_hash = _write_jsonl(train_path, rows)
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["file_hashes"]["sft_train.jsonl"] = replacement_hash
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DatasetIntegrityError, match="hidden field"):
        load_sft_dataset(data_dir, 500)
