import hashlib
import json
from pathlib import Path

import pytest

from dispute_agent.training.grpo_dataset import GRPODatasetError, load_grpo_dataset


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
    bundle = load_grpo_dataset(tmp_path := _dataset_dir(tmp_path), profile="smoke", curriculum_phase=1)

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
