#!/usr/bin/env python3
"""Generate leakage-safe synthetic dispute datasets.

Usage:
    python scripts/generate_data.py --seed 20260817 --output data/generated
    python scripts/generate_data.py --seed 20260817 --fixture-size 24 --output artifacts/data-smoke
    python scripts/generate_data.py --seed 20260817 --freeze-test --output data/generated
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dispute_agent.data.generator import (
    FactInstance,
    SFTProfile,
    fact_fingerprint,
    generate_fact_instances,
    plan_sft_profiles,
)
from dispute_agent.data.renderer import render_sft_trace
from dispute_agent.data.splits import DEFAULT_COUNTS, build_dataset_manifest
from dispute_agent.data.validators import validate_trace_messages
from dispute_agent.domain.schemas import Decision, Escalation


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decision_evidence_ids(instance: FactInstance, tool_names: tuple[str, ...]) -> list[str]:
    evidence_ids = [instance.observation.evidence[0].evidence_id]
    derived = {
        "check_logistics": f"logistics:{instance.observation.order_id}",
        "check_buyer_history": f"buyer_history:{instance.observation.buyer_id}",
        "check_merchant_history": f"merchant_history:{instance.observation.merchant_id}",
        "verify_evidence": instance.observation.evidence[0].evidence_id,
    }
    for tool_name in tool_names:
        evidence_id = derived[tool_name]
        if evidence_id not in evidence_ids:
            evidence_ids.append(evidence_id)
    return evidence_ids


def _make_decision(instance: FactInstance, tool_names: tuple[str, ...]) -> Decision | Escalation:
    gt = instance.ground_truth
    evidence_ids = _decision_evidence_ids(instance, tool_names)
    reason = "结合公开材料与已调用工具结果完成判断" if tool_names else "根据公开证据完成判断"
    if gt.should_escalate:
        return Escalation(
            action="escalate",
            confidence=0.7,
            evidence_ids=evidence_ids,
            reason=f"{reason}；证据不足或风险较高，建议人工复核",
        )
    low, high = gt.reasonable_compensation_range
    compensation = round((low + high) / 2, 2)
    return Decision(
        action="decide",
        liability=gt.true_liability,
        compensation=compensation,
        confidence=0.8,
        evidence_ids=evidence_ids,
        reason=reason,
    )


def _tool_arguments(instance: FactInstance, tool_name: str) -> dict:
    return {
        "check_logistics": {"order_id": instance.observation.order_id},
        "check_buyer_history": {"buyer_id": instance.observation.buyer_id},
        "check_merchant_history": {"merchant_id": instance.observation.merchant_id},
        "verify_evidence": {"evidence_id": instance.observation.evidence[0].evidence_id},
    }[tool_name]


def _apply_edge_case(instance: FactInstance, profile: SFTProfile) -> None:
    if profile.category != "edge_case":
        return
    instance.ground_truth.should_escalate = True
    if profile.edge_case == "tool_failure":
        instance.ground_truth.tool_timeout_rate = 1.0
        instance.ground_truth.tool_missing_rate = 0.0
    elif profile.edge_case == "high_risk":
        instance.ground_truth.risk_level = "high"


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
    return {
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


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _freeze_or_verify(output: Path) -> int:
    manifest_path = output / "manifest.json"
    frozen_path = output / "frozen_manifest.json"
    if not manifest_path.exists():
        print("FREEZE TEST FAILED: manifest.json does not exist")
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared_hashes = manifest.get("file_hashes", {})
    actual_hashes: dict[str, str] = {}
    for file_name in declared_hashes:
        split_path = output / file_name
        if not split_path.exists():
            print(f"FREEZE TEST FAILED: missing {file_name}")
            return 1
        actual_hashes[file_name] = _sha256_file(split_path)

    if actual_hashes != declared_hashes:
        print("FREEZE TEST FAILED: generated files do not match manifest")
        return 1

    snapshot = {
        "manifest_sha256": _sha256_file(manifest_path),
        "file_hashes": actual_hashes,
    }
    if not frozen_path.exists():
        frozen_path.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("FREEZE TEST PASSED: snapshot created")
        return 0

    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen != snapshot:
        print("FREEZE TEST FAILED: frozen dataset hashes changed")
        return 1
    print("FREEZE TEST PASSED")
    return 0


def _build_rows(seed: int, counts: dict[str, int], fixture_size: int | None) -> dict[str, list[dict]]:
    manifest = build_dataset_manifest(seed=seed, counts=counts, fixture_size=fixture_size)
    rows_by_split: dict[str, list[dict]] = {}
    cursor = 0
    for split_name, split_manifest in manifest.splits.items():
        count = split_manifest.count
        rows: list[dict] = []
        profiles = (
            plan_sft_profiles(count, seed + cursor)
            if split_name in {"sft_train", "sft_val"}
            else [None] * count
        )
        if split_name == "ood_test":
            bucket_cursor = 0
            for bucket, bucket_count in manifest.ood_counts.items():
                instances = generate_fact_instances(
                    seed + bucket_cursor,
                    bucket_count,
                    start_id=cursor + bucket_cursor,
                    ood_bucket=bucket,
                    language_shift=(bucket == "language_style"),
                    tool_noise=(bucket == "tool_noise"),
                )
                for inst in instances:
                    inst.split = split_name
                    rows.append(_render_row(inst))
                bucket_cursor += bucket_count
        else:
            instances = generate_fact_instances(seed, count, start_id=cursor, ood_bucket=None)
            for inst, profile in zip(instances, profiles, strict=True):
                inst.split = split_name
                rows.append(_render_row(inst, profile))
        rows_by_split[split_name] = rows
        cursor += count
    return rows_by_split


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--output", type=str, default="data/generated")
    parser.add_argument("--fixture-size", type=int, default=None)
    parser.add_argument("--freeze-test", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if args.freeze_test:
        return _freeze_or_verify(output)

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest = build_dataset_manifest(seed=args.seed, fixture_size=args.fixture_size)

    rows_by_split = _build_rows(args.seed, manifest.counts, args.fixture_size)
    file_hashes: dict[str, str] = {}
    for split_name, rows in rows_by_split.items():
        public_path = output / f"{split_name}.jsonl"
        hidden_path = output / f"{split_name}.ground_truth.jsonl"
        public_rows = [
            {key: value for key, value in row.items() if not key.startswith("_")}
            for row in rows
        ]
        hidden_rows = [
            {"case_id": row["case_id"], "ground_truth": row["_ground_truth"]}
            for row in rows
        ]
        _write_jsonl(public_path, public_rows)
        _write_jsonl(hidden_path, hidden_rows)
        file_hashes[public_path.name] = _sha256_file(public_path)
        file_hashes[hidden_path.name] = _sha256_file(hidden_path)

    manifest_dict = {
        "seed": args.seed,
        "schema_version": manifest.schema_version,
        "counts": manifest.counts,
        "ood_counts": manifest.ood_counts,
        "human_audit": {
            "count": manifest.human_audit.count,
            "fact_instance_ids": manifest.human_audit.fact_instance_ids,
        },
        "splits": {
            name: {
                "count": split.count,
                "fact_instance_ids": split.fact_instance_ids,
            }
            for name, split in manifest.splits.items()
        },
        "file_hashes": file_hashes,
        "generation_config": {"seed": args.seed, "fixture_size": args.fixture_size},
    }
    manifest_text = json.dumps(manifest_dict, ensure_ascii=False, indent=2, sort_keys=True)
    manifest_path.write_text(manifest_text + "\n", encoding="utf-8")

    all_rows = [row for rows in rows_by_split.values() for row in rows]
    fingerprints = [row["_fact_fingerprint"] for row in all_rows]
    fingerprint_counts = Counter(fingerprints)
    quality_report = {
        "total_rows": len(all_rows),
        "duplicate_fact_fingerprints": sum(count - 1 for count in fingerprint_counts.values() if count > 1),
        "trace_validation_errors": sum(
            len(validate_trace_messages(row["messages"])) for row in all_rows
        ),
        "sft_category_counts": dict(
            Counter(row["metadata"]["sft_category"] for row in rows_by_split["sft_train"])
        ),
        "ood_bucket_counts": dict(Counter(row["ood_bucket"] for row in rows_by_split["ood_test"])),
    }
    (output / "quality_report.json").write_text(
        json.dumps(quality_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if quality_report["duplicate_fact_fingerprints"] or quality_report["trace_validation_errors"]:
        print("DATA QUALITY FAILED: see quality_report.json")
        return 1

    total = sum(len(v) for v in rows_by_split.values())
    print(f"Generated {total} rows into {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
