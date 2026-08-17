#!/usr/bin/env python3
"""Generate leakage-safe synthetic dispute datasets.

Usage:
    python scripts/generate_data.py --seed 20260817 --output data/generated
    python scripts/generate_data.py --seed 20260817 --fixture-size 24 --output artifacts/data-smoke
    python scripts/generate_data.py --seed 20260817 --freeze-test --output data/generated
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dispute_agent.data.generator import FactInstance, generate_fact_instances
from dispute_agent.data.renderer import render_sft_trace
from dispute_agent.data.splits import DEFAULT_COUNTS, build_dataset_manifest
from dispute_agent.domain.schemas import Decision, Escalation


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_decision(instance: FactInstance) -> Decision | Escalation:
    gt = instance.ground_truth
    if gt.should_escalate:
        return Escalation(
            action="escalate",
            confidence=0.7,
            evidence_ids=[instance.observation.evidence[0].evidence_id],
            reason="证据不足或风险较高，建议人工复核",
        )
    low, high = gt.reasonable_compensation_range
    compensation = round((low + high) / 2, 2)
    return Decision(
        action="decide",
        liability=gt.true_liability,
        compensation=compensation,
        confidence=0.8,
        evidence_ids=[instance.observation.evidence[0].evidence_id],
        reason="根据公开证据判定责任",
    )


def _render_row(instance: FactInstance) -> dict:
    tool_plan = [("check_logistics", {"order_id": instance.observation.order_id})]
    return {
        "fact_instance_id": instance.fact_instance_id,
        "case_id": instance.case_id,
        "split": instance.split,
        "ood_bucket": instance.ood_bucket,
        "messages": render_sft_trace(instance, tool_plan=tool_plan, decision=_make_decision(instance)),
        "ground_truth": instance.ground_truth.model_dump(mode="json"),
        "metadata": instance.metadata,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_rows(seed: int, counts: dict[str, int], fixture_size: int | None) -> dict[str, list[dict]]:
    manifest = build_dataset_manifest(seed=seed, counts=counts, fixture_size=fixture_size)
    rows_by_split: dict[str, list[dict]] = {}
    cursor = 0
    for split_name, split_manifest in manifest.splits.items():
        count = split_manifest.count
        rows: list[dict] = []
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
            for inst in instances:
                inst.split = split_name
                rows.append(_render_row(inst))
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
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest = build_dataset_manifest(seed=args.seed, fixture_size=args.fixture_size)

    rows_by_split = _build_rows(args.seed, manifest.counts, args.fixture_size)
    file_hashes: dict[str, str] = {}
    for split_name, rows in rows_by_split.items():
        path = output / f"{split_name}.jsonl"
        _write_jsonl(path, rows)
        file_hashes[split_name] = _sha256_file(path)

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

    if args.freeze_test:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("file_hashes") != file_hashes:
            print("FREEZE TEST FAILED: file hashes changed")
            return 1
        print("FREEZE TEST PASSED")

    total = sum(len(v) for v in rows_by_split.values())
    print(f"Generated {total} rows into {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
