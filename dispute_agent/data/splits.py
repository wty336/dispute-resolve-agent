"""Deterministic dataset split manifests with strict fact-instance isolation."""
from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_COUNTS = {
    "sft_train": 1500,
    "sft_val": 150,
    "grpo_train": 700,
    "grpo_val": 100,
    "id_test": 400,
    "ood_test": 200,
}

OOD_BUCKETS = {
    "unseen_combination": 80,
    "language_style": 60,
    "tool_noise": 60,
}


@dataclass
class SplitManifest:
    name: str
    count: int
    fact_instance_ids: list[str] = field(default_factory=list)


@dataclass
class HumanAudit:
    count: int
    fact_instance_ids: list[str] = field(default_factory=list)


@dataclass
class DatasetManifest:
    counts: dict[str, int]
    ood_counts: dict[str, int]
    splits: dict[str, SplitManifest]
    human_audit: HumanAudit
    seed: int
    schema_version: str = "1.0"
    generation_config: dict = field(default_factory=dict)


def _scale_counts(counts: dict[str, int], target: int) -> dict[str, int]:
    """Scale a dict of counts to an exact target using largest remainder."""
    total = sum(counts.values())
    if total <= 0 or target <= 0:
        return {name: 0 for name in counts}
    raw = {name: value * target / total for name, value in counts.items()}
    scaled = {name: int(value) for name, value in raw.items()}
    remaining = target - sum(scaled.values())
    order = sorted(counts, key=lambda name: raw[name] - scaled[name], reverse=True)
    for i in range(remaining):
        scaled[order[i % len(order)]] += 1
    return scaled


def build_dataset_manifest(
    seed: int = 20260817,
    counts: dict[str, int] | None = None,
    fixture_size: int | None = None,
) -> DatasetManifest:
    """Build a deterministic manifest.

    When ``fixture_size`` is provided, all full-size counts are scaled down
    proportionally so the total is exactly ``fixture_size``.
    """
    counts = dict(DEFAULT_COUNTS if counts is None else counts)
    if fixture_size is not None:
        counts = _scale_counts(counts, fixture_size)

    total = sum(counts.values())
    ids = [f"fact-{i:06d}" for i in range(total)]
    cursor = 0
    splits: dict[str, SplitManifest] = {}
    for name, count in counts.items():
        split_ids = ids[cursor : cursor + count]
        cursor += count
        splits[name] = SplitManifest(name=name, count=count, fact_instance_ids=split_ids)

    ood_ids = splits["ood_test"].fact_instance_ids
    ood_counts = _scale_counts(OOD_BUCKETS, len(ood_ids))
    # Ensure ood buckets never exceed the actual ood split size.
    ood_cursor = 0
    for bucket, bucket_count in list(ood_counts.items()):
        ood_counts[bucket] = min(bucket_count, len(ood_ids) - ood_cursor)
        ood_cursor += ood_counts[bucket]

    id_test_ids = splits["id_test"].fact_instance_ids
    audit_count = min(100, len(id_test_ids) + len(ood_ids))
    audit_ids = id_test_ids[: audit_count // 2] + ood_ids[: audit_count - audit_count // 2]
    human_audit = HumanAudit(count=len(audit_ids), fact_instance_ids=audit_ids)

    return DatasetManifest(
        counts=counts,
        ood_counts=ood_counts,
        splits=splits,
        human_audit=human_audit,
        seed=seed,
        schema_version="1.0",
        generation_config={"seed": seed, "fixture_size": fixture_size},
    )
