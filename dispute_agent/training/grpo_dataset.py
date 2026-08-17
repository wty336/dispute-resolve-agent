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
    """Reconstruct a fresh, private environment for every rollout."""

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
    rows: list[dict[str, Any]] = []
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise GRPODatasetError(f"unable to read GRPO file: {path}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GRPODatasetError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise GRPODatasetError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _load_split(
    root: Path,
    split: str,
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, DisputeObservation], dict[str, DisputeGroundTruth], dict[str, str]]:
    public_path = root / f"{split}.jsonl"
    hidden_path = root / f"{split}.ground_truth.jsonl"
    hashes: dict[str, str] = {}
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

    hidden_by_id: dict[str, Any] = {}
    for row in hidden_rows:
        case_id = row.get("case_id")
        if not case_id or case_id in hidden_by_id:
            raise GRPODatasetError(f"invalid or duplicate hidden case_id in {split}: {case_id!r}")
        hidden_by_id[case_id] = row.get("ground_truth")

    observations: dict[str, DisputeObservation] = {}
    truths: dict[str, DisputeGroundTruth] = {}
    for row in public_rows:
        case_id = row.get("case_id")
        if not case_id or case_id in observations or case_id not in hidden_by_id:
            raise GRPODatasetError(f"invalid or unmatched case_id in {split}: {case_id!r}")
        try:
            observation = DisputeObservation.model_validate(row.get("observation"))
            truth = DisputeGroundTruth.model_validate(hidden_by_id[case_id])
        except Exception as exc:
            raise GRPODatasetError(f"invalid GRPO schema for case_id {case_id}") from exc
        if observation.case_id != case_id or truth.case_id != case_id:
            raise GRPODatasetError(f"case_id mismatch in {split}: {case_id}")
        observations[case_id] = observation
        truths[case_id] = truth
    if set(hidden_by_id) != set(observations):
        raise GRPODatasetError(f"public/hidden case_id mismatch in {split}")
    return public_rows, observations, truths, hashes


def _tasks(rows: list[dict[str, Any]], curriculum_phase: int) -> list[dict[str, Any]]:
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
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GRPODatasetError(f"invalid GRPO manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise GRPODatasetError("GRPO manifest must be an object")

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
    visible_train_rows = train_rows[:2] if profile == "smoke" else train_rows
    visible_val_rows = val_rows[: min(2, len(val_rows))] if profile == "smoke" else val_rows

    return GRPODatasetBundle(
        train_tasks=_tasks(visible_train_rows, curriculum_phase),
        val_tasks=_tasks(visible_val_rows, curriculum_phase),
        episode_source=EpisodeSource({**train_obs, **val_obs}, {**train_truth, **val_truth}),
        file_hashes={**train_hashes, **val_hashes},
        manifest_sha256=_sha256(manifest_path),
    )
