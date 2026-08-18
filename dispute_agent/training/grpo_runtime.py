"""Auditable Agent Lightning/verl GRPO runtime with lazy GPU dependencies."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import asyncio
import hashlib
import importlib
import json
from pathlib import Path
import statistics
from typing import Any

import yaml

from dispute_agent.training.grpo_config import GRPOConfig, load_grpo_config
from dispute_agent.training.grpo_dataset import GRPODatasetBundle, load_grpo_dataset
from dispute_agent.training.lightning_agent import build_lightning_agent


PACKAGE_EXPECTATIONS = {
    "torch": "2.8.0",
    "torchvision": "0.23.0",
    "transformers": "4.55.4",
    "peft": "0.18.1",
    "accelerate": "1.10.1",
    "flash-attn": "2.8.3",
    "vllm": "0.10.2",
    "verl": "0.5.0",
    "agentlightning": "0.3.0",
    "openai-agents": "0.6.0",
}


class GRPORuntimeError(RuntimeError):
    """Raised when a GRPO run would be unsafe or unauditable."""


@dataclass(frozen=True)
class GRPORunRequest:
    config_path: Path
    data_dir: Path
    output_root: Path
    run_id: str
    profile: str
    curriculum_phase: int
    input_adapter: Path | None = None
    max_steps: int | None = None
    resume: bool = False


@dataclass(frozen=True)
class GRPOTrainingResult:
    run_dir: Path
    manifest_path: Path
    dry_run: bool
    verl_config: dict[str, Any]


@dataclass(frozen=True)
class GRPORunPlan:
    request: GRPORunRequest
    config: GRPOConfig
    bundle: GRPODatasetBundle
    run_dir: Path
    verl_config: dict[str, Any]
    fingerprint: dict[str, Any]
    adapter_info: dict[str, Any] | None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _effective_config(config: GRPOConfig, adapter: Path | None) -> GRPOConfig:
    if adapter is None:
        return config
    raw = config.model_dump(mode="python")
    raw["actor_rollout_ref"]["model"]["lora_adapter_path"] = str(adapter)
    return GRPOConfig.model_validate(raw)


def _nearest_run_manifests(adapter: Path):
    current = adapter if adapter.is_dir() else adapter.parent
    yield from (current / "run_manifest.json",) if (current / "run_manifest.json").is_file() else ()
    for parent in current.parents:
        candidate = parent / "run_manifest.json"
        if candidate.is_file():
            yield candidate


def _provenance_target_modules(manifest: dict[str, Any]) -> str | None:
    candidates = [
        manifest.get("training", {}).get("trainer_spec", {}).get("peft_args", {}).get("target_modules"),
        manifest.get("resolved_config", {}).get("actor_rollout_ref", {}).get("model", {}).get("target_modules"),
        manifest.get("resolved_config", {}).get("lora", {}).get("target_modules"),
    ]
    for value in candidates:
        if isinstance(value, str):
            return value
    return None


def validate_input_adapter(adapter_dir: str | Path, config: GRPOConfig) -> dict[str, Any]:
    """Validate PEFT metadata, immutable weights, and all-linear provenance."""
    adapter = Path(adapter_dir)
    if not adapter.is_dir():
        raise GRPORuntimeError(f"input adapter directory is missing: {adapter}")
    config_path = adapter / "adapter_config.json"
    if not config_path.is_file():
        raise GRPORuntimeError("input adapter requires adapter_config.json")
    try:
        adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GRPORuntimeError("input adapter_config.json is invalid") from exc
    if not isinstance(adapter_config, dict):
        raise GRPORuntimeError("input adapter_config.json must be an object")
    if any(key in adapter_config and adapter_config[key] not in (None, False, 0) for key in ("quantization_config", "load_in_4bit", "load_in_8bit")):
        raise GRPORuntimeError("quantization fields are not allowed in the input adapter")

    model_cfg = config.actor_rollout_ref.model
    checks = {
        "base_model_name_or_path": (adapter_config.get("base_model_name_or_path"), model_cfg.path),
        "r": (adapter_config.get("r"), model_cfg.lora_rank),
        "lora_alpha": (adapter_config.get("lora_alpha"), model_cfg.lora_alpha),
        "lora_dropout": (adapter_config.get("lora_dropout", 0.0), model_cfg.lora_dropout),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            raise GRPORuntimeError(f"input adapter {name} mismatch: expected {expected!r}, got {actual!r}")
    weights = sorted(adapter.glob("*.safetensors"))
    if not weights:
        raise GRPORuntimeError("input adapter requires at least one safetensors weight")

    provenance = None
    provenance_path = None
    for candidate in _nearest_run_manifests(adapter):
        try:
            document = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if _provenance_target_modules(document) == "all-linear":
            provenance = "all-linear"
            provenance_path = str(candidate)
            break
    if provenance is None:
        raise GRPORuntimeError("input adapter is missing all-linear provenance")

    return {
        "path": str(adapter.resolve()),
        "adapter_config_sha256": _sha256(config_path),
        "weight_hashes": {path.name: _sha256(path) for path in weights},
        "provenance_path": provenance_path,
        "target_modules_provenance": provenance,
    }


def build_run_plan(request: GRPORunRequest, *, validate_adapter: bool = False) -> GRPORunPlan:
    if not request.run_id or Path(request.run_id).name != request.run_id:
        raise GRPORuntimeError("run_id must be a non-empty single directory name")
    if request.profile not in {"smoke", "formal"}:
        raise GRPORuntimeError(f"unknown profile: {request.profile}")
    if request.curriculum_phase not in {1, 2}:
        raise GRPORuntimeError("curriculum_phase must be 1 or 2")
    if request.max_steps is not None and request.max_steps <= 0:
        raise GRPORuntimeError("max_steps must be positive")

    config_path = Path(request.config_path)
    data_dir = Path(request.data_dir)
    config = _effective_config(load_grpo_config(config_path), request.input_adapter)
    bundle = load_grpo_dataset(data_dir, profile=request.profile, curriculum_phase=request.curriculum_phase)
    adapter_info = None
    if validate_adapter:
        adapter_info = validate_input_adapter(config.actor_rollout_ref.model.lora_adapter_path, config)
    verl_config = config.to_verl_config(profile=request.profile, output_dir=request.output_root / request.run_id)
    if request.profile == "smoke":
        verl_config["trainer"]["total_training_steps"] = 1
    if request.max_steps is not None:
        verl_config["trainer"]["total_training_steps"] = request.max_steps

    adapter_path = Path(config.actor_rollout_ref.model.lora_adapter_path)
    fingerprint = {
        "config_sha256": _sha256(config_path),
        "data_manifest_sha256": bundle.manifest_sha256,
        "data_file_hashes": bundle.file_hashes,
        "adapter_path": str(adapter_path),
        "adapter": adapter_info,
        "profile": request.profile,
        "curriculum_phase": request.curriculum_phase,
        "max_steps": request.max_steps,
    }
    return GRPORunPlan(
        request=request,
        config=config,
        bundle=bundle,
        run_dir=Path(request.output_root) / request.run_id,
        verl_config=verl_config,
        fingerprint=fingerprint,
        adapter_info=adapter_info,
    )


def _get(value: object, name: str, default: object = None) -> object:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _safe_annotations(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    allowed = {
        "case_id", "scenario_id", "curriculum_phase", "reward", "components",
        "tool_call_count", "thinking_enabled", "terminal", "event", "tool_name",
    }
    result: dict[str, Any] = {}
    for key in allowed:
        if key not in value:
            continue
        item = value[key]
        if key == "components" and isinstance(item, dict):
            component_names = {
                "liability", "compensation", "escalation", "grounding",
                "escalation_quality", "normalized_tool_cost", "invalid_action_penalty", "hard_failure",
            }
            result[key] = {name: item[name] for name in component_names if name in item and isinstance(item[name], (int, float, bool))}
        elif isinstance(item, (str, int, float, bool)) or item is None:
            result[key] = item
    return result


def _span_objects(spans: list[object]) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for span in spans:
        if _get(span, "name") != "agentlightning.object":
            continue
        attributes = _get(span, "attributes", {})
        if not isinstance(attributes, dict):
            continue
        raw = attributes.get("agentlightning.object.json")
        if not isinstance(raw, str):
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            continue
        objects.append(_safe_annotations(value))
    return objects


def _span_annotations(spans: list[object]) -> dict[str, Any]:
    annotations: dict[str, Any] = {}
    for value in _span_objects(spans):
        if "reward" in value:
            annotations.update(value)
    return annotations


def _status_text(status: object) -> str:
    value = getattr(status, "value", status)
    return str(value).lower()


def _latest_verl_checkpoint(run_dir: Path) -> Path | None:
    checkpoints_dir = run_dir / "checkpoints"
    candidates: list[tuple[int, Path]] = []
    if checkpoints_dir.is_dir():
        for candidate in checkpoints_dir.glob("global_step_*"):
            if not candidate.is_dir():
                continue
            try:
                step = int(candidate.name.removeprefix("global_step_"))
            except ValueError:
                continue
            candidates.append((step, candidate))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


async def _export_rollouts(
    store: object,
    metrics_path: Path,
    *,
    find_final_reward,
    find_reward_spans,
) -> dict[str, Any]:
    rollouts = list(await store.query_rollouts()) if hasattr(store, "query_rollouts") else []
    rewards: list[float] = []
    tool_call_counts: list[int] = []
    failures = 0
    model_span_count = 0
    object_span_count = 0
    tool_event_span_count = 0
    reward_span_count = 0
    multi_turn_rollout_count = 0
    tool_rollout_count = 0
    thinking_enabled_count = 0
    single_reward_rollout_count = 0
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        for rollout in rollouts:
            rollout_id = _get(rollout, "rollout_id", _get(rollout, "id", "unknown"))
            spans = list(await store.query_spans(rollout_id)) if hasattr(store, "query_spans") else []
            status = _status_text(_get(rollout, "status", "unknown"))
            reward_value = find_final_reward(spans)
            reward_spans = list(find_reward_spans(spans))
            reward_span_count += len(reward_spans)
            if len(reward_spans) == 1:
                single_reward_rollout_count += 1
            span_objects = _span_objects(spans)
            annotations = _span_annotations(spans)
            rollout_tool_event_spans = sum(
                1 for value in span_objects if value.get("event") == "tool_call"
            )
            tool_event_span_count += rollout_tool_event_spans
            rollout_object_spans = sum(
                1 for span in spans if _get(span, "name") == "agentlightning.object"
            )
            object_span_count += rollout_object_spans
            rollout_model_spans = sum(
                1 for span in spans if _get(span, "name") == "openai.chat.completion"
            )
            model_span_count += rollout_model_spans
            if rollout_model_spans >= 2:
                multi_turn_rollout_count += 1
            if isinstance(reward_value, (int, float)):
                rewards.append(float(reward_value))
            tool_call_count = annotations.get("tool_call_count")
            if isinstance(tool_call_count, int) and not isinstance(tool_call_count, bool):
                tool_call_counts.append(tool_call_count)
                if tool_call_count > 0:
                    tool_rollout_count += 1
            if annotations.get("thinking_enabled") is True:
                thinking_enabled_count += 1
            if status not in {"completed", "success", "succeeded"}:
                failures += 1
            record = {
                "rollout_id": str(rollout_id),
                "status": status,
                "final_reward": reward_value,
                "annotations": annotations,
                "span_count": len(spans),
                "model_span_count": rollout_model_spans,
                "object_span_count": rollout_object_spans,
                "tool_event_span_count": rollout_tool_event_spans,
                "reward_span_count": len(reward_spans),
                "started_at": _get(rollout, "start_time", _get(rollout, "started_at")),
                "completed_at": _get(rollout, "end_time", _get(rollout, "completed_at")),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {
        "rollout_count": len(rollouts),
        "completed_count": len(rollouts) - failures,
        "failure_count": failures,
        "reward_count": len(rewards),
        "reward_span_count": reward_span_count,
        "model_span_count": model_span_count,
        "object_span_count": object_span_count,
        "tool_event_span_count": tool_event_span_count,
        "multi_turn_rollout_count": multi_turn_rollout_count,
        "tool_rollout_count": tool_rollout_count,
        "thinking_enabled_count": thinking_enabled_count,
        "single_reward_rollout_count": single_reward_rollout_count,
        "tool_call_mean": statistics.fmean(tool_call_counts) if tool_call_counts else None,
        "reward_mean": statistics.fmean(rewards) if rewards else None,
        "reward_std": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0 if rewards else None,
    }


def run_grpo_training(
    request: GRPORunRequest,
    *,
    agl_module: object | None = None,
    dry_run: bool = False,
    validate_adapter: bool | None = None,
) -> GRPOTrainingResult:
    """Prepare or execute one auditable Agent Lightning/verl run."""
    if validate_adapter is None:
        validate_adapter = not dry_run
    try:
        plan = build_run_plan(request, validate_adapter=validate_adapter)
    except Exception as exc:
        if not dry_run and request.run_id and Path(request.run_id).name == request.run_id:
            failed_dir = Path(request.output_root) / request.run_id
            failed_manifest = failed_dir / "run_manifest.json"
            try:
                _write_json_atomic(failed_manifest, {
                    "status": "failed",
                    "started_at": _now(),
                    "completed_at": _now(),
                    "error": f"{type(exc).__name__}: {exc}",
                    "inputs": {
                        "config_path": str(Path(request.config_path).resolve()),
                        "data_dir": str(Path(request.data_dir).resolve()),
                        "input_adapter": str(request.input_adapter) if request.input_adapter else None,
                    },
                })
            except OSError:
                pass
        raise
    run_dir = plan.run_dir
    manifest_path = run_dir / "run_manifest.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint: Path | None = None
    if not dry_run:
        if request.resume:
            if not manifest_path.is_file():
                raise GRPORuntimeError("resume requires run_manifest.json")
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if previous.get("fingerprint") != plan.fingerprint:
                raise GRPORuntimeError("resume configuration, data, or adapter does not match original run")
            resume_checkpoint = _latest_verl_checkpoint(run_dir)
            if resume_checkpoint is None:
                raise GRPORuntimeError("resume requires an existing VERL global_step_* checkpoint")
            plan.verl_config["trainer"]["resume_mode"] = "auto"
        elif any(run_dir.iterdir()):
            raise GRPORuntimeError("output directory is non-empty; use --resume")
    _write_yaml(run_dir / "resolved_config.yaml", plan.config.model_dump(mode="python"))
    _write_yaml(run_dir / "verl_config.yaml", plan.verl_config)

    if dry_run:
        _write_json_atomic(run_dir / "run_plan.json", {
            "status": "planned",
            "created_at": _now(),
            "fingerprint": plan.fingerprint,
            "verl_config": plan.verl_config,
        })
        return GRPOTrainingResult(run_dir, manifest_path, True, plan.verl_config)

    manifest = {
        "status": "running",
        "started_at": _now(),
        "fingerprint": plan.fingerprint,
        "inputs": {
            "config_path": str(Path(request.config_path).resolve()),
            "data_dir": str(Path(request.data_dir).resolve()),
            "adapter": plan.adapter_info,
        },
        "resolved_config": plan.config.model_dump(mode="python"),
        "packages": PACKAGE_EXPECTATIONS,
        "resumed_from": str(resume_checkpoint.resolve()) if resume_checkpoint else None,
    }
    _write_json_atomic(manifest_path, manifest)

    try:
        agl = agl_module or importlib.import_module("agentlightning")
        store = agl.InMemoryLightningStore()
        algorithm = agl.VERL(plan.verl_config)
        trainer = agl.Trainer(
            algorithm=algorithm,
            n_runners=plan.config.agentlightning.n_runners,
            tracer=agl.OtelTracer(),
            adapter=agl.LlmProxyTraceToTriplet(),
            store=store,
        )
        agent_config_path = run_dir / "resolved_config.yaml" if request.input_adapter else Path(request.config_path)
        trainer.fit(
            build_lightning_agent(
                str(agent_config_path), str(request.data_dir), request.profile, agl_module=agl
            ),
            train_dataset=plan.bundle.train_tasks,
            val_dataset=plan.bundle.val_tasks,
        )
        final_reward_finder = getattr(agl, "find_final_reward", None)
        reward_spans_finder = getattr(agl, "find_reward_spans", None)
        if final_reward_finder is None or reward_spans_finder is None:
            from agentlightning.emitter import find_final_reward, find_reward_spans

            final_reward_finder = find_final_reward
            reward_spans_finder = find_reward_spans
        summary = asyncio.run(_export_rollouts(
            store,
            run_dir / "metrics" / "rollouts.jsonl",
            find_final_reward=final_reward_finder,
            find_reward_spans=reward_spans_finder,
        ))
        _write_json_atomic(run_dir / "metrics" / "summary.json", summary)
        manifest.update(status="completed", completed_at=_now(), metrics=summary)
        _write_json_atomic(manifest_path, manifest)
    except Exception as exc:
        manifest.update(status="failed", completed_at=_now(), error=f"{type(exc).__name__}: {exc}")
        _write_json_atomic(manifest_path, manifest)
        raise
    return GRPOTrainingResult(run_dir, manifest_path, False, plan.verl_config)
