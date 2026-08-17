"""Dependency-light dispute rollout core with a lazy Agent Lightning binding."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from dispute_agent.agent.runtime import build_runtime
from dispute_agent.rewards.engine import RewardEngine
from dispute_agent.training.grpo_config import GRPOConfig, load_grpo_config
from dispute_agent.training.grpo_dataset import EpisodeSource, load_grpo_dataset


def _value(obj: object, name: str, default: object = None) -> object:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _is_terminal_runtime_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "without submitting a terminal decision",
            "maximum turns",
            "max_turns",
            "turn limit",
        )
    )


async def run_dispute_rollout(
    task: Mapping[str, object],
    llm: object,
    *,
    episode_source: EpisodeSource,
    config: GRPOConfig,
    runtime_factory: Callable[..., Any] = build_runtime,
    reward_engine: RewardEngine | None = None,
    annotation_emitter: Callable[[dict[str, object]], None] | None = None,
) -> float:
    """Run one isolated episode and return exactly one scalar reward."""
    case_id = task.get("case_id")
    scenario_id = task.get("scenario_id")
    curriculum_phase = task.get("curriculum_phase")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("GRPO task requires a non-empty case_id")
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError("GRPO task requires a non-empty scenario_id")
    if curriculum_phase not in {1, 2}:
        raise ValueError("GRPO task curriculum_phase must be 1 or 2")

    episode = episode_source.create(case_id)
    endpoint = _value(llm, "endpoint") or _value(llm, "base_url")
    if not isinstance(endpoint, str) or not endpoint:
        raise ValueError("rollout LLM descriptor requires endpoint")
    api_key = _value(llm, "api_key") or "EMPTY"
    model = _value(llm, "model")
    if not isinstance(model, str) or not model:
        raise ValueError("rollout LLM descriptor requires model")

    if curriculum_phase == 1:
        allowed_tools: set[str] | None = set(config.curriculum.phase1_tools)
        max_rounds = config.curriculum.phase1_max_rounds
    else:
        allowed_tools = None
        max_rounds = config.curriculum.phase2_max_rounds
    if max_rounds <= 0:
        raise ValueError("curriculum max rounds must be positive")
    max_tokens_per_round = min(
        config.actor_rollout_ref.rollout.max_tokens_per_round,
        config.actor_rollout_ref.rollout.max_episode_tokens // max_rounds,
    )
    if max_tokens_per_round <= 0:
        raise ValueError("curriculum token budget must be positive")

    runtime = runtime_factory(base_url=endpoint, api_key=str(api_key), model=model)
    try:
        decision = await runtime.run(
            episode,
            enable_thinking=True,
            allowed_tools=allowed_tools,
            max_rounds=max_rounds,
            max_tokens_per_round=max_tokens_per_round,
        )
    except RuntimeError as exc:
        if not _is_terminal_runtime_error(exc):
            raise
        decision = None

    # A custom runtime may return a terminal decision without mutating the episode.
    if episode.terminal_decision is None and decision is not None and hasattr(decision, "action"):
        episode.submit(decision)

    scorer = reward_engine or RewardEngine()
    result = scorer.score(episode)
    total = float(result.total)
    if annotation_emitter is not None:
        annotation_emitter({
            "case_id": case_id,
            "scenario_id": scenario_id,
            "curriculum_phase": curriculum_phase,
            "reward": total,
            "components": result.components.model_dump(mode="json"),
            "tool_call_count": len(episode.tool_calls),
            "terminal": episode.terminal_decision is not None,
        })
    return total


def build_lightning_agent(
    config_path: str,
    data_dir: str,
    profile: str,
    *,
    agl_module: object | None = None,
):
    """Build the real Agent Lightning rollout wrapper without heavy imports."""
    agl = agl_module
    if agl is None:
        import agentlightning as agl

    config = load_grpo_config(config_path)
    bundle = load_grpo_dataset(data_dir, profile=profile, curriculum_phase=1)

    @agl.rollout
    async def rollout(task: Mapping[str, object], llm: object) -> float:
        return await run_dispute_rollout(
            task,
            llm,
            episode_source=bundle.episode_source,
            config=config,
            annotation_emitter=agl.emit_object,
        )

    return rollout
