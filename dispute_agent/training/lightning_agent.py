"""Agent Lightning rollout adapter for the dispute agent.

The adapter keeps a process-local episode repository.  Tasks only carry
``case_id`` and public observation; hidden ground truth is resolved inside the
process and never placed into the Lightning task payload.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from dispute_agent.agent.runtime import build_runtime
from dispute_agent.environment import EpisodeState
from dispute_agent.rewards.engine import RewardEngine


class EpisodeRepository:
    def __init__(self, episodes: dict[str, EpisodeState] | None = None) -> None:
        self._episodes: dict[str, EpisodeState] = episodes or {}

    def add(self, episode: EpisodeState) -> None:
        self._episodes[episode.case_id] = episode

    def get(self, case_id: str) -> EpisodeState:
        return self._episodes[case_id]


class LitDisputeAgent:
    """Minimal Agent Lightning-compatible rollout worker."""

    def __init__(
        self,
        repository: EpisodeRepository | None = None,
        reward_engine: RewardEngine | None = None,
        runtime_factory=None,
    ) -> None:
        self.repository = repository or EpisodeRepository()
        self.reward_engine = reward_engine or RewardEngine()
        self.runtime_factory = runtime_factory or build_runtime

    async def rollout(self, task: dict, resources: dict[str, Any], rollout: dict[str, Any]) -> float:
        case_id = task["case_id"]
        episode = self.repository.get(case_id)
        main_llm = resources["main_llm"]
        base_url = main_llm if isinstance(main_llm, str) else main_llm["base_url"]
        api_key = main_llm.get("api_key", "EMPTY") if isinstance(main_llm, dict) else "EMPTY"

        runtime = self.runtime_factory(base_url=base_url, api_key=api_key)
        await runtime.run(episode, enable_thinking=True)

        result = self.reward_engine.score(episode)
        return float(result.total)


@dataclass
class LightningRollout:
    agent: LitDisputeAgent
    task: dict
    resources: dict[str, Any]
    rollout: dict[str, Any] = field(default_factory=dict)
    emitted_reward_count: int = 0
    returned_reward_count: int = 0

    def emit_reward(self, reward: float) -> None:
        self.emitted_reward_count += 1

    def run(self) -> "LightningRolloutResult":
        reward = asyncio.run(self.agent.rollout(self.task, self.resources, self.rollout))
        self.returned_reward_count += 1
        return LightningRolloutResult(reward=reward)


@dataclass
class LightningRolloutResult:
    reward: float
