import json
from types import SimpleNamespace

import pytest

from dispute_agent.domain.schemas import Decision
from dispute_agent.training.grpo_config import load_grpo_config
from dispute_agent.training.grpo_dataset import EpisodeSource
from dispute_agent.training.lightning_agent import run_dispute_rollout


@pytest.fixture
def episode_source(make_episode):
    episode = make_episode(case_id="case-001")
    return EpisodeSource(
        observations={episode.case_id: episode.observation},
        ground_truth={episode.case_id: episode.ground_truth},
    )


@pytest.fixture
def fake_runtime_factory():
    class FakeRuntime:
        def __init__(self, factory, episode, kwargs):
            self.factory = factory
            self.episode = episode
            self.kwargs = kwargs

        async def run(self, episode, **kwargs):
            self.factory.episodes.append(episode)
            self.factory.calls.append(kwargs)
            episode.submit(Decision(
                action="decide",
                liability="merchant",
                compensation=50.0,
                confidence=0.9,
                evidence_ids=["chat:1"],
                reason="证据支持商家责任",
            ))

    def factory(**kwargs):
        episode = kwargs.pop("episode", None)
        runtime = FakeRuntime(factory, episode, kwargs)
        return runtime

    factory.episodes = []
    factory.calls = []
    factory.runtime_type = FakeRuntime
    return factory


@pytest.mark.asyncio
async def test_rollout_reconstructs_fresh_episode_and_returns_one_reward(
    episode_source, fake_runtime_factory
):
    annotations: list[dict[str, object]] = []
    task = {
        "case_id": "case-001",
        "scenario_id": "fact-001",
        "curriculum_phase": 1,
    }
    llm = SimpleNamespace(
        endpoint="http://127.0.0.1:8000/v1",
        api_key=None,
        model="Qwen/Qwen3-8B",
    )
    config = load_grpo_config("configs/grpo.yaml")

    first = await run_dispute_rollout(
        task,
        llm,
        episode_source=episode_source,
        config=config,
        runtime_factory=fake_runtime_factory,
        annotation_emitter=annotations.append,
    )
    second = await run_dispute_rollout(
        task,
        llm,
        episode_source=episode_source,
        config=config,
        runtime_factory=fake_runtime_factory,
        annotation_emitter=annotations.append,
    )

    assert isinstance(first, float)
    assert first == second
    assert fake_runtime_factory.episodes[0] is not fake_runtime_factory.episodes[1]
    assert fake_runtime_factory.calls[0]["allowed_tools"] == set(config.curriculum.phase1_tools)
    assert fake_runtime_factory.calls[0]["max_rounds"] == config.curriculum.phase1_max_rounds
    assert fake_runtime_factory.calls[0]["max_tokens_per_round"] == min(384, 1280 // 3)
    assert all("ground_truth" not in json.dumps(item) for item in annotations)


@pytest.mark.asyncio
async def test_missing_submission_maps_to_hard_failure(episode_source, monkeypatch):
    class NoSubmitRuntime:
        async def run(self, episode, **kwargs):
            return None

    def factory(**kwargs):
        return NoSubmitRuntime()

    llm = SimpleNamespace(endpoint="http://127.0.0.1:8000/v1", api_key=None, model="Qwen/Qwen3-8B")
    config = load_grpo_config("configs/grpo.yaml")
    reward = await run_dispute_rollout(
        {"case_id": "case-001", "scenario_id": "fact-001", "curriculum_phase": 1},
        llm,
        episode_source=episode_source,
        config=config,
        runtime_factory=factory,
    )
    assert reward == -1.5


@pytest.mark.asyncio
async def test_agents_sdk_max_turns_maps_to_hard_failure(episode_source):
    from agents.exceptions import MaxTurnsExceeded

    class TurnLimitedRuntime:
        async def run(self, episode, **kwargs):
            raise MaxTurnsExceeded("Maximum turns exceeded")

    llm = SimpleNamespace(endpoint="http://127.0.0.1:8000/v1", api_key=None, model="Qwen/Qwen3-8B")
    reward = await run_dispute_rollout(
        {"case_id": "case-001", "scenario_id": "fact-001", "curriculum_phase": 1},
        llm,
        episode_source=episode_source,
        config=load_grpo_config("configs/grpo.yaml"),
        runtime_factory=lambda **_: TurnLimitedRuntime(),
    )

    assert reward == -1.5
