import pytest

from dispute_agent.agent.runtime import build_runtime
from dispute_agent.training.lightning_agent import (
    EpisodeRepository,
    LightningRollout,
    LitDisputeAgent,
)


@pytest.fixture
def lightning_rollout(fake_model_server, episode):
    repository = EpisodeRepository()
    repository.add(episode)
    agent = LitDisputeAgent(
        repository=repository,
        runtime_factory=lambda base_url, api_key: build_runtime(
            base_url=base_url,
            api_key=api_key,
            http_client=fake_model_server.http_client,
        ),
    )
    resources = {"main_llm": {"base_url": fake_model_server.url, "api_key": "test"}}
    return LightningRollout(agent=agent, task={"case_id": episode.case_id}, resources=resources)


def test_rollout_returns_reward_once(lightning_rollout):
    result = lightning_rollout.run()
    assert isinstance(result.reward, float)
    assert lightning_rollout.emitted_reward_count == 0
    assert lightning_rollout.returned_reward_count == 1
