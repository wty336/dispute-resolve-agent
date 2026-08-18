"""单元测试和集成测试共用的 fixture。"""
import json

import httpx
import pytest
import pytest_asyncio

from dispute_agent.domain.schemas import (
    Decision,
    DisputeGroundTruth,
    DisputeObservation,
    Evidence,
)
from dispute_agent.environment import EpisodeState
from dispute_agent.tools import ToolRegistry


def _make_episode(
    case_id: str = "case-1",
    group_seed: int = 42,
    order_id: str = "o-1",
    buyer_id: str = "b-1",
    merchant_id: str = "m-1",
) -> EpisodeState:
    observation = DisputeObservation(
        case_id=case_id,
        order_id=order_id,
        buyer_id=buyer_id,
        merchant_id=merchant_id,
        item_name="测试商品",
        order_amount=100.0,
        claim_type="damaged",
        buyer_claim="收到商品时已破损，要求赔偿 89 元。",
        buyer_requested_amount=89.0,
        merchant_response="发货前已检查完好，可能是物流导致。",
        chat_log=["买家：商品破损。", "商家：发货前完好。"],
        evidence=[
            Evidence(
                evidence_id="chat:1",
                type="聊天记录",
                description="买家反馈破损",
                source="buyer",
                visible=True,
            )
        ],
    )
    ground_truth = DisputeGroundTruth(
        case_id=case_id,
        true_liability="merchant",
        true_loss=50.0,
        reasonable_compensation_range=(40.0, 60.0),
        buyer_strategy="honest",
        merchant_strategy="honest",
        should_escalate=False,
        tool_information_value={},
        risk_level="low",
    )
    registry = ToolRegistry(case_id=case_id, case_seed=group_seed)
    return EpisodeState(
        observation=observation,
        ground_truth=ground_truth,
        case_seed=group_seed,
        tool_registry=registry,
    )


@pytest.fixture
def make_episode():
    return _make_episode


@pytest.fixture
def episode() -> EpisodeState:
    return _make_episode()


@pytest.fixture
def tool_registry() -> ToolRegistry:
    return _make_episode().tool_registry


@pytest.fixture
def valid_decision() -> Decision:
    return Decision(
        action="decide",
        liability="merchant",
        compensation=50.0,
        confidence=0.9,
        evidence_ids=["chat:1", "logistics:o-1"],
        reason="物流异常且证据支持商家责任",
    )


class FakeModelServer:
    def __init__(self) -> None:
        self.url = "http://fake-model"
        self.requests: list[dict] = []
        self._call_count = 0
        self.http_client = httpx.AsyncClient(transport=httpx.MockTransport(self._handler))

    async def _handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        call = self._call_count
        self._call_count += 1
        if call == 0:
            return self._response(
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "check_logistics",
                            "arguments": json.dumps({"order_id": "o-1"}),
                        },
                    }
                ]
            )
        if call == 1:
            return self._response(
                tool_calls=[
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "submit_decision",
                            "arguments": json.dumps(
                                {
                                    "action": "decide",
                                    "liability": "merchant",
                                    "compensation": 50.0,
                                    "confidence": 0.9,
                                    "evidence_ids": ["chat:1", "logistics:o-1"],
                                    "reason": "物流异常且证据支持商家责任",
                                }
                            ),
                        },
                    }
                ]
            )
        return self._response(content="unexpected extra call")

    def _response(self, content: str | None = None, tool_calls: list | None = None) -> httpx.Response:
        message: dict = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        payload = {
            "id": f"chatcmpl-{self._call_count}",
            "object": "chat.completion",
            "created": 0,
            "model": "qwen3-8b",
            "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        return httpx.Response(200, json=payload)

    def requests_contain(self, key: str, value: str) -> bool:
        for request in self.requests:
            if _contains_key_value(request, key, value):
                return True
        return False

    def requests_contain_text(self, text: str) -> bool:
        return any(text in json.dumps(request, ensure_ascii=False) for request in self.requests)


def _contains_key_value(obj, key: str, value: str) -> bool:
    if isinstance(obj, dict):
        if obj.get(key) == value:
            return True
        return any(_contains_key_value(v, key, value) for v in obj.values())
    if isinstance(obj, list):
        return any(_contains_key_value(item, key, value) for item in obj)
    return False


@pytest_asyncio.fixture
async def fake_model_server():
    server = FakeModelServer()
    yield server
    await server.http_client.aclose()
