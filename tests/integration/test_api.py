import pytest
from fastapi.testclient import TestClient

from dispute_agent.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def public_case():
    return {
        "case_id": "api-case",
        "order_id": "o-1",
        "buyer_id": "b-1",
        "merchant_id": "m-1",
        "item_name": "测试商品",
        "order_amount": 100.0,
        "claim_type": "damaged",
        "buyer_claim": "商品破损，要求处理",
        "buyer_requested_amount": 50.0,
        "merchant_response": "发货前完好",
        "chat_log": ["买家：破损"],
        "evidence": [
            {
                "evidence_id": "e1",
                "type": "聊天记录",
                "description": "买家反馈",
                "source": "buyer",
                "visible": True,
            }
        ],
        "tool_results": [],
    }


def test_resolve_returns_decision_and_public_trace(client, public_case):
    response = client.post("/v1/disputes/resolve", json=public_case)
    assert response.status_code == 200
    body = response.json()
    assert body["decision"]["action"] in {"decide", "escalate"}
    assert set(body["trace"][0]) >= {"event", "latency_ms"}
    assert "thinking" not in str(body).lower()
    assert "ground_truth" not in str(body).lower()
