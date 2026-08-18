"""纠纷判责 Agent 的精简版 FastAPI 演示。"""
from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from dispute_agent.domain.schemas import DisputeObservation

app = FastAPI(title="Dispute Agent API", version="0.1.0")

TRACES: dict[str, dict] = {}


class ResolveResponse(BaseModel):
    trace_id: str
    decision: dict
    trace: list[dict]


class TraceResponse(BaseModel):
    trace_id: str
    trace: list[dict]


def _rule_decision(obs: DisputeObservation) -> dict:
    if obs.buyer_requested_amount > obs.order_amount * 0.9:
        return {
            "action": "escalate",
            "confidence": 0.7,
            "evidence_ids": [e.evidence_id for e in obs.evidence],
            "reason": "高额诉求或证据不足，转人工复核",
        }
    return {
        "action": "decide",
        "liability": "merchant",
        "compensation": round(obs.order_amount * 0.5, 2),
        "confidence": 0.8,
        "evidence_ids": [e.evidence_id for e in obs.evidence],
        "reason": "规则基线：根据公开信息判定商家责任",
    }


@app.post("/v1/disputes/resolve", response_model=ResolveResponse)
def resolve_dispute(payload: dict) -> ResolveResponse:
    obs = DisputeObservation.model_validate(payload)
    started = time.perf_counter()
    trace_id = str(uuid.uuid4())
    trace = [
        {"event": "observation_received", "latency_ms": round((time.perf_counter() - started) * 1000, 3)},
    ]
    decision = _rule_decision(obs)
    trace.append({"event": "decision_made", "latency_ms": round((time.perf_counter() - started) * 1000, 3)})
    TRACES[trace_id] = {"trace_id": trace_id, "trace": trace, "decision": decision}
    return ResolveResponse(trace_id=trace_id, decision=decision, trace=trace)


@app.get("/v1/traces/{trace_id}", response_model=TraceResponse)
def get_trace(trace_id: str) -> TraceResponse:
    record = TRACES.get(trace_id)
    if record is None:
        raise HTTPException(status_code=404, detail="trace not found")
    return TraceResponse(trace_id=trace_id, trace=record["trace"])


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}
