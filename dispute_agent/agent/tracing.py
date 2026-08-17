"""Local runtime tracing and trainable-span selection.

OpenAI Agents SDK tracing is disabled for cloud export.  The local recorder
stores runtime events (tools, guardrails, state transitions) but only ProxyLLM
model spans are declared trainable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainableSpan:
    span_id: str
    source: str
    model_resource: str
    token_ids: list[int]
    rollout_id: str | None = None
    attempt_id: str | None = None
    model_version: str | None = None


@dataclass
class RuntimeTrace:
    spans: list[TrainableSpan] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def proxy_model_call_count(self) -> int:
        return sum(1 for span in self.spans if span.source == "proxy_llm")


def select_trainable_spans(trace: RuntimeTrace, model_resource: str = "main_llm") -> list[TrainableSpan]:
    """Select ProxyLLM spans that belong to the given model resource."""
    return [
        span
        for span in trace.spans
        if span.source == "proxy_llm" and span.model_resource == model_resource and span.token_ids
    ]


class RuntimeTraceRecorder:
    """Records non-trainable runtime events and trainable proxy spans."""

    def __init__(self, namespace: str = "training") -> None:
        self.namespace = namespace
        self.trace = RuntimeTrace()

    def record_event(self, event: str, **details: Any) -> None:
        self.trace.events.append({"event": event, "namespace": self.namespace, **details})

    def record_proxy_span(
        self,
        *,
        span_id: str,
        token_ids: list[int],
        model_resource: str = "main_llm",
        rollout_id: str | None = None,
        attempt_id: str | None = None,
        model_version: str | None = None,
    ) -> TrainableSpan:
        span = TrainableSpan(
            span_id=span_id,
            source="proxy_llm",
            model_resource=model_resource,
            token_ids=token_ids,
            rollout_id=rollout_id,
            attempt_id=attempt_id,
            model_version=model_version,
        )
        self.trace.spans.append(span)
        return span
