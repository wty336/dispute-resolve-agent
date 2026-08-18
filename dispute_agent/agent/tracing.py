"""本地运行时追踪和可训练 span 选择。

OpenAI Agents SDK tracing 已关闭云端导出。本地记录器会保存运行时事件（工具、
护栏、状态转换），但只有 ProxyLLM 模型 span 会被声明为可训练。
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
    """选择属于指定模型资源的 ProxyLLM span。"""
    return [
        span
        for span in trace.spans
        if span.source == "proxy_llm" and span.model_resource == model_resource and span.token_ids
    ]


class RuntimeTraceRecorder:
    """记录不可训练的运行时事件和可训练的 proxy span。"""

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
