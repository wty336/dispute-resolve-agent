import pytest

from dispute_agent.agent.tracing import RuntimeTrace, RuntimeTraceRecorder, TrainableSpan, select_trainable_spans


@pytest.fixture
def trace() -> RuntimeTrace:
    recorder = RuntimeTraceRecorder(namespace="training")
    recorder.record_proxy_span(
        span_id="p1",
        token_ids=[1, 2, 3],
        model_resource="main_llm",
        rollout_id="r1",
        attempt_id="a1",
        model_version="v1",
    )
    recorder.record_proxy_span(
        span_id="p2",
        token_ids=[4, 5],
        model_resource="main_llm",
        rollout_id="r1",
        attempt_id="a2",
        model_version="v1",
    )
    recorder.trace.spans.append(
        TrainableSpan(
            span_id="s1",
            source="sdk_runtime",
            model_resource="main_llm",
            token_ids=[],
        )
    )
    recorder.record_event("tool_call", tool="check_logistics")
    return recorder.trace


def test_only_proxy_llm_spans_are_trainable(trace):
    trainable = select_trainable_spans(trace, model_resource="main_llm")
    assert len(trainable) == trace.proxy_model_call_count
    assert all(span.token_ids for span in trainable)
    assert all(span.source == "proxy_llm" for span in trainable)
    assert not any(span.source == "sdk_runtime" for span in trainable)
