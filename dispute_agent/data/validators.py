"""合成数据质量与泄漏安全校验辅助函数。"""
from __future__ import annotations

import json

from dispute_agent.domain.schemas import DisputeObservation

HIDDEN_KEYWORDS = (
    "true_liability",
    "should_escalate",
    "tool_information_value",
    "true_loss",
    "buyer_strategy",
    "merchant_strategy",
)


def validate_observation(observation: DisputeObservation) -> list[str]:
    errors: list[str] = []
    text = observation.model_dump_json().lower()
    for keyword in HIDDEN_KEYWORDS:
        if keyword in text:
            errors.append(f"public observation leaks hidden field {keyword!r}")
    return errors


def validate_trace_messages(messages: list[dict]) -> list[str]:
    errors: list[str] = []
    tool_message_ids = [m.get("tool_call_id") for m in messages if m.get("role") == "tool"]
    assistant_call_ids = [
        call["id"]
        for m in messages
        if m.get("role") == "assistant" and m.get("tool_calls")
        for call in m["tool_calls"]
    ]
    if len(tool_message_ids) != len(assistant_call_ids):
        errors.append("tool message count does not match assistant tool call count")
    if set(tool_message_ids) != set(assistant_call_ids):
        errors.append("tool_call_id mismatch between assistant calls and tool messages")

    text = json.dumps(messages, ensure_ascii=False).lower()
    for keyword in HIDDEN_KEYWORDS:
        if keyword in text:
            errors.append(f"trace leaks hidden field {keyword!r}")

    for message in messages:
        if message.get("role") == "user" and "tool_result" in message.get("content", ""):
            errors.append("tool result is disguised as a user message")
    return errors
