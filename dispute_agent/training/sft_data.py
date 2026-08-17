"""SFT data preprocessing with assistant-only loss masking."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MessageSpan:
    start: int
    end: int
    role: str


def preprocess_example(messages: list[dict], tokenizer, max_length: int | None = None) -> dict:
    """Tokenize with the native chat template and mask non-assistant tokens.

    The tokenizer must support ``apply_chat_template(..., return_dict=True,
    return_assistant_tokens_mask=True)``.  If that mask is unavailable the
    function falls back to encoding each message separately (approximate).
    """
    chat_template_kwargs = {"enable_thinking": False}
    kwargs = {
        "tokenize": True,
        "return_dict": True,
        "add_generation_prompt": False,
        "chat_template_kwargs": chat_template_kwargs,
    }
    try:
        encoded = tokenizer.apply_chat_template(messages, return_assistant_tokens_mask=True, **kwargs)
    except TypeError:
        encoded = tokenizer.apply_chat_template(messages, **kwargs)
        encoded = _fallback_spans(encoded, messages, tokenizer)

    input_ids = list(encoded["input_ids"])
    if max_length is not None:
        input_ids = input_ids[:max_length]

    if "assistant_tokens_mask" in encoded:
        mask = list(encoded["assistant_tokens_mask"])[: len(input_ids)]
        spans = _spans_from_mask(mask)
    else:
        spans = encoded.get("message_spans", [])

    labels = [
        token if any(span.role == "assistant" and span.start <= i < span.end for span in spans) else -100
        for i, token in enumerate(input_ids)
    ]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "message_spans": spans,
        "chat_template_kwargs": chat_template_kwargs,
    }


def _spans_from_mask(mask: list[bool]) -> list[MessageSpan]:
    spans: list[MessageSpan] = []
    in_assistant = False
    start = 0
    for i, flag in enumerate(mask + [False]):
        if flag != in_assistant:
            if i > start:
                spans.append(MessageSpan(start=start, end=i, role="assistant" if in_assistant else "other"))
            start = i
            in_assistant = flag
    return spans


def _fallback_spans(encoded: dict, messages: list[dict], tokenizer) -> dict:
    """Approximate per-message spans when the tokenizer lacks the mask."""
    input_ids = list(encoded["input_ids"])
    spans: list[MessageSpan] = []
    offset = 0
    for message in messages:
        content = message.get("content") or ""
        tokens = tokenizer.encode(content, add_special_tokens=False)
        end = min(offset + len(tokens), len(input_ids))
        spans.append(MessageSpan(start=offset, end=end, role=message["role"]))
        offset = end + 1
    encoded["message_spans"] = spans
    return encoded
