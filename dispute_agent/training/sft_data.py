"""Strict assistant-only preprocessing and template preflight for SFT."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


class SFTPreflightError(ValueError):
    """Raised when the tokenizer cannot prove safe assistant-only training."""


@dataclass(frozen=True)
class PreflightReport:
    checked_rows: int
    max_observed_length: int
    supervised_tokens: int


def preprocess_example(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int,
) -> dict[str, list[int]]:
    """Render one trace without truncation and mask every non-assistant token."""

    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    except (TypeError, ValueError) as exc:
        raise SFTPreflightError(
            "tokenizer chat template does not support native assistant token masks"
        ) from exc

    input_ids = list(encoded.get("input_ids", []))
    attention_mask = list(encoded.get("attention_mask", [1] * len(input_ids)))
    assistant_mask = encoded.get("assistant_masks")
    if assistant_mask is None:
        assistant_mask = encoded.get("assistant_tokens_mask")
    if assistant_mask is None:
        raise SFTPreflightError("tokenizer did not return an assistant token mask")
    assistant_mask = list(assistant_mask)

    if not input_ids:
        raise SFTPreflightError("chat template produced an empty sequence")
    if len(input_ids) > max_length:
        raise SFTPreflightError(
            f"rendered sequence length {len(input_ids)} exceeds max_length {max_length}"
        )
    if len(attention_mask) != len(input_ids):
        raise SFTPreflightError("attention mask length does not match input_ids")
    if len(assistant_mask) != len(input_ids):
        raise SFTPreflightError("assistant mask length does not match input_ids")

    labels = [token if bool(flag) else -100 for token, flag in zip(input_ids, assistant_mask)]
    if all(token == -100 for token in labels):
        raise SFTPreflightError("example contains no supervised assistant tokens")
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _audit_template(tokenizer: Any, max_length: int) -> None:
    messages = [
        {"role": "system", "content": "<SYSTEM_SENTINEL>"},
        {"role": "user", "content": "<USER_SENTINEL>"},
        {
            "role": "assistant",
            "content": "<ASSISTANT_SENTINEL>",
            "tool_calls": [
                {
                    "id": "audit_call",
                    "type": "function",
                    "function": {
                        "name": "audit_tool",
                        "arguments": '{"value":"audit"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "audit_call", "content": "<TOOL_SENTINEL>"},
        {"role": "assistant", "content": "<FINAL_SENTINEL>"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "audit_tool",
                "description": "template mask audit",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            },
        }
    ]
    encoded = preprocess_example(messages, tools, tokenizer, max_length)
    supervised_ids = [
        token for token, label in zip(encoded["input_ids"], encoded["labels"]) if label != -100
    ]
    supervised_text = tokenizer.decode(supervised_ids, skip_special_tokens=False)

    forbidden = ("<SYSTEM_SENTINEL>", "<USER_SENTINEL>", "<TOOL_SENTINEL>")
    if any(sentinel in supervised_text for sentinel in forbidden):
        raise SFTPreflightError("assistant mask supervises non-assistant template content")
    required = ("<ASSISTANT_SENTINEL>", "<FINAL_SENTINEL>")
    if any(sentinel not in supervised_text for sentinel in required):
        raise SFTPreflightError("assistant mask omits assistant template content")
    lowered = supervised_text.lower()
    if "<think>" in lowered or "</think>" in lowered:
        raise SFTPreflightError("thinking tags remain in supervised template content")


def preflight_dataset(
    rows: Iterable[dict[str, Any]], tokenizer: Any, max_length: int
) -> PreflightReport:
    """Audit the template and fully validate every selected public row."""

    _audit_template(tokenizer, max_length)
    checked_rows = 0
    max_observed_length = 0
    supervised_tokens = 0
    for index, row in enumerate(rows):
        case_id = row.get("case_id", f"row-{index}")
        try:
            encoded = preprocess_example(
                messages=row["messages"],
                tools=row["tools"],
                tokenizer=tokenizer,
                max_length=max_length,
            )
        except (KeyError, TypeError, SFTPreflightError) as exc:
            raise SFTPreflightError(f"preflight failed for {case_id}: {exc}") from exc
        checked_rows += 1
        max_observed_length = max(max_observed_length, len(encoded["input_ids"]))
        supervised_tokens += sum(label != -100 for label in encoded["labels"])

    if checked_rows == 0:
        raise SFTPreflightError("SFT dataset is empty")
    return PreflightReport(
        checked_rows=checked_rows,
        max_observed_length=max_observed_length,
        supervised_tokens=supervised_tokens,
    )
