import pytest

from dispute_agent.training.sft_data import SFTPreflightError, preflight_dataset


class AuditTokenizer:
    token_text = {
        10: "<SYSTEM_SENTINEL>",
        11: "<USER_SENTINEL>",
        12: "<think>\n\n</think><ASSISTANT_SENTINEL>",
        13: "<TOOL_SENTINEL>",
        14: "<FINAL_SENTINEL>",
    }

    def __init__(self, *, mask=None, length=5):
        self.mask = mask or [0, 0, 1, 0, 1]
        self.length = length

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        assert kwargs["return_assistant_tokens_mask"] is True
        input_ids = [10, 11, 12, 13, 14] + [99] * (self.length - 5)
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "assistant_masks": self.mask + [0] * (self.length - 5),
        }

    def decode(self, token_ids, **kwargs):
        return "".join(self.token_text.get(token_id, "x") for token_id in token_ids)


class NonemptyThinkingTokenizer(AuditTokenizer):
    token_text = {
        **AuditTokenizer.token_text,
        12: "<think>hidden reasoning</think><ASSISTANT_SENTINEL>",
    }


ROW = {
    "case_id": "case-1",
    "messages": [
        {"role": "user", "content": "case"},
        {"role": "assistant", "content": "answer"},
    ],
    "tools": [{"type": "function"}],
}


def test_preflight_reports_valid_assistant_only_dataset():
    report = preflight_dataset([ROW], AuditTokenizer(), max_length=8)

    assert report.checked_rows == 1
    assert report.max_observed_length == 5
    assert report.supervised_tokens == 2


@pytest.mark.parametrize(
    ("tokenizer", "message"),
    [
        (AuditTokenizer(mask=[1, 0, 1, 0, 1]), "non-assistant"),
        (AuditTokenizer(length=9), "exceeds max_length"),
        (NonemptyThinkingTokenizer(), "non-empty thinking"),
    ],
)
def test_preflight_rejects_unsafe_mask_or_overlength(tokenizer, message):
    with pytest.raises(SFTPreflightError, match=message):
        preflight_dataset([ROW], tokenizer, max_length=8)
