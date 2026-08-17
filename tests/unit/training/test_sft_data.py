from dispute_agent.training.sft_data import preprocess_example


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        assert kwargs["tools"] == [{"type": "function"}]
        return {
            "input_ids": [10, 11, 12, 13, 14],
            "attention_mask": [1, 1, 1, 1, 1],
            "assistant_masks": [0, 0, 1, 0, 1],
        }


def test_preprocess_uses_native_assistant_mask_and_disables_thinking():
    result = preprocess_example(
        messages=[
            {"role": "user", "content": "case"},
            {"role": "assistant", "content": "answer"},
        ],
        tools=[{"type": "function"}],
        tokenizer=FakeTokenizer(),
        max_length=8,
    )

    assert result["labels"] == [-100, -100, 12, -100, 14]
