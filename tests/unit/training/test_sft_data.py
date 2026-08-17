import pytest

from dispute_agent.training.sft_data import preprocess_example


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        return_dict=False,
        add_generation_prompt=False,
        chat_template_kwargs=None,
        return_assistant_tokens_mask=False,
    ):
        input_ids = list(range(10))
        result = {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}
        if return_assistant_tokens_mask:
            result["assistant_tokens_mask"] = [
                False, False, True, True, False, False, True, True, True, False,
            ]
        return result


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


@pytest.fixture
def messages():
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user"},
        {"role": "assistant", "content": None, "tool_calls": []},
        {"role": "tool", "tool_call_id": "call_0", "content": "result"},
        {"role": "assistant", "content": "final"},
    ]


@pytest.fixture
def preprocessed_example(tokenizer, messages):
    return preprocess_example(messages, tokenizer)


def test_only_assistant_actions_have_labels(preprocessed_example, tokenizer):
    labels = preprocessed_example["labels"]
    spans = preprocessed_example["message_spans"]
    for span in spans:
        supervised = any(token != -100 for token in labels[span.start:span.end])
        assert supervised is (span.role == "assistant")
    assert preprocessed_example["chat_template_kwargs"]["enable_thinking"] is False
