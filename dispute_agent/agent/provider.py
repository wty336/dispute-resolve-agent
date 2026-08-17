"""OpenAI-compatible model provider for the dispute agent.

The provider is intentionally explicit about the base URL so training can point
to Agent Lightning's ``main_llm`` ProxyLLM and evaluation can point to a plain
vLLM endpoint.
"""
from __future__ import annotations

from openai import AsyncOpenAI
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel


def create_chat_model(
    *,
    base_url: str,
    api_key: str,
    model: str = "qwen3-8b",
    http_client=None,
) -> OpenAIChatCompletionsModel:
    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        http_client=http_client,
    )
    return OpenAIChatCompletionsModel(model=model, openai_client=client)
