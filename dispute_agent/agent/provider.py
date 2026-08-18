"""兼容 OpenAI 协议的纠纷判责 Agent 模型提供方。

这里显式配置 base URL，训练时可指向 Agent Lightning 的 ``main_llm``
ProxyLLM，评估时可指向普通的 vLLM endpoint。
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
