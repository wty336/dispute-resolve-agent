"""Single-agent multi-turn runtime built on OpenAI Agents SDK.

The same runtime is used for training rollouts, evaluation, and the demo API.
It only sees public observations and tool results; hidden ground truth lives in
the episode and is never placed in prompts or tool arguments.
"""
from __future__ import annotations

from agents import Agent, ModelSettings, Runner, set_tracing_disabled

from dispute_agent.agent.provider import create_chat_model
from dispute_agent.agent.tools import build_agent_tools
from dispute_agent.domain.policies import MAX_ROUNDS


def _observation_to_text(episode) -> str:
    obs = episode.observation
    lines = [
        f"纠纷编号：{obs.case_id}",
        f"订单号：{obs.order_id}",
        f"买家ID：{obs.buyer_id}",
        f"商家ID：{obs.merchant_id}",
        f"商品：{obs.item_name}，金额：{obs.order_amount:.2f} 元",
        f"投诉类型：{obs.claim_type}",
        f"买家陈述：{obs.buyer_claim}",
        f"买家诉求金额：{obs.buyer_requested_amount:.2f} 元",
        f"商家回应：{obs.merchant_response}",
        "聊天记录：" + "；".join(obs.chat_log),
        "已有证据：" + "；".join(f"{e.evidence_id}:{e.description}" for e in obs.evidence),
        "可调用工具：check_logistics、check_buyer_history、check_merchant_history、verify_evidence、submit_decision",
    ]
    return "\n".join(lines)


class DisputeRuntime:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = "qwen3-8b",
        http_client=None,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.http_client = http_client
        set_tracing_disabled(True)

    async def run(self, episode, *, enable_thinking: bool = True):
        tools = build_agent_tools(episode)
        model = create_chat_model(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            http_client=self.http_client,
        )
        agent = Agent(
            name="dispute_resolution_agent",
            instructions=(
                "你是电商纠纷判责 Agent。先根据工单判断是否需要调查，"
                "可以调用工具获取更多公开信息，最后必须调用 submit_decision 提交决策。"
            ),
            model=model,
            tools=tools,
            tool_use_behavior={"stop_at_tool_names": ["submit_decision"]},
            model_settings=ModelSettings(
                temperature=0.6,
                top_p=0.95,
                extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
            ),
        )
        await Runner.run(
            agent,
            input=_observation_to_text(episode),
            max_turns=MAX_ROUNDS + 1,
        )
        if episode.terminal_decision is None:
            raise RuntimeError("agent finished without submitting a terminal decision")
        return episode.terminal_decision


def build_runtime(
    *,
    base_url: str,
    api_key: str,
    model: str = "qwen3-8b",
    http_client=None,
) -> DisputeRuntime:
    return DisputeRuntime(
        base_url=base_url,
        api_key=api_key,
        model=model,
        http_client=http_client,
    )
