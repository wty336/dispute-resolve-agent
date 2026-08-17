"""Deterministic template renderer for synthetic disputes.

The renderer uses standard OpenAI chat messages: assistant ``tool_calls`` and
``role=tool`` messages.  Tool results are never disguised as user messages.
"""
from __future__ import annotations

import json

from dispute_agent.data.generator import FactInstance
from dispute_agent.domain.schemas import Decision, Escalation
from dispute_agent.tools.simulators import simulate_tool

SYSTEM_PROMPT = (
    "你是一个电商纠纷判责 Agent。你可以调用调查工具获取公开信息，"
    "最后必须提交 decide 或 escalate 决策。不要编造证据。"
)


def render_observation(instance: FactInstance) -> str:
    obs = instance.observation
    lines = [
        f"纠纷编号：{obs.case_id}",
        f"订单号：{obs.order_id}",
        f"商品：{obs.item_name}，金额：{obs.order_amount:.2f} 元",
        f"投诉类型：{obs.claim_type}",
        f"买家陈述：{obs.buyer_claim}",
        f"买家诉求金额：{obs.buyer_requested_amount:.2f} 元",
        f"商家回应：{obs.merchant_response}",
        "聊天记录：" + "；".join(obs.chat_log),
        "已有证据：" + "；".join(f"{e.evidence_id}:{e.description}" for e in obs.evidence),
    ]
    return "\n".join(lines)


def render_sft_trace(
    instance: FactInstance,
    *,
    tool_plan: list[tuple[str, dict]] | None = None,
    decision: Decision | Escalation | None = None,
) -> list[dict]:
    """Render a native tool-protocol SFT trace for one fact instance."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_observation(instance)},
    ]

    for index, (tool_name, arguments) in enumerate(tool_plan or []):
        call_id = f"call_{index}"
        result = simulate_tool(
            tool_name,
            case_id=instance.case_id,
            arguments=arguments,
            case_seed=instance.fact_instance_id,
            ground_truth=instance.ground_truth,
        )
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(arguments, ensure_ascii=False, sort_keys=True),
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": result.payload,
            }
        )

    if decision is None:
        decision_text = "final"
    else:
        decision_text = json.dumps(decision.model_dump(mode="json"), ensure_ascii=False)
    messages.append({"role": "assistant", "content": decision_text})
    return messages
