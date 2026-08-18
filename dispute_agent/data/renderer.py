"""合成纠纷的确定性模板渲染器。

渲染器使用标准 OpenAI chat message：assistant ``tool_calls`` 和
``role=tool`` 消息。工具结果绝不会伪装成 user 消息。
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
    tool_result_overrides: dict[int, str] | None = None,
    decision: Decision | Escalation | None = None,
) -> list[dict]:
    """为一个事实实例渲染原生工具协议 SFT trace。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_observation(instance)},
    ]

    for index, (tool_name, arguments) in enumerate(tool_plan or []):
        call_id = f"call_{index}"
        previous_result_was_invalid = index > 0 and index - 1 in (tool_result_overrides or {})
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
                "content": (
                    "上一工具返回格式非法，改用其他证据源继续核验。"
                    if previous_result_was_invalid
                    else None
                ),
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
                "content": (tool_result_overrides or {}).get(index, result.payload),
            }
        )

    if decision is None:
        decision_text = "final"
    else:
        decision_text = json.dumps(decision.model_dump(mode="json"), ensure_ascii=False)
    messages.append({"role": "assistant", "content": decision_text})
    return messages
