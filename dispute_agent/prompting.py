"""统一的 prompt 模板与输出解析。

训练（SFT/RL）与推理（LLMAgent）共用同一套 prompt，避免 train-inference
不一致。模型输出为 JSON，字段与 `AgentDecision` 对应。
"""
from __future__ import annotations

import json
import re

from .models import DisputeCase

SYSTEM_PROMPT = (
    "你是电商平台的纠纷判责 Agent。你的目标不是单轮正确，而是平台长期收益最大化。"
    "你需要在公平、成本、买家满意度、商家配合度之间取得平衡。"
    "买家可能夸大损失甚至虚假投诉，商家可能推卸责任甚至虚假否认。"
)

ACTION_FORMAT = (
    '请输出 JSON：{"liability": "商家责任"|"买家责任"|"双方共担"|"无法认定", '
    '"compensation": 赔付金额(元,数字), "escalate": true/false, "reason": "简要依据"}'
)

# ---------- 多步工具调用 ----------
TOOL_LOOP_SYSTEM_PROMPT = (
    "你是电商平台的纠纷判责 Agent，目标不是单轮正确，而是平台长期收益最大化。"
    "你需要在公平、成本、买家满意度、商家配合度之间取得平衡。"
    "买家可能夸大损失甚至虚假投诉，商家可能推卸责任甚至虚假否认。"
    "你可以调用工具查询信息，但每次调用都有成本，应只调用必要工具。"
    "当你获得足够信息后，必须输出 final 决策。"
)

TOOL_ACTION_FORMAT = (
    "每一步只输出一个 JSON 对象：\n"
    "1) 需要查询信息时：{\"action\": \"tool_call\", \"tool\": \"工具名\", \"arguments\": {...}}\n"
    "2) 信息足够时：{\"action\": \"final\", \"liability\": \"商家责任\"|\"买家责任\"|"
    "\"双方共担\"|\"无法认定\", \"compensation\": 赔付金额(元,数字), "
    "\"escalate\": true/false, \"reason\": \"简要依据\"}\n"
    "不要输出其他文字。"
)


def build_case_info(case: DisputeCase) -> str:
    """把一单纠纷格式化为工单信息文本（不含输出要求）。"""
    chat_text = "\n".join(f"{m.role}: {m.content}" for m in case.chat_log)
    buyer_ev = "\n".join(
        f"- [{e.type}] {e.description}" for e in case.buyer_evidence
    ) or "（无）"
    merchant_ev = "\n".join(
        f"- [{e.type}] {e.description}" for e in case.merchant_evidence
    ) or "（无）"

    return f"""【订单信息】
订单号：{case.order_id}
商品：{case.item_name}
订单金额：{case.order_amount:.2f} 元

【用户投诉】
类型：{case.claim_type.value}
描述：{case.buyer_claim}
诉求金额：{case.buyer_requested_amount:.2f} 元

【商家回应】
{case.merchant_response}

【聊天记录】
{chat_text}

【买家举证】
{buyer_ev}

【商家举证】
{merchant_ev}"""


def build_case_prompt(case: DisputeCase) -> str:
    """把一单纠纷格式化为单步判责模型输入 prompt（user 部分）。"""
    return f"{build_case_info(case)}\n\n{ACTION_FORMAT}"


def build_full_prompt(case: DisputeCase, system_prompt: str = SYSTEM_PROMPT) -> str:
    """拼接 system + user 的完整文本（用于无 chat template 的场景）。"""
    return f"{system_prompt}\n\n{build_case_prompt(case)}"


def build_tool_loop_user_prompt(case: DisputeCase, tool_definitions_text: str) -> str:
    """构造多步工具调用循环的首轮 user prompt。"""
    return f"""{build_case_info(case)}

【可用工具】
{tool_definitions_text}

【输出要求】
{TOOL_ACTION_FORMAT}"""


def build_rl_user_prompt_with_tool_results(case: DisputeCase, tool_results_text: str) -> str:
    """构造 RL 阶段的 user prompt：工单 + 已执行的工具结果 + 只输出 final。

    RL 阶段不再让模型决定调用哪些工具（该能力由 SFT 阶段学习），
    而是把仿真器预执行的工具结果直接给模型，只训练“给定信息后的最终判责”。
    """
    return f"""{build_case_info(case)}

【工具查询结果】
{tool_results_text}

【输出要求】
请根据以上信息输出最终判责 JSON：
{{"action": "final", "liability": "商家责任"|"买家责任"|"双方共担"|"无法认定", "compensation": 赔付金额(元,数字), "escalate": true/false, "reason": "简要依据"}}
只输出该 JSON，不要输出其他内容。"""


def parse_action(text: str) -> dict:
    """解析模型输出的动作 JSON。失败返回空 dict。"""
    data = parse_agent_response(text)
    if not data:
        return {}
    return data


def parse_agent_response(text: str) -> dict:
    """从模型输出中解析 JSON。失败时返回空 dict。"""
    if not text:
        return {}
    # 去掉 ```json ... ``` 围栏
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    # 截取第一个 { 到最后一个 }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        data = json.loads(text[start : end + 1])
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}
