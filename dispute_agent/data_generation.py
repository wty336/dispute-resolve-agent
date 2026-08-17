"""训练数据生成。

利用仿真环境生成无限训练数据：
- SFT 数据：多步工具调用轨迹（messages 格式，兼容 ms-swift），
  标签由 Oracle 教师策略 + 教师工具选择生成。
- RL 数据：prompt 中预嵌入工具查询结果（内嵌 case_id 与 tool_cost），
  同时把完整 case（含 ground truth）保存到 `rl_cases.jsonl`，
  供 verl 自定义奖励函数按 case_id 查表计算长期收益奖励。
"""
from __future__ import annotations

import json
from pathlib import Path

from .case_generator import CaseGenerator
from .models import (
    BuyerStrategy,
    ChatMessage,
    ClaimType,
    DisputeCase,
    Evidence,
    Liability,
    MerchantStrategy,
    Party,
)
from .oracle import OracleAgent
from .prompting import (
    SYSTEM_PROMPT,
    TOOL_LOOP_SYSTEM_PROMPT,
    build_case_prompt,
    build_rl_user_prompt_with_tool_results,
    build_tool_loop_user_prompt,
)
from .legacy_tools import (
    execute_tool,
    format_tool_definitions,
    format_tool_result_message,
)


# ---------- 序列化 ----------
def case_to_dict(case: DisputeCase) -> dict:
    """把 DisputeCase 序列化为可 JSON 存储的 dict。"""
    return {
        "case_id": case.case_id,
        "order_id": case.order_id,
        "buyer_id": case.buyer_id,
        "merchant_id": case.merchant_id,
        "item_name": case.item_name,
        "order_amount": case.order_amount,
        "claim_type": case.claim_type.value,
        "buyer_claim": case.buyer_claim,
        "buyer_requested_amount": case.buyer_requested_amount,
        "merchant_response": case.merchant_response,
        "chat_log": [{"role": m.role, "content": m.content} for m in case.chat_log],
        "buyer_evidence": [
            {
                "type": e.type,
                "description": e.description,
                "party": e.party.value,
                "strength": e.strength,
                "verified": e.verified,
            }
            for e in case.buyer_evidence
        ],
        "merchant_evidence": [
            {
                "type": e.type,
                "description": e.description,
                "party": e.party.value,
                "strength": e.strength,
                "verified": e.verified,
            }
            for e in case.merchant_evidence
        ],
        "true_liability": case.true_liability.value,
        "true_buyer_loss": case.true_buyer_loss,
        "buyer_strategy": case.buyer_strategy.value,
        "merchant_strategy": case.merchant_strategy.value,
    }


def case_from_dict(d: dict) -> DisputeCase:
    """从 dict 恢复 DisputeCase。"""
    return DisputeCase(
        case_id=d["case_id"],
        order_id=d["order_id"],
        buyer_id=d.get("buyer_id", ""),
        merchant_id=d.get("merchant_id", ""),
        item_name=d["item_name"],
        order_amount=float(d["order_amount"]),
        claim_type=ClaimType(d["claim_type"]),
        buyer_claim=d["buyer_claim"],
        buyer_requested_amount=float(d["buyer_requested_amount"]),
        merchant_response=d["merchant_response"],
        chat_log=[ChatMessage(role=m["role"], content=m["content"]) for m in d["chat_log"]],
        buyer_evidence=[
            Evidence(
                type=e["type"],
                description=e["description"],
                party=Party(e["party"]),
                strength=float(e["strength"]),
                verified=bool(e["verified"]),
            )
            for e in d["buyer_evidence"]
        ],
        merchant_evidence=[
            Evidence(
                type=e["type"],
                description=e["description"],
                party=Party(e["party"]),
                strength=float(e["strength"]),
                verified=bool(e["verified"]),
            )
            for e in d["merchant_evidence"]
        ],
        true_liability=Liability(d["true_liability"]),
        true_buyer_loss=float(d["true_buyer_loss"]),
        buyer_strategy=BuyerStrategy(d["buyer_strategy"]),
        merchant_strategy=MerchantStrategy(d["merchant_strategy"]),
    )


def decision_to_json(decision) -> str:
    """把 AgentDecision 序列化为模型可输出的 JSON 字符串（单步判责格式）。"""
    return json.dumps(
        {
            "liability": decision.liability.value,
            "compensation": decision.compensation,
            "escalate": decision.escalate,
            "reason": decision.reason,
        },
        ensure_ascii=False,
    )


def decision_to_final_json(decision) -> str:
    """把 AgentDecision 序列化为多步轨迹中的 final 动作 JSON。"""
    return json.dumps(
        {
            "action": "final",
            "liability": decision.liability.value,
            "compensation": decision.compensation,
            "escalate": decision.escalate,
            "reason": decision.reason,
        },
        ensure_ascii=False,
    )


# ---------- 教师工具选择 ----------
def select_teacher_tools(case: DisputeCase) -> list[tuple[str, dict]]:
    """Oracle 教师根据 ground truth 选择需要调用的工具。

    规则：
    - 破损/未收到货 → 必查物流
    - 商家推卸/虚假否认 → 查商家历史
    - 买家夸大/虚假投诉 → 查买家历史
    - 金额较大或证据冲突 → 核验关键证据
    """
    tools: list[tuple[str, dict]] = []
    if case.claim_type in (ClaimType.DAMAGED, ClaimType.NOT_RECEIVED):
        tools.append(("check_order_logistics", {"order_id": case.order_id}))
    if case.merchant_strategy in (MerchantStrategy.EVASIVE, MerchantStrategy.DENY):
        tools.append(("check_merchant_history", {"merchant_id": case.merchant_id}))
    if case.buyer_strategy in (BuyerStrategy.EXAGGERATE, BuyerStrategy.FRAUD):
        tools.append(("check_buyer_history", {"buyer_id": case.buyer_id}))

    # 证据核验：金额大或双方证据强度接近时，核验买家举证；商家否认时再核验商家举证
    if case.order_amount >= 200 or (
        abs(
            (sum(e.strength for e in case.buyer_evidence) / max(1, len(case.buyer_evidence)))
            - (sum(e.strength for e in case.merchant_evidence) / max(1, len(case.merchant_evidence)))
        )
        < 0.15
    ):
        tools.append(("verify_evidence", {"evidence_type": "买家举证"}))
        if case.merchant_strategy == MerchantStrategy.DENY:
            tools.append(("verify_evidence", {"evidence_type": "商家举证"}))

    # 最多 4 次调用，避免轨迹过长
    return tools[:4]


def make_sft_example(case: DisputeCase, oracle: OracleAgent | None = None) -> dict:
    """构造一条单步 SFT 样本（messages 格式，判责直接输出）。"""
    oracle = oracle or OracleAgent()
    decision = oracle.decide(case)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_case_prompt(case)},
            {"role": "assistant", "content": decision_to_json(decision)},
        ]
    }


def make_sft_tool_example(case: DisputeCase, oracle: OracleAgent | None = None) -> dict:
    """构造一条多步工具调用 SFT 样本。

    messages 结构：
        system: TOOL_LOOP_SYSTEM_PROMPT
        user:   工单 + 可用工具 + 输出要求
        assistant: tool_call JSON
        tool:   工具结果（由仿真工具层执行）
        ...（重复）
        assistant: final JSON（Oracle 教师决策）
    """
    oracle = oracle or OracleAgent()
    messages: list[dict] = [
        {"role": "system", "content": TOOL_LOOP_SYSTEM_PROMPT},
        {"role": "user", "content": build_tool_loop_user_prompt(case, format_tool_definitions())},
    ]
    for tool_name, args in select_teacher_tools(case):
        call_json = json.dumps(
            {"action": "tool_call", "tool": tool_name, "arguments": args},
            ensure_ascii=False,
        )
        result = execute_tool(case, tool_name, args)
        messages.append({"role": "assistant", "content": call_json})
        # 用 user 角色回填工具结果，兼容 OpenAI/vLLM 与 ms-swift
        messages.append({"role": "user", "content": format_tool_result_message(result)})

    decision = oracle.decide(case)
    messages.append({"role": "assistant", "content": decision_to_final_json(decision)})
    return {"messages": messages}


def make_rl_example(case: DisputeCase) -> dict:
    """构造一条单步 RL prompt 样本（不含工具，仅用于快速调试）。"""
    prompt = f"案例编号：{case.case_id}\n{build_case_prompt(case)}"
    return {"prompt": prompt, "case_id": case.case_id, "tool_cost": 0.0}


def make_rl_tool_example(case: DisputeCase) -> dict:
    """构造一条多步 RL 样本：prompt 中预嵌入工具查询结果，只训练最终判责。

    prompt 内嵌 `案例编号：Cxxxxxx`，并包含仿真器预执行的工具结果。
    RL 奖励会从 extra_info 中读取工具成本并扣除。
    """
    tools = select_teacher_tools(case)
    results = [execute_tool(case, name, args) for name, args in tools]
    tool_results_text = "\n".join(format_tool_result_message(r) for r in results) or "（未调用工具）"
    total_tool_cost = sum(r.cost for r in results)
    prompt = (
        f"案例编号：{case.case_id}\n"
        f"{build_rl_user_prompt_with_tool_results(case, tool_results_text)}"
    )
    return {
        "prompt": prompt,
        "case_id": case.case_id,
        "tool_cost": round(total_tool_cost, 2),
    }


# ---------- 批量生成 ----------
def generate_sft_data(
    n_cases: int,
    seed: int = 42,
    output_path: str | Path = "data/sft.jsonl",
    style: str = "multi_tool",
) -> list[DisputeCase]:
    """生成 SFT 训练数据，返回对应 case 列表。

    Args:
        style: "multi_tool"（默认，多步工具调用）或 "single"（单步直接判责）。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gen = CaseGenerator(seed=seed)
    oracle = OracleAgent()
    cases: list[DisputeCase] = []
    with output_path.open("w", encoding="utf-8") as f:
        for case in gen.generate_batch(n_cases):
            cases.append(case)
            if style == "multi_tool":
                example = make_sft_tool_example(case, oracle)
            else:
                example = make_sft_example(case, oracle)
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    return cases


def generate_rl_data(
    n_cases: int,
    seed: int = 42,
    output_dir: str | Path = "data",
    multi_tool: bool = True,
) -> list[DisputeCase]:
    """生成 RL 训练数据与 case 表。

    写入：
    - data/rl.jsonl：{"prompt": ..., "case_id": ..., "tool_cost": ...}
    - data/rl_cases.jsonl：完整 case（含 ground truth），供 RewardEngine 加载

    Args:
        multi_tool: True（默认）时 prompt 中预嵌入工具查询结果，RL 只训练最终判责；
                    False 时使用无工具的原始工单 prompt（调试用）。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gen = CaseGenerator(seed=seed)
    cases: list[DisputeCase] = []
    with (output_dir / "rl.jsonl").open("w", encoding="utf-8") as f_prompt, \
         (output_dir / "rl_cases.jsonl").open("w", encoding="utf-8") as f_case:
        for case in gen.generate_batch(n_cases):
            cases.append(case)
            example = make_rl_tool_example(case) if multi_tool else make_rl_example(case)
            f_prompt.write(json.dumps(example, ensure_ascii=False) + "\n")
            f_case.write(json.dumps(case_to_dict(case), ensure_ascii=False) + "\n")
    return cases


def load_cases_from_jsonl(path: str | Path) -> list[DisputeCase]:
    """从 rl_cases.jsonl 恢复 case 列表。"""
    cases: list[DisputeCase] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(case_from_dict(json.loads(line)))
    return cases
