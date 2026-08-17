"""仿真工具层。

推理侧 ToolLoopAgent 和 SFT/RL 数据生成共用本模块：
- 工具定义一致；
- 工具执行结果带噪声但可复现（用 case_id + tool_name 做随机种子，
  保证同一案例无论何时查询，结果一致）。
- 每次工具调用都有成本，最终计入长期收益（在 payoff 中通过轨迹成本体现）。
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass

from .models import (
    BuyerStrategy,
    ClaimType,
    DisputeCase,
    Liability,
    MerchantStrategy,
)

# 工具定义（name, description, arguments, cost 元/次）
TOOL_DEFINITIONS = [
    {
        "name": "check_order_logistics",
        "description": "查询订单物流签收状态，核实买家是否签收、是否本人签收",
        "arguments": {"order_id": "订单号"},
        "cost": 2.0,
    },
    {
        "name": "check_merchant_history",
        "description": "查询商家历史纠纷率、处罚记录与配合度",
        "arguments": {"merchant_id": "商家ID"},
        "cost": 3.0,
    },
    {
        "name": "check_buyer_history",
        "description": "查询买家历史投诉率、被驳回率与信誉记录",
        "arguments": {"buyer_id": "买家ID"},
        "cost": 3.0,
    },
    {
        "name": "verify_evidence",
        "description": "核验买家/商家提交的证据真伪（成本较高，按需使用）",
        "arguments": {"evidence_type": "买家举证/商家举证"},
        "cost": 8.0,
    },
]

TOOL_COSTS = {t["name"]: t["cost"] for t in TOOL_DEFINITIONS}


@dataclass
class ToolResult:
    """一次工具调用的结果。"""

    tool_name: str
    arguments: dict
    content: str
    cost: float
    success: bool = True


def format_tool_definitions() -> str:
    """把工具定义格式化为 prompt 可用的 JSON 文本。"""
    return json.dumps(TOOL_DEFINITIONS, ensure_ascii=False, indent=2)


def _stable_seed(case_id: str, salt: str) -> int:
    """生成稳定随机种子（不依赖 PYTHONHASHSEED）。"""
    digest = hashlib.md5(f"{case_id}:{salt}".encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


def execute_tool(case: DisputeCase, tool_name: str, arguments: dict | None = None) -> ToolResult:
    """执行工具并返回带噪声的结果。

    噪声与 case 绑定：同一案例同一工具，结果可复现。
    """
    arguments = arguments or {}
    cost = TOOL_COSTS.get(tool_name, 0.0)

    if tool_name == "check_order_logistics":
        return _check_logistics(case, cost)
    if tool_name == "check_merchant_history":
        return _check_merchant_history(case, cost)
    if tool_name == "check_buyer_history":
        return _check_buyer_history(case, cost)
    if tool_name == "verify_evidence":
        return _verify_evidence(case, arguments, cost)

    return ToolResult(
        tool_name=tool_name,
        arguments=arguments,
        content=f"未知工具：{tool_name}",
        cost=0.0,
        success=False,
    )


def _rng_for(case: DisputeCase, salt: str) -> random.Random:
    return random.Random(_stable_seed(case.case_id, salt))


# ---------- 各工具实现 ----------
def _check_logistics(case: DisputeCase, cost: float) -> ToolResult:
    rng = _rng_for(case, "logistics")
    if case.claim_type == ClaimType.NOT_RECEIVED:
        if case.true_liability in (Liability.BUYER, Liability.NONE):
            status = "物流显示已签收，签收人：本人/家人代收"
        else:
            status = "物流显示已签收，但签收底单模糊，存在他人代签可能"
    elif case.claim_type == ClaimType.DAMAGED:
        status = "物流显示已签收，外包装有轻微破损记录"
    else:
        status = "物流显示已签收，签收后超过 24 小时无异常上报"

    # 噪声：10% 概率信息不完整
    if rng.random() < 0.10:
        status = "物流信息缺失，仅有发货扫描记录"
    return ToolResult(
        tool_name="check_order_logistics",
        arguments={"order_id": case.order_id},
        content=status,
        cost=cost,
    )


def _check_merchant_history(case: DisputeCase, cost: float) -> ToolResult:
    rng = _rng_for(case, "merchant_history")
    if case.merchant_strategy in (MerchantStrategy.EVASIVE, MerchantStrategy.DENY):
        base_rate = rng.uniform(0.12, 0.30)
        record = "近90天纠纷率偏高，有 2 起未配合平台处理的记录"
    else:
        base_rate = rng.uniform(0.01, 0.06)
        record = "近90天纠纷率低，历史配合度良好"
    return ToolResult(
        tool_name="check_merchant_history",
        arguments={"merchant_id": case.merchant_id},
        content=f"{record}（历史纠纷率约 {base_rate:.1%}）",
        cost=cost,
    )


def _check_buyer_history(case: DisputeCase, cost: float) -> ToolResult:
    rng = _rng_for(case, "buyer_history")
    if case.buyer_strategy == BuyerStrategy.FRAUD:
        rate = rng.uniform(0.35, 0.60)
        content = f"买家近90天投诉率 {rate:.1%}，多起被平台驳回"
    elif case.buyer_strategy == BuyerStrategy.EXAGGERATE:
        rate = rng.uniform(0.12, 0.25)
        content = f"买家近90天投诉率 {rate:.1%}，历史诉求金额普遍偏高"
    else:
        rate = rng.uniform(0.01, 0.08)
        content = f"买家近90天投诉率 {rate:.1%}，历史信誉良好"
    return ToolResult(
        tool_name="check_buyer_history",
        arguments={"buyer_id": case.buyer_id},
        content=content,
        cost=cost,
    )


def _verify_evidence(case: DisputeCase, arguments: dict, cost: float) -> ToolResult:
    evidence_type = arguments.get("evidence_type", "买家举证")
    rng = _rng_for(case, f"evidence:{evidence_type}")
    if "买家" in evidence_type:
        strength = sum(e.strength for e in case.buyer_evidence) / max(1, len(case.buyer_evidence))
        # 买家证据越强越可能核验为真
        p_true = strength
    else:
        strength = sum(e.strength for e in case.merchant_evidence) / max(1, len(case.merchant_evidence))
        p_true = strength

    roll = rng.random()
    if roll < p_true:
        verdict = "核验通过：证据未发现伪造痕迹"
    elif roll < p_true + 0.25:
        verdict = "核验存疑：存在修图/时间不一致的迹象"
    else:
        verdict = "核验失败：证据存在明显伪造/剪辑痕迹"
    return ToolResult(
        tool_name="verify_evidence",
        arguments=arguments,
        content=verdict,
        cost=cost,
    )


def format_tool_result_message(result: ToolResult) -> str:
    """把工具结果格式化为回填给模型的消息内容。"""
    return (
        f"工具 {result.tool_name} 返回：{result.content}\n"
        f"（本次调用成本 ¥{result.cost:.2f}）"
    )
