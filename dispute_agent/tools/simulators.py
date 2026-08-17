"""Deterministic tool simulators.

Tool results are reproducible for a fixed ``(case_id, tool_name, arguments,
case_seed)`` tuple.  The simulators may read hidden ground truth internally to
calibrate noise, but they never place hidden labels into the returned payload.
"""
from __future__ import annotations

import hashlib
import json
import random

from dispute_agent.domain.schemas import ToolResult

TOOL_COSTS = {
    "check_logistics": 2.0,
    "check_buyer_history": 3.0,
    "check_merchant_history": 3.0,
    "verify_evidence": 8.0,
}

TOOL_ARGUMENT_SCHEMAS = {
    "check_logistics": {"order_id": str},
    "check_buyer_history": {"buyer_id": str},
    "check_merchant_history": {"merchant_id": str},
    "verify_evidence": {"evidence_id": str},
}


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _canonical_json(value: dict) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def simulate_tool(
    tool_name: str,
    *,
    case_id: str,
    arguments: dict,
    case_seed: str | int | None,
    ground_truth: object | None = None,
) -> ToolResult:
    """Return a deterministic simulated tool result.

    ``ground_truth`` is accepted for future calibrated noise; this minimal
    implementation only uses it to choose wording and never leaks hidden
    attribute names.
    """
    args = {k: str(v) for k, v in arguments.items()}
    seed = _stable_seed(case_id, tool_name, _canonical_json(args), str(case_seed))
    rng = random.Random(seed)
    cost = TOOL_COSTS.get(tool_name, 0.0)

    if tool_name == "check_logistics":
        payload = _simulate_logistics(rng, args)
    elif tool_name == "check_buyer_history":
        payload = _simulate_buyer_history(rng, args)
    elif tool_name == "check_merchant_history":
        payload = _simulate_merchant_history(rng, args)
    elif tool_name == "verify_evidence":
        payload = _simulate_verify_evidence(rng, args)
    else:
        raise KeyError(f"unknown tool: {tool_name}")

    return ToolResult(
        tool_name=tool_name,
        arguments=args,
        payload=payload,
        cached=False,
        cost=cost,
        success=True,
    )


def _simulate_logistics(rng: random.Random, args: dict) -> str:
    roll = rng.random()
    if roll < 0.1:
        status = "物流信息缺失，仅有发货扫描记录"
    elif roll < 0.45:
        status = "物流显示已签收，签收底单模糊，存在他人代签可能"
    else:
        status = "物流显示已签收，签收人与订单地址一致"
    return f"订单 {args['order_id']} 物流核验：{status}。证据ID：logistics:{args['order_id']}"


def _simulate_buyer_history(rng: random.Random, args: dict) -> str:
    rate = rng.uniform(0.01, 0.08) if rng.random() < 0.7 else rng.uniform(0.12, 0.35)
    if rate < 0.08:
        tag = "历史信誉良好"
    elif rate < 0.25:
        tag = "历史诉求金额普遍偏高"
    else:
        tag = "多起投诉被平台驳回"
    return f"买家 {args['buyer_id']} 近90天投诉率约 {rate:.1%}，{tag}。证据ID：buyer_history:{args['buyer_id']}"


def _simulate_merchant_history(rng: random.Random, args: dict) -> str:
    rate = rng.uniform(0.01, 0.06) if rng.random() < 0.7 else rng.uniform(0.12, 0.25)
    if rate < 0.06:
        tag = "历史配合度良好"
    else:
        tag = "有未配合平台处理的记录"
    return f"商家 {args['merchant_id']} 近90天纠纷率约 {rate:.1%}，{tag}。证据ID：merchant_history:{args['merchant_id']}"


def _simulate_verify_evidence(rng: random.Random, args: dict) -> str:
    roll = rng.random()
    if roll < 0.6:
        verdict = "核验通过：证据未发现伪造痕迹"
    elif roll < 0.85:
        verdict = "核验存疑：存在修图/时间不一致迹象"
    else:
        verdict = "核验失败：证据存在明显伪造痕迹"
    return f"证据 {args['evidence_id']} 核验：{verdict}。证据ID：{args['evidence_id']}"
