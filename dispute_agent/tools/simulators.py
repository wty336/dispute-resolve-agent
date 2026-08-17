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
    """Return a deterministic, hidden-fact-conditioned tool result."""
    args = {k: str(v) for k, v in arguments.items()}
    seed = _stable_seed(case_id, tool_name, _canonical_json(args), str(case_seed))
    rng = random.Random(seed)
    cost = TOOL_COSTS.get(tool_name, 0.0)
    timeout_rate = float(getattr(ground_truth, "tool_timeout_rate", 0.02))
    missing_rate = float(getattr(ground_truth, "tool_missing_rate", 0.03))
    noise_rate = float(getattr(ground_truth, "tool_noise_rate", 0.08))

    failure_roll = rng.random()
    if failure_roll < timeout_rate:
        return ToolResult(
            tool_name=tool_name,
            arguments=args,
            payload=f"{tool_name} 查询超时，请稍后重试。",
            cached=False,
            cost=cost,
            success=False,
            error_type="timeout",
        )
    if failure_roll < timeout_rate + missing_rate:
        return ToolResult(
            tool_name=tool_name,
            arguments=args,
            payload=f"{tool_name} 暂无可用记录。",
            cached=False,
            cost=cost,
            success=False,
            error_type="missing",
        )

    if tool_name == "check_logistics":
        payload = _simulate_logistics(rng, args, ground_truth, noise_rate)
    elif tool_name == "check_buyer_history":
        payload = _simulate_buyer_history(rng, args, ground_truth, noise_rate)
    elif tool_name == "check_merchant_history":
        payload = _simulate_merchant_history(rng, args, ground_truth, noise_rate)
    elif tool_name == "verify_evidence":
        payload = _simulate_verify_evidence(rng, args, ground_truth, noise_rate)
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


def _noisy_choice(rng: random.Random, truth: str, choices: tuple[str, ...], noise_rate: float) -> str:
    if rng.random() >= noise_rate:
        return truth
    alternatives = tuple(choice for choice in choices if choice != truth)
    return rng.choice(alternatives) if alternatives else truth


def _simulate_logistics(rng: random.Random, args: dict, ground_truth: object | None, noise_rate: float) -> str:
    states = ("delivered_verified", "delivered_disputed", "lost", "damaged_in_transit")
    truth = str(getattr(ground_truth, "logistics_state", "delivered_verified"))
    state = _noisy_choice(rng, truth, states, noise_rate)
    status_by_state = {
        "delivered_verified": "物流显示已签收，签收人与订单地址一致",
        "delivered_disputed": "物流显示已签收，但签收底单模糊，存在他人代签可能",
        "lost": "物流轨迹长时间停滞，承运方登记为运输丢失",
        "damaged_in_transit": "承运方记录外包装破损，并有异常处理扫描",
    }
    status = status_by_state[state]
    return f"订单 {args['order_id']} 物流核验：{status}。证据ID：logistics:{args['order_id']}"


def _simulate_buyer_history(rng: random.Random, args: dict, ground_truth: object | None, noise_rate: float) -> str:
    strategies = ("honest", "exaggerate", "fraud")
    truth = str(getattr(ground_truth, "buyer_strategy", "honest"))
    strategy = _noisy_choice(rng, truth, strategies, noise_rate)
    if strategy == "honest":
        rate, tag = rng.uniform(0.01, 0.07), "历史信誉良好"
    elif strategy == "exaggerate":
        rate, tag = rng.uniform(0.12, 0.24), "历史诉求金额普遍偏高"
    else:
        rate, tag = rng.uniform(0.26, 0.40), "多起投诉被平台驳回"
    return f"买家 {args['buyer_id']} 近90天投诉率约 {rate:.1%}，{tag}。证据ID：buyer_history:{args['buyer_id']}"


def _simulate_merchant_history(rng: random.Random, args: dict, ground_truth: object | None, noise_rate: float) -> str:
    strategies = ("honest", "evasive", "deny")
    truth = str(getattr(ground_truth, "merchant_strategy", "honest"))
    strategy = _noisy_choice(rng, truth, strategies, noise_rate)
    if strategy == "honest":
        rate, tag = rng.uniform(0.01, 0.05), "历史配合度良好"
    elif strategy == "evasive":
        rate, tag = rng.uniform(0.10, 0.18), "多次延迟提交售后材料"
    else:
        rate, tag = rng.uniform(0.18, 0.30), "有未配合平台处理的记录"
    return f"商家 {args['merchant_id']} 近90天纠纷率约 {rate:.1%}，{tag}。证据ID：merchant_history:{args['merchant_id']}"


def _simulate_verify_evidence(
    rng: random.Random,
    args: dict,
    ground_truth: object | None,
    noise_rate: float,
) -> str:
    authenticity = getattr(ground_truth, "evidence_authenticity", {}) or {}
    truth = str(authenticity.get(args["evidence_id"], "suspicious"))
    state = _noisy_choice(rng, truth, ("authentic", "suspicious", "forged"), noise_rate)
    verdict_by_state = {
        "authentic": "核验通过：证据未发现伪造痕迹",
        "suspicious": "核验存疑：存在修图或时间不一致迹象",
        "forged": "核验失败：证据存在明显伪造痕迹",
    }
    verdict = verdict_by_state[state]
    return f"证据 {args['evidence_id']} 核验：{verdict}。证据ID：{args['evidence_id']}"
