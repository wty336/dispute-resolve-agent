"""Structured synthetic dispute fact generator.

The generator first samples hidden ground truth, then renders a public
observation.  It never writes hidden fields into the public observation.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from dispute_agent.domain.schemas import (
    DisputeGroundTruth,
    DisputeObservation,
    Evidence,
    Liability,
)

ITEMS = [
    ("蓝牙耳机", 199.0),
    ("电热水壶", 89.0),
    ("连衣裙", 259.0),
    ("机械键盘", 349.0),
    ("移动电源", 129.0),
    ("运动鞋", 399.0),
    ("智能手环", 249.0),
    ("保温杯", 79.0),
    ("台灯", 119.0),
    ("双肩包", 159.0),
]

CLAIM_TYPES = ["quality", "not_as_described", "damaged", "not_received", "aftersales"]


@dataclass
class FactInstance:
    fact_instance_id: str
    case_id: str
    observation: DisputeObservation
    ground_truth: DisputeGroundTruth
    ood_bucket: str | None = None
    split: str | None = None
    metadata: dict = field(default_factory=dict)


def generate_fact_instances(
    seed: int,
    n: int,
    *,
    start_id: int = 0,
    ood_bucket: str | None = None,
    language_shift: bool = False,
    tool_noise: bool = False,
) -> list[FactInstance]:
    """Generate ``n`` deterministic fact instances."""
    rng = random.Random(seed)
    instances: list[FactInstance] = []
    for offset in range(n):
        index = start_id + offset
        fact_id = f"fact-{index:06d}"
        case_id = f"case-{index:06d}"
        item_name, order_amount = rng.choice(ITEMS)
        claim_type = rng.choice(CLAIM_TYPES)
        order_id = f"o-{index:06d}"
        buyer_id = f"b-{index:06d}"
        merchant_id = f"m-{index:06d}"
        true_liability = rng.choice(list(Liability))
        true_loss = round(order_amount * rng.uniform(0.2, 0.8), 2)
        low = round(max(0.0, true_loss * 0.8), 2)
        high = round(min(order_amount, true_loss * 1.2), 2)
        should_escalate = rng.random() < 0.15
        risk_level = rng.choice(["low", "medium", "high"])

        observation = DisputeObservation(
            case_id=case_id,
            order_id=order_id,
            buyer_id=buyer_id,
            merchant_id=merchant_id,
            item_name=item_name,
            order_amount=order_amount,
            claim_type=claim_type,
            buyer_claim=f"商品出现问题：{claim_type}，要求处理。",
            buyer_requested_amount=round(order_amount * rng.uniform(0.3, 1.0), 2),
            merchant_response="商家反馈商品发出前完好。",
            chat_log=[
                f"买家：{claim_type}，请处理。",
                "商家：我们核实后回复。",
            ],
            evidence=[
                Evidence(
                    evidence_id=f"chat:{index}",
                    type="聊天记录",
                    description="买家与商家沟通记录",
                    source="buyer",
                    visible=True,
                )
            ],
        )
        ground_truth = DisputeGroundTruth(
            case_id=case_id,
            true_liability=true_liability,
            true_loss=true_loss,
            reasonable_compensation_range=(low, high),
            buyer_strategy=rng.choice(["honest", "exaggerate", "fraud"]),
            merchant_strategy=rng.choice(["honest", "evasive", "deny"]),
            should_escalate=should_escalate,
            tool_information_value={
                "check_logistics": rng.random(),
                "check_buyer_history": rng.random(),
                "check_merchant_history": rng.random(),
                "verify_evidence": rng.random(),
            },
            risk_level=risk_level,
        )
        if language_shift:
            observation.buyer_claim = f"亲，{observation.buyer_claim} 麻烦尽快处理！"
        instances.append(
            FactInstance(
                fact_instance_id=fact_id,
                case_id=case_id,
                observation=observation,
                ground_truth=ground_truth,
                ood_bucket=ood_bucket,
                metadata={"tool_noise": tool_noise, "language_shift": language_shift},
            )
        )
    return instances
