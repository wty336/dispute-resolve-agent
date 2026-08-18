"""结构化合成纠纷事实生成器。

生成器先采样隐藏真值，再渲染公开观测；绝不会把隐藏字段写入公开观测。
"""
from __future__ import annotations

import hashlib
import json
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

# 这些组合不会参与常规生成，只用于 OOD 未见组合桶。
RESERVED_OOD_COMBINATIONS = (
    ("not_received", Liability.BUYER, "high"),
    ("damaged", Liability.NONE, "high"),
    ("quality", Liability.SPLIT, "high"),
    ("aftersales", Liability.BUYER, "high"),
)


@dataclass
class FactInstance:
    fact_instance_id: str
    case_id: str
    observation: DisputeObservation
    ground_truth: DisputeGroundTruth
    ood_bucket: str | None = None
    split: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SFTProfile:
    category: str
    tool_names: tuple[str, ...] = ()
    edge_case: str | None = None


SFT_CATEGORY_COUNTS = {"direct": 800, "multi_tool": 500, "edge_case": 200}
SFT_TOOL_NAMES = (
    "check_logistics",
    "check_buyer_history",
    "check_merchant_history",
    "verify_evidence",
)


def _scaled_counts(counts: dict[str, int], target: int) -> dict[str, int]:
    total = sum(counts.values())
    raw = {name: value * target / total for name, value in counts.items()}
    scaled = {name: int(value) for name, value in raw.items()}
    for name in sorted(counts, key=lambda key: raw[key] - scaled[key], reverse=True)[: target - sum(scaled.values())]:
        scaled[name] += 1
    return scaled


def plan_sft_profiles(count: int, seed: int) -> list[SFTProfile]:
    """创建确定性的 direct/multi-tool/edge-case SFT 混合数据。"""
    counts = _scaled_counts(SFT_CATEGORY_COUNTS, count)
    profiles: list[SFTProfile] = [SFTProfile(category="direct") for _ in range(counts["direct"])]
    for index in range(counts["multi_tool"]):
        tool_count = index % 4 + 1
        start = index % len(SFT_TOOL_NAMES)
        tools = tuple(SFT_TOOL_NAMES[(start + offset) % len(SFT_TOOL_NAMES)] for offset in range(tool_count))
        profiles.append(SFTProfile(category="multi_tool", tool_names=tools))

    edge_cases = ("manual_escalation", "tool_failure", "illegal_result_recovery", "high_risk")
    for index in range(counts["edge_case"]):
        edge_case = edge_cases[index % len(edge_cases)]
        tools = {
            "manual_escalation": (),
            "tool_failure": ("check_logistics",),
            "illegal_result_recovery": ("verify_evidence", "check_buyer_history"),
            "high_risk": ("check_buyer_history", "check_merchant_history", "verify_evidence"),
        }[edge_case]
        profiles.append(SFTProfile(category="edge_case", tool_names=tools, edge_case=edge_case))

    random.Random(seed).shuffle(profiles)
    return profiles


def _instance_rng(seed: int, index: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{index}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def fact_fingerprint(instance: FactInstance) -> str:
    """对结构化事实计算哈希，同时排除 ID 和语言渲染内容。"""
    payload = {
        "item_name": instance.observation.item_name,
        "order_amount": instance.observation.order_amount,
        "claim_type": instance.observation.claim_type,
        "buyer_requested_amount": instance.observation.buyer_requested_amount,
        "true_liability": instance.ground_truth.true_liability.value,
        "true_loss": instance.ground_truth.true_loss,
        "buyer_strategy": instance.ground_truth.buyer_strategy,
        "merchant_strategy": instance.ground_truth.merchant_strategy,
        "should_escalate": instance.ground_truth.should_escalate,
        "risk_level": instance.ground_truth.risk_level,
        "logistics_state": instance.ground_truth.logistics_state,
        "evidence_authenticity": instance.ground_truth.evidence_authenticity,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _logistics_state(claim_type: str, liability: Liability, rng: random.Random) -> str:
    if claim_type == "not_received":
        return "lost" if liability in {Liability.MERCHANT, Liability.SPLIT} else "delivered_verified"
    if claim_type == "damaged":
        return "damaged_in_transit" if liability in {Liability.MERCHANT, Liability.SPLIT} else "delivered_verified"
    return rng.choice(["delivered_verified", "delivered_disputed"])


def generate_fact_instances(
    seed: int,
    n: int,
    *,
    start_id: int = 0,
    ood_bucket: str | None = None,
    language_shift: bool = False,
    tool_noise: bool = False,
) -> list[FactInstance]:
    """生成 ``n`` 个确定性的事实实例。"""
    instances: list[FactInstance] = []
    for offset in range(n):
        index = start_id + offset
        rng = _instance_rng(seed, index)
        fact_id = f"fact-{index:06d}"
        case_id = f"case-{index:06d}"
        item_name, order_amount = rng.choice(ITEMS)
        unseen_combination = ood_bucket == "unseen_combination"
        if unseen_combination:
            claim_type, true_liability, risk_level = RESERVED_OOD_COMBINATIONS[
                index % len(RESERVED_OOD_COMBINATIONS)
            ]
        else:
            claim_type = rng.choice(CLAIM_TYPES)
            true_liability = rng.choice(list(Liability))
            risk_level = rng.choice(["low", "medium", "high"])
            if (claim_type, true_liability, risk_level) in RESERVED_OOD_COMBINATIONS:
                risk_level = "medium"
        order_id = f"o-{index:06d}"
        buyer_id = f"b-{index:06d}"
        merchant_id = f"m-{index:06d}"
        true_loss = round(order_amount * rng.uniform(0.2, 0.8), 2)
        if true_liability == Liability.MERCHANT:
            low = round(max(0.0, true_loss * 0.8), 2)
            high = round(min(order_amount, true_loss * 1.2), 2)
        elif true_liability == Liability.SPLIT:
            low = round(max(0.0, true_loss * 0.35), 2)
            high = round(min(order_amount, true_loss * 0.65), 2)
        else:
            low, high = 0.0, 0.0
        buyer_strategy = rng.choice(["honest", "exaggerate", "fraud"])
        merchant_strategy = rng.choice(["honest", "evasive", "deny"])
        authenticity = (
            "forged" if buyer_strategy == "fraud" else "suspicious" if buyer_strategy == "exaggerate" else "authentic"
        )
        should_escalate = risk_level == "high" or authenticity == "forged" or rng.random() < 0.08
        evidence_id = f"chat:{index}"
        merchant_fault = true_liability in {Liability.MERCHANT, Liability.SPLIT}
        buyer_fault = true_liability in {Liability.BUYER, Liability.SPLIT}
        # 公开受理证据有参考价值但并不完美。保留假阳性和假阴性观测，避免句式
        # 模板退化成隐藏责任的查表规则。
        observed_merchant_fault = rng.random() < (0.78 if merchant_fault else 0.22)
        observed_buyer_fault = rng.random() < (0.78 if buyer_fault else 0.22)
        merchant_cue = rng.choice(
            (
                "出库质检照片缺失",
                "包装记录显示防护材料不足",
                "商家未能提供完整履约凭证",
            )
            if observed_merchant_fault
            else (
                "出库质检记录完整",
                "包装与发货记录未见异常",
                "商家已提交完整履约凭证",
            )
        )
        buyer_cue = rng.choice(
            (
                "售后记录显示商品经使用后出现问题",
                "签收后较长时间才首次反馈异常",
                "沟通记录显示存在保管或操作不当",
            )
            if observed_buyer_fault
            else (
                "买家在签收后立即反馈且未使用商品",
                "开箱记录与首次投诉时间一致",
                "现有记录未发现买家操作不当",
            )
        )
        evidence_description = f"{merchant_cue}；{buyer_cue}"

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
                    evidence_id=evidence_id,
                    type="平台初审记录",
                    description=evidence_description,
                    source="platform",
                    visible=True,
                )
            ],
        )
        ground_truth = DisputeGroundTruth(
            case_id=case_id,
            true_liability=true_liability,
            true_loss=true_loss,
            reasonable_compensation_range=(low, high),
            buyer_strategy=buyer_strategy,
            merchant_strategy=merchant_strategy,
            should_escalate=should_escalate,
            tool_information_value={
                "check_logistics": 0.9 if claim_type in {"not_received", "damaged"} else 0.2,
                "check_buyer_history": 0.8 if buyer_strategy != "honest" else 0.3,
                "check_merchant_history": 0.8 if merchant_strategy != "honest" else 0.3,
                "verify_evidence": 0.9 if authenticity != "authentic" else 0.4,
            },
            risk_level=risk_level,
            logistics_state=_logistics_state(claim_type, true_liability, rng),
            evidence_authenticity={evidence_id: authenticity},
            tool_noise_rate=0.40 if tool_noise else 0.08,
            tool_timeout_rate=0.15 if tool_noise else 0.02,
            tool_missing_rate=0.20 if tool_noise else 0.03,
        )
        if language_shift:
            styles = (
                f"亲，{observation.buyer_claim} 麻烦尽快处理！",
                f"我这单真的很急：{observation.buyer_claim}",
                f"平台您好，现正式反馈：{observation.buyer_claim}",
            )
            observation.buyer_claim = styles[index % len(styles)]
        instances.append(
            FactInstance(
                fact_instance_id=fact_id,
                case_id=case_id,
                observation=observation,
                ground_truth=ground_truth,
                ood_bucket=ood_bucket,
                metadata={
                    "tool_noise": tool_noise,
                    "language_shift": language_shift,
                    "unseen_combination": unseen_combination,
                },
            )
        )
    return instances
