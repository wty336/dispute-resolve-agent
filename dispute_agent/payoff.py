"""长期收益核算模型。

平台目标不是单轮判对，而是长期收益最大化。每轮收益由以下部分组成：

    U = buyer_LTV * buyer_repurchase_prob
      + merchant_LTV * merchant_retention_prob
      - immediate_cost
      - risk_cost
      + reputation_gain

其中：
- immediate_cost：赔付金额 + 人工升级成本 + 基础处理成本
- risk_cost：该赔不赔的升级/监管风险 + 不该赔乱赔的道德风险
- reputation_gain：判责与事实一致带来的平台信任积累
"""
from __future__ import annotations

from dataclasses import dataclass

from .models import AgentDecision, CaseOutcome, DisputeCase, Liability


@dataclass
class PayoffConfig:
    """收益模型参数，可按业务口径调整。"""

    buyer_ltv: float = 800.0  # 买家生命周期价值（元）
    merchant_ltv: float = 1500.0  # 商家生命周期价值（元）
    base_processing_cost: float = 5.0  # 单均基础处理成本
    manual_upgrade_cost: float = 80.0  # 人工升级额外成本
    wrong_deny_risk_factor: float = 1.2  # 该赔不赔 → 升级/监管风险系数
    moral_hazard_factor: float = 1.5  # 不该赔乱赔 → 道德风险系数
    reputation_scale: float = 60.0  # 判责正确带来的口碑收益
    wrong_reputation_penalty: float = 30.0  # 判责方向错误的口碑损失
    gamma: float = 0.98  # 折现因子（用于多轮累计，单轮核算时不需要）


def liability_match_score(decision: AgentDecision, case: DisputeCase) -> float:
    """判责与事实的匹配度，0~1。"""
    d = decision.liability
    t = case.true_liability
    if d == t:
        return 1.0
    # 一方责任 vs 双方共担：方向大体接近
    if {d, t} == {Liability.MERCHANT, Liability.SPLIT}:
        return 0.6
    if {d, t} == {Liability.BUYER, Liability.SPLIT}:
        return 0.6
    # 无法认定：部分正确（保守处理比完全判反好）
    if d == Liability.NONE or t == Liability.NONE:
        return 0.3
    # 商家责任 vs 买家责任：完全判反
    return 0.0


def _sigmoid(x: float) -> float:
    """将 [-inf, inf] 映射到 [0, 1]。"""
    import math

    return 1.0 / (1.0 + math.exp(-x))


def compute_outcome(case: DisputeCase, decision: AgentDecision, config: PayoffConfig | None = None) -> CaseOutcome:
    """计算单轮决策的收益核算结果。"""
    config = config or PayoffConfig()
    notes: list[str] = []

    accuracy = liability_match_score(decision, case)
    t = case.true_liability
    request = max(case.buyer_requested_amount, 1.0)
    comp = decision.compensation

    # ---------- 买家满意度 ----------
    # 基础分 + 诉求满足程度 + 判责方向修正
    claim_satisfied = comp / request
    buyer_base = 0.25 + 0.75 * claim_satisfied
    if t in (Liability.MERCHANT, Liability.SPLIT):
        # 买家确实有损失：被判支持会显著提升满意度
        buyer_satisfaction = buyer_base + 0.10 * accuracy
        if decision.liability in (Liability.MERCHANT, Liability.SPLIT):
            notes.append("买家有理且获得支持")
        else:
            buyer_satisfaction -= 0.25
            notes.append("买家有理但未获支持，满意度下降")
    else:
        # 买家无真实损失：被判支持会短期满意，但长期滋生道德风险
        buyer_satisfaction = buyer_base
        if decision.liability in (Liability.MERCHANT, Liability.SPLIT):
            notes.append("买家无真实损失但获得赔付，短期满意")
    buyer_satisfaction = min(1.0, max(0.0, buyer_satisfaction))

    # ---------- 商家满意度 ----------
    merchant_base = 0.6
    if decision.liability in (Liability.MERCHANT, Liability.SPLIT):
        payout_ratio = comp / max(case.order_amount, 1.0)
        merchant_satisfaction = merchant_base - 0.35 * min(1.0, payout_ratio) - 0.15 * accuracy
        if t == Liability.BUYER:
            merchant_satisfaction -= 0.20
            notes.append("商家实际无责却被判担责")
    else:
        merchant_satisfaction = merchant_base + 0.20 * accuracy
        if t in (Liability.MERCHANT, Liability.SPLIT):
            merchant_satisfaction -= 0.10
            notes.append("商家有责但未被认定")
    merchant_satisfaction = min(1.0, max(0.0, merchant_satisfaction))

    # ---------- 留存/复购概率 ----------
    buyer_repurchase_prob = _sigmoid(6.0 * (buyer_satisfaction - 0.45))
    merchant_retention_prob = _sigmoid(6.0 * (merchant_satisfaction - 0.45))

    # ---------- 成本 ----------
    base_cost = config.base_processing_cost
    escalate_cost = config.manual_upgrade_cost if decision.escalate else 0.0
    immediate_cost = base_cost + escalate_cost + comp + decision.tool_cost

    # ---------- 风险成本 ----------
    risk_cost = 0.0
    if t in (Liability.MERCHANT, Liability.SPLIT):
        deserved = min(case.true_buyer_loss, case.order_amount)
        shortfall = max(0.0, deserved - comp)
        risk_cost += config.wrong_deny_risk_factor * shortfall
        if shortfall > 1e-6:
            notes.append(f"该赔不赔缺口 {shortfall:.2f} 元，存在升级/监管风险")
    if t in (Liability.BUYER, Liability.NONE):
        # 买家不应得赔付：赔得越多，道德风险越大
        risk_cost += config.moral_hazard_factor * comp
        if comp > 1e-6:
            notes.append(f"向无真实损失的买家赔付 {comp:.2f} 元，滋生道德风险")

    # ---------- 口碑 ----------
    if accuracy >= 0.99:
        reputation_gain = config.reputation_scale
    elif accuracy <= 0.01:
        reputation_gain = -config.wrong_reputation_penalty
    else:
        reputation_gain = (accuracy - 0.5) * config.reputation_scale

    # ---------- 长期价值 ----------
    long_term_value = (
        config.buyer_ltv * buyer_repurchase_prob
        + config.merchant_ltv * merchant_retention_prob
        - immediate_cost
        - risk_cost
        + reputation_gain
    )

    return CaseOutcome(
        buyer_satisfaction=round(buyer_satisfaction, 4),
        merchant_satisfaction=round(merchant_satisfaction, 4),
        buyer_repurchase_prob=round(buyer_repurchase_prob, 4),
        merchant_retention_prob=round(merchant_retention_prob, 4),
        immediate_cost=round(immediate_cost, 2),
        risk_cost=round(risk_cost, 2),
        reputation_gain=round(reputation_gain, 2),
        long_term_value=round(long_term_value, 2),
        notes=notes,
    )
