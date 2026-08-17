"""Oracle 教师策略。

Oracle 可以读取仿真 ground truth（真实责任、真实损失），因此它代表
“信息完全透明”的理论最优策略，主要用于：
1. 生成 SFT 训练标签（教小模型从可观察信息推断合理判责）；
2. 在评估中作为长期收益的近似上限。

Oracle 决策原则：按真实损失赔付，不按买家虚高诉求赔付；证据冲突或
责任无法认定时人工升级。
"""
from __future__ import annotations

from .models import AgentDecision, DisputeCase, Liability
from .platform_agent import BasePlatformAgent, _avg_evidence_strength


class OracleAgent(BasePlatformAgent):
    """基于 ground truth 的教师/上限策略。"""

    name = "Oracle上限"

    def decide(self, case: DisputeCase) -> AgentDecision:
        t = case.true_liability
        amount = case.order_amount
        request = case.buyer_requested_amount

        # ---- 责任判定：直接使用真实责任 ----
        liability = t

        # ---- 赔付：按真实损失，而不是买家诉求 ----
        if t == Liability.MERCHANT:
            comp = min(case.true_buyer_loss, amount)
        elif t == Liability.SPLIT:
            comp = min(case.true_buyer_loss, amount) * 0.5
        else:
            comp = 0.0

        # ---- 人工升级：事实无法认定、证据冲突、或诉求明显超过真实损失 ----
        buyer_ev = _avg_evidence_strength(case, "buyer")
        merchant_ev = _avg_evidence_strength(case, "merchant")
        evidence_conflict = abs(buyer_ev - merchant_ev) < 0.15
        exaggerated_request = request > max(case.true_buyer_loss * 1.5, amount * 1.0)
        escalate = (
            t == Liability.NONE
            or evidence_conflict
            or exaggerated_request
            or request > amount * 1.5
        )

        if t == Liability.NONE:
            reason = "事实无法认定，建议人工升级"
        elif evidence_conflict:
            reason = f"双方证据接近，虽有真实责任方({t.value})，但建议人工复核"
        else:
            reason = f"按真实损失核定赔付 {comp:.2f} 元（责任：{t.value}）"

        return AgentDecision(
            liability=liability,
            compensation=round(comp, 2),
            escalate=escalate,
            reason=reason,
        )
