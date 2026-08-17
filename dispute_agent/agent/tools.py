"""OpenAI Agents SDK function tools for the dispute runtime."""
from __future__ import annotations

from agents import function_tool

from dispute_agent.domain.policies import MAX_COMPENSATION_RATIO
from dispute_agent.domain.schemas import Decision, Escalation


def build_agent_tools(episode, allowed_tools: set[str] | None = None) -> list:
    """Build episode-bound investigation tools and always expose submission."""
    builders = {
        "check_logistics": _check_logistics_tool,
        "check_buyer_history": _check_buyer_history_tool,
        "check_merchant_history": _check_merchant_history_tool,
        "verify_evidence": _verify_evidence_tool,
    }
    if allowed_tools is None:
        selected = set(builders)
    else:
        unknown = set(allowed_tools) - set(builders)
        if unknown:
            raise ValueError(f"unknown investigation tool(s): {sorted(unknown)}")
        selected = set(allowed_tools)
    tools = [builders[name](episode) for name in builders if name in selected]
    tools.append(_submit_decision_tool(episode))
    return tools


def _check_logistics_tool(episode):
    @function_tool
    def check_logistics(order_id: str) -> str:
        """查询订单物流签收状态。"""
        result = episode.call("check_logistics", {"order_id": order_id})
        return result.payload

    return check_logistics


def _check_buyer_history_tool(episode):
    @function_tool
    def check_buyer_history(buyer_id: str) -> str:
        """查询买家历史投诉记录。"""
        result = episode.call("check_buyer_history", {"buyer_id": buyer_id})
        return result.payload

    return check_buyer_history


def _check_merchant_history_tool(episode):
    @function_tool
    def check_merchant_history(merchant_id: str) -> str:
        """查询商家历史纠纷记录。"""
        result = episode.call("check_merchant_history", {"merchant_id": merchant_id})
        return result.payload

    return check_merchant_history


def _verify_evidence_tool(episode):
    @function_tool
    def verify_evidence(evidence_id: str) -> str:
        """核验一条证据的真实性。"""
        result = episode.call("verify_evidence", {"evidence_id": evidence_id})
        return result.payload

    return verify_evidence


def _submit_decision_tool(episode):
    @function_tool
    def submit_decision(
        action: str,
        confidence: float,
        evidence_ids: list[str],
        reason: str,
        liability: str | None = None,
        compensation: float | None = None,
    ) -> str:
        """提交最终判责或人工升级决策。action 为 decide 或 escalate。"""
        if action == "decide":
            if liability is None or compensation is None:
                raise ValueError("decide 必须提供 liability 和 compensation")
            decision = Decision(
                action="decide",
                liability=liability,
                compensation=compensation,
                confidence=confidence,
                evidence_ids=evidence_ids,
                reason=reason,
            )
        elif action == "escalate":
            decision = Escalation(
                action="escalate",
                confidence=confidence,
                evidence_ids=evidence_ids,
                reason=reason,
            )
        else:
            raise ValueError("action 必须为 decide 或 escalate")

        if any(evidence_id not in episode.visible_evidence for evidence_id in evidence_ids):
            raise ValueError("引用了不可见证据")
        if decision.action == "decide" and decision.compensation > episode.observation.order_amount * MAX_COMPENSATION_RATIO:
            raise ValueError("赔付金额超过上限")

        episode.submit(decision)
        return f"{decision.action}:{getattr(decision, 'liability', 'none')}"

    return submit_decision
