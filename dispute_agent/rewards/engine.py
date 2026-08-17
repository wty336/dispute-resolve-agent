"""Auditable rule-based reward engine for training.

The training reward never imports business utility and never reads an oracle
tool-information value.  It only uses hidden ground truth (liability,
reasonable compensation range, escalation risk) plus agent-visible state.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from dispute_agent.domain.policies import (
    HARD_FAILURE_REWARD,
    INVALID_ACTION_PENALTY,
    MAX_ACTION_PENALTY,
    MAX_COMPENSATION_RATIO,
    MAX_TOOL_COST,
    REPEAT_CALL_PENALTY,
    REWARD_MAX,
    REWARD_MIN,
)
from dispute_agent.domain.schemas import Decision, Escalation, Liability


class RewardComponents(BaseModel):
    liability: float = 0.0
    compensation: float = 0.0
    escalation: float = 0.0
    grounding: float = 0.0
    escalation_quality: float = 0.0
    normalized_tool_cost: float = 0.0
    invalid_action_penalty: float = 0.0
    hard_failure: bool = False


class RewardResult(BaseModel):
    components: RewardComponents
    total: float = Field(ge=-1.5, le=1.0)


class RewardEngine:
    """Compute the training reward for one finished episode."""

    def score(self, episode) -> RewardResult:
        decision = episode.terminal_decision
        if decision is None:
            return self._hard_failure()

        gt = episode.ground_truth
        order_amount = episode.observation.order_amount

        if decision.action == "decide" and decision.compensation > order_amount * MAX_COMPENSATION_RATIO:
            return self._hard_failure()

        components = RewardComponents(
            normalized_tool_cost=min(1.0, max(0.0, episode.cumulative_cost / MAX_TOOL_COST)),
            invalid_action_penalty=max(
                MAX_ACTION_PENALTY,
                INVALID_ACTION_PENALTY * episode.invalid_actions.illegal_calls
                + REPEAT_CALL_PENALTY * episode.invalid_actions.repeat_calls,
            ),
        )

        if decision.action == "decide":
            components.liability = self._liability_score(decision.liability, gt.true_liability)
            components.compensation = self._compensation_score(
                decision.compensation, gt.reasonable_compensation_range, order_amount
            )
            components.escalation = 1.0 if not gt.should_escalate else -1.0
            components.grounding = self._grounding_score(decision.evidence_ids, episode.visible_evidence)
            total = (
                0.45 * components.liability
                + 0.25 * components.compensation
                + 0.15 * components.escalation
                + 0.15 * components.grounding
                - 0.10 * components.normalized_tool_cost
                + components.invalid_action_penalty
            )
        else:
            components.escalation = 1.0 if gt.should_escalate else -1.0
            components.grounding = self._grounding_score(decision.evidence_ids, episode.visible_evidence)
            components.escalation_quality = self._escalation_quality_score(decision, gt, episode)
            total = (
                0.60 * components.escalation
                + 0.25 * components.grounding
                + 0.15 * components.escalation_quality
                - 0.10 * components.normalized_tool_cost
                + components.invalid_action_penalty
            )

        return RewardResult(
            components=components,
            total=max(REWARD_MIN, min(REWARD_MAX, total)),
        )

    def _hard_failure(self) -> RewardResult:
        return RewardResult(
            components=RewardComponents(hard_failure=True),
            total=HARD_FAILURE_REWARD,
        )

    def _liability_score(self, predicted: Liability, truth: Liability) -> float:
        if predicted == truth:
            return 1.0
        if {predicted, truth} == {Liability.MERCHANT, Liability.SPLIT}:
            return 0.4
        if {predicted, truth} == {Liability.BUYER, Liability.SPLIT}:
            return 0.4
        if {predicted, truth} == {Liability.MERCHANT, Liability.BUYER}:
            return -1.0
        return -0.3

    def _compensation_score(self, compensation: float, reasonable_range: tuple[float, float], order_amount: float) -> float:
        low, high = reasonable_range
        if low <= compensation <= high:
            return 1.0
        if compensation < low:
            return max(-1.0, 1.0 - (low - compensation) / max(order_amount, 1e-9))
        return max(-1.0, 1.0 - (compensation - high) / max(order_amount, 1e-9))

    def _grounding_score(self, evidence_ids: list[str], visible_evidence: dict) -> float:
        if not evidence_ids:
            return -0.5
        if any(evidence_id not in visible_evidence for evidence_id in evidence_ids):
            return -1.0
        return 1.0

    def _escalation_quality_score(self, decision: Escalation, gt, episode) -> float:
        # Structured, non-LLM risk-rule quality score.
        high_risk = getattr(gt, "risk_level", "medium") == "high"
        evidence_conflict = "冲突" in decision.reason or "证据" in decision.reason
        if high_risk or evidence_conflict:
            return 1.0
        if len(decision.reason) >= 10:
            return 0.0
        return -1.0
