"""仿真环境：生成案例 → Agent 决策 → 收益核算。"""
from __future__ import annotations

from dataclasses import dataclass, field

from .case_generator import CaseGenerator
from .models import AgentDecision, CaseOutcome, DisputeCase
from .payoff import PayoffConfig, compute_outcome
from .platform_agent import BasePlatformAgent


@dataclass
class StepRecord:
    """单轮仿真记录。"""

    case: DisputeCase
    decision: AgentDecision
    outcome: CaseOutcome


@dataclass
class RunReport:
    """多轮仿真汇总。"""

    agent_name: str
    n_cases: int
    total_long_term_value: float = 0.0
    avg_long_term_value: float = 0.0
    avg_compensation: float = 0.0
    avg_immediate_cost: float = 0.0
    avg_risk_cost: float = 0.0
    avg_reputation_gain: float = 0.0
    avg_buyer_satisfaction: float = 0.0
    avg_merchant_satisfaction: float = 0.0
    avg_buyer_repurchase_prob: float = 0.0
    avg_merchant_retention_prob: float = 0.0
    accuracy: float = 0.0
    escalate_rate: float = 0.0
    records: list[StepRecord] = field(default_factory=list)


class DisputeEnvironment:
    """纠纷判责仿真环境。"""

    def __init__(
        self,
        seed: int | None = None,
        payoff_config: PayoffConfig | None = None,
        generator: CaseGenerator | None = None,
    ) -> None:
        self.generator = generator or CaseGenerator(seed=seed)
        self.payoff_config = payoff_config or PayoffConfig()

    def step(self, agent: BasePlatformAgent, case: DisputeCase | None = None) -> StepRecord:
        """执行一轮：给定 case（或自动生成），Agent 决策并核算收益。"""
        if case is None:
            case = self.generator.generate()
        decision = agent.decide(case)
        outcome = compute_outcome(case, decision, self.payoff_config)
        return StepRecord(case=case, decision=decision, outcome=outcome)

    def run(self, agent: BasePlatformAgent, n_cases: int = 1000) -> RunReport:
        """对同一 Agent 连续仿真 n_cases 轮，返回汇总报告。"""
        report = RunReport(agent_name=agent.name, n_cases=n_cases)
        total_ltv = 0.0
        total_accuracy = 0.0
        total_escalate = 0

        for _ in range(n_cases):
            record = self.step(agent)
            report.records.append(record)
            o = record.outcome
            total_ltv += o.long_term_value
            total_accuracy += 1.0 if record.decision.liability == record.case.true_liability else 0.0
            total_escalate += int(record.decision.escalate)

            report.avg_compensation += record.decision.compensation
            report.avg_immediate_cost += o.immediate_cost
            report.avg_risk_cost += o.risk_cost
            report.avg_reputation_gain += o.reputation_gain
            report.avg_buyer_satisfaction += o.buyer_satisfaction
            report.avg_merchant_satisfaction += o.merchant_satisfaction
            report.avg_buyer_repurchase_prob += o.buyer_repurchase_prob
            report.avg_merchant_retention_prob += o.merchant_retention_prob

        report.total_long_term_value = round(total_ltv, 2)
        report.avg_long_term_value = round(total_ltv / n_cases, 2)
        report.avg_compensation = round(report.avg_compensation / n_cases, 2)
        report.avg_immediate_cost = round(report.avg_immediate_cost / n_cases, 2)
        report.avg_risk_cost = round(report.avg_risk_cost / n_cases, 2)
        report.avg_reputation_gain = round(report.avg_reputation_gain / n_cases, 2)
        report.avg_buyer_satisfaction = round(report.avg_buyer_satisfaction / n_cases, 4)
        report.avg_merchant_satisfaction = round(report.avg_merchant_satisfaction / n_cases, 4)
        report.avg_buyer_repurchase_prob = round(report.avg_buyer_repurchase_prob / n_cases, 4)
        report.avg_merchant_retention_prob = round(report.avg_merchant_retention_prob / n_cases, 4)
        report.accuracy = round(total_accuracy / n_cases, 4)
        report.escalate_rate = round(total_escalate / n_cases, 4)
        return report
