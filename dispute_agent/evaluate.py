"""多策略长期收益对比评估。"""
from __future__ import annotations

from .legacy_environment import DisputeEnvironment, RunReport
from .platform_agent import BasePlatformAgent


def evaluate_agents(
    agents: dict[str, BasePlatformAgent],
    n_cases: int = 1000,
    seed: int = 42,
) -> list[RunReport]:
    """在相同案例序列上评估多个平台策略。

    为保证公平，每个 Agent 使用独立但同种子的案例生成器。
    """
    reports: list[RunReport] = []
    for name, agent in agents.items():
        env = DisputeEnvironment(seed=seed)
        report = env.run(agent, n_cases=n_cases)
        report.agent_name = name
        reports.append(report)
    return reports


def print_report_table(reports: list[RunReport]) -> None:
    """打印对比表格（不依赖第三方库）。"""
    header = (
        f"{'策略':<10} {'长期收益':>12} {'单均收益':>12} {'判责准确率':>10} "
        f"{'平均赔付':>10} {'风险成本':>10} {'买家留存':>10} {'商家留存':>10} {'人工升级率':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in reports:
        print(
            f"{r.agent_name:<10} "
            f"{r.total_long_term_value:>12.2f} "
            f"{r.avg_long_term_value:>12.2f} "
            f"{r.accuracy:>10.2%} "
            f"{r.avg_compensation:>10.2f} "
            f"{r.avg_risk_cost:>10.2f} "
            f"{r.avg_buyer_repurchase_prob:>10.4f} "
            f"{r.avg_merchant_retention_prob:>10.4f} "
            f"{r.escalate_rate:>10.2%}"
        )


def find_best_agent(reports: list[RunReport]) -> RunReport:
    """返回长期收益最高的策略报告。"""
    return max(reports, key=lambda r: r.total_long_term_value)
