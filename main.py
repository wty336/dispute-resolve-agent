"""入口脚本。

用法：
    python main.py             # 单案 demo + 多策略对比评估
    python main.py --llm       # 额外加入 LLM Agent（需配置 OPENAI_API_KEY）
    python main.py --cases 200 # 评估案例数
"""
from __future__ import annotations

import argparse
import sys

from dispute_agent.legacy_environment import DisputeEnvironment
from dispute_agent.evaluate import evaluate_agents, find_best_agent, print_report_table
from dispute_agent.oracle import OracleAgent
from dispute_agent.platform_agent import (
    LLMAgent,
    ProConsumerAgent,
    ProMerchantAgent,
    RuleBasedAgent,
)


def print_case_demo(case, decision, outcome) -> None:
    """打印单个案例的演示结果。"""
    print("=" * 70)
    print(f"案例 {case.case_id} | 订单 {case.order_id} | {case.item_name} | "
          f"订单金额 ¥{case.order_amount:.2f}")
    print(f"投诉类型：{case.claim_type.value}")
    print(f"买家投诉：{case.buyer_claim}")
    print(f"诉求金额：¥{case.buyer_requested_amount:.2f}")
    print(f"商家回应：{case.merchant_response}")
    print(f"聊天记录：")
    for msg in case.chat_log:
        print(f"  [{msg.role}] {msg.content}")
    print(f"买家举证：{len(case.buyer_evidence)} 项；商家举证：{len(case.merchant_evidence)} 项")
    print("-" * 70)
    print(f"Agent 决策 -> 责任判定：{decision.liability.value} | "
          f"赔付：¥{decision.compensation:.2f} | 人工升级：{'是' if decision.escalate else '否'}")
    print(f"判定依据：{decision.reason}")
    print(f"收益核算 -> 买家满意：{outcome.buyer_satisfaction:.2f} | "
          f"商家满意：{outcome.merchant_satisfaction:.2f}")
    print(f"            买家复购：{outcome.buyer_repurchase_prob:.2f} | "
          f"商家留存：{outcome.merchant_retention_prob:.2f}")
    print(f"            即时成本：¥{outcome.immediate_cost:.2f} | "
          f"风险成本：¥{outcome.risk_cost:.2f} | 口碑：{outcome.reputation_gain:+.2f}")
    print(f"            长期收益：¥{outcome.long_term_value:.2f}")
    if outcome.notes:
        print(f"            备注：{'；'.join(outcome.notes)}")
    print()


def demo() -> None:
    """生成 3 个案例，展示规则基线 Agent 的决策与收益核算。"""
    print("\n" + "=" * 70)
    print("一、单案 Demo（规则基线 Agent）")
    print("=" * 70)
    env = DisputeEnvironment(seed=42)
    agent = RuleBasedAgent()
    for _ in range(3):
        record = env.step(agent)
        print_case_demo(record.case, record.decision, record.outcome)


def evaluate(llm: bool = False, n_cases: int = 1000) -> None:
    """对比不同平台策略的长期收益。"""
    print("\n" + "=" * 70)
    print(f"二、多策略长期收益对比（{n_cases} 个仿真案例，种子 42）")
    print("=" * 70)

    agents = {
        "规则基线": RuleBasedAgent(),
        "偏买家": ProConsumerAgent(),
        "偏商家": ProMerchantAgent(),
        "Oracle上限": OracleAgent(),
    }
    if llm:
        try:
            agents["LLM"] = LLMAgent()
        except Exception as exc:  # noqa: BLE001
            print(f"跳过 LLM Agent：{exc}")

    reports = evaluate_agents(agents, n_cases=n_cases, seed=42)
    print_report_table(reports)
    best = find_best_agent(reports)
    print(f"\n长期收益最高的策略：{best.agent_name} "
          f"（总收益 ¥{best.total_long_term_value:.2f}）")


def main() -> None:
    parser = argparse.ArgumentParser(description="电商纠纷判责 Agent 博弈仿真")
    parser.add_argument("--llm", action="store_true", help="加入 LLM Agent 对比")
    parser.add_argument("--cases", type=int, default=1000, help="评估案例数")
    args = parser.parse_args()

    demo()
    evaluate(llm=args.llm, n_cases=args.cases)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
