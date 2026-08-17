"""评估本地训练产物（SFT/RL 后的 checkpoint）。

在仿真环境上运行模型，输出与规则基线、Oracle 上限的长期收益对比。

用法：
    python scripts/evaluate_model.py --model_path outputs/rl --n_cases 500 --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dispute_agent.legacy_environment import DisputeEnvironment  # noqa: E402
from dispute_agent.evaluate import find_best_agent, print_report_table  # noqa: E402
from dispute_agent.oracle import OracleAgent  # noqa: E402
from dispute_agent.platform_agent import (  # noqa: E402
    LLMAgent,
    LocalModelAgent,
    ProConsumerAgent,
    ProMerchantAgent,
    RuleBasedAgent,
    ToolLoopAgent,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="评估本地训练模型")
    parser.add_argument("--model_path", default=None, help="本地 checkpoint 路径（transformers 加载）")
    parser.add_argument("--vllm_url", default=None, help="vLLM OpenAI 兼容接口地址，如 http://localhost:8000/v1")
    parser.add_argument("--vllm_model", default=None, help="vLLM 部署的模型名（默认不传）")
    parser.add_argument("--tool_loop", action=argparse.BooleanOptionalAction, default=True,
                        help="使用多步工具循环 Agent（默认开启）")
    parser.add_argument("--n_cases", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    if args.vllm_url:
        print(f"使用 vLLM 接口：{args.vllm_url}")
        if args.tool_loop:
            local_agent = ToolLoopAgent(
                model=args.vllm_model or "dispute-7b",
                api_key="EMPTY",  # vLLM 通常不需要真实 key
                base_url=args.vllm_url,
            )
        else:
            local_agent = LLMAgent(
                model=args.vllm_model or "dispute-7b",
                api_key="EMPTY",
                base_url=args.vllm_url,
            )
    else:
        if not args.model_path:
            parser.error("必须提供 --model_path 或 --vllm_url")
        print(f"加载本地模型：{args.model_path}")
        local_agent = LocalModelAgent(
            model_path=args.model_path,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )

    reports = []
    for name, agent in [
        ("本地模型", local_agent),
        ("规则基线", RuleBasedAgent()),
        ("偏买家", ProConsumerAgent()),
        ("偏商家", ProMerchantAgent()),
        ("Oracle上限", OracleAgent()),
    ]:
        env = DisputeEnvironment(seed=args.seed)
        report = env.run(agent, n_cases=args.n_cases)
        report.agent_name = name
        reports.append(report)
        print(f"{name}: 完成 {args.n_cases} 轮评估")

    print()
    print_report_table(reports)
    best = find_best_agent(reports)
    print(f"\n长期收益最高的策略：{best.agent_name}")


if __name__ == "__main__":
    main()
