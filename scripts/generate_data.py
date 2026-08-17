"""生成 SFT / RL 训练数据。

用法：
    # 多步工具调用 SFT（默认）
    python scripts/generate_data.py --mode sft --n 5000 --seed 42 --output data/sft.jsonl
    # 单步判责 SFT（调试/对照）
    python scripts/generate_data.py --mode sft --n 2000 --style single --output data/sft_single.jsonl
    # RL：prompt 中预嵌入工具查询结果，模型只训练最终判责
    python scripts/generate_data.py --mode rl --n 8000 --seed 42 --output_dir data
    # 两者
    python scripts/generate_data.py --mode both --n 8000 --seed 42 --output_dir data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许从项目根目录直接运行脚本
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dispute_agent.data_generation import generate_rl_data, generate_sft_data  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="生成电商纠纷判责训练数据")
    parser.add_argument("--mode", choices=["sft", "rl", "both"], default="both")
    parser.add_argument("--n", type=int, default=2000, help="案例数量")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/sft.jsonl", help="SFT 输出文件")
    parser.add_argument("--output_dir", default="data", help="RL 输出目录")
    parser.add_argument("--style", choices=["multi_tool", "single"], default="multi_tool",
                        help="SFT 数据风格：multi_tool=多步工具调用，single=单步判责")
    parser.add_argument("--no_multi_tool_rl", action="store_true",
                        help="RL 数据使用无工具原始工单（调试用）")
    args = parser.parse_args()

    if args.mode in ("sft", "both"):
        cases = generate_sft_data(
            args.n, seed=args.seed, output_path=args.output, style=args.style
        )
        print(f"已生成 {len(cases)} 条 SFT 数据({args.style}) -> {args.output}")

    if args.mode in ("rl", "both"):
        cases = generate_rl_data(
            args.n,
            seed=args.seed,
            output_dir=args.output_dir,
            multi_tool=not args.no_multi_tool_rl,
        )
        print(f"已生成 {len(cases)} 条 RL 数据 -> {args.output_dir}/rl.jsonl")
        print(f"case 表 -> {args.output_dir}/rl_cases.jsonl")


if __name__ == "__main__":
    main()
