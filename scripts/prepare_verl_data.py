"""把 RL 数据转成 verl 可用的 parquet 格式。

输入：data/rl.jsonl（generate_data.py 生成，含 prompt 与 case_id）
输出：
    data/rl_train.parquet
    data/rl_val.parquet

verl 需要的列：
    prompt        : 模型输入 prompt（文本；verl 训练时用 tokenizer 处理）
    data_source   : 数据集标识，自定义奖励函数按此过滤
    ability       : 能力标签（可留空）
    reward_model  : 奖励模型标识（本方案不使用 RM，留空）
    extra_info    : JSON 字符串，内含 case_id，自定义奖励函数据此查 case 表

用法：
    python scripts/prepare_verl_data.py --input data/rl.jsonl --output_dir data --val_ratio 0.05

注意：如果你的 verl 版本要求 tokenized 数据（input_ids/attention_mask 列），
请在本脚本基础上，加载模型 tokenizer 后对 prompt 列做 encode 再存 parquet，
或使用 verl 自带的数据预处理脚本。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 verl 训练/验证 parquet")
    parser.add_argument("--input", default="data/rl.jsonl")
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError as exc:
        print("缺少 pandas，请先安装：pip install pandas pyarrow")
        sys.exit(1)

    rows = []
    with Path(args.input).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            rows.append({
                "prompt": item["prompt"],
                "data_source": "dispute_resolve",
                "ability": "dispute_judgement",
                "reward_model": "",
                "extra_info": json.dumps(
                    {
                        "case_id": item["case_id"],
                        "tool_cost": item.get("tool_cost", 0.0),
                    },
                    ensure_ascii=False,
                ),
            })

    df = pd.DataFrame(rows)
    # 打乱并切分
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    n_val = max(1, int(len(df) * args.val_ratio))
    val_df = df.iloc[:n_val]
    train_df = df.iloc[n_val:]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "rl_train.parquet"
    val_path = out_dir / "rl_val.parquet"
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"总样本：{len(df)}")
    print(f"训练集：{len(train_df)} -> {train_path}")
    print(f"验证集：{len(val_df)} -> {val_path}")


if __name__ == "__main__":
    main()
