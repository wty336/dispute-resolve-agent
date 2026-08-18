#!/usr/bin/env python3
"""运行冻结的统一评估协议。

用法：
    python scripts/evaluate.py --config configs/evaluation.yaml --models all --output artifacts/evaluation
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dispute_agent.evaluation.metrics import compute_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation.yaml")
    parser.add_argument("--models", default="all")
    parser.add_argument("--output", default="artifacts/evaluation")
    args = parser.parse_args()

    # 在真实 checkpoint 可用前使用占位预测。
    predictions = [
        {
            "true_liability": "merchant",
            "pred_liability": "merchant",
            "true_compensation": 100.0,
            "pred_compensation": 100.0,
            "escalation_true": False,
            "escalation_pred": False,
            "evidence_ids": ["e1"],
            "visible_evidence_ids": ["e1"],
            "tool_calls": 1,
            "tool_cost": 2.0,
            "necessary_tool": True,
            "used_necessary_tool": True,
            "episode_success": True,
        }
    ]
    report = compute_metrics(predictions)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(report.model_dump(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "summary.md").write_text("# Evaluation Summary\n\nPlaceholder until real checkpoints.\n", encoding="utf-8")
    print(f"Evaluation written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
