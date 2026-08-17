#!/usr/bin/env python3
"""Run monitored Agent Lightning + VERL Agentic GRPO.

Usage:
    python scripts/train_agentic_grpo.py --config configs/grpo.yaml --dry-run
    python scripts/train_agentic_grpo.py --config configs/grpo.yaml --ablation no-tool-cost
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dispute_agent.training.grpo_config import load_grpo_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grpo.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ablation", choices=["full", "no-tool-cost"], default="full")
    args = parser.parse_args()

    cfg = load_grpo_config(args.config)
    if args.dry_run:
        print("GRPO dry-run plan")
        print(f"  adv_estimator={cfg.algorithm.adv_estimator}")
        print(f"  lora_adapter_path={cfg.actor_rollout_ref.model.lora_adapter_path}")
        print(f"  lora_rank={cfg.actor_rollout_ref.model.lora_rank} alpha={cfg.actor_rollout_ref.model.lora_alpha}")
        print(f"  rollout_n={cfg.actor_rollout_ref.rollout.n} tp={cfg.actor_rollout_ref.rollout.tensor_model_parallel_size}")
        print(f"  n_gpus_per_node={cfg.trainer.n_gpus_per_node}")
        print(f"  curriculum phase1_max_rounds={cfg.curriculum.phase1_max_rounds} phase2_max_rounds={cfg.curriculum.phase2_max_rounds}")
        print(f"  ablation={args.ablation}")
        if args.ablation == "no-tool-cost":
            print("  NOTE: tool-cost coefficient will be zeroed; all other config remains identical.")
        return 0

    # Actual training launch is implemented on the training machine after Phase 0.
    print("GRPO training launch is reserved for the dual-4090 environment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
