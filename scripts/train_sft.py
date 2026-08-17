#!/usr/bin/env python3
"""Run TRL BF16 LoRA SFT for the dispute agent.

Usage:
    python scripts/train_sft.py --config configs/sft.yaml --train-size 500
    python scripts/train_sft.py --config configs/sft.yaml --fixture --max-steps 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dispute_agent.training.train_sft import load_sft_config, train_sft


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--train-size", type=int, choices=[500, 1000, 1500], default=500)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_sft_config(args.config)
    output = train_sft(
        config,
        train_size=args.train_size,
        fixture=args.fixture,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
    )
    print(f"SFT dry-run complete: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
