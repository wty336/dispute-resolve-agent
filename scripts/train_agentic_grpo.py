#!/usr/bin/env python3
"""Prepare or execute the Agent Lightning 0.3 + verl 0.5 GRPO run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dispute_agent.training.grpo_runtime import GRPORunRequest, run_grpo_training


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grpo.yaml")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-root", default="outputs/grpo")
    parser.add_argument("--run-id")
    parser.add_argument("--profile", choices=["smoke", "formal"], default="smoke")
    parser.add_argument("--curriculum-phase", type=int, choices=[1, 2], default=1)
    parser.add_argument("--input-adapter")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not args.dry_run and not args.run_id:
        parser.error("--run-id is required outside --dry-run")
    request = GRPORunRequest(
        config_path=Path(args.config),
        data_dir=Path(args.data_dir),
        output_root=Path(args.output_root),
        run_id=args.run_id or "dry-run",
        profile=args.profile,
        curriculum_phase=args.curriculum_phase,
        input_adapter=Path(args.input_adapter) if args.input_adapter else None,
        max_steps=args.max_steps,
        resume=args.resume,
    )
    result = run_grpo_training(request, dry_run=args.dry_run)
    print(json.dumps({
        "status": "planned" if result.dry_run else "completed",
        "run_dir": str(result.run_dir),
        "manifest": str(result.manifest_path),
        "dry_run": result.dry_run,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
