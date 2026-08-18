#!/usr/bin/env python3
"""校验或运行 Qwen3-8B BF16 LoRA SFT。"""
from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dispute_agent.training.sft_dataset import load_sft_dataset
from dispute_agent.training.sft_runtime import RealSFTBackend
from dispute_agent.training.train_sft import load_sft_config, run_sft_training


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _environment() -> dict:
    packages = ("torch", "transformers", "trl", "peft", "accelerate")
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    gpu_names: list[str] = []
    gpu_memory_bytes: list[int] = []
    cuda_version = None
    try:
        import torch

        cuda_version = torch.version.cuda
        gpu_names = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
        gpu_memory_bytes = [
            torch.cuda.get_device_properties(index).total_memory
            for index in range(torch.cuda.device_count())
        ]
    except (ImportError, RuntimeError):
        pass
    return {
        "python": sys.version.split()[0],
        "packages": versions,
        "cuda": cuda_version,
        "gpu_names": gpu_names,
        "gpu_memory_bytes": gpu_memory_bytes,
        "argv": sys.argv,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--train-size", type=int, choices=[500, 1000, 1500], default=500)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fixture", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    config = load_sft_config(args.config)
    data_dir = args.data_dir or config.data_dir
    bundle = load_sft_dataset(data_dir, None if args.fixture else args.train_size)
    if args.fixture:
        print(
            f"SFT fixture passed: train={len(bundle.train_rows)} "
            f"val={len(bundle.val_rows)}; no model loaded"
        )
        return 0

    backend = RealSFTBackend()
    if args.preflight:
        report = backend.preflight(config, bundle)
        print(
            f"SFT preflight passed: rows={report.checked_rows} "
            f"max_tokens={report.max_observed_length} "
            f"supervised_tokens={report.supervised_tokens}"
        )
        return 0

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    output = Path(args.output_dir or Path(config.output_root) / f"sft-{args.train_size}")
    best_name = f"{output.name}-best" if args.output_dir else f"sft-{args.train_size}-best"
    best = output.parent / best_name
    result = run_sft_training(
        config,
        bundle,
        train_size=args.train_size,
        output_dir=output,
        best_dir=best,
        backend=backend,
        world_size=world_size,
        rank=rank,
        max_steps=args.max_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
        git_commit=_git_commit(),
        environment=_environment(),
    )
    if rank == 0:
        print(f"SFT training complete: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
