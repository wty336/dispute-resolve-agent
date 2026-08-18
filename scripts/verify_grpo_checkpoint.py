#!/usr/bin/env python3
"""独立验证可重新加载的 BF16 LoRA checkpoint。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_checkpoint(
    *,
    base_model: str,
    adapter_dir: Path,
    public_prompt_file: Path,
    evidence_out: Path,
) -> bool:
    evidence: dict[str, object] = {
        "base_model": base_model,
        "adapter_dir": str(adapter_dir.resolve()),
        "adapter_config_sha256": None,
        "weight_hashes": {},
        "device": None,
        "dtype": "bfloat16",
        "generated_token_count": 0,
        "status": "failed",
    }
    try:
        config_path = adapter_dir / "adapter_config.json"
        weights = sorted(adapter_dir.glob("*.safetensors"))
        if not config_path.is_file() or not weights:
            raise RuntimeError("adapter_config.json and safetensors weights are required")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise RuntimeError("adapter_config.json must be an object")
        if config.get("base_model_name_or_path") != base_model:
            raise RuntimeError("adapter base model does not match requested base model")
        if config.get("r") != 32 or config.get("lora_alpha") != 64 or config.get("lora_dropout", 0.0) != 0.0:
            raise RuntimeError("adapter is not the required BF16 LoRA r=32 alpha=64 dropout=0 contract")
        evidence["adapter_config_sha256"] = _sha256(config_path)
        evidence["weight_hashes"] = {path.name: _sha256(path) for path in weights}
        if not public_prompt_file.is_file():
            raise RuntimeError(f"public prompt file is missing: {public_prompt_file}")

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for checkpoint verification")
        device = torch.device("cuda:0")
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = PeftModel.from_pretrained(base, adapter_dir, torch_dtype=torch.bfloat16)
        prompt = public_prompt_file.read_text(encoding="utf-8")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=1, do_sample=False)
        evidence["device"] = str(device)
        evidence["generated_token_count"] = int(generated.shape[-1] - inputs["input_ids"].shape[-1])
        if evidence["generated_token_count"] < 1:
            raise RuntimeError("checkpoint generated no token")
        evidence["status"] = "passed"
        return True
    except Exception as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        return False
    finally:
        evidence_out.parent.mkdir(parents=True, exist_ok=True)
        evidence_out.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--public-prompt-file", required=True, type=Path)
    parser.add_argument("--evidence-out", required=True, type=Path)
    args = parser.parse_args()
    return 0 if verify_checkpoint(
        base_model=args.base_model,
        adapter_dir=args.adapter_dir,
        public_prompt_file=args.public_prompt_file,
        evidence_out=args.evidence_out,
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
