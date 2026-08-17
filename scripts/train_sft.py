"""SFT 训练已迁移到 ms-swift。

请使用：scripts/train_sft_ms_swift.sh
该脚本调用 `swift sft`，对 Qwen2.5-7B-Instruct 做 LoRA SFT。
"""
from __future__ import annotations


def main() -> None:
    print(__doc__)


if __name__ == "__main__":
    main()
