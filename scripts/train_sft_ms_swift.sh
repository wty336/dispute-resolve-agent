#!/usr/bin/env bash
set -e
# ============================================================
# SFT：ms-swift + Qwen2.5-7B-Instruct + LoRA（两张 4090 友好）
#
# 先安装：pip install ms-swift
# 数据：data/sft.jsonl（messages 格式，本项目可直接生成）
#
# 显存说明：
#   - 7B 基座 bf16 约 14GB；LoRA 优化器/梯度很小
#   - 双卡 DDP：把 --nproc_per_node 2 加上即可，单卡也能跑
# ============================================================

MODEL=Qwen/Qwen2.5-7B-Instruct
DATA=data/sft.jsonl
OUTPUT_DIR=outputs/sft-7b-lora

swift sft \
    --model "$MODEL" \
    --dataset "$DATA" \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --max_length 2048 \
    --gradient_checkpointing true \
    --logging_steps 10 \
    --save_steps 200 \
    --output_dir "$OUTPUT_DIR"

# 双卡 DDP 运行方式（可选）：
#   NPROC_PER_NODE=2 swift sft \
#       --model "$MODEL" \
#       --dataset "$DATA" \
#       --train_type lora \
#       ... 同上 ...
