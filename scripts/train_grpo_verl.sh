#!/usr/bin/env bash
set -e
# ============================================================
# GRPO：verl + Qwen2.5-7B-Instruct（LoRA）+ 双 4090
#
# 前置：
#   1) 已完成 SFT：outputs/sft-7b-lora（或任意 7B 基座）
#   2) 已生成 RL 数据：python scripts/generate_data.py --mode rl ...
#   3) 已转 parquet：python scripts/prepare_verl_data.py ...
#   4) pip install verl vllm ray pyarrow pandas
#
# 显存布局（双 4090 24GB）：
#   - GPU0：vLLM rollout（gpu_memory_utilization=0.55 约 13GB，留余量）
#   - GPU1：actor/ref 训练（LoRA，只优化低秩参数）
#   - 若 OOM，优先调小 max_prompt_length / max_response_length /
#     ppo_micro_batch_size_per_gpu，或开启 offload
# ============================================================

# 项目根目录（按实际路径修改）
cd "$(dirname "$0")/.." || exit 1

export DISPUTE_CASE_TABLE_PATH=data/rl_cases.jsonl
export PYTHONPATH="$PWD:$PYTHONPATH"

# 7B LoRA SFT 后的模型路径；未完成 SFT 可先改用基座模型
MODEL_PATH=outputs/sft-7b-lora

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/rl_train.parquet \
    data.val_files=data/rl_val.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=1536 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=true \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    algorithm.use_kl_in_reward=false \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=dispute-resolve \
    trainer.experiment_name=grpo-7b-lora \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.total_epochs=2 \
    reward_model.enable=false \
    custom_reward_function.path=dispute_agent/verl_reward.py \
    custom_reward_function.name=compute_score

# 说明：
# 1) 若你的 verl 版本支持 batch reward function，把上面最后两行改为：
#    custom_reward_function.name=compute_score_batch
# 2) 若 verl 版本不支持 custom_reward_function 而使用 reward_manager，
#    请参考 verl 文档配置，但奖励函数文件保持不变。
# 3) LoRA 开关：部分 verl 版本支持
#    actor_rollout_ref.model.lora_rank=32
#    actor_rollout_ref.model.lora_alpha=64
#    请按你安装的 verl 版本打开；若版本较旧，需在 SFT 后合并 LoRA 权重，
#    或改用全参数（2×4090 不够）的方案。
