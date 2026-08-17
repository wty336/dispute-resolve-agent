# 电商纠纷判责 Agent —— 多方博弈与长期收益仿真

## 1. 项目定位

这是一个**业务博弈决策 Agent** 项目。平台 Agent 在「买家 / 商家 / 平台」三方
利益冲突下做出判责决策，优化目标不是单轮正确率，而是**长期收益最大化**。

| 博弈方 | 利益诉求 | 可能策略 |
| --- | --- | --- |
| 买家 | 获得赔付、被公平对待 | 诚实 / 夸大损失 / 虚假投诉 |
| 商家 | 不承担不应承担的赔付 | 诚实 / 推卸责任 / 虚假否认 |
| 平台 Agent | 公平、成本、双边满意度、长期留存 | 规则基线 / 偏买家 / 偏商家 / LLM 决策 |

## 2. 场景定义

**输入**
- 订单信息（商品、金额、订单号、买家/商家 ID）
- 用户投诉（投诉类型、诉求金额、描述）
- 商家举证（描述、凭证）
- 聊天记录

**输出**
- 责任判定（商家责任 / 买家责任 / 双方共担 / 无法认定）
- 赔付方案（金额）
- 是否需要人工升级

**博弈点**
- 买家可能夸大损失甚至虚假投诉
- 商家可能推卸责任甚至虚假否认
- 平台需要平衡公平、成本、用户满意度、商家配合度

## 3. 多步 Agent 设计（工具调用）

判责不是一步拍板，而是“先查证、后判责”的多步流程：

```
输入工单
  → Step1 调用 check_order_logistics   查物流签收（成本 ¥2，有噪声）
  → Step2 调用 check_merchant_history  查商家历史纠纷率（成本 ¥3）
  → Step3 调用 check_buyer_history     查买家历史投诉率（成本 ¥3）
  → Step4 调用 verify_evidence         核验证据（成本 ¥8）
  → Step5 输出 final 判责 JSON
```

- 工具由 `dispute_agent/tools.py` 统一仿真，结果带噪声但可复现（同一案例同一工具结果一致）。
- 每次工具调用都有成本，推理侧计入 `AgentDecision.tool_cost`，RL 奖励中从长期收益扣除。
- 训练/推理共用同一套工具定义与结果格式，避免 train-inference 不一致。

**训练职责分工**：
- **SFT**：教模型“何时调用哪些工具、拿到结果后如何判责”（多轮 messages 轨迹）。
- **RL**：prompt 中预嵌入工具查询结果，只训练“给定查证信息后的最终判责”。这样 verl 无需做任意环境交互，最稳定。

## 4. 长期收益建模

平台 Agent 的长期收益：

```
U = Σ γ^t * (
      buyer_LTV   * buyer_repurchase_prob
    + merchant_LTV * merchant_retention_prob
    - compensation
    - manual_cost
    - risk_cost
    + reputation_gain
)
```

- `buyer_LTV` / `merchant_LTV`：双边用户生命周期价值
- `risk_cost`：该赔不赔 → 投诉升级/监管风险；不该赔乱赔 → 道德风险
- `reputation_gain`：判责与事实一致带来平台信任积累

## 5. 项目结构

```
dispute‑resolve‑agent/
├── README.md
├── requirements.txt
├── main.py                       # 入口：单案 demo + 多策略对比评估
├── dispute_agent/
│   ├── __init__.py
│   ├── models.py                 # 数据模型
│   ├── case_generator.py         # 纠纷案例生成（含 ground truth）
│   ├── prompting.py              # 统一 prompt 模板与输出解析（训练/推理一致）
│   ├── tools.py                  # 仿真工具层（物流/商家/买家/证据，带噪声带成本）
│   ├── payoff.py                 # 长期收益核算
│   ├── platform_agent.py         # 平台判责 Agent（规则基线 / 多步工具循环 / LLM 接口）
│   ├── oracle.py                 # Oracle 教师策略（基于 ground truth 的理论上限）
│   ├── reward.py                 # RL 奖励函数（长期收益 + 判责匹配度）
│   ├── verl_reward.py            # verl 自定义奖励函数入口
│   ├── environment.py            # 仿真环境
│   ├── evaluate.py               # 多轮评估与策略对比
│   └── data_generation.py        # SFT / RL 训练数据生成与序列化
├── scripts/
│   ├── generate_data.py          # 生成训练数据 CLI（SFT messages / RL prompt）
│   ├── train_sft_ms_swift.sh     # SFT：ms-swift + 7B LoRA
│   ├── prepare_verl_data.py      # RL 数据转 verl parquet
│   ├── train_grpo_verl.sh        # GRPO：verl + 7B LoRA（双 4090）
│   ├── train_sft.py              # 已迁移（占位说明）
│   ├── train_rl.py               # 已迁移（占位说明）
│   └── evaluate_model.py         # 评估本地模型或 vLLM OpenAI 接口
└── tests/
    └── test_smoke.py
```

## 6. 快速开始（仿真评估）

```bash
cd "E:/研究生/study_project/dispute‑resolve‑agent"
python main.py
```

运行后输出：
1. 3 个示例纠纷案例及不同 Agent 的判责结果
2. 1000 个仿真案例下，不同平台策略的长期收益对比（含 Oracle 理论上限）

## 7. 接入真实 LLM / vLLM

`platform_agent.py` 中的 `LLMAgent` 已实现 OpenAI 兼容接口：

```bash
pip install openai
export OPENAI_API_KEY=sk-xxx    # Windows PowerShell 用 $env:OPENAI_API_KEY="sk-xxx"
python main.py --llm
```

也可以将 `LLMAgent` 替换为任何支持 JSON 输出的本地模型服务。

## 8. SFT + 强化学习训练 7B 模型

目标：训练 Qwen2.5-7B-Instruct，使其在“多步查证 + 判责”框架下做出
长期收益最大化的决策。

### 8.1 整体流程

```
                  ┌────────────────────────────────────────────┐
                  │     仿真环境（无限数据源）                    │
                  │  CaseGenerator: 买家/商家博弈策略 + ground truth │
                  └───────────────┬────────────────────────────┘
                                  │
                  ┌───────────────▼───────────────┐
                  │ Oracle 教师策略（读 ground truth）│
                  │ + 教师工具选择 select_teacher_tools │
                  └───────────────┬───────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │ SFT                     │ RL                      │ 推理/评估
        ▼                         ▼                         ▼
  data/sft.jsonl           data/rl.jsonl + rl_cases      ToolLoopAgent
  多轮 messages：            prompt 预嵌入工具结果           + vLLM
  tool_call → user(结果) → final  (case_id + tool_cost)          多步工具循环
        │                         │                         │
        ▼                         ▼                         ▼
  train_sft_ms_swift.sh     prepare_verl_data.py      evaluate_model.py
  (ms-swift LoRA)           train_grpo_verl.sh        (长期收益对比)
        │                         │
        └────────────┬────────────┘
                     ▼
            outputs/rl（最终 7B 模型）
```

### 8.2 生成训练数据

```bash
# 多步工具调用 SFT 数据（默认，messages 含 tool_call/tool/final 轨迹）
python scripts/generate_data.py --mode sft --n 5000 --seed 42 --output data/sft.jsonl

# 单步判责 SFT（调试/对照）
python scripts/generate_data.py --mode sft --n 2000 --seed 42 --style single --output data/sft_single.jsonl

# RL 数据：prompt 预嵌入工具查询结果，模型只输出 final 判责
python scripts/generate_data.py --mode rl --n 8000 --seed 42 --output_dir data

# RL 数据转 verl parquet
python scripts/prepare_verl_data.py --input data/rl.jsonl --output_dir data
```

数据格式：
- SFT（多步）：`{"messages": [system, user, assistant(tool_call), user(工具结果), ..., assistant(final)]}`
- RL：`{"prompt": "案例编号：C000001\n工单+工具查询结果+输出要求", "case_id": "C000001", "tool_cost": 16.0}`
- RL case 表：`data/rl_cases.jsonl`（含 ground truth，供奖励函数查表）

### 8.3 SFT 微调（ms-swift + 7B LoRA）

```bash
# 先安装：pip install ms-swift
bash scripts/train_sft_ms_swift.sh
```

脚本内容等价于：
```bash
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset data/sft.jsonl \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --lora_rank 32 --lora_alpha 64 \
    --max_length 2048 \
    --gradient_checkpointing true \
    --output_dir outputs/sft-7b-lora
```

双 4090：LoRA SFT 显存充裕，可加 `NPROC_PER_NODE=2 swift sft ...` 双卡 DDP。

### 8.4 GRPO 强化学习（verl + 7B LoRA）

奖励函数在 `dispute_agent/reward.py` 与 `dispute_agent/verl_reward.py`：

```
reward = 判责匹配度(0~1) + (长期收益 - 工具成本) / 1000
```

其中长期收益来自 `payoff.compute_outcome`，工具成本从 RL 数据
`extra_info` 中读入。解析失败给 -1.0。

```bash
bash scripts/train_grpo_verl.sh
```

脚本会调用 `python3 -m verl.trainer.main_ppo`，双卡配置：
- GPU0：vLLM rollout（`gpu_memory_utilization=0.55`）
- GPU1：actor/ref 训练（LoRA）

**注意**：verl 不同版本的 LoRA 开关与 reward 函数注册方式略有差异，
`train_grpo_verl.sh` 末尾有对应注释，请按你的 verl 版本微调。

### 8.5 推理与评估

**方式 A：vLLM + 多步工具循环（推荐，与训练格式一致）**

```bash
# 1) 启动 vLLM（OpenAI 兼容接口）
vllm serve outputs/rl --served-model-name dispute-7b --port 8000

# 2) 在仿真环境评估多步 Agent
python scripts/evaluate_model.py \
    --vllm_url http://localhost:8000/v1 \
    --vllm_model dispute-7b \
    --n_cases 500 --seed 42
```

**方式 B：本地 transformers 加载（调试用，不推荐 7B 实时推理）**

```bash
python scripts/evaluate_model.py --model_path outputs/rl --n_cases 100
```

**方式 C：单步 LLM Agent（不调用工具，对照用）**

```bash
python scripts/evaluate_model.py \
    --vllm_url http://localhost:8000/v1 \
    --vllm_model dispute-7b \
    --no-tool_loop
```

## 9. 扩展方向

- 用真实历史工单拟合收益模型参数
- 把 `payoff.PayoffConfig` 改为可学习/可调参，做多目标权衡分析
- 加入商家申诉、平台复议等多轮交互
- 引入买家/商家信誉分与重复博弈记忆
- RL 阶段可迁移到 verl / Ray + vLLM 多卡 GRPO
