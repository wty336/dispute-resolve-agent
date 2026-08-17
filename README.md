# 基于 SFT 与 Agentic GRPO 的电商纠纷智能判责 Agent

本项目将一个单步电商纠纷仿真原型迁移为可复现的 **Qwen3-8B BF16 LoRA** 项目：

- 使用 **TRL** 完成小规模高质量 SFT；
- 使用 **OpenAI Agents SDK** 构建真实多轮工具 Agent；
- 通过 **Agent Lightning + VERL** 完成 Agentic GRPO；
- 在冻结的 ID/OOD 测试集上统一评测。

> 当前仓库包含完整代码、配置、测试与本地可运行的 smoke 流程。
> 真实 Qwen3-8B 训练和 Agent Lightning/VERL 训练结果需在双 4090 环境完成
> **Phase 0** 门禁后填写，仓库不包含虚构指标。

## 1. 问题定义

平台 Agent 在「买家 / 商家 / 平台」三方利益冲突下处理电商纠纷。Agent 只能看到公开工单与工具返回结果，不能读取隐藏的 Ground Truth。它需要：

- 自主决定是否调用物流、买家历史、商家历史、证据核验工具；
- 在证据充分时提交 `decide`（责任 + 赔付）；
- 在证据冲突、高金额、高风险或信息不足时提交 `escalate`（人工升级）。

## 2. 架构

```text
结构化 Ground Truth
    ├── 渲染为公开 Observation ──→ SFT / GRPO / Agent
    ├── 生成带噪声工具结果 ─────→ Episode Environment / Agent Tools
    └── 计算标签与奖励 ─────────→ Reward / Evaluation（模型不可见）

OpenAI Agents SDK Agent
    ├── 通过 main_llm ProxyLLM 调用 Qwen3
    ├── 与 Episode Environment 多轮交互
    └── 由 Agent Lightning 记录 Trace 与 Reward
                    ↓
             LightningStore
                    ↓
            VERL / GRPO 参数更新
                    ↓
             新 LoRA adapter checkpoint
```

## 3. 目录结构

```text
configs/                     # sft.yaml / grpo.yaml / evaluation.yaml
constraints/train.txt        # 训练依赖版本固定
dispute_agent/
  domain/                    # 公开/隐藏 Schema、策略常量
  data/                      # 合成数据生成、渲染、切分、校验
  tools/                     # 确定性工具仿真与注册表
  environment/               # Episode 状态机
  rewards/                   # 训练奖励 + 离线业务效用
  agent/                     # OpenAI Agents SDK Runtime + Tracing
  training/                  # SFT 数据/配置、Agent Lightning、GRPO、监控
  evaluation/                # 指标、Bootstrap、公平评测 Runner
  api/                       # 最小 FastAPI 演示
scripts/                     # generate_data / train_sft / train_agentic_grpo / evaluate / phase0_smoke
tests/                       # unit / integration / leakage / evaluation
```

## 4. 数据规模

| 数据集 | 数量 | 用途 |
| --- | ---: | --- |
| SFT 训练 | 1,500 | 领域判责与工具调用 |
| SFT 验证 | 150 | checkpoint 选择 |
| Agentic GRPO 训练 | 700 | 多轮策略优化 |
| GRPO 验证 | 100 | 奖励与策略坍缩监控 |
| ID 测试 | 400 | 同分布评测 |
| OOD 测试 | 200 | 鲁棒性评测（80/60/60） |
| 人工审核 | 100 | 从测试集中分层抽取 |

生成、渲染、切分和泄漏校验均有自动化测试。

## 5. SFT / GRPO 配置

- 模型：`Qwen/Qwen3-8B`
- 精度：BF16，无 4/8-bit 量化
- LoRA：`rank=32, alpha=64, dropout=0, target_modules=all-linear`
- SFT：non-thinking，只监督 assistant 的 tool call / 动作 / 最终决策
- GRPO：`adv_estimator=grpo`，rollout `n=4`，TP=2，双 4090，
  从 `sft-1500-best` 的只读副本继续

## 6. Phase 0 证据

`scripts/phase0_smoke.py` 定义七项门禁：

1. `sdk_vllm_multiturn`
2. `thinking_tool_compatibility`
3. `trace_complete`
4. `trl_adapter_loaded_by_verl`
5. `grpo_update_reload`
6. `dual_gpu_no_oom`
7. `single_model_span_and_reward`

本地 `--fixture` 模式可生成报告；真实门禁需在 Ubuntu 22.04 / Python 3.11 / 双 RTX 4090 上运行。

## 7. 统一评测命令

```bash
python scripts/generate_data.py --seed 20260817 --fixture-size 24 --output artifacts/data-smoke
python scripts/train_sft.py --config configs/sft.yaml --fixture --max-steps 1
python scripts/train_agentic_grpo.py --config configs/grpo.yaml --dry-run
python scripts/evaluate.py --config configs/evaluation.yaml --models all --output artifacts/evaluation
pytest -q
```

## 8. 真实结果表

> 待训练完成后填写。当前不预填任何提升数字。

| 模型 | 判责 Macro-F1 | 赔付 MAE | Episode 成功率 | 平均工具成本 |
| --- | --- | --- | --- | --- |
| RuleBased | - | - | - | - |
| Base | - | - | - | - |
| SFT-500 | - | - | - | - |
| SFT-1000 | - | - | - | - |
| SFT-1500 | - | - | - | - |
| SFT-1500 + GRPO | - | - | - | - |
| Oracle | - | - | - | - |

## 9. 局限

- 合成数据来自模板 + 仿真器，不等同于真实工单。
- 当前仓库尚未在双 4090 上完成 Phase 0 与正式训练。
- Agent Lightning/VERL 的版本兼容性以 Phase 0 实测为准。
- 未进行多随机种子完整训练，Bootstrap 只提供单种子下的 95% CI。

## 10. 双 4090 复现方式

```bash
# Ubuntu 22.04, Python 3.11, CUDA 12.x
uv pip install -e ".[dev,train]" -c constraints/train.txt
python scripts/phase0_smoke.py --cases 20 --report artifacts/phase0/report.json
pytest tests/integration/test_phase0_contract.py -q
```

## 11. 典型 Trace

FastAPI 演示接口会返回：

```json
{
  "trace_id": "...",
  "decision": {"action": "decide", "liability": "merchant", "compensation": 50.0},
  "trace": [
    {"event": "observation_received", "latency_ms": 0.1},
    {"event": "decision_made", "latency_ms": 0.2}
  ]
}
```

Trace 不包含 thinking、Ground Truth 或 token IDs。
