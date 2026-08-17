# 简历表述证据清单

> 简历中的每一条表述都必须能链接到仓库中的配置、日志、定向测试或真实训练产物。
> 未实际跑通的 Agent Lightning/VERL、checkpoint 和指标不得写成“已完成”。

## 当前可引用（本地已实现）

| 表述 | 证据位置 | 状态 |
| --- | --- | --- |
| 公开/隐藏领域协议与泄漏隔离 | `dispute_agent/domain/`, `tests/leakage/` | implemented_locally |
| 确定性工具仿真与多轮 Episode 状态机 | `dispute_agent/tools/`, `dispute_agent/environment/` | implemented_locally |
| 可审计训练奖励与离线业务效用 | `dispute_agent/rewards/` | implemented_locally |
| 泄漏安全合成数据管线 | `dispute_agent/data/`, `scripts/generate_data.py` | implemented_locally |
| OpenAI Agents SDK 多轮工具 Runtime | `dispute_agent/agent/` | implemented_locally |
| Agent Lightning rollout 核心与懒绑定 | `dispute_agent/training/lightning_agent.py` | implemented_locally |
| TRL Qwen3-8B BF16 LoRA SFT 入口 | `dispute_agent/training/sft_runtime.py`, `configs/sft.yaml` | implemented_locally |
| Agent Lightning 0.3.0 + verl 0.5.0 GRPO 入口 | `dispute_agent/training/grpo_runtime.py`, `configs/grpo.yaml` | implemented_locally |

## 必须等服务器真实产物后才能引用

| 表述 | 必需证据 | 当前状态 |
| --- | --- | --- |
| 双 4090 Phase 0 门禁全部通过 | `outputs/grpo/<phase0-run>/phase0_report.json`，所有 gate 为 `passed` | not_run |
| GRPO 发生真实 optimizer update | `run_manifest.json`、verl checkpoint、`metrics/summary.json` | not_run |
| GRPO checkpoint 可独立重载并生成 token | `verify_grpo_checkpoint.py` 的 evidence JSON | not_run |
| 正式训练样本数与实测指标 | `grpo_train=700`、`grpo_val=100`、真实 metrics | not_run |

在 Phase 0 报告、checkpoint 和指标存在前，只能表述“已实现并通过本地契约验证”，不能表述“完成 Agentic GRPO 训练”。
