# 简历表述证据清单

> 原则：每一条简历表述都能链接到仓库中的 config、日志、测试或评测产物。
> 未实际跑通的 Agent Lightning/VERL 或指标不得出现在简历中。

## 可立即引用的本地证据

| 表述 | 证据位置 |
| --- | --- |
| 实现公开/隐藏领域协议与泄漏隔离 | `dispute_agent/domain/`, `tests/leakage/test_hidden_state.py` |
| 确定性工具仿真与多轮 Episode 状态机 | `dispute_agent/tools/`, `dispute_agent/environment/`, `tests/unit/tools/`, `tests/integration/test_episode_state.py` |
| 可审计训练奖励与离线业务效用 | `dispute_agent/rewards/`, `tests/unit/rewards/` |
| 泄漏安全合成数据管线 | `dispute_agent/data/`, `tests/unit/data/`, `tests/leakage/test_dataset_leakage.py` |
| OpenAI Agents SDK 多轮工具 Runtime | `dispute_agent/agent/`, `tests/unit/agent/`, `tests/integration/test_agent_episode.py` |
| Agent Lightning rollout 适配与单次奖励返回 | `dispute_agent/training/lightning_agent.py`, `tests/integration/test_lightning_rollout.py` |
| TRL Qwen3-8B BF16 LoRA SFT 真实训练入口 | `dispute_agent/training/sft_runtime.py`, `configs/sft.yaml`, `constraints/sft.txt` |
| SFT 数据与运行可追溯性 | `run_manifest.json`, `metrics.json`, `sft-{size}-best/adapter_config.json` |
| Agent Lightning + VERL GRPO 配置与坍缩监控 | `configs/grpo.yaml`, `dispute_agent/training/grpo_config.py`, `dispute_agent/training/monitor.py`, `tests/unit/training/` |
| 统一公平评测协议 | `dispute_agent/evaluation/`, `tests/evaluation/` |
| 最小 FastAPI 演示与公开 Trace | `dispute_agent/api/`, `tests/integration/test_api.py` |

## 待训练机完成后补充

| 表述 | 证据位置 |
| --- | --- |
| 双 4090 Phase 0 门禁全部通过 | `artifacts/phase0/report.json`, `constraints/phase0-lock.txt` |
| SFT 规模消融真实指标 | `artifacts/evaluation/metrics.json`, `docs/experiments.md` |
| Agentic GRPO 真实更新与 checkpoint 可重载 | `artifacts/phase0/report.json`, 训练日志 |
| 统一评测矩阵 | `artifacts/evaluation/summary.md` |
