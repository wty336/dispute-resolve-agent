# 实验记录

> 本文件只记录真实执行过的实验与结果。未运行的训练实验保持 `not_run`，不填零值或猜测性指标。

## 本地实现状态

| 项目 | 状态 | 证据 |
| --- | --- | --- |
| GRPO 公共观测与隐藏标签隔离 | implemented_locally | `scripts/generate_data.py`, `tests/leakage/test_dataset_leakage.py` |
| manifest 校验与 fresh EpisodeSource | implemented_locally | `dispute_agent/training/grpo_dataset.py` |
| Agent Lightning/verl 训练入口与 dry-run | implemented_locally | `dispute_agent/training/grpo_runtime.py`, `scripts/train_agentic_grpo.py` |
| 双 RTX 4090 Phase 0 | not_run | `outputs/grpo/<run-id>/phase0_report.json`（待生成） |
| 正式 GRPO 与实测指标 | not_run | `outputs/grpo/<run-id>/metrics/summary.json`（待生成） |

## 运行记录模板

| run id | git commit | config hash | data hash | adapter hash | curriculum phase | optimizer updates | valid rollout rate | reward mean/std | component means | tool-call mean | failure rate | checkpoint path/hash | Phase 0 report |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | --- | --- |
| not_run | not_run | not_run | not_run | not_run | not_run | not_run | not_run | not_run | not_run | not_run | not_run | not_run | not_run |

完成一次服务器实验后，追加真实的日期、环境版本、commit SHA、数据/配置/adapter
哈希、命令、checkpoint 路径和指标；不要把 fixture 或 dry-run 写成训练结果。
