# 实验记录

> 本文件只记录真实执行过的实验与结果。未运行的训练实验不填写数字。

## 已完成的本地非 GPU 验证

- [x] Task 0–11 代码与测试：`pytest -q` 全部通过。
- [x] 数据 fixture 生成：`python scripts/generate_data.py --seed 20260817 --fixture-size 24 --output artifacts/data-smoke`。
- [x] SFT fixture 数据/配置检查：`python scripts/train_sft.py --config configs/sft.yaml --data-dir artifacts/data-smoke --fixture`；未加载模型，未产生训练指标。
- [x] GRPO dry-run：`python scripts/train_agentic_grpo.py --config configs/grpo.yaml --dry-run`。

## 待训练机完成

- [ ] Phase 0 七项门禁（`scripts/phase0_smoke.py`）。
- [ ] Qwen3 tokenizer non-thinking / assistant-mask preflight。
- [ ] 双 RTX 4090 两步 BF16 LoRA smoke 与 checkpoint 续训。
- [ ] SFT-500 / SFT-1000 / SFT-1500 正式训练。
- [ ] Agentic GRPO 主实验与无工具成本消融。
- [ ] 冻结测试集统一评测与 Bootstrap 95% CI。
- [ ] 最终 README 结果表。

## 记录格式

每次训练实验完成后，在本文件追加：

- 日期、环境、commit SHA；
- 数据 manifest SHA-256；
- SFT / GRPO 配置 hash；
- 评测命令与产物路径；
- 真实指标表。
