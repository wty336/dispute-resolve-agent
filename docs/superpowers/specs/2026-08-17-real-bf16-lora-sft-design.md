# 真实 BF16 LoRA SFT 训练链路设计

日期：2026-08-17  
状态：已确认，待实施计划

## 1. 目标与范围

本阶段把当前仅生成占位文件的 SFT 入口替换为可在双 RTX 4090 服务器上运行的真实训练链路。训练基于 `Qwen/Qwen3-8B`、TRL `SFTTrainer` 和 PEFT LoRA，使用 BF16，不使用 4-bit 或 8-bit 量化。

单次命令只训练一个数据规模，并支持 `500`、`1000`、`1500` 三档。三档训练集使用同一排序下的嵌套子集，验证集固定使用完整 150 条，以保证规模消融可比较。

本阶段包含：

- 公开 SFT 数据加载、哈希和协议校验；
- Qwen3 non-thinking 工具对话模板预检；
- assistant-only loss mask 校验；
- 双卡 BF16 LoRA 训练、验证、checkpoint 和断点续训；
- 最优 PEFT adapter、指标和可复现实验清单；
- 不下载大模型、不占用 GPU 的最小本地测试。

本阶段不包含：

- SFT 后的任务级生成评测；
- Agent Lightning 或 VERL 启动；
- Agentic GRPO 训练；
- 自动串行运行三档实验；
- 模型或训练产物自动上传到远程服务。

## 2. 技术路线

采用 TRL 原生对话式 SFT：

- 数据样本保留 `messages` 字段中的 system、user、assistant tool call、tool result 和最终 assistant 决策；
- 数据加载时为每条样本补充统一的 `tools` JSON Schema；
- `SFTTrainer` 使用 `assistant_only_loss=True`；
- PEFT 使用 `LoraConfig`，参数为 `r=32`、`lora_alpha=64`、`lora_dropout=0`、`target_modules="all-linear"`；
- Qwen3 训练模板必须以 non-thinking 方式渲染，不监督 `<think>` 内容；
- 训练前对真实 tokenizer 和至少一组完整工具轨迹执行 token、assistant mask 和长度预检。

TRL 官方支持带工具调用的 conversational dataset，并要求样本同时包含消息和工具 Schema；Qwen3 属于 TRL 可自动补齐 assistant generation mask 的已知模型家族。实现仍须通过运行时预检确认模板行为，不能仅依赖版本声明。

参考：

- <https://huggingface.co/docs/trl/sft_trainer>
- <https://huggingface.co/docs/trl/chat_templates>
- <https://huggingface.co/docs/peft/package_reference/lora>
- <https://huggingface.co/Qwen/Qwen3-8B>

## 3. 数据流

```text
manifest.json + sft_train.jsonl + sft_val.jsonl
        |
        v
文件哈希、数量、公开字段和消息协议校验
        |
        v
从固定顺序中截取前 500 / 1000 / 1500 条训练样本
验证集固定为完整 150 条
        |
        v
为每条样本补充统一 tools JSON Schema
        |
        v
Qwen3 non-thinking 模板、assistant mask、最大长度预检
        |
        v
TRL SFTTrainer + PEFT BF16 LoRA
        |
        v
checkpoint + best adapter + metrics + run_manifest
```

训练代码只读取公开的 `sft_train.jsonl` 和 `sft_val.jsonl`。任何 `*.ground_truth.jsonl` 文件都不能进入训练 Dataset，也不能出现在 run manifest 的训练输入列表中。

## 4. 命令行接口

保留单一入口 `scripts/train_sft.py`：

```bash
# 本地 fixture：不加载模型，不声称完成真实训练
python scripts/train_sft.py --config configs/sft.yaml --fixture

# 服务器预检：加载 tokenizer，不加载 8B 模型权重
python scripts/train_sft.py --config configs/sft.yaml --train-size 500 --preflight

# 双卡短训练
accelerate launch --num_processes 2 scripts/train_sft.py \
  --config configs/sft.yaml \
  --train-size 500 \
  --max-steps 2 \
  --output-dir checkpoints/sft/smoke-500

# 正式单档训练
accelerate launch --num_processes 2 scripts/train_sft.py \
  --config configs/sft.yaml \
  --train-size 500
```

CLI 约束：

- `--train-size` 仅接受 `500`、`1000`、`1500`；
- `--max-steps` 默认不设置，正式训练按 epoch 完成；
- `--fixture`、`--preflight` 和正式训练互斥；
- `--resume-from-checkpoint latest|PATH` 显式启用断点续训；
- 输出目录非空时默认拒绝启动，除非是合法续训；
- 不提供隐式覆盖开关，避免误删 checkpoint；
- 正式训练成功时输出文案必须明确为 `training complete`，fixture 和 preflight 使用不同文案。

## 5. 训练配置

默认配置：

| 项目 | 值 |
| --- | --- |
| Base model | `Qwen/Qwen3-8B` |
| Precision | BF16 |
| Quantization | none |
| LoRA rank / alpha / dropout | `32 / 64 / 0` |
| LoRA target | `all-linear` |
| Epochs | `3` |
| Learning rate | `2e-4` |
| Scheduler | cosine |
| Warmup ratio | `0.03` |
| Per-device train batch | `1` |
| Global effective batch | `16` |
| Max sequence length | `2048` |
| Gradient checkpointing | enabled |
| Attention implementation | PyTorch SDPA |
| Packing | disabled |
| Loss | assistant messages only |
| Validation/checkpoint | every epoch |
| Best-model metric | lowest `eval_loss` |
| Checkpoint retention | latest two |

脚本根据 `WORLD_SIZE` 计算梯度累积次数：

```text
gradient_accumulation_steps = global_batch_size / (world_size * per_device_batch_size)
```

若无法整除则在构建 Trainer 前失败。双卡、每卡 batch 1 时梯度累积为 8；这样单卡或双卡启动不会悄悄改变有效 batch。

不强制安装 FlashAttention，避免将编译和 CUDA 版本问题引入首个真实训练阶段。PyTorch SDPA 作为默认实现；后续只有在有真实吞吐瓶颈证据时才增加 FlashAttention 消融。

## 6. 模板与 loss mask 门禁

预检必须使用目标 Qwen3 tokenizer 和训练时采用的同一模板路径完成以下检查：

1. 工具定义可被模板接受；
2. system、user、assistant tool call、tool result 和最终 assistant 消息都能渲染；
3. non-thinking 模式下不产生可训练的 `<think>` 内容；
4. `assistant_masks` 至少包含一个训练 token；
5. system、user 和 tool 结果 token 不进入 assistant loss；
6. assistant tool call 和最终 assistant 决策进入 loss；
7. 每条样本 token 长度不超过 2048。

禁止静默截断。若存在超长样本，错误信息列出 case ID、实际长度和上限，并在训练前退出。这样不会截掉最终判责 JSON 或破坏工具调用闭合关系。

## 7. 分布式运行与资源边界

双 RTX 4090 使用 Accelerate/Trainer 的 DDP 启动，每个进程各加载一份 BF16 base model，并训练各自同步的 LoRA 参数。只允许主进程写最终 adapter、指标和 run manifest。

正式训练前检查：

- PyTorch 能识别 CUDA；
- GPU 支持 BF16；
- 进程数与可见 GPU 数相符；
- 运行设备不是 CPU；
- 未启用 4-bit 或 8-bit 加载；
- TRL、Transformers、PEFT 和 Accelerate 版本可读取；
- 输出目录满足新训练或续训条件。

SFT 依赖单独锁定到 `constraints/sft.txt`，不在本阶段同时安装 VERL、vLLM 和 Agent Lightning。PEFT adapter 与 GRPO 环境的兼容性由后续 Phase 0 的 `trl_adapter_loaded_by_verl` 门禁验证。

## 8. Checkpoint 与断点续训

每个 epoch 执行验证和保存，最多保留两个 Trainer checkpoint。`load_best_model_at_end=True`，以最低 `eval_loss` 选择最终 adapter。

续训规则：

- `--resume-from-checkpoint latest` 解析输出目录中编号最大的完整 checkpoint；
- 显式路径必须位于当前 run 输出目录内；
- checkpoint 必须包含 Trainer 恢复优化器、调度器和随机状态所需文件；
- 当前数据哈希、训练规模、模型、LoRA 配置和关键优化参数必须与原 run manifest 一致；
- 不一致时拒绝恢复并报告差异；
- 未显式指定 resume 时，非空输出目录导致失败。

参考：<https://huggingface.co/docs/transformers/trainer_recipes>

## 9. 输出产物

每档训练使用独立路径：

```text
checkpoints/sft/
  sft-500/
    checkpoint-*/
    trainer_state.json
    run_manifest.json
    metrics.json
  sft-500-best/
    adapter_model.safetensors
    adapter_config.json
    tokenizer files
```

`run_manifest.json` 至少记录：

- Git commit SHA 和完整执行参数；
- base model、LoRA 和 Trainer 配置；
- 数据文件路径、SHA-256、样本数和 dataset manifest SHA-256；
- PyTorch、Transformers、TRL、PEFT、Accelerate、Python 和 CUDA 版本；
- GPU 型号、数量、总显存；
- world size、梯度累积和全局有效 batch；
- 开始/结束时间、训练状态、是否续训和恢复点；
- 最优 checkpoint、train loss、eval loss 和训练耗时。

`metrics.json` 保存 Trainer 返回的原始数值指标。`sft-{size}-best` 只包含后续推理和 GRPO 所需的 adapter 与 tokenizer 文件，不复制 base model 权重。

只有满足以下条件才创建或更新 `sft-{size}-best`：

- Trainer 正常结束；
- 最终训练与验证损失均为有限数；
- adapter 文件完整；
- 主进程成功写入 metrics 和完成态 run manifest。

## 10. 错误处理

以下情况在加载 8B 模型权重前失败：

- 数据文件或 manifest 缺失；
- 文件哈希与 manifest 不一致；
- 样本数不符；
- 公开训练样本包含隐藏字段；
- 消息或工具调用协议非法；
- tokenizer 模板或 assistant mask 门禁失败；
- 样本超过最大长度；
- 输出目录冲突；
- 续训配置或数据与原 run 不一致；
- CUDA、BF16、GPU 数量或依赖版本不满足要求。

训练中出现 NaN/Inf 时，写入失败状态和诊断信息，保留已有 checkpoint 供调查，但不生成 `*-best`。多进程错误由主进程输出人类可读摘要，底层异常仍保留在日志中。

## 11. 最小测试策略

遵循“只测试高风险边界”的原则，不下载模型、不执行 GPU 训练，仅新增三组核心测试：

1. **数据加载测试**
   - 500/1000/1500 为嵌套子集；
   - 验证集始终为 150 条；
   - 训练输入不含 ground truth；
   - 每条样本具有统一工具 Schema；
   - manifest 哈希不一致时拒绝加载。

2. **模板预检测试**
   - 使用轻量假 tokenizer 验证合法 assistant mask；
   - 空 assistant mask、错误角色掩码和超长样本会失败；
   - non-thinking 门禁能发现被监督的 `<think>` token。

3. **训练编排测试**
   - 使用注入的轻量假 Trainer/PEFT 后端；
   - 检查 BF16、非量化 LoRA、全局 batch 和 Trainer 参数映射；
   - 检查输出冲突、合法续训、run manifest 和 best adapter 产物；
   - 不导入或下载真实 8B 模型。

现有完整测试集仍作为回归门禁。服务器上的真实 tokenizer preflight、两步双卡 smoke 和完整训练属于运行验证，不伪装为本地自动化测试。

## 12. 服务器验收顺序

1. 安装并记录 SFT 锁定依赖；
2. 生成正式数据并执行 freeze 验证；
3. 对 500 档运行真实 tokenizer preflight；
4. 双卡执行 2 个 optimizer step 的 smoke；
5. 检查峰值显存、checkpoint、续训和 adapter 可加载性；
6. 依次人工启动 500、1000、1500 三档正式训练；
7. 每档训练后保存真实 run manifest 和指标；
8. 后续统一评测阶段比较 Base、SFT-500、SFT-1000、SFT-1500。

## 13. 完成标准

本阶段完成需同时满足：

- 当前占位训练逻辑被真实 TRL/PEFT 路径替换；
- 三档单次训练参数可用，数据子集可复现；
- 真实 Qwen3 tokenizer preflight 能验证工具模板和 assistant mask；
- 双卡命令、输出冲突和断点续训语义明确；
- 本地最小测试和现有完整测试通过；
- 服务器两步 smoke 能产生可恢复 checkpoint 和可加载 PEFT adapter；
- 正式训练不会读取隐藏 ground truth；
- 所有简历指标仍只在真实训练与统一评测后填写。
