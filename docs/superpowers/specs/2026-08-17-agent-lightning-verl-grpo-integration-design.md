# Agent Lightning 0.3 + verl 0.5 Agentic GRPO 真实训练链路设计

**日期：** 2026-08-17  
**状态：** 已批准  
**目标环境：** Ubuntu 22.04、Python 3.11、CUDA 12.8、双 RTX 4090

## 1. 背景与目标

仓库已经具备电商纠纷环境、确定性工具、OpenAI Agents SDK 多轮运行时、组合奖励函数和真实 BF16 LoRA SFT 入口，但 Agentic GRPO 仍是占位实现：当前 `LitDisputeAgent` 没有使用 Agent Lightning 的真实 Agent/Trainer 接口，训练脚本只能输出 dry-run，Phase 0 脚本还会把未执行的门禁直接写成通过。

本阶段将占位实现替换为可在双 RTX 4090 上运行的真实 Agentic GRPO 链路。Agent Lightning 负责 rollout 编排、OpenAI 兼容模型代理、Trace 采集和训练样本适配；verl 负责 GRPO 优势计算、BF16 LoRA 参数更新和 checkpoint；OpenAI Agents SDK 继续负责业务 Agent 的多轮工具循环。

本阶段的目标是：

- 使用 Agent Lightning `0.3.0`、verl `0.5.0` 和 vLLM `0.10.2` 的稳定组合；
- 从只读的最佳 SFT-1500 adapter 副本继续 BF16 LoRA GRPO；
- 保持现有 Agent、工具、环境和奖励公式为唯一业务实现；
- 在本地完成可审计代码、dry-run 和轻量测试；
- 在服务器完成一次真实参数更新、adapter 重载和双卡显存门禁；
- 只有取得真实训练证据后，才在简历中声称跑通 Agent Lightning + verl。

本阶段不迁移到尚未正式发布且接口完全重构的 Agent Lightning v1.0，也不并行维护 v0.3 和 v1.0 两套实现。

## 2. 版本与依赖边界

完整训练环境固定为：

| 组件 | 版本 |
|---|---:|
| Python | 3.11 |
| PyTorch | 2.8.0 |
| torchvision | 0.23.0 |
| Agent Lightning | 0.3.0 |
| verl | 0.5.0 |
| vLLM | 0.10.2 |
| OpenAI Agents SDK | 0.6.0 |
| CUDA | 12.8 |

SFT 环境与完整 GRPO 环境继续分离。训练模块对 Agent Lightning、verl、vLLM 和 GPU 包使用延迟导入，使本地 `--dry-run`、配置检查和基础测试不要求安装完整 GPU 依赖。当前 Windows 电脑只验证项目自身的配置、数据和 rollout 核心逻辑；Agent Lightning 官方运行时绑定及分布式行为留到受支持的 Ubuntu 服务器验证。

依赖解析或运行时版本不一致时立即失败，不自动升级框架，不自动切换到 QLoRA、单卡训练或其他基座模型。

## 3. 总体架构

真实调用链如下：

```text
scripts/train_agentic_grpo.py
  -> GRPOTrainingRuntime
      -> load and validate frozen train/val tasks
      -> build Agent Lightning / verl resolved config
      -> agl.VERL(config)
      -> agl.Trainer(...)
      -> trainer.fit(dispute_rollout, train_dataset, val_dataset)

dispute_rollout
  -> rebuild a fresh EpisodeState from case_id
  -> receive rollout-scoped main_llm from Agent Lightning
  -> run the existing OpenAI Agents SDK DisputeRuntime
  -> execute deterministic environment tools
  -> compute RewardEngine result once
  -> attach non-secret diagnostic annotations
  -> return exactly one float reward

Agent Lightning Trace
  -> LlmProxyTraceToTriplet / trajectory aggregation
  -> verl GRPO advantage and policy loss
  -> BF16 LoRA update
  -> reloadable adapter checkpoint
```

组件边界如下：

- `dispute_agent/agent/` 只负责业务 Agent 和 OpenAI Agents SDK 运行时，不引用 verl。
- `dispute_agent/environment/` 负责 Episode 状态、工具结果、成本和隐藏 Ground Truth。
- `dispute_agent/rewards/` 保持唯一奖励实现，不在训练层复制奖励公式。
- `dispute_agent/training/lightning_agent.py` 提供真实 `@agl.rollout` 函数式适配，不再保留模拟的 `LightningRollout`。
- `dispute_agent/training/grpo_dataset.py` 负责冻结数据校验、task 构造和独立 Episode 重建。
- `dispute_agent/training/grpo_runtime.py` 负责 Agent Lightning、verl、Trace adapter、训练状态和产物。
- `scripts/train_agentic_grpo.py` 是统一用户入口，不包含底层训练实现。

## 4. Agent Lightning 适配

采用官方推荐的函数式 `@agl.rollout` 方式。rollout 函数接收任务和 Agent Lightning 注入的 LLM 资源，将 `llm.endpoint`、`llm.model` 和训练专用 API key 传给现有 `build_runtime`。函数内部继续调用现有 `DisputeRuntime.run(..., enable_thinking=True)`，不重新实现工具循环。

Agent Lightning 自动为当前 rollout 和 attempt 解析独立代理地址。所有 OpenAI Agents SDK 模型请求必须经过该地址，使模型请求、token IDs、响应、log probability 和最终奖励能关联到同一个 rollout。

rollout 的返回值是唯一的最终标量奖励。代码不得在返回奖励的同时调用 `emit_reward`。奖励分量、工具成本、终止类型和轮数可以用非奖励 annotation 保存，但 annotation 不得包含隐藏答案。

Trainer 使用：

- `agl.VERL(resolved_verl_config)`；
- `agl.Trainer`；
- `agl.OtelTracer()`；
- `agl.LlmProxyTraceToTriplet()`；
- 一个本地 runner 起步，Phase 0 验证稳定后再按显存和吞吐增加 runner 数。

## 5. 数据与 Episode 隔离

Agent Lightning task 只包含：

```json
{
  "case_id": "case_xxx",
  "scenario_id": "scenario_xxx",
  "curriculum_phase": 1
}
```

task 不包含责任标签、赔付区间、是否应升级、证据真实性或其他 Ground Truth。rollout worker 根据 `case_id` 从固定数据目录加载公开记录和 Ground Truth sidecar，并在环境边界内创建全新的 `EpisodeState`。

每个 rollout 必须重建 Episode，不允许从全局仓库返回同一个可变对象。同一 GRPO 组的四个 rollout 共享相同案例、初态和确定性工具结果，但累计成本、已见证据、非法动作计数和终局决策互不共享。组内差异只来自模型采样。

训练前执行以下数据门禁：

1. 公开数据与 Ground Truth sidecar 的 `case_id` 一一对应；
2. 每个文件的 hash 与冻结 manifest 一致；
3. SFT、GRPO、ID test 和 OOD test 按 `scenario_id` 隔离；
4. task 和可持久化 Trace 不含隐藏字段名或隐藏值；
5. 训练集为 700 个案例，验证集为 100 个案例。

## 6. 奖励与异常语义

`RewardEngine` 继续返回总奖励和现有分量：责任判断、赔付、升级判断、证据 grounding、升级质量、工具成本和非法动作惩罚。GRPO 只优化最终 Episode 标量奖励；本阶段不声称实现逐工具步骤的独立 credit assignment。

异常分两类处理：

- **业务失败：** 非法动作、工具预算耗尽、未形成合法终局决策等，转化为环境定义的确定性失败结果或最低奖励。
- **基础设施失败：** vLLM 断连、Agent Lightning 代理异常、worker 崩溃、Trace 缺少模型 span 等，使 rollout 失败并按有限次数重试，不转化为低奖励训练样本。

每次训练使用全新的 Episode，重试也不得复用已被修改的 Episode。

## 7. 模型、LoRA 与双卡拓扑

模型配置固定为：

| 参数 | 值 |
|---|---|
| Base model | `Qwen/Qwen3-8B` |
| 初始 adapter | 最佳 `SFT-1500` adapter 的只读副本 |
| dtype | BF16 |
| quantization | `null` |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0 |
| target modules | `all-linear` |
| rollout load format | `safetensors` |

verl 通过 `actor_rollout_ref.model.lora_adapter_path` 加载 SFT adapter。启动前校验 adapter 的基座模型、rank、alpha、dropout 和 target modules。SFT 原始 adapter 不被覆盖，GRPO 输出写入独立目录。

两张 4090 共同用于 FSDP actor 和 TP=2 的 vLLM rollout，并由 hybrid engine 在训练与生成阶段切换资源。LoRA 模式下 reference policy 使用 actor 的无 adapter 基座行为，避免额外常驻一份完整 reference model。启用梯度检查点，并以 parameter/optimizer CPU offload 作为双 24GB 显存的初始安全配置。

vLLM 的 `gpu_memory_utilization` 从 `0.30` 到 `0.35` 开始 Phase 0。发生 OOM 时，依次降低并发 task 数、runner 数、vLLM KV cache 占比和 micro batch；不得改变 BF16 LoRA、`n=4`、模型或奖励语义来掩盖兼容问题。

## 8. GRPO 与轨迹配置

核心配置为：

- `algorithm.adv_estimator=grpo`；
- rollout `n=4`；
- thinking 模式开启；
- 单轮最多生成 384 tokens；
- Episode 模型生成总预算 1,280 tokens；
- 每个正式训练 step 初始取 2 个不同案例，共 8 个 rollout；
- actor micro batch 每卡 1；
- LoRA 学习率从 `1e-5` 起；
- 轻量 KL 约束系数从 `0.001` 起；
- trajectory-level aggregation；
- rollout-level reward/advantage 语义；
- trajectory 的初始 prompt 与累计 response/context 上限分别从 2,048 和 4,096 tokens 起测。

trajectory 聚合只在相邻模型调用满足精确 token 前缀连续时合并。工具 observation 作为上下文保留但不计入策略 loss。被丢弃或截断的轨迹数量必须记录；若正式训练出现非零截断，先修正长度预算后再解释实验结果。

课程学习保持两阶段：第一阶段开放物流、买家历史和商家历史三个工具并限制三轮；稳定后开放证据核验并扩展到五轮。课程只改变可用工具和最大轮数，不改变数据划分、奖励公式或 LoRA 配置。

## 9. Smoke 与正式配置

使用同一个训练入口和同一套运行时代码，只通过 profile 改变规模：

| Profile | 用途 | 数据 | `n` | 更新 |
|---|---|---:|---:|---:|
| `smoke` | Phase 0 | 2 个训练案例和最小验证集 | 4 | 1 次真实更新 |
| `formal` | 正式训练 | 700 train / 100 val | 4 | 完整 epoch |

`smoke` 不改变模型、LoRA、Trace 聚合、奖励、工具协议和 checkpoint 格式，因此能验证正式链路。正式训练开始前必须完成 smoke adapter 的参数变化和重载验证。

## 10. 训练产物与断点续训

每次运行写入：

```text
artifacts/grpo/<run-id>/
  resolved_config.yaml
  run_manifest.json
  metrics.jsonl
  phase0_report.json
  traces/
  checkpoints/
```

`run_manifest.json` 包含框架版本、GPU、模型、SFT adapter hash、数据 hash、随机种子、profile、resolved config hash、开始/结束时间和 `running/completed/failed` 状态。失败时记录失败阶段、异常类型和最后可用 checkpoint。

断点续训只允许数据 hash、基座模型、初始 adapter、LoRA 参数和核心算法配置与原运行一致。恢复动作写入同一 run manifest 的事件记录，不能静默覆盖先前状态。

checkpoint 至少包含可重载 LoRA adapter、优化器/调度器状态和训练进度。简历证据使用导出的 adapter、resolved config、指标和 Phase 0 报告，不使用只有日志文本的占位证明。

## 11. 监控与停止条件

记录以下指标：

- group reward mean/std；
- 零方差组比例；
- 有效、失败和重试 rollout 比例；
- Episode 成功率；
- 各奖励分量；
- KL、entropy、梯度范数和学习率；
- 工具调用数、工具成本和生成长度；
- 被丢弃或截断的 trajectory 数；
- 每张 GPU 峰值显存。

连续 50 step 的零方差组比例超过 30% 时，监控结果标记为失败并阻止进入下一正式阶段。首版不修改 Agent Lightning/verl 内部 Trainer 来夸大实现自动恢复；只有真实接入外部停止机制后，才声称训练能自动暂停。

## 12. Phase 0 门禁

`phase0_report.json` 中每项门禁只能是 `passed`、`failed` 或 `not_run`，并附证据路径或失败原因。门禁包括：

1. OpenAI Agents SDK 通过 Agent Lightning/vLLM 代理完成多轮工具调用；
2. thinking 与工具调用格式兼容；
3. Trace 包含模型调用、工具结果、终止状态和唯一奖励；
4. TRL SFT adapter 被 verl 正确加载；
5. 完成一次真实 BF16 LoRA GRPO 更新且 adapter 参数变化；
6. 更新后的 adapter 能重新加载并推理；
7. 两张 4090 在目标配置下无 OOM；
8. task 和 Trace 不泄漏隐藏 Ground Truth。

任何门禁未执行都不得写成通过。所有关键门禁通过后才允许启动 `formal` profile。

## 13. 测试策略

本地只增加五类聚焦测试，不加载真实模型：

1. 项目配置能转换为预期的 Agent Lightning/verl 字典；
2. task 不包含隐藏字段，公开/sidecar 映射完整；
3. 同一 `case_id` 的多个 rollout 获得互相独立的 Episode；
4. fake LLM 和 fake Runtime 下，rollout 核心函数只返回一次奖励；Agent Lightning 装饰器绑定由服务器 Phase 0 验证；
5. dry-run 和 Phase 0 报告不会将未执行项写成通过。

服务器只执行一次必要的 Phase 0 真更新和重载验证。Phase 0 通过后不重复进行无信息增益的 GPU 测试，直接进入正式训练。

## 14. 文件变更范围

计划修改：

- `dispute_agent/training/lightning_agent.py`
- `dispute_agent/training/grpo_config.py`
- `scripts/train_agentic_grpo.py`
- `scripts/phase0_smoke.py`
- `configs/grpo.yaml`
- `constraints/train.txt`
- `README.md`
- `docs/experiments.md`
- `docs/resume-evidence-checklist.md`

计划新增：

- `dispute_agent/training/grpo_dataset.py`
- `dispute_agent/training/grpo_runtime.py`
- 对应的少量训练测试文件

本阶段不修改 SFT 训练实现、业务判责 Schema、工具语义、奖励公式和 FastAPI 演示接口。

## 15. 完成标准与简历口径

本地实现完成需要满足：

- dry-run 输出完整 resolved config 和关键派生值；
- fake rollout 经过项目 Runtime 核心路径返回唯一奖励，且训练入口能在依赖可用时绑定真实 Agent Lightning 装饰器；
- 数据隔离、配置和 Phase 0 状态测试通过；
- 仓库不包含生成数据、模型权重或伪造训练指标。

服务器实现完成需要满足：

- 双 4090 完成一次真实 BF16 LoRA GRPO 更新；
- 更新前后 adapter 参数 hash 不同；
- checkpoint 能重新加载并完成至少一个真实 Agent rollout；
- Phase 0 全部关键门禁通过并有证据；
- 正式训练和统一评测产生真实结果。

完成本地代码但未进行服务器训练时，只能表述为“实现 Agent Lightning + verl Agentic GRPO 训练链路”。只有服务器证据齐全后，才能表述为“在双 RTX 4090 上完成 Agentic GRPO 训练并取得相应指标”。
