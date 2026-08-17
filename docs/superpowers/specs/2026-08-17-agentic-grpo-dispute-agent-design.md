# 基于 SFT 与 Agentic GRPO 的电商纠纷判责 Agent：设计文档

日期：2026-08-17

## 1. 项目目标

本项目面向大模型算法实习岗位，构建一个在部分可观测电商纠纷环境中自主调查并完成判责的 7B/8B Agent。模型先通过工具调用 SFT 学习领域输出格式和基本调查行为，再通过 Agentic GRPO 联合优化判责质量、赔付效用、人工升级合理性与调查成本。运行时使用 OpenAI Agents SDK 构建单 Agent 工具循环，Agent Lightning 采集完整交互轨迹并驱动 GRPO 训练，vLLM 提供本地 Qwen3 推理服务。

项目需要回答的核心问题是：

> 高质量少量 SFT 数据与多目标 Agentic GRPO，能否让模型在不读取隐藏事实的前提下，学会选择有价值的调查工具，并在判责质量和调查成本之间取得更好的权衡？

### 1.1 成功标准

- 真实完成 Base、SFT、SFT + Agentic GRPO 三阶段评测。
- 固定测试集上报告判责、赔付、业务效用、工具策略和安全指标。
- 至少完成一个 SFT 数据规模消融和一个 GRPO 奖励消融。
- Agent 能端到端接收工单、调用工具、提交判责或升级人工。
- 保存可复现配置、训练日志摘要、结果图表与典型交互轨迹。

### 1.2 非目标

- 不构建买家、商家、裁判多个对话 Agent。
- 不追求生产级数据库、前端或多租户部署。
- 不使用未经实际运行的训练结果或虚构指标。
- 不将 LLM-as-a-Judge 作为在线训练主奖励。

## 2. 总体架构

系统分为六层：

1. **数据与仿真层**：生成结构化纠纷事实、公开工单、隐藏标签和可复现工具结果。
2. **SFT 训练层**：使用 TRL 对 Qwen3-8B 进行 BF16 LoRA SFT。
3. **Agent 运行层**：使用 OpenAI Agents SDK 定义单 Agent、工具、状态、输入/输出 Guardrail 和运行 Trace。
4. **强化学习环境层**：维护 Episode 状态、执行工具、记录成本并依据隐藏事实计算奖励。
5. **Agent 强化学习层**：Agent Lightning 将 Agent 运行轨迹写入 LightningStore，并由 VERL/GRPO 后端读取轨迹、完成 BF16 LoRA 参数更新和资源版本切换。
6. **服务与评测层**：vLLM 提供本地模型服务，评测器在 ID、OOD 和人工审核集上统一比较基线与训练模型。

核心数据流为：

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

训练 rollout 和部署演示复用同一个 OpenAI Agents SDK Agent、领域模型、工具注册表、参数 Schema 和决策 Schema。训练环境只额外注入隐藏奖励接口，运行时 Agent 无权读取 Ground Truth。

### 2.1 框架职责边界

- TRL 只负责 SFT，不承担最终 Agentic GRPO 主线。
- OpenAI Agents SDK 只负责编排 Agent，不参与模型权重训练。
- Agent Lightning 负责任务分发、轨迹与奖励采集、训练资源同步，不重复实现业务工具。
- VERL/GRPO 负责策略优化与 LoRA 权重更新。
- vLLM 只负责 rollout、评测和演示推理，不保存业务状态。

不同时引入 LangGraph、CrewAI 等第二套 Agent 编排框架，避免为了展示技术栈而重复实现相同能力。

## 3. 数据模型与隔离

### 3.1 公开观察

`DisputeObservation` 仅包含：

- 订单号、买卖双方匿名 ID、商品、金额；
- 投诉类型、买家陈述、诉求金额；
- 商家回应、聊天记录；
- 原始证据的类型、描述和来源；
- Agent 已实际调用工具后获得的结果。

### 3.2 隐藏事实

`DisputeGroundTruth` 单独保存：

- 真实责任；
- 真实损失与合理赔付区间；
- 买卖双方隐藏策略；
- 证据真实性和真实证明力；
- 是否应人工升级；
- 各工具在该案例上的潜在信息价值。

Ground Truth 不进入消息、Prompt、工具参数或 Agent 状态。只有数据生成器、仿真工具内部、奖励引擎和评测器可以读取。

### 3.3 工具结果

工具结果由 `case_id + tool_name + arguments` 决定，可复现且包含经过校准的误报、漏报和信息缺失。工具不能直接返回“真实责任”“买家欺诈”等隐藏标签。

同一 GRPO 组内的四个 rollout 使用相同案例和相同工具结果，使组内差异只来自模型策略。

## 4. 合成数据设计

### 4.1 数据规模

| 数据集 | 数量 | 用途 |
| --- | ---: | --- |
| SFT 训练 | 1,500 | 领域判责与工具调用 |
| SFT 验证 | 150 | checkpoint 选择 |
| Agentic GRPO 训练 | 700 | 多轮策略优化 |
| GRPO 验证 | 100 | 奖励与策略坍缩监控 |
| ID 测试 | 400 | 同分布评测 |
| OOD 测试 | 200 | 鲁棒性评测 |
| 人工审核 | 100 | 从测试集中抽取的重点子集 |

SFT、GRPO 与测试案例按结构化场景 ID 隔离；同一事实实例的不同语言改写不能跨集合。

### 4.2 生成流程

```text
场景变量采样
→ 结构化 Ground Truth
→ 公开信息与隐藏信息拆分
→ 教师模型自然语言渲染
→ 工具结果仿真
→ 自动校验与去重
→ 难度分级和数据切分
→ 人工抽查
```

场景变量覆盖投诉类型、真实责任、金额档位、双方行为、证据冲突、证据缺失、工具异常和案例难度。自然语言渲染通过可配置的 OpenAI 兼容教师端点完成；单元测试使用确定性模板渲染器，不依赖外部服务。

### 4.3 SFT 组成

1,500 条训练样本包括：

- 800 条信息完整的直接判责样本；
- 500 条包含一至四次实际工具调用的多轮轨迹；
- 200 条人工升级、工具失败、非法结果恢复和高风险案例。

工具轨迹采用标准 assistant tool call 与 tool message，不再把工具调用伪装成普通文本 JSON。最终输出包含责任、赔付、是否升级、置信度、证据摘要和简短理由；理由只表达可核验依据，不要求生成隐藏思维链。

### 4.4 数据质量

生成后执行：

- JSON Schema 和工具参数校验；
- 金额、责任、赔付区间一致性校验；
- Ground Truth 泄漏扫描；
- 文本近重复和结构化事实重复检测；
- 标签、难度和工具分布统计；
- 轨迹可执行性校验。

OOD 测试集固定由三种互斥分布偏移构成：80 条训练中未出现的场景变量组合、60 条语言表达和聊天风格变化、60 条工具噪声/缺失/超时强度变化。人工审核集从 ID 和 OOD 测试集分层抽取 100 条，记录事实自洽性、判责合理性、赔付合理性、工具必要性和语言自然度。审核修正完成后冻结测试文件并记录 SHA-256；此后不得依据测试结果修改奖励或训练数据。

## 5. SFT 训练设计

基座模型为 `Qwen/Qwen3-8B`，基础权重以 BF16 加载并使用 LoRA 训练。SFT 和 GRPO 统一使用 `rank=32`、`alpha=64`、`dropout=0`、`target_modules=all-linear`；不使用 4-bit/8-bit 量化。SFT 样本采用 non-thinking chat template，只对 assistant 的标准 tool call、调用结果后的 assistant 动作和最终判责计算 loss，system、user 与 tool message 全部 mask。这只选择 SFT 的输出路径，不删除基座模型已有的 thinking 能力。两张 24GB 4090 采用双卡数据并行完成 SFT，并启用梯度检查点、较小的单卡微批次和梯度累积。训练配置、依赖版本、随机种子和数据哈希写入实验记录。

Agentic GRPO 从最佳 SFT adapter 的副本启动，不合并回基础权重；VERL 通过 `actor_rollout_ref.model.lora_adapter_path` 加载该 adapter，并保持相同的 LoRA 配置、tokenizer 和 chat template。原始 SFT adapter 只读保存，GRPO 只更新副本。

Agentic GRPO 在 rollout 中启用 thinking。VERL hybrid engine 统一管理策略训练和内部 vLLM rollout 服务，两张 4090 共同承担 FSDP actor/reference 与 TP=2 rollout，不采用固定的一卡训练、另一卡推理拓扑。具体配置以 Phase 0 显存实测为准。模型可以使用 Qwen3 thinking 模式规划调查动作，但奖励只评价可观测的工具轨迹和最终结果，不直接评价、展示或要求复现隐藏思维内容。每轮 generation 最多 384 tokens，整个 Episode 的模型生成预算最多 1,280 tokens。优先通过限制上下文长度、并发 rollout 数量和 KV Cache 占用解决显存压力；必要时使用参数或优化器 CPU offload，但不改用量化训练。

数据规模消融分别使用 500、1,000 和 1,500 条嵌套训练子集，三组保持验证集、训练步数口径和评测集一致。正式 Agentic GRPO 从 SFT-1500 的最佳验证 checkpoint 启动。

SFT 主要学习：

- 公开信息与证据的结构化理解；
- 标准工具调用协议；
- 调用结果后的继续调查或终止决策；
- 合法、简洁的最终判责格式；
- 高风险和信息不足时的人工升级行为。

## 6. Agentic GRPO 与 Agent Lightning

### 6.1 Episode 状态

每个 rollout 保存：

- 当前 `case_id` 和公开 Observation；
- 消息与已获得证据；
- 工具调用历史、累计成本和缓存结果；
- 当前轮数、非法动作次数和风险标志；
- 最终决策、结束状态和结束原因。

### 6.2 可用动作与终止语义

- `check_logistics(order_id)`
- `check_buyer_history(buyer_id)`
- `check_merchant_history(merchant_id)`
- `verify_evidence(evidence_id)`
- `submit_decision(action, liability, compensation, confidence, evidence_ids, reason)`

`submit_decision` 使用判别联合 Schema：

- `action="decide"` 时，`liability` 和 `compensation` 必填；
- `action="escalate"` 时，`liability` 和 `compensation` 必须为 `null`，只提交置信度、证据和升级理由；
- 两种动作都必须引用 Agent 实际可见的 `evidence_ids`，不得同时“判责并升级”。

`submit_decision` 是终止工具：参数通过工具 Schema 与工具级 Guardrail 后，将结构化决策写入 Episode，并通知 Runner 结束。每个 Episode 最多五轮，最多四次调查工具调用。重复调用返回缓存结果并产生动作惩罚；超过最大轮数仍未提交终止动作则 Episode 失败。

### 6.3 Agent Lightning 适配

每个训练任务向 rollout 函数提供 `case_id` 和公开 Observation。Agent Lightning 的 VERL 集成从训练配置部署首个模型端点，并向 rollout 提供名为 `main_llm` 的 ProxyLLM 资源；OpenAI Agents SDK provider/adapter 必须使用当前 rollout 和 attempt 对应的代理地址，保证请求、响应与训练样本正确关联。rollout 函数启动同一个 OpenAI Agents SDK Agent，将工具调用、工具结果、最终决策、终止原因、成本和奖励分量记录为可关联的 Trace；隐藏 Ground Truth 只在环境侧用于计算最终奖励。

训练 Trace 采用单一来源原则：只有 ProxyLLM 产生、带 rollout/attempt 标识和 token IDs 的模型调用 span 可以进入 VERL Adapter。OpenAI Agents SDK Trace 只记录工具、Guardrail、状态、延迟等运行时事件，关闭默认云端导出或写入独立本地命名空间，不作为训练样本。rollout 只 `return` 一次最终标量奖励，不再同时调用 `emit_reward`。

Agent Lightning 负责以下协议：

- 将任务、Trace、最终奖励和各奖励分量写入 LightningStore；
- 保证训练算法可以按 `case_id` 聚合同一 GRPO 组的 rollout；
- 由 VERL 管理 rollout 服务和训练权重同步，避免 Episode 中途切换策略；
- VERL Adapter 只匹配 `main_llm` 的 ProxyLLM 模型 span，排除 SDK 观测 span；
- 训练与评测 Trace 使用不同命名空间，防止测试数据进入训练。

VERL 配置使用 `adv_estimator=grpo`，完成 log probability、组内相对优势和 LoRA 参数更新，`actor_rollout_ref.rollout.n=4` 定义每个任务的四个采样。当前 Agent Lightning VERL 集成默认将轨迹最终标量奖励传播到该轨迹的模型调用，因此本项目优化的是完整 Episode 的组合奖励；各奖励分量作为 Trace annotation 和监控指标保存，但不声称实现逐工具步骤的独立 credit assignment。若需要改变奖励传播或轨迹聚合方式，必须作为额外算法实验单独实现和验证，不纳入最小可行主线。

Agent Lightning 不重新计算业务奖励，也不在 Agent 运行代码中隐藏第二套训练逻辑。

GRPO 使用每组四个 rollout。同组使用相同案例、初态和确定性工具结果，但采用不同采样随机性。初始训练先开放物流、买家历史和商家历史三个工具并限制三轮；稳定后启用证据核验并扩展到五轮。该课程只改变环境难度，不插入单轮 GRPO 过渡阶段。

### 6.4 Phase 0 兼容性门槛

在 Qwen3-8B 正式训练前，先使用同系列小模型和 20 个案例验证完整链路。必须同时满足：

1. OpenAI Agents SDK 能通过 provider/adapter 调用本地 vLLM，并完成多轮工具调用；
2. Qwen3 thinking 与 tool call 不互相破坏，最终结构化决策可稳定解析；
3. Agent Lightning Trace 能恢复完整的任务、动作、工具结果、成本和奖励；
4. TRL 生成的 SFT adapter 能通过 `lora_adapter_path` 被 VERL 加载，且加载前后固定样例输出一致；
5. VERL/GRPO 能在两张 4090 上完成至少一次 BF16 LoRA 参数更新，并产出可重新加载的 adapter checkpoint；
6. 两张 4090 在目标上下文和最小并发下无显存溢出；
7. 同一次 rollout 不产生重复模型训练 span，最终奖励只写入一次。

任何一项失败时，先修复 adapter 或缩小 rollout 配置；若 Agent Lightning 与 VERL 的关键链路仍无法稳定运行，则降级为 TRL 原生 Agentic GRPO 环境。降级方案不与主方案并行维护，也不在简历中声称使用未实际跑通的框架。

## 7. 奖励设计

训练奖励不包含业务效用或 Oracle 预设的工具信息价值。工具是否值得调用，只通过最终任务质量、证据完整性、动作惩罚和工具成本学习。

当 `action="decide"` 时：

```text
R_decide = 0.45 R_liability
         + 0.25 R_compensation
         + 0.15 R_escalation
         + 0.15 R_grounding
         - 0.10 normalized_tool_cost
         + P_invalid_action
```

当 `action="escalate"` 时：

```text
R_escalate = 0.60 R_escalation
           + 0.25 R_grounding
           + 0.15 R_escalation_quality
           - 0.10 normalized_tool_cost
           + P_invalid_action
```

总奖励裁剪到 `[-1.5, 1]`。

- `R_liability`：责任完全正确为 1；单方责任与双方共担相邻为 0.4；买卖双方责任判反为 -1；其他错误为 -0.3。
- `R_compensation`：落在合理赔付区间为 1，偏离区间后按相对订单金额线性下降到 -1。
- `R_escalation`：是否升级与隐藏的风险标签一致为 1，否则为 -1。
- `R_grounding`：按有效证据引用比例计分；无引用但需要证据时为 -0.5，引用任何不存在或不可见证据时为 -1。
- `R_escalation_quality`：升级理由与 Agent 可见的证据冲突、信息不足、高金额、疑似欺诈或工具连续失败相对应为 1；理由空泛为 0；虚构风险为 -1。该项完全由结构化规则计算，不使用 LLM Judge。
- `normalized_tool_cost`：累计成本除以 Episode 最大允许成本。
- `P_invalid_action`：每次非法调用扣 0.2、每次重复调用扣 0.1，总计裁剪到 `[-0.4, 0]`。

非法终止 Schema、负赔付、超权限赔付、`decide` 缺少责任/赔付、`escalate` 仍提交责任/赔付、未提交终止动作等触发硬惩罚，奖励直接为 -1.5。

上线训练前使用手工构造轨迹验证奖励排序：

```text
正确且调查合理
> 正确但调查冗余
> 正确但过度赔付
> 合理人工升级
> 判责错误
> 非法或未完成输出
```

训练时单独记录每个奖励分量，监控“从不调用工具”“调用全部工具”“全部人工升级”和“全部支持买家”等策略坍缩。业务效用不参与训练，只在离线评测中使用固定成本公式计算。

## 8. OpenAI Agents SDK 运行时

运行时只实现一个 `DisputeResolutionAgent`，不使用多 Agent handoff。其组成包括：

1. **输入 Guardrail**：检查字段、权限、订单金额和输入完整性；
2. **模型策略**：通过自定义 provider/adapter 调用 vLLM 中的 Qwen3 + LoRA，选择调查工具或提交决策；
3. **函数工具**：复用物流、买家历史、商家历史和证据核验后端，并以终止工具 `submit_decision` 提交判责或升级；
4. **会话状态**：保存已获得证据、累计成本、轮数、非法动作和风险标志；
5. **工具级 Guardrail**：在执行 `submit_decision` 前校验判别联合 Schema、赔付上限、证据引用和升级条件；
6. **输出 Guardrail**：校验最终对外响应与 Episode 中已提交的结构化决策一致；
7. **Tracing**：SDK 记录工具、Guardrail、状态和延迟；ProxyLLM 单独记录可训练的模型请求、响应和 token IDs。

字段完整且证据充分的简单案件可以使用 non-thinking；证据冲突、高金额或低置信度案件使用 thinking。无论采用哪种模式，产品 Trace 和演示界面均不展示隐藏思维内容，只展示可审计的工具动作、证据和最终理由。

异常策略：

- 工具超时允许重试一次，之后切换证据来源或升级人工；
- 连续两次非法调用后升级人工；
- 重复调用返回缓存并记录无效行为；
- 模型格式修复只尝试一次；
- 超过最大轮数、超权限赔付和高风险案件进入人工复核；
- 模型服务失败时使用规则策略回退，并在结果中明确标记 `fallback`。

训练时由 VERL 管理 vLLM 和 Agent Lightning ProxyLLM；独立评测与演示时使用普通 vLLM OpenAI 兼容接口。OpenAI Agents SDK 的 provider/adapter 同时支持这两种 base URL，使业务 Agent 不依赖具体传输协议。FastAPI 仅提供单案例判责和 Trace 查询接口。演示界面只展示工单、动作、工具结果、累计成本、最终判责和人工升级状态。

## 9. 评测与实验

### 9.1 公平评测协议

Base、SFT-500、SFT-1000、SFT-1500 和 SFT-1500 + Agentic GRPO 使用完全相同的 OpenAI Agents SDK Agent、工具后端、最大轮数、工具成本、测试案例和终止 Schema，只替换模型/adapter 权重。主实验统一使用 thinking，固定 `temperature=0.6`、`top_p=0.95`、`top_k=20` 和推理随机种子。RuleBased 与 Oracle 只作为下界/上界锚点，不共享模型推理过程。

### 9.2 主实验

- RuleBased；
- Base Model；
- SFT-500；
- SFT-1000；
- SFT-1500；
- SFT-1500 + Agentic GRPO；
- Oracle。

### 9.3 GRPO 消融

- 完整奖励；
- 去掉工具成本惩罚。

最终 Agentic GRPO checkpoint 还需在相同测试案例上分别以 thinking 和 non-thinking 模式推理。thinking 使用 `temperature=0.6`、`top_p=0.95`、`top_k=20`；non-thinking 使用 `temperature=0.7`、`top_p=0.8`、`top_k=20`。两组固定推理随机种子，对比任务成功率、判责 Macro-F1、平均工具成本、平均生成 tokens 和端到端延迟。该对比不额外训练模型，用于衡量运行时思考预算的收益与成本。

### 9.4 指标与训练监控

判责指标：Accuracy、Macro-F1、类别 Precision/Recall、混淆矩阵。

业务指标：赔付 MAE、过度赔付率、赔付不足率、归一化业务效用。

离线业务效用沿用固定、可审计的长期价值模型：

```text
U = buyer_LTV × buyer_repurchase_probability
  + merchant_LTV × merchant_retention_probability
  - compensation
  - tool_cost
  - manual_review_cost
  - risk_cost
  + reputation_gain
```

买家复购率、商家留存率、风险成本和口碑项均由结构化 Ground Truth 与固定配置计算，不使用 LLM Judge。`action="escalate"` 时，离线评测假定人工最终采用 Oracle 判责与赔付，再额外扣除人工审核成本和处理延迟成本；训练奖励仍只使用第 7 节的规则。所有效用参数在首次训练前冻结，并以 RuleBased 和 Oracle 的测试集效用为锚点归一化。

Agent 指标：Episode 成功率、平均调用数、平均成本、必要工具召回率、无效调用率、重复调用率、超限率、工具失败恢复率。

安全指标：人工升级 Precision/Recall/F1、非法决策率、证据幻觉率、超权限赔付率、低置信度错误率。

GRPO 训练额外记录 `group_reward_std`、零方差组比例、有效 rollout 比例、Episode 成功率、KL、生成长度、工具调用数和各奖励分量。连续 50 个训练 step 的零方差组比例超过 30% 时暂停训练，检查采样温度、案例难度和奖励离散度后再恢复，不在无有效优势信号时继续消耗算力。

所有指标分别在 ID、OOD 和人工审核子集上报告。完整训练使用一个固定随机种子，并通过测试集 Bootstrap 报告 95% 置信区间；README 明确说明未进行多随机种子完整训练。

## 10. 测试策略

- **数据测试**：公开字段不可访问隐藏属性，跨集合无事实重复，渲染后金额与 ID 一致。
- **工具测试**：结果可复现、参数校验正确、缓存和成本累计正确、无隐藏标签直出。
- **奖励测试**：`decide`/`escalate` 分支、手工轨迹排序、各分量边界、动作惩罚、裁剪、硬惩罚和无 NaN。
- **环境测试**：组内初态一致、不同 rollout 状态隔离、最大轮数终止、终止工具调用后不可继续动作。
- **Provider 测试**：vLLM 响应、thinking、tool call、tool result 回填和结构化输出在 adapter 中往返不丢失。
- **LoRA 接续测试**：TRL adapter 配置与 VERL 配置一致、加载前后固定样例输出一致、一次 GRPO 更新后 adapter 可重载。
- **Agent Lightning 测试**：Trace 完整性、任务与 adapter 关联、奖励仅写入一次、ProxyLLM span 唯一、SDK span 被训练 Adapter 排除、组内 rollout 聚合。
- **Agent 集成测试**：OpenAI Agents SDK 正常工具循环、Guardrail、超时、非法参数、格式修复、人工升级和模型回退。
- **评测测试**：指标在固定样例上的预期值、ID/OOD 分组正确、报告可重复生成。

## 11. 建议目录

```text
dispute-resolve-agent/
├── configs/
│   ├── sft.yaml
│   ├── grpo.yaml
│   └── evaluation.yaml
├── dispute_agent/
│   ├── domain/
│   ├── data/
│   ├── tools/
│   ├── environment/
│   ├── rewards/
│   ├── training/
│   │   ├── sft/
│   │   └── agentic_grpo/
│   ├── agent/
│   │   ├── runtime/
│   │   ├── providers/
│   │   └── tracing/
│   ├── evaluation/
│   └── api/
├── scripts/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── reward/
│   └── leakage/
├── docs/
├── README.md
└── requirements.txt
```

仓库不提交模型权重和完整生成数据，只提交小型样例、生成器、固定配置、依赖锁定、实验日志摘要、结果图表和典型 Trace。

### 11.1 旧雏形迁移边界

现有代码仅作为领域仿真原型，不继续维护旧训练协议。实施时：

- 保留并测试案例生成、领域枚举、工具仿真和业务成本模型中仍符合新 Schema 的逻辑；
- 删除“预执行工具并把结果嵌入 RL prompt”的单步 RL 数据链路；
- 将 Qwen2.5-7B 和 ms-swift 脚本替换为 Qwen3-8B、TRL SFT 与 Agent Lightning/VERL 配置；
- 将普通文本 JSON tool call 和伪装成 user 的工具结果替换为标准 assistant `tool_calls` 与 `role=tool` 消息；
- 不在旧 `PlatformAgent` 类层级上继续叠加框架，重新以领域层、OpenAI Agents SDK 运行时和训练适配层建立清晰边界。

## 12. 主要风险与控制

- **显存不足或 thinking 输出膨胀**：Agent Lightning/VERL 官方示例常使用更大显存设备，两张 24GB 4090 运行 8B Agent RL 属于必须实测的高风险配置。坚持 BF16 LoRA，启用梯度检查点、小微批次和梯度累积，并限制上下文、单轮及 Episode 生成预算、并发 rollout 与 vLLM KV Cache；必要时使用 CPU offload，不同时承担额外 Judge 模型。Phase 0 未通过时先缩短上下文和降低并发，而不是改为 QLoRA。
- **奖励投机**：硬约束、分量日志、奖励消融、OOD 测试和高奖励轨迹人工检查。
- **合成数据模板化**：结构化采样与语言渲染分离、近重复检测、测试专属语言风格。
- **工具策略坍缩或 GRPO 无优势信号**：控制工具成本权重，记录必要工具召回率、`group_reward_std` 和零方差组比例，并比较无成本惩罚消融。
- **框架与模型协议不兼容**：先完成 Phase 0，单独测试 OpenAI Agents SDK provider/adapter、Qwen3 thinking/tool call 和 Agent Lightning Trace，不在 8B 正式训练中首次验证集成。
- **Agent Lightning/VERL 版本耦合**：固定 Python、Transformers、TRL、PEFT、vLLM、OpenAI Agents SDK、Agent Lightning 和 VERL 版本，并保存云端环境清单与最小可运行样例。
- **项目范围失控**：优先完成数据隔离、SFT、Agentic GRPO 和统一评测；FastAPI 和演示界面只做最小功能。

## 13. 简历表述原则

项目名称：**基于 SFT 与 Agentic GRPO 的电商纠纷智能判责 Agent**。

简历描述围绕三项真实工作展开：高质量小规模工具调用数据、Agentic GRPO 多目标奖励与环境、OpenAI Agents SDK + Agent Lightning + vLLM 端到端落地。TRL 仅作为 SFT 实现出现；VERL/GRPO、Agent Lightning 和 OpenAI Agents SDK 只有在 Phase 0 与正式训练真实跑通后才能写入简历。指标提升仅在实验完成后从固定评测报告中填写，不预设数字。

## 14. 参考资料

- [TRL 官方文档](https://huggingface.co/docs/trl/main/en/index)
- [Agent Lightning 官方仓库](https://github.com/microsoft/agent-lightning)
- [Agent Lightning：Write Agents](https://microsoft.github.io/agent-lightning/latest/tutorials/write-agents/)
- [Agent Lightning Algorithm Zoo](https://github.com/microsoft/agent-lightning/blob/main/docs/algorithm-zoo/index.md)
- [OpenAI Agents SDK Quickstart](https://developers.openai.com/api/docs/guides/agents/quickstart)
- [OpenAI Agents SDK Models and Providers](https://developers.openai.com/api/docs/guides/agents/models)
- [VERL 官方仓库](https://github.com/verl-project/verl)
- [VERL Qwen3-8B GRPO LoRA 双卡示例](https://github.com/verl-project/verl/blob/main/examples/tuning/lora/run_qwen3_8b_fsdp.sh)
- [vLLM 官方文档](https://docs.vllm.ai/)
- [Qwen3 Function Calling](https://github.com/QwenLM/Qwen3/blob/main/docs/source/framework/function_call.md)
