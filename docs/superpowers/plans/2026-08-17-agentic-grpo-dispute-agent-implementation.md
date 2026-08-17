# Agentic GRPO Dispute Resolution Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有单步电商纠纷原型迁移为一个可复现的 Qwen3-8B BF16 LoRA 项目：用 TRL 完成小规模高质量 SFT，用 OpenAI Agents SDK 构建真实多轮工具 Agent，并通过 Agent Lightning + VERL 完成 Agentic GRPO，最后在冻结的 ID/OOD 测试集上统一评测。

**Architecture:** 结构化 Ground Truth、公开 Observation 与确定性工具仿真严格分层；同一 OpenAI Agents SDK Runtime 同时服务训练、评测和演示。TRL 只训练 SFT adapter；Agent Lightning 的 `main_llm` ProxyLLM 是唯一训练模型 span 来源，VERL 从只读 SFT adapter 的副本继续进行 GRPO；业务奖励只在环境侧读取隐藏事实。Phase 0 在双 RTX 4090 上验证完整链路，未通过前不启动 Qwen3-8B 正式训练。

**Tech Stack:** Python 3.11、Pydantic 2、PyTorch BF16、Qwen3-8B、PEFT LoRA、TRL、OpenAI Agents SDK、Agent Lightning、VERL、vLLM、FastAPI、pytest、Hugging Face Datasets/Parquet、YAML。

---

## 实施约束

- 需求基线：`docs/superpowers/specs/2026-08-17-agentic-grpo-dispute-agent-design.md`，当前 SHA-256 为 `C476A5B6661B864A4EB48DB7EF9116C74AD1D5AFAE0D8963572AD801B529DA64`。
- 当前目录不是 Git 仓库，因此 Task 0 先建立可恢复的基线；完成 Task 0 后，从基线分支创建独立 worktree，再执行 Task 1 及以后任务。
- 训练目标环境固定为 Ubuntu 22.04、Python 3.11、CUDA 12.x、两张 24GB RTX 4090。Windows 本地只运行数据、领域、奖励和评测单元测试。
- 不提交模型权重、完整数据或密钥。提交生成器、少量样例、固定配置、锁文件、日志摘要、图表和脱敏 Trace。
- 所有新增行为先写失败测试，再写最小实现。每个任务的提交必须只包含该任务列出的文件。
- 旧 `PlatformAgent`、ms-swift 和“预执行工具后做单步 RL”的链路仅在新链路有等价测试后删除。

## 目标目录

```text
configs/
  sft.yaml
  grpo.yaml
  evaluation.yaml
constraints/
  train.txt
dispute_agent/
  domain/{__init__.py,schemas.py,policies.py}
  data/{__init__.py,generator.py,renderer.py,splits.py,validators.py}
  tools/{__init__.py,registry.py,simulators.py}
  environment/{__init__.py,episode.py}
  rewards/{__init__.py,engine.py,business_utility.py}
  agent/{__init__.py,provider.py,tools.py,runtime.py,tracing.py}
  training/{__init__.py,sft_data.py,train_sft.py,lightning_agent.py,grpo_config.py,monitor.py}
  evaluation/{__init__.py,metrics.py,bootstrap.py,runner.py}
  api/{__init__.py,app.py}
scripts/{generate_data.py,phase0_smoke.py,train_sft.py,train_agentic_grpo.py,evaluate.py}
tests/
  unit/{domain,tools,rewards,data,agent,training}/
  integration/
  leakage/
  evaluation/
```

## Task 0：建立 Git 基线和可复现开发环境

**Files:**

- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `constraints/train.txt`
- Modify: `requirements.txt`
- Test: `tests/test_project_contract.py`

- [ ] **Step 1: 先写忽略规则，再初始化仓库并记录旧原型基线**

先创建 `.gitignore`，覆盖 `.venv/`、`__pycache__/`、`.pytest_cache/`、`data/generated/`、`checkpoints/`、`wandb/`、`.env`、模型文件后缀、`artifacts/**/*.parquet`、`artifacts/**/raw/` 和 `artifacts/**/checkpoints/`。不要忽略 `artifacts/phase0/*.json|txt`、评测 Markdown 和图表，因为它们是可审计证据。

Run:

```powershell
git init
git add .
git commit -m "chore: checkpoint legacy dispute prototype"
```

Expected: `git status --short` 无输出；旧原型可通过首个 commit 恢复。

- [ ] **Step 2: 写项目契约失败测试**

```python
# tests/test_project_contract.py
from pathlib import Path
import tomllib


ROOT = Path(__file__).parents[1]


def test_python_and_training_contract_are_declared():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    assert project["project"]["requires-python"] == ">=3.11,<3.12"
    pins = (ROOT / "constraints/train.txt").read_text("utf-8")
    for pin in (
        "torch==2.8.0",
        "vllm==0.10.2",
        "verl==0.5.0",
        "agentlightning==0.3.0",
        "openai-agents==0.6.0",
    ):
        assert pin in pins
```

- [ ] **Step 3: 运行测试并确认失败**

Run: `python -m pytest tests/test_project_contract.py -q`

Expected: FAIL，提示 `pyproject.toml` 或 `constraints/train.txt` 不存在。

- [ ] **Step 4: 声明本地依赖、训练兼容矩阵和忽略项**

`pyproject.toml` 至少包含：

```toml
[project]
name = "dispute-resolve-agent"
version = "0.1.0"
requires-python = ">=3.11,<3.12"
dependencies = [
  "pydantic>=2.11,<3",
  "numpy>=2.0,<3",
  "pyyaml>=6.0,<7",
  "datasets>=4.0,<5",
  "pyarrow>=20,<22",
  "scikit-learn>=1.7,<2",
  "matplotlib>=3.10,<4",
  "fastapi>=0.116,<1",
  "uvicorn>=0.35,<1",
]

[project.optional-dependencies]
dev = ["pytest>=8.4,<9", "pytest-asyncio>=1.1,<2", "httpx>=0.28,<1"]
train = [
  "torch==2.8.0",
  "torchvision==0.23.0",
  "transformers>=4.55,<5",
  "trl>=0.20,<1",
  "peft>=0.17,<1",
  "accelerate>=1.10,<2",
  "vllm==0.10.2",
  "verl==0.5.0",
  "openai-agents==0.6.0",
  "agentlightning==0.3.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
```

`constraints/train.txt` 固定 Phase 0 的初始兼容矩阵：

```text
torch==2.8.0
torchvision==0.23.0
vllm==0.10.2
verl==0.5.0
agentlightning==0.3.0
openai-agents==0.6.0
```

补全 Step 1 的 `.gitignore`，但保留脱敏报告和图表可提交。`requirements.txt` 只保留一行 `-e .[dev]`，训练机用 `uv pip install -e ".[dev,train]" -c constraints/train.txt`。

- [ ] **Step 5: 运行测试并生成锁定证据**

Run:

```powershell
python -m pytest tests/test_project_contract.py -q
python -m pytest tests -q
```

Expected: 项目契约 PASS；旧 9 个测试仍 PASS。Phase 0 成功后再将训练机的 `uv pip freeze` 保存为 `artifacts/phase0/environment.txt`，不要在未验证的平台上伪造锁文件。

- [ ] **Step 6: 提交并切换独立 worktree**

```powershell
git add .gitignore pyproject.toml requirements.txt constraints/train.txt tests/test_project_contract.py
git commit -m "chore: establish reproducible project baseline"
git worktree add ..\dispute-agent-implementation -b feat/agentic-grpo
```

Expected: 后续任务在 `..\dispute-agent-implementation` 中执行。

## Task 1：建立公开/隐藏领域协议和判别联合决策 Schema

**Files:**

- Create: `dispute_agent/domain/__init__.py`
- Create: `dispute_agent/domain/schemas.py`
- Create: `dispute_agent/domain/policies.py`
- Test: `tests/unit/domain/test_schemas.py`
- Test: `tests/leakage/test_hidden_state.py`

- [ ] **Step 1: 写决定与升级互斥的失败测试**

```python
# tests/unit/domain/test_schemas.py
import pytest
from pydantic import TypeAdapter, ValidationError
from dispute_agent.domain.schemas import Decision, Escalation, TerminalDecision


adapter = TypeAdapter(TerminalDecision)


def test_decide_requires_liability_and_compensation():
    value = adapter.validate_python({
        "action": "decide", "liability": "merchant", "compensation": 89.0,
        "confidence": 0.91, "evidence_ids": ["logistics:1"], "reason": "物流签收异常",
    })
    assert isinstance(value, Decision)


def test_escalate_rejects_liability_and_compensation():
    with pytest.raises(ValidationError):
        adapter.validate_python({
            "action": "escalate", "liability": "buyer", "compensation": 0,
            "confidence": 0.3, "evidence_ids": ["chat:2"], "reason": "证据冲突",
        })
```

- [ ] **Step 2: 写隐藏状态隔离失败测试**

```python
# tests/leakage/test_hidden_state.py
from dispute_agent.domain.schemas import DisputeGroundTruth, DisputeObservation


def test_public_observation_has_no_ground_truth_fields():
    forbidden = set(DisputeGroundTruth.model_fields) & set(DisputeObservation.model_fields)
    assert forbidden == {"case_id"}
    schema = DisputeObservation.model_json_schema()
    text = str(schema).lower()
    assert "true_liability" not in text
    assert "should_escalate" not in text
    assert "tool_information_value" not in text
```

- [ ] **Step 3: 运行测试并确认导入失败**

Run: `python -m pytest tests/unit/domain/test_schemas.py tests/leakage/test_hidden_state.py -q`

Expected: FAIL，`dispute_agent.domain` 不存在。

- [ ] **Step 4: 实现最小领域模型**

在 `schemas.py` 定义 `Liability`、`Evidence`、`DisputeObservation`、`DisputeGroundTruth`、`ToolResult`、`Decision`、`Escalation`。终止类型必须是：

```python
from typing import Annotated, Literal
from pydantic import BaseModel, Field


class Decision(BaseModel):
    action: Literal["decide"]
    liability: Liability
    compensation: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[str] = Field(min_length=1)
    reason: str = Field(min_length=1, max_length=500)


class Escalation(BaseModel):
    action: Literal["escalate"]
    liability: None = None
    compensation: None = None
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[str] = Field(min_length=1)
    reason: str = Field(min_length=1, max_length=500)


TerminalDecision = Annotated[Decision | Escalation, Field(discriminator="action")]
```

`policies.py` 集中定义赔付上限、最大 5 轮、最多 4 次调查调用、连续 2 次非法动作升级等常量；禁止这些规则散落在 Runtime 和 Reward 中。

- [ ] **Step 5: 运行测试并提交**

Run: `python -m pytest tests/unit/domain tests/leakage/test_hidden_state.py -q`

Expected: PASS。

```powershell
git add dispute_agent/domain tests/unit/domain tests/leakage/test_hidden_state.py
git commit -m "feat: define isolated dispute domain contracts"
```

## Task 2：实现确定性工具仿真和多轮 Episode 状态机

**Files:**

- Create: `dispute_agent/tools/__init__.py`
- Create: `dispute_agent/tools/registry.py`
- Create: `dispute_agent/tools/simulators.py`
- Create: `dispute_agent/environment/__init__.py`
- Create: `dispute_agent/environment/episode.py`
- Test: `tests/unit/tools/test_simulators.py`
- Test: `tests/integration/test_episode_state.py`

- [ ] **Step 1: 写确定性、缓存和成本失败测试**

```python
# tests/unit/tools/test_simulators.py
def test_same_case_tool_and_arguments_return_same_result(tool_registry, episode):
    first = tool_registry.execute(episode, "check_logistics", {"order_id": "o-1"})
    second = tool_registry.execute(episode, "check_logistics", {"order_id": "o-1"})
    assert first.payload == second.payload
    assert second.cached is True
    assert episode.invalid_actions.repeat_calls == 1
    assert "true_liability" not in str(first.payload).lower()
```

- [ ] **Step 2: 写终止后不可继续动作失败测试**

```python
# tests/integration/test_episode_state.py
import pytest


def test_terminal_decision_closes_episode(episode, valid_decision):
    episode.submit(valid_decision)
    assert episode.done and episode.end_reason == "submitted"
    with pytest.raises(RuntimeError, match="already terminated"):
        episode.record_tool_call("check_logistics", {"order_id": "o-1"})
```

- [ ] **Step 3: 运行测试并确认失败**

Run: `python -m pytest tests/unit/tools/test_simulators.py tests/integration/test_episode_state.py -q`

Expected: FAIL，工具注册表和 Episode 尚未实现。

- [ ] **Step 4: 实现工具注册表与状态机**

`ToolRegistry` 只注册四个调查工具：`check_logistics`、`check_buyer_history`、`check_merchant_history`、`verify_evidence`。用规范化 JSON 参数和 `sha256(case_id + tool_name + arguments + case_seed)` 生成确定性随机源。`EpisodeState` 保存公开 observation、可见 evidence、调用历史、缓存、累计成本、轮数、非法动作和终止结果；Ground Truth 作为私有环境依赖，不暴露在 `model_dump()`、Prompt 或工具参数中。

每次工具调用按以下顺序执行：验证 Episode 未终止 → 参数 Schema 校验 → 检查预算 → 命中缓存或仿真 → 将新 evidence 加入可见集合 → 累计成本与轮数。重复调用返回缓存、成本不重复收费，但记一次重复动作。

- [ ] **Step 5: 补充组内一致与组间隔离测试**

```python
def test_grpo_group_shares_tool_outcomes_but_not_mutable_state(make_episode):
    episodes = [make_episode(case_id="case-7", group_seed=42) for _ in range(4)]
    results = [e.call("verify_evidence", {"evidence_id": "ev-1"}) for e in episodes]
    assert len({r.model_dump_json() for r in results}) == 1
    episodes[0].invalid_actions.illegal_calls += 1
    assert episodes[1].invalid_actions.illegal_calls == 0
```

- [ ] **Step 6: 运行测试并提交**

Run: `python -m pytest tests/unit/tools tests/integration/test_episode_state.py -q`

Expected: PASS。

```powershell
git add dispute_agent/tools dispute_agent/environment tests/unit/tools tests/integration/test_episode_state.py
git commit -m "feat: add deterministic tools and episode state machine"
```

## Task 3：实现可审计的训练奖励与离线业务效用

**Files:**

- Create: `configs/evaluation.yaml`
- Create: `dispute_agent/rewards/__init__.py`
- Create: `dispute_agent/rewards/engine.py`
- Create: `dispute_agent/rewards/business_utility.py`
- Test: `tests/unit/rewards/test_engine.py`
- Test: `tests/unit/rewards/test_business_utility.py`

- [ ] **Step 1: 写硬惩罚和分支公式失败测试**

```python
# tests/unit/rewards/test_engine.py
import pytest
from dispute_agent.rewards.engine import RewardEngine


def test_decide_reward_uses_approved_weights(decide_episode):
    result = RewardEngine().score(decide_episode)
    expected = (
        0.45 * result.components.liability
        + 0.25 * result.components.compensation
        + 0.15 * result.components.escalation
        + 0.15 * result.components.grounding
        - 0.10 * result.components.normalized_tool_cost
        + result.components.invalid_action_penalty
    )
    assert result.total == pytest.approx(max(-1.5, min(1.0, expected)))


def test_missing_terminal_decision_is_hard_failure(unfinished_episode):
    assert RewardEngine().score(unfinished_episode).total == -1.5
```

- [ ] **Step 2: 写奖励排序失败测试**

```python
def test_handcrafted_reward_ranking(reward_fixture):
    names = [
        "correct_efficient", "correct_redundant", "correct_overpay",
        "reasonable_escalation", "wrong_liability", "invalid_output",
    ]
    scores = [RewardEngine().score(reward_fixture(name)).total for name in names]
    assert scores == sorted(scores, reverse=True)
```

- [ ] **Step 3: 运行测试并确认失败**

Run: `python -m pytest tests/unit/rewards -q`

Expected: FAIL，奖励模块不存在。

- [ ] **Step 4: 实现决定/升级两套奖励**

`RewardBreakdown` 明确保存 `liability`、`compensation`、`escalation`、`grounding`、`escalation_quality`、`normalized_tool_cost`、`invalid_action_penalty`、`hard_failure` 和 `total`。两条分支精确为：

```python
decide_total = (
    0.45 * liability
    + 0.25 * compensation
    + 0.15 * escalation
    + 0.15 * grounding
    - 0.10 * normalized_tool_cost
    + invalid_action_penalty
)
escalate_total = (
    0.60 * escalation
    + 0.25 * grounding
    + 0.15 * escalation_quality
    - 0.10 * normalized_tool_cost
    + invalid_action_penalty
)
```

责任完全正确为 1，单方责任与双方共担相邻为 0.4，买卖双方判反为 -1，其他错误为 -0.3；赔付在合理区间为 1，离开区间后按相对订单金额线性降至 -1；升级标签一致为 1、否则 -1；grounding 按有效可见证据引用比例计分，无引用但需要证据为 -0.5，引用不存在/不可见证据为 -1；升级质量只用结构化风险规则给 1/0/-1。每次非法调用 `-0.2`、每次重复调用 `-0.1`，动作惩罚裁剪至 `[-0.4, 0]`，总奖励裁剪至 `[-1.5, 1]`。非法终止 Schema、负/越权赔付、矛盾升级和未终止直接返回 `-1.5`。

训练奖励不得导入 `business_utility.py`，不得使用 Oracle 的工具信息价值，也不得调用 LLM Judge。

- [ ] **Step 5: 实现离线业务效用并测试升级语义**

```python
# tests/unit/rewards/test_business_utility.py
def test_escalation_uses_oracle_outcome_and_charges_review_and_delay(config, case):
    utility = score_business_utility(case, escalation(case), config)
    direct_oracle = score_business_utility(case, oracle_decision(case), config)
    assert utility.raw == pytest.approx(
        direct_oracle.raw - config.manual_review_cost - config.review_delay_cost
    )
```

实现固定公式并以 RuleBased/Oracle 锚点归一化；在本任务创建 `configs/evaluation.yaml` 的 `business_utility` 段，明确 buyer/merchant LTV、复购/留存映射、人工审核、延迟、风险、口碑和工具成本参数。首次训练前连同测试 manifest 一起冻结。

- [ ] **Step 6: 运行测试并提交**

Run: `python -m pytest tests/unit/rewards -q`

Expected: PASS，且测试输出无 NaN/Inf。

```powershell
git add configs/evaluation.yaml dispute_agent/rewards tests/unit/rewards
git commit -m "feat: implement auditable rewards and offline utility"
```

## Task 4：重建合成数据、标准工具消息、隔离切分与冻结清单

**Files:**

- Create: `dispute_agent/data/__init__.py`
- Create: `dispute_agent/data/generator.py`
- Create: `dispute_agent/data/renderer.py`
- Create: `dispute_agent/data/splits.py`
- Create: `dispute_agent/data/validators.py`
- Rewrite: `scripts/generate_data.py`
- Test: `tests/unit/data/test_splits.py`
- Test: `tests/unit/data/test_messages.py`
- Test: `tests/leakage/test_dataset_leakage.py`

- [ ] **Step 1: 写标准工具消息失败测试**

```python
# tests/unit/data/test_messages.py
def test_sft_trace_uses_native_tool_protocol(rendered_trace):
    assistant_call = next(m for m in rendered_trace if m["role"] == "assistant" and m.get("tool_calls"))
    call_id = assistant_call["tool_calls"][0]["id"]
    tool_message = next(m for m in rendered_trace if m["role"] == "tool")
    assert tool_message["tool_call_id"] == call_id
    assert not any(m["role"] == "user" and "tool_result" in m.get("content", "") for m in rendered_trace)
```

- [ ] **Step 2: 写结构化隔离和 OOD 配额失败测试**

```python
# tests/unit/data/test_splits.py
def test_exact_dataset_sizes_and_ood_buckets(dataset_manifest):
    assert dataset_manifest.counts == {
        "sft_train": 1500, "sft_val": 150, "grpo_train": 700,
        "grpo_val": 100, "id_test": 400, "ood_test": 200,
    }
    assert dataset_manifest.ood_counts == {
        "unseen_combination": 80, "language_style": 60, "tool_noise": 60,
    }
    assert dataset_manifest.human_audit.count == 100


def test_fact_instances_do_not_cross_splits(dataset_manifest):
    instances = [set(split.fact_instance_ids) for split in dataset_manifest.splits.values()]
    assert all(not left & right for i, left in enumerate(instances) for right in instances[i + 1:])
```

- [ ] **Step 3: 运行测试并确认失败**

Run: `python -m pytest tests/unit/data tests/leakage/test_dataset_leakage.py -q`

Expected: FAIL，新数据层尚未实现。

- [ ] **Step 4: 实现结构化生成、渲染和验证管线**

保留旧 `case_generator.py` 中可复用的场景采样思想，但输出 Task 1 的新 Schema。生成步骤固定为：场景变量 → Ground Truth → public observation → 确定性工具结果 → 模板/教师端点渲染 → Schema、金额、泄漏、近重复和轨迹可执行性校验。教师端点只改写语言，不产生标签。

SFT 训练精确分成 800 条直接判责、500 条 1–4 次工具轨迹、200 条升级/失败/非法恢复/高风险轨迹。工具调用使用 assistant `tool_calls`，结果使用 `role=tool`；Reason 只包含可核验证据，不生成隐藏思维链。

- [ ] **Step 5: 实现不可变 manifest 与人工抽查流程**

`scripts/generate_data.py` 接受 `--seed 20260817 --output data/generated`，写 Parquet/JSONL、`manifest.json` 和每个文件的 SHA-256。人工审核从 ID/OOD 分层抽取 100 条，审核字段固定为事实自洽、判责合理、赔付合理、工具必要、语言自然。审核修正后执行 `--freeze-test`; 若冻结文件已存在且哈希变化，命令必须非零退出。

- [ ] **Step 6: 生成小型测试 fixture 并验证**

Run:

```powershell
python scripts/generate_data.py --seed 20260817 --fixture-size 24 --output artifacts/data-smoke
python -m pytest tests/unit/data tests/leakage -q
```

Expected: 24 条 fixture 通过全部校验，`manifest.json` 包含种子、Schema 版本、生成配置与文件哈希；正式配额测试通过确定性 dry-run manifest，不要求本地生成全部教师文本。

- [ ] **Step 7: 提交**

```powershell
git add dispute_agent/data scripts/generate_data.py tests/unit/data tests/leakage
git commit -m "feat: build leakage-safe synthetic data pipeline"
```

## Task 5：用 OpenAI Agents SDK 实现唯一的多轮 Agent Runtime

**Files:**

- Create: `dispute_agent/agent/__init__.py`
- Create: `dispute_agent/agent/provider.py`
- Create: `dispute_agent/agent/tools.py`
- Create: `dispute_agent/agent/runtime.py`
- Test: `tests/unit/agent/test_provider.py`
- Test: `tests/integration/test_agent_episode.py`

- [ ] **Step 1: 用假 OpenAI 兼容服务写 provider 往返失败测试**

测试响应依次返回 `check_logistics` tool call 和 `submit_decision` tool call，断言 Runtime 将 tool result 以标准协议回填，并且没有把 Ground Truth 序列化到任何请求。

```python
@pytest.mark.asyncio
async def test_runtime_executes_tool_then_terminal_decision(fake_model_server, episode):
    runtime = build_runtime(base_url=fake_model_server.url, api_key="test")
    result = await runtime.run(episode, enable_thinking=True)
    assert [c.name for c in episode.tool_calls] == ["check_logistics"]
    assert result.action == "decide"
    assert fake_model_server.requests_contain("role", "tool")
    assert not fake_model_server.requests_contain_text("true_liability")
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/unit/agent/test_provider.py tests/integration/test_agent_episode.py -q`

Expected: FAIL，SDK Runtime 尚不存在。

- [ ] **Step 3: 实现 provider 与函数工具**

使用 `AsyncOpenAI(base_url=..., api_key=...)` 和 OpenAI Agents SDK 的 Chat Completions model adapter。provider 必须由调用方显式传入 base URL：训练传 Agent Lightning `main_llm` ProxyLLM 地址，独立评测传普通 vLLM 地址。`enable_thinking` 通过 Qwen chat template 参数控制，不在 Prompt 中模拟。

四个调查函数工具只调用 Task 2 的 Registry。`submit_decision` 使用 Task 1 的判别联合类型，在工具级 Guardrail 中验证赔付上限和 evidence 可见性，通过后写入 Episode。

- [ ] **Step 4: 先做 SDK 终止语义的表征测试**

用固定假模型确认当前固定版本中“终止工具”是否由 `StopAtTools` 或等价 API 完成，并断言 `submit_decision` 后模型调用次数不再增加。将验证过的 SDK API 写进实现，禁止猜测私有接口。

- [ ] **Step 5: 实现输入/工具/输出 Guardrail 与异常策略**

输入 Guardrail 检查字段、金额和权限；工具超时只重试一次；连续两次非法调用、最大轮数、高风险或越权赔付升级人工；格式修复只尝试一次；模型服务失败返回带 `fallback=True` 的规则结果。输出 Guardrail 断言对外响应与 Episode terminal decision 完全一致。

- [ ] **Step 6: 运行测试并提交**

Run: `python -m pytest tests/unit/agent tests/integration/test_agent_episode.py -q`

Expected: PASS；`submit_decision` 后无额外模型调用。

```powershell
git add dispute_agent/agent tests/unit/agent tests/integration/test_agent_episode.py
git commit -m "feat: build OpenAI Agents SDK dispute runtime"
```

## Task 6：建立单一训练 Trace 来源和 Agent Lightning rollout 适配

**Files:**

- Create: `dispute_agent/agent/tracing.py`
- Create: `dispute_agent/training/__init__.py`
- Create: `dispute_agent/training/lightning_agent.py`
- Test: `tests/unit/agent/test_tracing.py`
- Test: `tests/integration/test_lightning_rollout.py`

- [ ] **Step 1: 写 span 去重和单奖励失败测试**

```python
# tests/unit/agent/test_tracing.py
def test_only_proxy_llm_spans_are_trainable(trace):
    trainable = select_trainable_spans(trace, model_resource="main_llm")
    assert len(trainable) == trace.proxy_model_call_count
    assert all(span.token_ids for span in trainable)
    assert all(span.source == "proxy_llm" for span in trainable)
    assert not any(span.source == "sdk_runtime" for span in trainable)


def test_rollout_returns_reward_once(lightning_rollout):
    result = lightning_rollout.run()
    assert isinstance(result.reward, float)
    assert lightning_rollout.emitted_reward_count == 0
    assert lightning_rollout.returned_reward_count == 1
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/unit/agent/test_tracing.py tests/integration/test_lightning_rollout.py -q`

Expected: FAIL，Tracing/Lightning adapter 尚不存在。

- [ ] **Step 3: 实现本地 SDK 运行 Trace**

关闭 SDK 默认云端 Trace 导出。`RuntimeTraceRecorder` 只记录工具、Guardrail、Episode 状态转换、延迟、终止原因、成本和奖励分量；不保存 thinking 内容，不把这些事件声明成可训练模型 span。训练和评测使用不同 namespace。

- [ ] **Step 4: 实现 Agent Lightning rollout**

`LitDisputeAgent.rollout(task, resources, rollout)` 从 `resources["main_llm"]` 获取与当前 rollout/attempt 绑定的 ProxyLLM 地址，构造 Task 5 Runtime，运行一个完整 Episode，然后调用 Task 3 RewardEngine。函数只 `return` 一次最终标量奖励；奖励分量写 annotation，不调用第二个 reward API。

每个任务仅包含 `case_id` 和公开 Observation；Ground Truth 由进程内环境仓库按 `case_id` 读取，绝不放进 Lightning task payload。

- [ ] **Step 5: 增加四 rollout 聚合测试**

固定同一 `case_id`、相同工具结果、四个不同采样 seed；断言存储中四条轨迹属于同一 group，且模型 span 都带 rollout/attempt 标识、token IDs 和 model version。

- [ ] **Step 6: 运行测试并提交**

Run: `python -m pytest tests/unit/agent/test_tracing.py tests/integration/test_lightning_rollout.py -q`

Expected: PASS，无重复 trainable span，无重复奖励。

```powershell
git add dispute_agent/agent/tracing.py dispute_agent/training tests/unit/agent/test_tracing.py tests/integration/test_lightning_rollout.py
git commit -m "feat: adapt dispute episodes to Agent Lightning traces"
```

## Task 7：实现 non-thinking、assistant-only 的 TRL BF16 LoRA SFT

**Files:**

- Create: `configs/sft.yaml`
- Create: `dispute_agent/training/sft_data.py`
- Create: `dispute_agent/training/train_sft.py`
- Rewrite: `scripts/train_sft.py`
- Test: `tests/unit/training/test_sft_data.py`
- Test: `tests/unit/training/test_sft_config.py`

- [ ] **Step 1: 写 loss mask 和工具协议失败测试**

```python
# tests/unit/training/test_sft_data.py
def test_only_assistant_actions_have_labels(preprocessed_example, tokenizer):
    labels = preprocessed_example["labels"]
    spans = preprocessed_example["message_spans"]
    for span in spans:
        supervised = any(token != -100 for token in labels[span.start:span.end])
        assert supervised is (span.role == "assistant")
    assert preprocessed_example["chat_template_kwargs"]["enable_thinking"] is False
```

- [ ] **Step 2: 写 LoRA 配置失败测试**

```python
# tests/unit/training/test_sft_config.py
def test_sft_is_bf16_lora_not_qlora(load_sft_config):
    cfg = load_sft_config()
    assert cfg.model == "Qwen/Qwen3-8B"
    assert cfg.bf16 is True and cfg.load_in_4bit is False and cfg.load_in_8bit is False
    assert (cfg.lora.rank, cfg.lora.alpha, cfg.lora.dropout) == (32, 64, 0)
    assert cfg.lora.target_modules == "all-linear"
    assert cfg.assistant_only_loss is True
```

- [ ] **Step 3: 运行测试并确认失败**

Run: `python -m pytest tests/unit/training/test_sft_data.py tests/unit/training/test_sft_config.py -q`

Expected: FAIL，SFT 新实现不存在。

- [ ] **Step 4: 实现配置与预处理**

`configs/sft.yaml` 固定 Qwen3-8B、BF16、LoRA r=32/alpha=64/dropout=0/all-linear、gradient checkpointing、seed=20260817、最大长度和小微批次。数据预处理调用 tokenizer 原生 chat template，传 `enable_thinking=False`，并用 assistant token mask 只监督 assistant tool call、工具结果后的 assistant 动作和最终决策；system/user/tool 全部为 `-100`。

- [ ] **Step 5: 实现三个嵌套规模的训练入口**

`scripts/train_sft.py --train-size {500,1000,1500}` 只允许三个值；500 是 1000 的前缀子集，1000 是 1500 的前缀子集，三者使用同一验证集、seed 和有效优化步数口径。使用 TRL `SFTTrainer` 和 PEFT `LoraConfig`，只保存 adapter、tokenizer、resolved config、数据 hash 和验证指标。

- [ ] **Step 6: 用 tiny fixture 做 CPU dry-run**

Run:

```powershell
python scripts/train_sft.py --config configs/sft.yaml --fixture --max-steps 1
python -m pytest tests/unit/training/test_sft_data.py tests/unit/training/test_sft_config.py -q
```

Expected: 预处理与配置测试 PASS；fixture 产生 adapter 目录，不下载或训练 Qwen3-8B。

- [ ] **Step 7: 提交**

```powershell
git add configs/sft.yaml dispute_agent/training/sft_data.py dispute_agent/training/train_sft.py scripts/train_sft.py tests/unit/training
git commit -m "feat: add TRL BF16 LoRA SFT pipeline"
```

## Task 8：建立双 4090 Phase 0 兼容性门禁

**Files:**

- Create: `scripts/phase0_smoke.py`
- Create: `tests/integration/test_phase0_contract.py`
- Create at runtime: `artifacts/phase0/report.json`
- Create at runtime: `artifacts/phase0/environment.txt`
- Create at runtime: `constraints/phase0-lock.txt`

- [ ] **Step 1: 写七项门禁失败测试**

```python
# tests/integration/test_phase0_contract.py
REQUIRED = {
    "sdk_vllm_multiturn", "thinking_tool_compatibility", "trace_complete",
    "trl_adapter_loaded_by_verl", "grpo_update_reload", "dual_gpu_no_oom",
    "single_model_span_and_reward",
}


def test_phase0_report_passes_every_gate(phase0_report):
    assert set(phase0_report["gates"]) == REQUIRED
    assert all(gate["passed"] for gate in phase0_report["gates"].values())
    assert phase0_report["gpu_count"] == 2
    assert phase0_report["quantization"] == "none"
```

- [ ] **Step 2: 实现可恢复的 Phase 0 runner**

脚本使用同系列小 Qwen3 模型和 20 个固定案例；每个 gate 单独写状态、开始/结束时间、日志路径和失败原因，所以网络或云实例中断后可从未通过 gate 恢复。adapter 接续 gate 必须比较 TRL adapter 被 VERL 加载前后的固定样例输出，并验证一次 GRPO 更新后的 adapter 可重新加载。

- [ ] **Step 3: 在云端创建环境并记录版本**

Run on Ubuntu 22.04/Python 3.11:

```bash
uv pip install -e ".[dev,train]" -c constraints/train.txt
uv pip freeze > constraints/phase0-lock.txt
cp constraints/phase0-lock.txt artifacts/phase0/environment.txt
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
```

Expected: 两行 RTX 4090、每张约 24GB；环境清单包含固定的 Agent Lightning、OpenAI Agents SDK、VERL、vLLM 和 torch 版本。

- [ ] **Step 4: 启动 Qwen tool serving 并跑门禁**

独立 vLLM 表征测试使用：

```bash
vllm serve Qwen/Qwen3-0.6B --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser deepseek_r1
```

随后由脚本启动 Agent Lightning/VERL 管理的训练 rollout：

```bash
torchrun --nproc-per-node=2 scripts/phase0_smoke.py --cases 20 --report artifacts/phase0/report.json
pytest tests/integration/test_phase0_contract.py -q
```

Expected: 七项全部 PASS，至少一个 BF16 LoRA 参数在 update 前后数值改变，checkpoint 可重载，两张 GPU 无 OOM；同一 rollout 仅有 ProxyLLM 模型 span，reward 只返回一次。

- [ ] **Step 5: 处理门禁失败而不扩大范围**

按顺序只调整上下文长度、rollout 并发、KV cache 占用、micro-batch、gradient accumulation，必要时启用参数/优化器 CPU offload；不改 QLoRA。若 Agent Lightning + VERL 关键链路仍不稳定，记录失败证据并将主线单次降级为 TRL 原生 Agentic GRPO，删除未跑通框架的简历表述，不并行维护两套 RL。

- [ ] **Step 6: 提交门禁代码和脱敏结果**

```bash
git add scripts/phase0_smoke.py tests/integration/test_phase0_contract.py constraints/phase0-lock.txt artifacts/phase0/report.json artifacts/phase0/environment.txt
git commit -m "test: verify dual-4090 agentic training compatibility"
```

## Task 9：实现 Agent Lightning + VERL Agentic GRPO 配置和坍缩监控

**Files:**

- Create: `configs/grpo.yaml`
- Create: `dispute_agent/training/grpo_config.py`
- Create: `dispute_agent/training/monitor.py`
- Create: `scripts/train_agentic_grpo.py`
- Test: `tests/unit/training/test_grpo_config.py`
- Test: `tests/unit/training/test_monitor.py`

- [ ] **Step 1: 写 adapter 接续和双卡拓扑失败测试**

```python
# tests/unit/training/test_grpo_config.py
def test_grpo_continues_from_sft_adapter(load_grpo_config):
    cfg = load_grpo_config()
    assert cfg.algorithm.adv_estimator == "grpo"
    assert cfg.actor_rollout_ref.model.lora_adapter_path.endswith("sft-1500-best")
    assert cfg.actor_rollout_ref.model.lora_rank == 32
    assert cfg.actor_rollout_ref.model.lora_alpha == 64
    assert cfg.actor_rollout_ref.rollout.n == 4
    assert cfg.actor_rollout_ref.rollout.tensor_model_parallel_size == 2
    assert cfg.trainer.n_gpus_per_node == 2
    assert cfg.quantization is None
```

- [ ] **Step 2: 写零方差暂停失败测试**

```python
# tests/unit/training/test_monitor.py
def test_monitor_pauses_after_fifty_bad_steps():
    monitor = CollapseMonitor(window=50, max_zero_variance_ratio=0.30)
    for _ in range(50):
        monitor.observe(group_rewards=[0.4, 0.4, 0.4, 0.4], valid_rollouts=4)
    assert monitor.should_pause is True
    assert monitor.reason == "zero_variance_group_ratio"
```

- [ ] **Step 3: 运行测试并确认失败**

Run: `python -m pytest tests/unit/training/test_grpo_config.py tests/unit/training/test_monitor.py -q`

Expected: FAIL，GRPO 配置与监控尚不存在。

- [ ] **Step 4: 实现正式 GRPO 配置解析和启动前校验**

`configs/grpo.yaml` 固定：`adv_estimator=grpo`、rollout `n=4`、thinking 开启、单轮最多 384 tokens、Episode 最多 1,280 生成 tokens、LoRA r=32/alpha=64、`lora_adapter_path` 指向 SFT-1500 best 的只读副本、FSDP actor/reference、TP=2 rollout、两 GPU、无 reward model、无量化。启动器必须验证 tokenizer/chat template 与 SFT manifest 一致，原 SFT adapter 不可写，GRPO 输出是副本目录。

- [ ] **Step 5: 实现课程和监控**

课程阶段 1 只开放物流/买家历史/商家历史、最多 3 轮；稳定后开放证据核验并扩展至 5 轮。稳定条件由配置中的有效 rollout、Episode 成功率和 group reward std 阈值决定。每 step 保存 `group_reward_std`、零方差组比例、有效 rollout、Episode 成功率、KL、生成长度、工具数和每个奖励分量；50-step 窗口零方差比例超过 30% 时保存 checkpoint 并暂停。

- [ ] **Step 6: 增加奖励消融入口**

`--ablation full` 与 `--ablation no-tool-cost` 只改变工具成本系数；数据、seed、SFT adapter、课程和其他权重保持一致。启动日志打印 resolved config diff，防止消融误改其他参数。

- [ ] **Step 7: 运行配置 dry-run 与测试**

Run:

```powershell
python scripts/train_agentic_grpo.py --config configs/grpo.yaml --dry-run
python -m pytest tests/unit/training/test_grpo_config.py tests/unit/training/test_monitor.py -q
```

Expected: dry-run 输出两 GPU hybrid-engine 计划、SFT adapter 路径、n=4 和课程阶段；测试 PASS。

- [ ] **Step 8: 提交**

```powershell
git add configs/grpo.yaml dispute_agent/training/grpo_config.py dispute_agent/training/monitor.py scripts/train_agentic_grpo.py tests/unit/training
git commit -m "feat: configure monitored Agent Lightning VERL GRPO"
```

## Task 10：实现统一公平评测、Bootstrap 和报告产物

**Files:**

- Modify: `configs/evaluation.yaml`
- Create: `dispute_agent/evaluation/__init__.py`
- Create: `dispute_agent/evaluation/metrics.py`
- Create: `dispute_agent/evaluation/bootstrap.py`
- Create: `dispute_agent/evaluation/runner.py`
- Rewrite: `scripts/evaluate.py`
- Test: `tests/evaluation/test_metrics.py`
- Test: `tests/evaluation/test_fair_protocol.py`

- [ ] **Step 1: 写固定样例指标失败测试**

```python
# tests/evaluation/test_metrics.py
def test_metrics_cover_task_business_agent_and_safety(fixed_predictions):
    report = compute_metrics(fixed_predictions)
    assert report.liability.macro_f1 == pytest.approx(0.666666, rel=1e-5)
    assert report.business.compensation_mae == pytest.approx(25.0)
    assert 0 <= report.agent.necessary_tool_recall <= 1
    assert 0 <= report.safety.escalation_f1 <= 1
    assert report.safety.evidence_hallucination_rate == 0
```

- [ ] **Step 2: 写公平协议失败测试**

```python
# tests/evaluation/test_fair_protocol.py
def test_all_model_variants_share_runtime_contract(resolved_runs):
    fields = ("case_hash", "tool_registry_hash", "max_rounds", "tool_budget", "terminal_schema_hash")
    baseline = {field: getattr(resolved_runs[0], field) for field in fields}
    assert all({field: getattr(run, field) for field in fields} == baseline for run in resolved_runs[1:])
```

- [ ] **Step 3: 运行测试并确认失败**

Run: `python -m pytest tests/evaluation -q`

Expected: FAIL，评测层尚不存在。

- [ ] **Step 4: 实现指标与分层 Bootstrap**

报告判责 Accuracy/Macro-F1/每类 Precision/Recall/混淆矩阵；赔付 MAE、过赔/少赔、归一化业务效用；Episode 成功、调用数/成本、必要工具召回、无效/重复/超限/失败恢复；升级 P/R/F1、非法决策、证据幻觉、越权赔付和低置信错误。按案例进行固定 seed Bootstrap，输出 95% CI，并分别报告 ID、OOD 与人工审核子集。

- [ ] **Step 5: 实现同 Runtime 的模型矩阵**

运行 RuleBased、Base、SFT-500、SFT-1000、SFT-1500、SFT-1500+GRPO、Oracle。模型组全部复用 Task 5 Runtime、同一冻结案例、工具、预算、终止 Schema；只替换 adapter。主实验 thinking 参数固定 `temperature=0.6, top_p=0.95, top_k=20, seed=20260817`。最终 GRPO 另跑 non-thinking：`temperature=0.7, top_p=0.8, top_k=20`。

- [ ] **Step 6: 生成机器可读与简历可引用产物**

Run:

```bash
python scripts/evaluate.py --config configs/evaluation.yaml --models all --output artifacts/evaluation
pytest tests/evaluation -q
```

Expected: `metrics.json`、`summary.md`、`predictions.parquet`、`confusion_matrix.png`、`tool_cost_tradeoff.png`、典型成功/失败 Trace；报告标注单随机种子完整训练，不预填不存在的提升数字。

- [ ] **Step 7: 提交**

```bash
git add configs/evaluation.yaml dispute_agent/evaluation scripts/evaluate.py tests/evaluation
git add artifacts/evaluation/summary.md artifacts/evaluation/*.png
git commit -m "feat: add frozen unified evaluation protocol"
```

## Task 11：增加最小 FastAPI 演示与可审计 Trace 查询

**Files:**

- Create: `dispute_agent/api/__init__.py`
- Create: `dispute_agent/api/app.py`
- Test: `tests/integration/test_api.py`

- [ ] **Step 1: 写 API 失败测试**

```python
# tests/integration/test_api.py
def test_resolve_returns_decision_and_public_trace(client, public_case):
    response = client.post("/v1/disputes/resolve", json=public_case)
    assert response.status_code == 200
    body = response.json()
    assert body["decision"]["action"] in {"decide", "escalate"}
    assert set(body["trace"][0]) >= {"event", "latency_ms"}
    assert "thinking" not in str(body).lower()
    assert "ground_truth" not in str(body).lower()
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/integration/test_api.py -q`

Expected: FAIL，API 尚不存在。

- [ ] **Step 3: 实现两个最小端点**

`POST /v1/disputes/resolve` 调用 Task 5 Runtime；`GET /v1/traces/{trace_id}` 只返回公开工单、工具动作、脱敏工具结果、累计成本、最终决策、升级状态和延迟。不得返回 Ground Truth、奖励内部标签、token IDs 或 thinking 内容。

- [ ] **Step 4: 运行测试和本地 smoke**

Run:

```powershell
python -m pytest tests/integration/test_api.py -q
python -m uvicorn dispute_agent.api.app:app --host 127.0.0.1 --port 8000
```

Expected: 测试 PASS；OpenAPI 页面只有上述两个业务端点和健康检查。手工 smoke 后停止服务。

- [ ] **Step 5: 提交**

```powershell
git add dispute_agent/api tests/integration/test_api.py
git commit -m "feat: expose minimal dispute agent API"
```

## Task 12：删除旧协议、整理文档并执行最终验收

**Files:**

- Delete: `dispute_agent/platform_agent.py`
- Delete: `dispute_agent/data_generation.py`
- Delete: `dispute_agent/verl_reward.py`
- Delete: `dispute_agent/case_generator.py`
- Delete: `dispute_agent/environment.py`
- Delete: `dispute_agent/evaluate.py`
- Delete: `dispute_agent/models.py`
- Delete: `dispute_agent/oracle.py`
- Delete: `dispute_agent/payoff.py`
- Delete: `dispute_agent/prompting.py`
- Delete: `dispute_agent/reward.py`
- Delete: `dispute_agent/tools.py`
- Delete: `scripts/train_sft_ms_swift.sh`
- Delete: `scripts/train_grpo_verl.sh`
- Delete: `scripts/train_rl.py`
- Delete: `scripts/prepare_verl_data.py`
- Delete: `scripts/evaluate_model.py`
- Delete: `main.py`
- Modify: `dispute_agent/__init__.py`
- Rewrite: `README.md`
- Create: `docs/experiments.md`
- Create: `docs/resume-evidence-checklist.md`
- Test: `tests/test_no_legacy_protocol.py`

- [ ] **Step 1: 写旧协议不存在的失败测试**

```python
# tests/test_no_legacy_protocol.py
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_legacy_single_step_protocol_is_removed():
    forbidden_files = [
        "dispute_agent/platform_agent.py", "dispute_agent/data_generation.py",
        "dispute_agent/verl_reward.py", "dispute_agent/tools.py",
        "dispute_agent/environment.py", "scripts/train_sft_ms_swift.sh",
        "scripts/train_grpo_verl.sh", "scripts/train_rl.py",
        "scripts/prepare_verl_data.py", "scripts/evaluate_model.py", "main.py",
    ]
    assert not [path for path in forbidden_files if (ROOT / path).exists()]
    source = "\n".join(p.read_text("utf-8") for p in (ROOT / "dispute_agent").rglob("*.py"))
    assert "Qwen2.5-7B" not in source
    assert '"role": "user", "content": tool_result' not in source
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/test_no_legacy_protocol.py -q`

Expected: FAIL，旧文件仍存在。

- [ ] **Step 3: 删除被替代实现并保留有价值概念**

确认 Task 1–10 全部 PASS 后，删除列出的旧文件。旧 `models.py`、`case_generator.py`、`environment.py`、`oracle.py`、`payoff.py`、`reward.py`、`tools.py` 中仍有价值的枚举、场景分布和成本规则必须先迁入新模块并由新测试覆盖；迁移完成后删除或缩成兼容导出，不能保留第二套业务真相。

- [ ] **Step 4: 重写 README 和实验事实清单**

README 必须包含：问题定义、架构图、数据规模、SFT/GRPO 配置、Phase 0 证据、统一评测命令、真实结果表、局限性、双 4090 复现方式和典型 Trace。`docs/resume-evidence-checklist.md` 对每条简历表述链接到 config、日志或评测产物；未实际跑通的 Agent Lightning/VERL 或指标不得出现。

- [ ] **Step 5: 执行全量本地验收**

Run:

```powershell
python -m pytest -q
python scripts/generate_data.py --seed 20260817 --fixture-size 24 --output artifacts/final-data-smoke
python scripts/train_sft.py --config configs/sft.yaml --fixture --max-steps 1
python scripts/train_agentic_grpo.py --config configs/grpo.yaml --dry-run
```

Expected: 全部测试 PASS；无旧协议引用；三个 smoke 命令成功。

- [ ] **Step 6: 在训练机执行最终验收**

Run:

```bash
pytest -q
pytest tests/integration/test_phase0_contract.py -q
python scripts/evaluate.py --config configs/evaluation.yaml --models all --output artifacts/evaluation
```

Expected: Phase 0 七项 PASS，评测矩阵完整，报告中的 checkpoint hash、数据 hash 和配置 hash 均可追溯。

- [ ] **Step 7: 检查敏感信息与大文件**

Run:

```powershell
git status --short
git diff --check
git ls-files | Select-String -Pattern '\.(safetensors|bin|pt|pth)$'
git grep -n -E "(sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY)"
rg -n "TODO|TBD|FIXME" . --glob '!docs/superpowers/plans/*'
```

Expected: 无模型权重、密钥、尾随空格或未解决占位符；只有预期的源码、配置、少量样例和脱敏结果。

- [ ] **Step 8: 提交最终迁移**

```powershell
git add -A
git commit -m "docs: complete agentic dispute agent migration"
git status --short
```

Expected: 工作树干净。此时再使用 `superpowers:verification-before-completion` 做完成前验证，并使用 `superpowers:finishing-a-development-branch` 选择合并、PR 或保留分支。

## 里程碑与放行条件

1. **M1 本地领域闭环（Task 0–4）**：数据隔离、工具、状态机、奖励和切分测试全部通过。
2. **M2 Agent 闭环（Task 5–7）**：同一 Runtime 能执行真实多轮工具协议；SFT mask 和 BF16 LoRA 配置可验证。
3. **M3 云端兼容性（Task 8）**：七项 Phase 0 门禁全部通过；否则不得启动 Qwen3-8B 正式训练。
4. **M4 算法实验（Task 9–10）**：完成 SFT 规模消融、正式 GRPO、无工具成本消融和冻结测试集评测。
5. **M5 简历交付（Task 11–12）**：演示可运行，README 中每个技术栈和数字都有真实产物支撑。

## 推荐训练顺序和算力止损点

1. 先在本地完成所有非 GPU 测试与 fixture dry-run。
2. 租双 4090 后先跑 Phase 0；若 4–6 小时内仍不能完成一次更新与重载，停止计费并根据门禁报告定位。
3. Phase 0 通过后依次训练 SFT-500、SFT-1000、SFT-1500，先比较验证集再选择正式 GRPO 起点。
4. 只复制最佳 SFT-1500 adapter 给 GRPO，保留原 adapter 只读。
5. 先跑小步数 GRPO sanity run，确认 reward variance、有效 rollout 和显存，再跑完整实验。
6. 完整奖励主实验通过后才跑“去工具成本”消融；统一评测最后一次性读取冻结测试集。

## 完成定义

- Base、SFT-500、SFT-1000、SFT-1500、SFT-1500+Agentic GRPO 在相同 Runtime/工具/预算/测试集上有真实结果。
- SFT 与 GRPO 均为 Qwen3-8B BF16 LoRA r=32/alpha=64/dropout=0/all-linear，无 4/8-bit 量化。
- GRPO 从未合并的最佳 SFT adapter 副本继续，Agent Lightning/VERL 完成至少一次真实更新并可重载。
- ID 400、OOD 200（80/60/60）与人工审核 100 的冻结哈希可核验。
- 奖励分支、硬惩罚、trace 单一来源、reward 单次返回和 50-step 零方差暂停均有自动化测试。
- README 不含虚构指标；简历中的每项结论能从配置、日志、Trace 或评测报告复核。
