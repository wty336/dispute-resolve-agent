# Python 代码注释中文化实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or **superpowers:executing-plans** to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将项目 Python 代码中的英文注释和 docstring 翻译为中文，同时保持运行逻辑和所有运行时字符串不变。

**Architecture:** 只处理 Python 注释语法和 docstring 文本，不触碰表达式、字符串字面量、名称或配置。按包、脚本、测试三组逐文件修改，最后用语法检查和差异检查确认范围。

**Tech Stack:** Python 3.11, `ast`/文本差异检查, `compileall`, Git。

---

### Task 1: 翻译 `dispute_agent` 包注释

**Files:**
- Modify: `dispute_agent/__init__.py`
- Modify: `dispute_agent/agent/__init__.py`
- Modify: `dispute_agent/agent/provider.py`
- Modify: `dispute_agent/agent/runtime.py`
- Modify: `dispute_agent/agent/tools.py`
- Modify: `dispute_agent/agent/tracing.py`
- Modify: `dispute_agent/api/__init__.py`
- Modify: `dispute_agent/api/app.py`
- Modify: `dispute_agent/data/__init__.py`
- Modify: `dispute_agent/data/generator.py`
- Modify: `dispute_agent/data/renderer.py`
- Modify: `dispute_agent/data/splits.py`
- Modify: `dispute_agent/data/validators.py`
- Modify: `dispute_agent/domain/__init__.py`
- Modify: `dispute_agent/domain/policies.py`
- Modify: `dispute_agent/domain/schemas.py`
- Modify: `dispute_agent/environment/__init__.py`
- Modify: `dispute_agent/environment/episode.py`
- Modify: `dispute_agent/evaluation/__init__.py`
- Modify: `dispute_agent/evaluation/bootstrap.py`
- Modify: `dispute_agent/evaluation/metrics.py`
- Modify: `dispute_agent/evaluation/runner.py`
- Modify: `dispute_agent/rewards/__init__.py`
- Modify: `dispute_agent/rewards/business_utility.py`
- Modify: `dispute_agent/rewards/engine.py`
- Modify: `dispute_agent/tools/__init__.py`
- Modify: `dispute_agent/tools/registry.py`
- Modify: `dispute_agent/tools/simulators.py`
- Modify: `dispute_agent/training/__init__.py`
- Modify: `dispute_agent/training/grpo_config.py`
- Modify: `dispute_agent/training/grpo_dataset.py`
- Modify: `dispute_agent/training/grpo_runtime.py`
- Modify: `dispute_agent/training/lightning_agent.py`
- Modify: `dispute_agent/training/monitor.py`
- Modify: `dispute_agent/training/phase0.py`
- Modify: `dispute_agent/training/sft_data.py`
- Modify: `dispute_agent/training/sft_dataset.py`
- Modify: `dispute_agent/training/sft_runtime.py`
- Modify: `dispute_agent/training/train_sft.py`

- [ ] **Step 1: 逐文件翻译英文模块说明、类/函数 docstring 和 `#` 注释**

保留代码结构，仅替换注释文本。例如：

```python
"""Auditable rule-based reward engine for training."""
# Reward constants
```

改为：

```python
"""用于训练的可审计规则奖励引擎。"""
# 奖励常量
```

不得改动 `"""` 边界、代码字符串、名称、协议字段或异常信息。

- [ ] **Step 2: 检查包目录差异范围**

运行：

```powershell
git diff -- dispute_agent
```

预期：差异只包含 docstring 和注释行的文本变化。

- [ ] **Step 3: 提交包注释翻译**

```bash
git add dispute_agent
git commit -m "docs: translate package comments to chinese"
```

### Task 2: 翻译脚本注释

**Files:**
- Modify: `scripts/evaluate.py`
- Modify: `scripts/generate_data.py`
- Modify: `scripts/phase0_smoke.py`
- Modify: `scripts/train_agentic_grpo.py`
- Modify: `scripts/train_sft.py`
- Modify: `scripts/verify_grpo_checkpoint.py`

- [ ] **Step 1: 翻译脚本级 docstring 和行注释**

保留 shebang、命令行参数、帮助文本中的技术名称和代码字符串；只翻译注释语句及模块 docstring 中的自然语言。

- [ ] **Step 2: 检查脚本差异范围**

```powershell
git diff -- scripts
```

预期：没有参数、导入、控制流或字符串字面量变化。

- [ ] **Step 3: 提交脚本注释翻译**

```bash
git add scripts
git commit -m "docs: translate script comments to chinese"
```

### Task 3: 翻译测试注释并做最终验收

**Files:**
- Modify: `tests/conftest.py`

- [ ] **Step 1: 翻译测试 fixture 的英文 docstring**

仅修改 `tests/conftest.py` 的注释/docstring；测试断言、fixture 名称和测试逻辑保持不变。

- [ ] **Step 2: 运行语法检查**

```powershell
python -m compileall -q dispute_agent scripts tests
```

预期：命令退出码为 0。

- [ ] **Step 3: 检查最终 diff**

```powershell
git diff HEAD~3 --stat
git diff HEAD~3 --check
git status --short
```

预期：工作树干净；差异只涉及 Python 注释/docstring 和本计划/设计记录。

- [ ] **Step 4: 提交测试注释翻译**

```bash
git add tests/conftest.py
git commit -m "docs: translate test comments to chinese"
```

不运行完整测试套件，因为本任务不改变业务逻辑；`compileall` 和 diff 范围检查足以验证本次变更。
