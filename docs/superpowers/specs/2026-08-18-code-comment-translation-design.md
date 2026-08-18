# Python 代码注释中文化设计

## 目标

将项目 Python 代码中的英文注释和 docstring 翻译为中文，提升中文开发和简历展示场景下的可读性。

## 范围

- 处理 `dispute_agent/`、`scripts/`、`tests/` 下的 `.py` 文件。
- 翻译模块、类、函数 docstring，以及行内 `#` 注释。
- 已有中文注释保持不变。

## 不变项

- 不修改业务逻辑、控制流、类型标注或格式化结构。
- 不修改代码字符串、异常信息、CLI 参数、类名、函数名、协议字段或 shebang。
- 不翻译 Markdown、YAML、TOML 或其他非 Python 文件。

## 验收

- 通过 Python `compileall` 语法检查。
- `git diff` 仅包含 Python 注释/docstring 与本设计记录的变化。
- 不新增与注释翻译无关的测试或依赖。
