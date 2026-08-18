# DeepSeek 语言增强层设计

## 目标

在现有确定性合成数据管线之上，增加一个可选的 DeepSeek API 语言改写阶段，提升买家与商家公开描述的语言多样性，同时保持隐藏真值隔离、工具参数一致、数据可复现和 API 不可用时的可回退性。

## 非目标

- 不让 DeepSeek 生成责任、赔付、升级人工等隐藏标签。
- 不让在线 API 调用成为默认数据生成依赖。
- 不直接让模型重写整个 SFT trace 或工具协议。
- 不把 API key、原始响应或隐藏真值写入仓库。

## 方案

数据流保持为：

```text
FactInstance(hidden truth + public observation)
  -> DeepSeek receives public fields only
  -> structured public rewrite
  -> invariant/leakage validation
  -> apply rewrite to observation
  -> render SFT trace and GRPO observation row
  -> manifest/hash freeze
```

默认生成流程不调用 DeepSeek。通过 `--enrich-language` 显式开启增强；未提供 API key、请求失败、JSON 无法解析或校验失败时，单条样本回退到原始模板，并在运行报告中记录失败计数。

## DeepSeek 接入

- 使用 OpenAI-compatible Chat Completions 接口。
- 默认 base URL：`https://api.deepseek.com`。
- 默认模型：`deepseek-v4-flash`，用于非思考、低成本的公开文本改写。
- API key 只从 `DEEPSEEK_API_KEY` 环境变量读取；base URL 和模型可分别由 `DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL` 覆盖。
- 请求使用 JSON Output，并在提示中明确要求输出 JSON。
- 通过 `extra_body.thinking.type=disabled` 关闭改写任务的思考输出；不保存 reasoning 内容。
- 每条请求设置超时、有限重试和最大 token；不做无限重试。

## 改写协议

输入只包含公开字段：`buyer_claim`、`merchant_response`、`chat_log`、公开 evidence description、商品、金额和投诉类型，以及一个独立的 style profile。

输出只允许：

```json
{
  "buyer_claim": "...",
  "merchant_response": "...",
  "chat_log": ["...", "..."],
  "evidence_descriptions": {"evidence_id": "..."}
}
```

提示约束：

1. 保留所有 ID、金额、投诉类型、证据 ID 和已有事件。
2. 不新增时间、金额、证据、责任结论、赔付结论或升级人工结论。
3. 不输出 `true_liability`、`true_loss`、`reasonable_compensation_range`、`should_escalate` 等隐藏字段。
4. 只改变表达方式、句式、礼貌程度、叙述顺序和对话风格。
5. 返回合法 JSON，不返回解释文字。

初始 style profile 覆盖平台正式、买家简短、买家时间线、买家情绪但事实清楚、商家配合、商家防御、商家推诿和聊天碎片化；买家与商家风格独立采样。

## 缓存与审计

缓存键由 `source_public_hash + style + prompt_version + model` 构成。缓存记录不包含隐藏真值，保存请求状态、模型、提示版本、输出 hash 和失败原因。最终数据 manifest 纳入公开 JSONL、隐藏 JSONL、质量报告和语言增强缓存的 hash。

增强比例默认 50%，保留 50% 原始模板作为可控基线；比例可在配置中调整。增强只在 `FactInstance` 层发生一次，随后 SFT 和 GRPO 从同一份公开 observation 渲染，避免两套语言不一致。

## 校验与回退

每条 DeepSeek 输出依次经过：JSON/schema 校验、ID/金额/投诉类型不变量校验、证据 ID 校验、隐藏字段/判责词扫描、公开 observation 和 SFT trace 原有校验。任何失败都回退到未改写 observation，不丢弃事实样本；最终报告记录 `requested/succeeded/fallback` 计数。

## 验证范围

- 不带 `--enrich-language` 的现有生成和测试结果保持不变。
- 使用 fake DeepSeek client 测试成功响应、非法 JSON、事实漂移、API 异常和缓存命中。
- 使用 24 条 fixture 检查 SFT/GRPO 输出、泄漏校验和增强统计。
- 不在本地测试中调用真实 DeepSeek API；真实 API 只在用户提供 key 后运行小批量 smoke。

