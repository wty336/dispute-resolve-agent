"""verl 自定义奖励函数。

在 verl 配置中指定：
    custom_reward_function.path=dispute_agent/verl_reward.py
    custom_reward_function.name=compute_score

verl 会把 parquet 中 `extra_info` 列的内容传给奖励函数；本项目的
`extra_info` 形如 {"case_id": "C000001"}，奖励函数据此加载 case 表
（data/rl_cases.jsonl），并调用 `dispute_agent.reward.compute_reward`
计算「判责匹配度 + 长期收益/1000」。

case 表路径通过环境变量 `DISPUTE_CASE_TABLE_PATH` 指定；未指定时默认
`data/rl_cases.jsonl`。
"""
from __future__ import annotations

import json
import os

_CASE_STORE: dict | None = None


def _load_case_store() -> dict:
    """加载 case 表并缓存（ray worker 内每个进程加载一次）。"""
    global _CASE_STORE
    if _CASE_STORE is None:
        from dispute_agent.data_generation import load_cases_from_jsonl

        path = os.environ.get("DISPUTE_CASE_TABLE_PATH", "data/rl_cases.jsonl")
        cases = load_cases_from_jsonl(path)
        _CASE_STORE = {case.case_id: case for case in cases}
    return _CASE_STORE


def _extract_case_id(extra_info) -> str | None:
    """从 extra_info 中解析 case_id，兼容 dict 与 JSON 字符串。"""
    if extra_info is None:
        return None
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except json.JSONDecodeError:
            return None
    if isinstance(extra_info, dict):
        return extra_info.get("case_id")
    return None


def _extract_tool_cost(extra_info) -> float:
    """从 extra_info 中读取预执行工具成本，兼容 dict 与 JSON 字符串。"""
    if extra_info is None:
        return 0.0
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except json.JSONDecodeError:
            return 0.0
    if isinstance(extra_info, dict):
        try:
            return float(extra_info.get("tool_cost", 0.0))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth=None,
    extra_info=None,
) -> float:
    """单条奖励：verl 旧版/新版通用的逐条调用接口。"""
    if data_source != "dispute_resolve":
        return 0.0  # 非本项目数据不评分

    case_id = _extract_case_id(extra_info)
    if case_id is None:
        return -1.0

    case = _load_case_store().get(case_id)
    if case is None:
        return -1.0

    from dispute_agent.reward import compute_reward

    tool_cost = _extract_tool_cost(extra_info)
    return compute_reward(case, solution_str, tool_cost=tool_cost)


def compute_score_batch(
    data_sources: list[str],
    solution_strs: list[str],
    ground_truths=None,
    extra_infos=None,
) -> list[float]:
    """批式奖励：verl 0.3+ 支持 batch reward function 时的接口。

    如果训练配置使用逐条奖励，可忽略本函数。
    """
    if extra_infos is None:
        extra_infos = [None] * len(solution_strs)
    if ground_truths is None:
        ground_truths = [None] * len(solution_strs)
    return [
        compute_score(ds, sol, gt, ei)
        for ds, sol, gt, ei in zip(data_sources, solution_strs, ground_truths, extra_infos)
    ]
