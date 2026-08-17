"""Tool registry for the dispute resolution agent.

The registry owns tool argument validation, deterministic simulation, caching,
and interaction with the episode state machine.  Only the four investigation
tools are registered; ``submit_decision`` is handled by the agent runtime.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from dispute_agent.domain.policies import MAX_INVESTIGATION_CALLS
from dispute_agent.domain.schemas import Evidence, ToolResult
from dispute_agent.tools.simulators import (
    TOOL_ARGUMENT_SCHEMAS,
    TOOL_COSTS,
    simulate_tool,
)

INVESTIGATION_TOOLS = tuple(TOOL_COSTS.keys())


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict[str, str]
    result: ToolResult
    cached: bool = False


def _canonical_json(value: dict) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


class ToolRegistry:
    """Deterministic registry bound to a case id and a group seed."""

    def __init__(self, case_id: str, case_seed: str | int | None = None) -> None:
        self.case_id = case_id
        self.case_seed = case_seed

    def execute(self, episode, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        if episode.done:
            raise RuntimeError("episode already terminated")
        if tool_name not in TOOL_COSTS:
            raise KeyError(f"unknown tool: {tool_name}")
        self._validate_arguments(tool_name, arguments)

        key = (tool_name, _canonical_json(arguments))
        if key in episode.cache:
            cached_result = episode.cache[key].model_copy(update={"cached": True, "cost": 0.0})
            episode.tool_calls.append(
                ToolCallRecord(name=tool_name, arguments=dict(arguments), result=cached_result, cached=True)
            )
            episode.invalid_actions.repeat_calls += 1
            return cached_result

        investigation_calls = sum(
            1 for call in episode.tool_calls if not call.cached and call.name in INVESTIGATION_TOOLS
        )
        if investigation_calls >= MAX_INVESTIGATION_CALLS:
            raise RuntimeError("investigation tool budget exhausted")

        result = simulate_tool(
            tool_name,
            case_id=self.case_id,
            arguments=arguments,
            case_seed=self.case_seed,
            ground_truth=episode.ground_truth,
        )
        episode.cache[key] = result
        episode.record_tool_call(tool_name, arguments)
        episode.add_tool_result(result)
        return result

    def _validate_arguments(self, tool_name: str, arguments: dict[str, Any]) -> None:
        schema = TOOL_ARGUMENT_SCHEMAS[tool_name]
        if set(schema) != set(arguments):
            raise ValueError(f"invalid arguments for {tool_name}: expected {sorted(schema)}, got {sorted(arguments)}")
        for key, expected_type in schema.items():
            if not isinstance(arguments.get(key), expected_type):
                raise ValueError(f"argument {key!r} for {tool_name} must be {expected_type.__name__}")
