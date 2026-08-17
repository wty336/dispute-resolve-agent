"""Episode state machine for a single dispute resolution run.

The episode keeps the public observation and the hidden ground truth separate.
Ground truth is only used by tools, rewards, and evaluation; it is never
included in ``model_dump()`` or exposed to the agent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dispute_agent.domain.schemas import (
    Decision,
    DisputeGroundTruth,
    DisputeObservation,
    Escalation,
    Evidence,
    TerminalDecision,
    ToolResult,
)
from dispute_agent.tools.registry import ToolCallRecord, ToolRegistry


@dataclass
class InvalidActionCounts:
    illegal_calls: int = 0
    repeat_calls: int = 0
    consecutive_illegal: int = 0


class EpisodeState:
    """Public, mutable state for one agent episode."""

    def __init__(
        self,
        *,
        observation: DisputeObservation,
        ground_truth: DisputeGroundTruth,
        case_seed: str | int | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.observation = observation
        self._ground_truth = ground_truth
        self.case_seed = case_seed
        self.tool_registry = tool_registry or ToolRegistry(observation.case_id, case_seed)
        self.cache: dict[tuple[str, str], ToolResult] = {}
        self.visible_evidence: dict[str, Evidence] = {
            evidence.evidence_id: evidence for evidence in observation.evidence
        }
        self.tool_calls: list[ToolCallRecord] = []
        self.cumulative_cost: float = 0.0
        self.round: int = 0
        self.invalid_actions = InvalidActionCounts()
        self.done: bool = False
        self.end_reason: str | None = None
        self.terminal_decision: Decision | Escalation | None = None

    @property
    def ground_truth(self) -> DisputeGroundTruth:
        return self._ground_truth

    @property
    def case_id(self) -> str:
        return self.observation.case_id

    def call(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        return self.tool_registry.execute(self, tool_name, arguments)

    def record_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> None:
        if self.done:
            raise RuntimeError("episode already terminated")
        result = ToolResult(
            tool_name=tool_name,
            arguments={str(k): str(v) for k, v in arguments.items()},
            payload="",
            cached=False,
            cost=0.0,
            success=True,
        )
        self.tool_calls.append(ToolCallRecord(name=tool_name, arguments=dict(arguments), result=result))
        self.round += 1

    def add_tool_result(self, result: ToolResult) -> None:
        self.cumulative_cost += result.cost
        evidence_id = self._evidence_id_for(result)
        if evidence_id and evidence_id not in self.visible_evidence:
            self.visible_evidence[evidence_id] = Evidence(
                evidence_id=evidence_id,
                type=result.tool_name,
                description=result.payload,
                source=self._source_for(result.tool_name),
                visible=True,
            )

    def submit(self, decision: TerminalDecision) -> None:
        if self.done:
            raise RuntimeError("episode already terminated")
        self.terminal_decision = decision
        self.done = True
        self.end_reason = "submitted"

    def model_dump(self) -> dict[str, Any]:
        return {
            "observation": self.observation.model_dump(),
            "visible_evidence": [e.model_dump() for e in self.visible_evidence.values()],
            "tool_calls": [
                {
                    "name": call.name,
                    "arguments": call.arguments,
                    "cached": call.cached,
                    "payload": call.result.payload,
                }
                for call in self.tool_calls
            ],
            "cumulative_cost": self.cumulative_cost,
            "round": self.round,
            "invalid_actions": {
                "illegal_calls": self.invalid_actions.illegal_calls,
                "repeat_calls": self.invalid_actions.repeat_calls,
                "consecutive_illegal": self.invalid_actions.consecutive_illegal,
            },
            "done": self.done,
            "end_reason": self.end_reason,
            "terminal_decision": (
                self.terminal_decision.model_dump(mode="json") if self.terminal_decision else None
            ),
        }

    def _evidence_id_for(self, result: ToolResult) -> str | None:
        if result.tool_name == "verify_evidence":
            return result.arguments.get("evidence_id")
        if result.tool_name == "check_logistics":
            order_id = result.arguments.get("order_id")
            return f"logistics:{order_id}" if order_id else None
        if result.tool_name == "check_buyer_history":
            buyer_id = result.arguments.get("buyer_id")
            return f"buyer_history:{buyer_id}" if buyer_id else None
        if result.tool_name == "check_merchant_history":
            merchant_id = result.arguments.get("merchant_id")
            return f"merchant_history:{merchant_id}" if merchant_id else None
        return None

    def _source_for(self, tool_name: str) -> str:
        if tool_name == "check_logistics":
            return "logistics"
        if tool_name == "check_buyer_history":
            return "buyer"
        if tool_name == "check_merchant_history":
            return "merchant"
        return "platform"
