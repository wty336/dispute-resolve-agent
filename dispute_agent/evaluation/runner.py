"""统一评估运行器和公平协议的 resolved run。"""
from __future__ import annotations

from dataclasses import dataclass, field

from dispute_agent.domain.policies import MAX_COMPENSATION_RATIO, MAX_ROUNDS, MAX_INVESTIGATION_CALLS
from dispute_agent.domain.schemas import TerminalDecision


@dataclass
class ResolvedRun:
    model: str
    case_hash: str
    tool_registry_hash: str
    max_rounds: int
    tool_budget: int
    terminal_schema_hash: str
    extra: dict = field(default_factory=dict)


def resolve_runs(model_names: list[str], case_hash: str = "case-hash") -> list[ResolvedRun]:
    schema_hash = hash(TerminalDecision) % (10**12)
    return [
        ResolvedRun(
            model=name,
            case_hash=case_hash,
            tool_registry_hash="tool-registry-hash",
            max_rounds=MAX_ROUNDS,
            tool_budget=MAX_INVESTIGATION_CALLS,
            terminal_schema_hash=str(schema_hash),
        )
        for name in model_names
    ]
