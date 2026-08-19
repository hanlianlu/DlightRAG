# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Foreground delegate_research: one child session inside the parent run."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from dlightrag_agent.session.ids import SessionId
from dlightrag_agent.tools import AgentTool, ToolResult, current_tool_call
from pydantic import BaseModel, ConfigDict, Field

from dlightrag.answer.evidence import EvidenceDelta

ChildStatus = Literal["succeeded", "failed", "cancelled"]


class DelegateInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    objective: str = Field(
        min_length=1, description="One concrete research question for the child."
    )


@dataclass(frozen=True, slots=True)
class ChildOutcome:
    """Distilled child result the parent tool returns."""

    status: ChildStatus
    summary: str
    handles: tuple[str, ...] = ()
    usage: Mapping[str, int] | None = None
    delta: EvidenceDelta | None = None
    child_session_id: str = ""


@dataclass
class DelegateHost:
    """Late-bound parent context for one research run's delegate tool."""

    parent_session_id: SessionId | None = None
    run_id: str = ""
    owner_id: str = ""
    persist: Callable[..., Awaitable[Any]] | None = None
    load_child: Callable[..., Awaitable[Any]] | None = None
    finish_child: Callable[..., Awaitable[Any]] | None = None
    run_child: Callable[[SessionId, str, str], Awaitable[ChildOutcome]] | None = None


def delegate_research_tool(*, host: DelegateHost) -> AgentTool:
    async def execute(raw: BaseModel) -> ToolResult:
        args = raw if isinstance(raw, DelegateInput) else DelegateInput.model_validate(raw)
        return await _run_delegate(host, args.objective)

    return AgentTool(
        "delegate_research",
        "Delegate one concrete research question to a child session. "
        "The child cannot write files, run bash, or delegate further.",
        DelegateInput,
        execute,
        replay_policy="safe",
    )


async def _run_delegate(host: DelegateHost, objective: str) -> ToolResult:
    call = current_tool_call()
    if host.parent_session_id is None or not host.run_id:
        raise RuntimeError("delegate_research is not bound to a parent session")
    if host.run_child is None:
        raise RuntimeError("delegate_research has no child runner")
    call_id = call.call_id if call is not None else "anonymous"
    child_id = SessionId.deterministic(
        run_id=host.run_id, name=f"delegate:{host.parent_session_id.value}:{call_id}"
    )
    existing = None
    if host.load_child is not None:
        existing = await host.load_child(
            owner_id=host.owner_id, run_id=host.run_id, child_session_id=child_id.value
        )
    if existing is not None and existing.get("status") in {"succeeded", "failed", "cancelled"}:
        return ToolResult(
            content=str(existing.get("summary") or "Child session already finished."),
            details={
                "child_session_id": child_id.value,
                "status": str(existing.get("status")),
                "replayed": True,
            },
        )
    if host.persist is not None:
        await host.persist(
            owner_id=host.owner_id,
            run_id=host.run_id,
            child_session_id=child_id.value,
            parent_session_id=host.parent_session_id.value,
            parent_call_id=call_id,
        )
    outcome = await host.run_child(child_id, objective, call_id)
    if host.finish_child is not None:
        await host.finish_child(
            owner_id=host.owner_id,
            run_id=host.run_id,
            child_session_id=child_id.value,
            status=outcome.status,
            summary=outcome.summary,
        )
    return _parent_result(outcome)


def _parent_result(outcome: ChildOutcome) -> ToolResult:
    lines = [outcome.summary.strip() or f"Child session {outcome.status}."]
    if outcome.handles:
        lines.append("Evidence handles:")
        lines.extend(f"- {item}" for item in outcome.handles)
    if outcome.usage:
        usage = ", ".join(f"{key}={value}" for key, value in outcome.usage.items())
        lines.append(f"Usage: {usage}")
    if outcome.delta is not None and outcome.delta.changed:
        lines.append(
            "Evidence delta: "
            f"chunks={outcome.delta.new_chunks} "
            f"entities={outcome.delta.new_entities} "
            f"relationships={outcome.delta.new_relationships}"
        )
    details: dict[str, Any] = {
        "child_session_id": outcome.child_session_id,
        "status": outcome.status,
        "evidence_handles": list(outcome.handles),
    }
    if outcome.usage is not None:
        details["usage"] = dict(outcome.usage)
    if outcome.delta is not None:
        details["evidence_delta"] = {
            "new_chunks": outcome.delta.new_chunks,
            "new_entities": outcome.delta.new_entities,
            "new_relationships": outcome.delta.new_relationships,
        }
    return ToolResult(content="\n".join(lines), details=details)


__all__ = [
    "ChildOutcome",
    "ChildStatus",
    "DelegateHost",
    "DelegateInput",
    "delegate_research_tool",
]
