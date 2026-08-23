# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""First-class foreground child Agent tools.

A spawn call may launch one or many ordinary child Agent Sessions in parallel,
but it does not return until every child has settled. There is no detached
scheduler: host task references exist only so concurrent status/wait/cancel
calls can address foreground work in the same parent run.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dlightrag.agent.session.ids import SessionId
from dlightrag.agent.tools import AgentTool, ToolResult, current_tool_call
from dlightrag.answer.evidence import EvidenceDelta

type ChildStatus = Literal["running", "succeeded", "failed", "cancelled"]
type ChildContextMode = Literal["isolated", "parent"]
type ChildModelRole = Literal["query", "extract"]


class ChildRequest(BaseModel):
    """One bounded child invocation selected by the parent model."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    objective: str = Field(min_length=1, description="One concrete child objective.")
    context: ChildContextMode = Field(
        default="isolated",
        description="isolated starts from the objective; parent also receives parent context.",
    )
    model_role: ChildModelRole = Field(
        default="query", description="Configured tool-capable model role for the child."
    )
    tools: tuple[str, ...] | None = Field(
        default=None,
        description="Optional inherited tool-name subset; spawn tools are always removed.",
    )


class SpawnAgentInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    children: tuple[ChildRequest, ...] = Field(
        min_length=1,
        max_length=8,
        description="One or more foreground child requests, run in parallel when possible.",
    )

    @model_validator(mode="after")
    def _unique_objectives(self) -> SpawnAgentInput:
        if len({child.objective for child in self.children}) != len(self.children):
            raise ValueError("child objectives must be unique within one spawn call")
        return self


class ChildControlInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    child_session_id: str = Field(min_length=1, description="Child session id from spawn_agent.")


@dataclass(frozen=True, slots=True)
class ChildOutcome:
    """Distilled child result adopted by the parent."""

    status: ChildStatus
    summary: str
    handles: tuple[str, ...] = ()
    usage: Mapping[str, int] | None = None
    delta: EvidenceDelta | None = None
    child_session_id: str = ""
    evidence_state: Mapping[str, Any] | None = None


@dataclass
class SubagentHost:
    """Late-bound lineage, persistence, and foreground-task state."""

    parent_session_id: SessionId | None = None
    run_id: str = ""
    owner_id: str = ""
    depth: int = 0
    max_depth: int = 1
    max_concurrency: int = 4
    persist: Callable[..., Awaitable[Any]] | None = None
    load_child: Callable[..., Awaitable[Any]] | None = None
    finish_child: Callable[..., Awaitable[Any]] | None = None
    run_child: Callable[[SessionId, ChildRequest, str], Awaitable[ChildOutcome]] | None = None
    adopt_evidence: Callable[[Mapping[str, Any], str, str], tuple[str, ...]] | None = None
    record_usage: Callable[[Mapping[str, int]], None] | None = None
    tasks: dict[str, asyncio.Task[ChildOutcome]] = field(default_factory=dict)
    outcomes: dict[str, ChildOutcome] = field(default_factory=dict)


def subagent_tools(*, host: SubagentHost) -> tuple[AgentTool, ...]:
    """Return spawn/status/wait/cancel tools over one foreground roster."""

    async def spawn(raw: BaseModel) -> ToolResult:
        args = cast(SpawnAgentInput, raw)
        return await _spawn(host, args)

    async def status(raw: BaseModel) -> ToolResult:
        args = cast(ChildControlInput, raw)
        outcome = await _status(host, args.child_session_id)
        return _single_result(outcome)

    async def wait(raw: BaseModel) -> ToolResult:
        args = cast(ChildControlInput, raw)
        task = host.tasks.get(args.child_session_id)
        if task is not None:
            outcome = await asyncio.shield(task)
        else:
            outcome = await _status(host, args.child_session_id)
        return _single_result(outcome)

    async def cancel(raw: BaseModel) -> ToolResult:
        args = cast(ChildControlInput, raw)
        task = host.tasks.get(args.child_session_id)
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            outcome = ChildOutcome(
                status="cancelled",
                summary="Child session cancelled.",
                child_session_id=args.child_session_id,
            )
            host.outcomes[args.child_session_id] = outcome
        else:
            outcome = await _status(host, args.child_session_id)
        return _single_result(outcome)

    return (
        AgentTool(
            "spawn_agent",
            "Run one or many foreground child Agent Sessions and wait for all results.",
            SpawnAgentInput,
            spawn,
            replay_policy="safe",
        ),
        AgentTool(
            "subagent_status",
            "Read one foreground or completed child session status.",
            ChildControlInput,
            status,
            replay_policy="safe",
        ),
        AgentTool(
            "wait_subagent",
            "Wait for one known foreground child session.",
            ChildControlInput,
            wait,
            replay_policy="safe",
        ),
        AgentTool(
            "cancel_subagent",
            "Cancel one known foreground child session.",
            ChildControlInput,
            cancel,
            replay_policy="safe",
        ),
    )


async def _spawn(host: SubagentHost, args: SpawnAgentInput) -> ToolResult:
    if host.parent_session_id is None or not host.run_id:
        raise RuntimeError("spawn_agent is not bound to a parent session")
    if host.run_child is None:
        raise RuntimeError("spawn_agent has no child runner")
    if host.depth >= host.max_depth:
        return ToolResult(content=f"Child depth limit reached ({host.max_depth}).")
    parent_session_id = host.parent_session_id
    run_child = host.run_child
    call = current_tool_call()
    call_id = call.call_id if call is not None else "anonymous"
    semaphore = asyncio.Semaphore(max(1, host.max_concurrency))
    child_ids = [
        child_session_id(
            run_id=host.run_id,
            parent_session_id=parent_session_id,
            call_id=call_id,
            position=position,
        )
        for position in range(len(args.children))
    ]

    async def run_one(child_id: SessionId, request: ChildRequest) -> ChildOutcome:
        if host.persist is not None:
            await host.persist(
                owner_id=host.owner_id,
                run_id=host.run_id,
                child_session_id=child_id.value,
                parent_session_id=parent_session_id.value,
                parent_call_id=call_id,
                objective=request.objective,
                context_mode=request.context,
                model_role=request.model_role,
                tools=request.tools,
            )
        try:
            async with semaphore:
                outcome = await run_child(child_id, request, call_id)
        except asyncio.CancelledError:
            cancelled = ChildOutcome(
                status="cancelled",
                summary="Child session cancelled.",
                child_session_id=child_id.value,
            )
            host.outcomes[child_id.value] = cancelled
            if host.finish_child is not None:
                await host.finish_child(
                    owner_id=host.owner_id,
                    run_id=host.run_id,
                    child_session_id=child_id.value,
                    status="cancelled",
                    summary=cancelled.summary,
                    usage=None,
                )
            return cancelled
        if outcome.evidence_state is not None and host.adopt_evidence is not None:
            outcome = replace(
                outcome,
                handles=host.adopt_evidence(outcome.evidence_state, child_id.value, call_id),
            )
        if outcome.usage is not None and host.record_usage is not None:
            host.record_usage(outcome.usage)
        if host.finish_child is not None:
            await host.finish_child(
                owner_id=host.owner_id,
                run_id=host.run_id,
                child_session_id=child_id.value,
                status=outcome.status,
                summary=outcome.summary,
                usage=outcome.usage,
            )
        host.outcomes[child_id.value] = outcome
        return outcome

    tasks = [
        asyncio.create_task(
            run_one(child_id, request),
            name=f"agent-child:{child_id.value}",
        )
        for child_id, request in zip(child_ids, args.children, strict=True)
    ]
    host.tasks.update(
        (child_id.value, task) for child_id, task in zip(child_ids, tasks, strict=True)
    )
    try:
        outcomes = await asyncio.gather(*tasks)
    finally:
        for child_id in child_ids:
            task = host.tasks.pop(child_id.value, None)
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return _many_result(tuple(outcomes))


async def _status(host: SubagentHost, child_id: str) -> ChildOutcome:
    task = host.tasks.get(child_id)
    if task is not None:
        if not task.done():
            return ChildOutcome(
                status="running",
                summary="Child session is running.",
                child_session_id=child_id,
            )
        return task.result()
    if child_id in host.outcomes:
        return host.outcomes[child_id]
    if host.load_child is not None:
        row = await host.load_child(
            owner_id=host.owner_id,
            run_id=host.run_id,
            child_session_id=child_id,
        )
        if row is not None:
            return ChildOutcome(
                status=str(row.get("status") or "failed"),  # type: ignore[arg-type]
                summary=str(row.get("summary") or ""),
                child_session_id=child_id,
            )
    return ChildOutcome(
        status="failed",
        summary="Unknown child session.",
        child_session_id=child_id,
    )


def child_session_id(
    *,
    run_id: str,
    parent_session_id: SessionId,
    call_id: str,
    position: int = 0,
) -> SessionId:
    """Deterministic child SessionId for one parent call and child position."""
    suffix = f":{position}" if position else ""
    return SessionId.deterministic(
        run_id=run_id,
        name=f"child:{parent_session_id.value}:{call_id}{suffix}",
    )


def _single_result(outcome: ChildOutcome) -> ToolResult:
    return _many_result((outcome,))


def _many_result(outcomes: tuple[ChildOutcome, ...]) -> ToolResult:
    lines: list[str] = []
    children: list[dict[str, Any]] = []
    inclusive_usage: dict[str, int] = {}
    for outcome in outcomes:
        lines.append(
            f"Child {outcome.child_session_id} [{outcome.status}]: "
            f"{outcome.summary.strip() or '(no summary)'}"
        )
        if outcome.handles:
            lines.extend(f"- adopted {item}" for item in outcome.handles)
        if outcome.usage:
            for key, value in outcome.usage.items():
                inclusive_usage[key] = inclusive_usage.get(key, 0) + int(value)
        children.append(
            {
                "child_session_id": outcome.child_session_id,
                "status": outcome.status,
                "evidence_handles": list(outcome.handles),
                "usage": dict(outcome.usage or {}),
            }
        )
    return ToolResult(
        content="\n".join(lines),
        details={"children": children, "inclusive_usage": inclusive_usage},
    )


__all__ = [
    "ChildContextMode",
    "ChildControlInput",
    "ChildModelRole",
    "ChildOutcome",
    "ChildRequest",
    "ChildStatus",
    "SpawnAgentInput",
    "SubagentHost",
    "child_session_id",
    "subagent_tools",
]
