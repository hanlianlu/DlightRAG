# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Research tools for Memory Write and deeper recall."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from dlightrag_agent.tools import AgentTool, ToolResult
from dlightrag_memory import (
    Memory,
    MemoryProvenance,
    MemoryUnavailableError,
    MemoryWriteRejectedError,
)
from pydantic import BaseModel, ConfigDict, Field

from dlightrag.answer.memory import memory_owner_allowed

MemoryKindInput = Literal["preference", "fact"]


class RememberInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    kind: MemoryKindInput = Field(description="preference or fact")
    body: str = Field(min_length=1, max_length=500, description="What to remember.")
    confidence: float = Field(gt=0, le=1, description="Confidence in (0, 1].")
    supersedes_id: str | None = Field(default=None, description="Memory id this replaces.")


class ForgetInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    memory_id: str | None = Field(default=None, description="Id of the memory to delete.")
    body: str | None = Field(default=None, description="Exact body to delete if id is unknown.")


class RecallInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass
class MemoryHost:
    """Late-bound owner context for one research run's memory tools."""

    owner_id: str = ""
    auth_mode: str = "none"
    run_id: str = ""
    session_id: str = ""
    memory: Memory | None = None
    enabled: bool = True


def remember_tool(*, host: MemoryHost) -> AgentTool:
    async def execute(raw: BaseModel) -> ToolResult:
        args = raw if isinstance(raw, RememberInput) else RememberInput.model_validate(raw)
        if not memory_owner_allowed(host.auth_mode):
            return ToolResult(content="Long-term memory requires a JWT owner.")
        if not host.enabled:
            return ToolResult(content="Memory is disabled for this owner.")
        try:
            written = await _memory(host).remember(
                owner_id=host.owner_id,
                kind=args.kind,
                body=args.body,
                confidence=args.confidence,
                provenance=MemoryProvenance(run_id=host.run_id, session_id=host.session_id),
                supersedes_id=args.supersedes_id,
            )
        except MemoryWriteRejectedError as exc:
            return ToolResult(content=str(exc.public_message))
        if written is None:
            return ToolResult(content="Memory was not stored.")
        return ToolResult(
            content=f"Remembered {written.kind} {written.memory_id}.",
            details={"memory_id": written.memory_id, "kind": written.kind},
        )

    return AgentTool(
        "remember",
        "Store one owner-scoped preference or fact for later conversations. Not evidence.",
        RememberInput,
        execute,
        replay_policy="never",
    )


def forget_tool(*, host: MemoryHost) -> AgentTool:
    async def execute(raw: BaseModel) -> ToolResult:
        args = raw if isinstance(raw, ForgetInput) else ForgetInput.model_validate(raw)
        if not memory_owner_allowed(host.auth_mode):
            return ToolResult(content="Long-term memory requires a JWT owner.")
        if not host.enabled:
            return ToolResult(content="Memory is disabled for this owner.")
        try:
            await _memory(host).forget(
                owner_id=host.owner_id,
                memory_id=args.memory_id,
                body=args.body,
                provenance=MemoryProvenance(run_id=host.run_id, session_id=host.session_id),
            )
        except MemoryWriteRejectedError as exc:
            return ToolResult(content=str(exc.public_message))
        return ToolResult(content="Forgotten.")

    return AgentTool(
        "forget",
        "Permanently delete one remembered preference or fact.",
        ForgetInput,
        execute,
        replay_policy="safe",
    )


def recall_memory_tool(*, host: MemoryHost) -> AgentTool:
    async def execute(_raw: BaseModel) -> ToolResult:
        if host.memory is None:
            return ToolResult(content="Memory store is not bound.")
        if not memory_owner_allowed(host.auth_mode):
            return ToolResult(content="Long-term memory requires a JWT owner.")
        if not host.enabled:
            return ToolResult(content="Memory is disabled for this owner.")
        rows = await host.memory.list_active(owner_id=host.owner_id)
        if not rows:
            return ToolResult(content="No stored memories.")
        lines = [f"- {row.memory_id} ({row.kind}) {row.body}" for row in rows[:50]]
        return ToolResult(content="Stored memories:\n" + "\n".join(lines))

    return AgentTool(
        "recall_memory",
        "List stored memories beyond the automatic recent set. Not evidence.",
        RecallInput,
        execute,
        replay_policy="safe",
    )


def _memory(host: MemoryHost) -> Memory:
    if host.memory is None:
        raise MemoryUnavailableError()
    return host.memory


__all__ = [
    "ForgetInput",
    "MemoryHost",
    "RecallInput",
    "RememberInput",
    "forget_tool",
    "recall_memory_tool",
    "remember_tool",
]
