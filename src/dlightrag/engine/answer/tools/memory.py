# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Parent Research tools for owner Profile Memory."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from dlightrag_memory import (
    Memory,
    MemoryOperationReceipt,
    MemoryProvenance,
    MemoryWriteRejectedError,
)
from pydantic import BaseModel, ConfigDict, Field

from dlightrag.engine.agent.tools import AgentTool, ToolDeclaration, ToolResult, ToolRuntime
from dlightrag.engine.answer.memory import recall_sections

MemoryKindInput = Literal["preference", "fact"]
_MEMORY_MUTATION_LIMIT = 10
_INACTIVE = "Profile Memory is not active for this owner."


class RememberInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    kind: MemoryKindInput = Field(description="preference or fact")
    body: str = Field(min_length=1, max_length=500, description="What to remember.")
    supersedes_id: str | None = Field(default=None, description="Active memory id this replaces.")


class ForgetInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    memory_id: str | None = Field(default=None, description="Id of the memory to forget.")
    body: str | None = Field(default=None, description="Exact body if the id is unknown.")


class RecallInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1, description="What to recall memories for.")


@dataclass
class MemoryHost:
    """Late-bound owner capability for one parent Research run."""

    owner_id: str = ""
    run_id: str = ""
    session_id: str = ""
    memory: Memory | None = None
    enabled: bool = True
    epoch: int = 0
    capability_current: Callable[..., Awaitable[bool]] | None = None


async def _available(host: MemoryHost, *, settlement: object | None = None) -> bool:
    if not host.enabled:
        return False
    if host.capability_current is None:
        return True
    return await host.capability_current(
        owner_id=host.owner_id,
        epoch=host.epoch,
        settlement=settlement,
    )


async def _require_available(host: MemoryHost, settlement: object | None) -> None:
    """Recheck the run epoch inside the store's atomic settlement."""
    if not await _available(host, settlement=settlement):
        raise MemoryWriteRejectedError(_INACTIVE)


def remember_declaration() -> ToolDeclaration:
    return ToolDeclaration(
        "remember",
        "Store one durable owner preference or fact for future conversations. "
        "Use when the user explicitly asks to remember eligible stable information, or "
        "for one minimally inferred repeated preference. Never store task state, "
        "research claims, evidence, citations, "
        "transcripts, credentials, or private keys. Recall first and pass supersedes_id "
        "when correcting an existing memory.",
        RememberInput,
        replay_policy="replayable",
    )


def remember_tool(*, host: MemoryHost) -> AgentTool:
    async def execute(raw: BaseModel, runtime: ToolRuntime) -> ToolResult:
        args = raw if isinstance(raw, RememberInput) else RememberInput.model_validate(raw)
        if host.memory is None or not await _available(host):
            return _rejected("remember", _INACTIVE)
        try:
            receipt = await host.memory.remember(
                owner_id=host.owner_id,
                kind=args.kind,
                body=args.body,
                provenance=_provenance(host),
                supersedes_id=args.supersedes_id,
                idempotency_key=_idempotency_key(host, runtime),
                mutation_scope=host.run_id,
                mutation_limit=_MEMORY_MUTATION_LIMIT,
                guard=lambda settlement: _require_available(host, settlement),
            )
        except MemoryWriteRejectedError as exc:
            return _rejected("remember", str(exc.public_message))
        return _receipt_result(receipt)

    return remember_declaration().bind(execute)


def forget_declaration() -> ToolDeclaration:
    return ToolDeclaration(
        "forget",
        "Forget one active Profile Memory by id or exact body. Recall first when the id is unknown.",
        ForgetInput,
        replay_policy="replayable",
    )


def forget_tool(*, host: MemoryHost) -> AgentTool:
    async def execute(raw: BaseModel, runtime: ToolRuntime) -> ToolResult:
        args = raw if isinstance(raw, ForgetInput) else ForgetInput.model_validate(raw)
        if host.memory is None or not await _available(host):
            return _rejected("forget", _INACTIVE)
        try:
            receipt = await host.memory.forget(
                owner_id=host.owner_id,
                memory_id=args.memory_id,
                body=args.body,
                provenance=_provenance(host),
                idempotency_key=_idempotency_key(host, runtime),
                mutation_scope=host.run_id,
                mutation_limit=_MEMORY_MUTATION_LIMIT,
                guard=lambda settlement: _require_available(host, settlement),
            )
        except MemoryWriteRejectedError as exc:
            return _rejected("forget", str(exc.public_message))
        return _receipt_result(receipt)

    return forget_declaration().bind(execute)


def recall_memory_declaration() -> ToolDeclaration:
    return ToolDeclaration(
        "recall_memory",
        "Recall the owner's standing preferences and the facts relevant to a query, plus "
        "the newest other memories, with the ids needed before replacing or forgetting "
        "one. Context only; never evidence or a citation.",
        RecallInput,
        replay_policy="replayable",
    )


def recall_memory_tool(*, host: MemoryHost) -> AgentTool:
    async def execute(raw: BaseModel, runtime: ToolRuntime) -> ToolResult:
        args = raw if isinstance(raw, RecallInput) else RecallInput.model_validate(raw)
        if host.memory is None or not await _available(host):
            return ToolResult.text(_INACTIVE, is_error=True)
        await runtime.emit_update(ToolResult.text("", subject=args.query))
        result = await host.memory.lookup(owner_id=host.owner_id, query=args.query)
        lines = recall_sections(result, ids=True)
        if not lines:
            return ToolResult.text("No remembered preferences or facts.")
        return ToolResult.text("\n".join(lines))

    return recall_memory_declaration().bind(execute)


def _receipt_result(receipt: MemoryOperationReceipt) -> ToolResult:
    if receipt.outcome == "changed":
        verb = "Remembered" if receipt.action == "remember" else "Forgot"
        text = f"{verb} Profile Memory {receipt.memory_id or receipt.change_id}."
    elif receipt.outcome == "unchanged":
        text = "Profile Memory was already in that state."
    else:
        text = "Profile Memory changed concurrently; recall it again before retrying."
    return ToolResult.text(
        text,
        details={"memory_operation": _receipt_details(receipt)},
        is_error=receipt.outcome == "conflict",
    )


def _rejected(operation: str, message: str) -> ToolResult:
    return ToolResult.text(
        message,
        details={
            "memory_operation": {
                "operation": operation,
                "outcome": "rejected",
            }
        },
        is_error=True,
    )


def _receipt_details(receipt: MemoryOperationReceipt) -> dict[str, object]:
    return {
        "body": receipt.body,
        "change_id": receipt.change_id,
        "kind": receipt.kind,
        "memory_ids": list(receipt.memory_ids),
        "operation": receipt.action,
        "outcome": receipt.outcome,
        "supersedes_id": receipt.supersedes_id,
        "target_change_id": receipt.target_change_id,
    }


def _provenance(host: MemoryHost) -> MemoryProvenance:
    return MemoryProvenance(
        origin_kind="answer_run",
        origin_id=host.run_id,
        run_id=host.run_id,
        session_id=host.session_id,
    )


def _idempotency_key(host: MemoryHost, runtime: ToolRuntime) -> str:
    return f"answer:{host.run_id}:{host.session_id}:{runtime.call_id}"


__all__ = [
    "ForgetInput",
    "MemoryHost",
    "RecallInput",
    "RememberInput",
    "forget_tool",
    "recall_memory_tool",
    "remember_tool",
]
