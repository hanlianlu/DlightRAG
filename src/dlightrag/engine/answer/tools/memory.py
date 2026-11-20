# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Parent Research tools for owner Profile Memory."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from dlightrag_memory import (
    Memory,
    MemoryKind,
    MemoryOperationReceipt,
    MemoryProvenance,
    MemoryWriteRejectedError,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator

from dlightrag.engine.agent.tools.contracts import (
    AgentTool,
    ToolDeclaration,
    ToolResult,
    ToolRuntime,
)
from dlightrag.engine.answer.memory import MEMORY_INACTIVE, RECALL_TIME_NOTE, recall_sections

_MEMORY_MUTATION_LIMIT = 10
MEMORY_REMEMBER = "memory_remember"
MEMORY_FORGET = "memory_forget"
MEMORY_UNDO = "memory_undo"
MEMORY_LOOKUP = "memory_lookup"
#: The memory Tools that change the owner's profile: Run authority a Child never holds.
MEMORY_MUTATION_TOOLS = (MEMORY_REMEMBER, MEMORY_FORGET, MEMORY_UNDO)


def _is_a_memory_id(value: str | None) -> str | None:
    """Refuse an argument that cannot be a memory id, which storage would raise on.

    A check, not a schema constraint: a Run's tool schemas are pinned when it is accepted.
    An empty string is the argument left out, as models often send for an optional field.
    """
    if not value:
        return None
    try:
        uuid.UUID(value)
    except ValueError:
        raise ValueError(f"{value!r} is not a memory id; call memory_lookup for ids") from None
    return value


class RememberInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    kind: MemoryKind
    body: str = Field(min_length=1, max_length=500, description="What to remember.")
    supersedes_id: str | None = Field(default=None, description="Active memory id this replaces.")

    _supersedes_id_is_an_id = field_validator("supersedes_id")(_is_a_memory_id)


class ForgetInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    memory_id: str | None = Field(default=None, description="Id of the memory to forget.")
    body: str | None = Field(default=None, description="Exact body if the id is unknown.")

    _memory_id_is_an_id = field_validator("memory_id")(_is_a_memory_id)


class UndoInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    memory_id: str | None = Field(
        default=None, description="Id of the active memory whose last change to undo."
    )
    body: str | None = Field(default=None, description="Exact body if the id is unknown.")

    _memory_id_is_an_id = field_validator("memory_id")(_is_a_memory_id)


class LookupInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1, description="What to look memories up for.")


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
        raise MemoryWriteRejectedError(MEMORY_INACTIVE)


def remember_declaration() -> ToolDeclaration:
    return ToolDeclaration(
        MEMORY_REMEMBER,
        "Store one durable memory for future conversations: a preference (what the owner "
        "wants of an answer), a fact (what the owner is like), or an experience (what you "
        "learned to be true about your own work, worth carrying forward). Use when the user "
        "explicitly asks to remember eligible stable information, for one minimally "
        "inferred repeated preference, or for one experience that would have saved this "
        "work. Never store task state, research claims, evidence, citations, "
        "transcripts, credentials, or private keys. Look up first and pass supersedes_id "
        "when correcting an existing memory.",
        RememberInput,
        replay_policy="replayable",
    )


def remember_tool(*, host: MemoryHost) -> AgentTool:
    async def execute(raw: BaseModel, runtime: ToolRuntime) -> ToolResult:
        args = raw if isinstance(raw, RememberInput) else RememberInput.model_validate(raw)
        if host.memory is None or not await _available(host):
            return _rejected("remember", MEMORY_INACTIVE)
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
        MEMORY_FORGET,
        "Forget one active Profile Memory by id or exact body. Look up first when the id is "
        "unknown.",
        ForgetInput,
        replay_policy="replayable",
    )


def forget_tool(*, host: MemoryHost) -> AgentTool:
    async def execute(raw: BaseModel, runtime: ToolRuntime) -> ToolResult:
        args = raw if isinstance(raw, ForgetInput) else ForgetInput.model_validate(raw)
        if host.memory is None or not await _available(host):
            return _rejected("forget", MEMORY_INACTIVE)
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
        if receipt.outcome == "unchanged":
            return await _forget_unmatched(host.memory, host.owner_id, receipt, args)
        return _receipt_result(receipt)

    return forget_declaration().bind(execute)


def undo_declaration() -> ToolDeclaration:
    return ToolDeclaration(
        MEMORY_UNDO,
        "Undo the last change to one active Profile Memory, named by id or exact body: a "
        "new memory is forgotten, a replaced one comes back, a merged one splits back into "
        "its sources. One change per call, and an undo cannot itself be undone.",
        UndoInput,
        replay_policy="replayable",
    )


def undo_tool(*, host: MemoryHost) -> AgentTool:
    async def execute(raw: BaseModel, runtime: ToolRuntime) -> ToolResult:
        args = raw if isinstance(raw, UndoInput) else UndoInput.model_validate(raw)
        if host.memory is None or not await _available(host):
            return _rejected("undo", MEMORY_INACTIVE)
        try:
            receipt = await host.memory.undo(
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
            return _rejected("undo", str(exc.public_message))
        if receipt.outcome == "conflict" and receipt.target_change_id is None:
            return await _undo_unmatched(host.memory, host.owner_id, receipt, args)
        return _receipt_result(receipt)

    return undo_declaration().bind(execute)


def lookup_declaration() -> ToolDeclaration:
    return ToolDeclaration(
        MEMORY_LOOKUP,
        "Look up the owner's memories relevant to a query — including standing memories "
        "the injected block left out — plus the newest other memories, with the ids needed "
        "before replacing, forgetting or undoing one. Context only; never evidence or a "
        "citation.",
        LookupInput,
        replay_policy="replayable",
    )


def lookup_tool(*, host: MemoryHost) -> AgentTool:
    async def execute(raw: BaseModel, runtime: ToolRuntime) -> ToolResult:
        args = raw if isinstance(raw, LookupInput) else LookupInput.model_validate(raw)
        if host.memory is None or not await _available(host):
            return ToolResult.text(MEMORY_INACTIVE, is_error=True)
        await runtime.emit_update(ToolResult.text("", subject=args.query))
        result = await host.memory.lookup(owner_id=host.owner_id, query=args.query)
        lines = recall_sections(result, ids=True)
        if not lines:
            return ToolResult.text("No remembered memories.")
        return ToolResult.text(
            "\n".join((f"Remembered about this owner ({RECALL_TIME_NOTE}):", *lines))
        )

    return lookup_declaration().bind(execute)


def _receipt_result(receipt: MemoryOperationReceipt) -> ToolResult:
    if receipt.outcome == "changed":
        verb = {"remember": "Remembered", "forget": "Forgot", "undo": "Undid"}.get(
            receipt.action, "Changed"
        )
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


async def _forget_unmatched(
    memory: Memory, owner_id: str, receipt: MemoryOperationReceipt, args: ForgetInput
) -> ToolResult:
    """Say why a forget matched no active memory: none by that name, or one already gone.

    Only an ``unchanged`` receipt comes here; a replayed ``changed`` one is read back as the
    original receipt and never reaches the check.
    """
    details = {"memory_operation": _receipt_details(receipt)}
    if args.memory_id is None:
        return ToolResult.text(
            "No active Profile Memory has exactly that body. Call memory_lookup to find its id.",
            details=details,
            is_error=True,
        )
    record = await memory.get(owner_id=owner_id, memory_id=args.memory_id)
    if record is None:
        return ToolResult.text(
            f"No Profile Memory has id {args.memory_id}. Call memory_lookup for current ids.",
            details=details,
            is_error=True,
        )
    if record.status == "forgotten":
        return ToolResult.text(
            f"Profile Memory {record.memory_id} was already forgotten.", details=details
        )
    if record.status == "superseded":
        return ToolResult.text(
            f"Profile Memory {record.memory_id} was already replaced by a newer memory; "
            "call memory_lookup to see it.",
            details=details,
        )
    return _receipt_result(receipt)


async def _undo_unmatched(
    memory: Memory, owner_id: str, receipt: MemoryOperationReceipt, args: UndoInput
) -> ToolResult:
    """Say why an undo matched no active memory: none by that name, or one no longer active.

    Only a ``conflict`` receipt without a target change comes here, which is what the store settles
    when it finds no active record; every other conflict names the change it could not compensate
    and keeps the concurrency message. A replayed ``changed`` receipt is read back as the original
    and never reaches the check.
    """
    details = {"memory_operation": _receipt_details(receipt)}
    if args.memory_id is None:
        return ToolResult.text(
            "No active Profile Memory has exactly that body. Call memory_lookup to find its id.",
            details=details,
            is_error=True,
        )
    record = await memory.get(owner_id=owner_id, memory_id=args.memory_id)
    if record is None:
        return ToolResult.text(
            f"No Profile Memory has id {args.memory_id}. Call memory_lookup for current ids.",
            details=details,
            is_error=True,
        )
    if record.status == "forgotten":
        return ToolResult.text(
            f"Profile Memory {record.memory_id} was forgotten, so it is not active; memory_undo "
            "names an active memory. Call memory_lookup for current ids.",
            details=details,
            is_error=True,
        )
    if record.status == "superseded":
        return ToolResult.text(
            f"Profile Memory {record.memory_id} was replaced by a newer memory, so it is not "
            "active; memory_undo names an active memory. Call memory_lookup to find the newer one.",
            details=details,
            is_error=True,
        )
    return _receipt_result(receipt)


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
    "MEMORY_FORGET",
    "MEMORY_LOOKUP",
    "MEMORY_MUTATION_TOOLS",
    "MEMORY_REMEMBER",
    "MEMORY_UNDO",
    "ForgetInput",
    "LookupInput",
    "MemoryHost",
    "RememberInput",
    "UndoInput",
    "forget_declaration",
    "forget_tool",
    "lookup_declaration",
    "lookup_tool",
    "remember_declaration",
    "remember_tool",
    "undo_declaration",
    "undo_tool",
]
