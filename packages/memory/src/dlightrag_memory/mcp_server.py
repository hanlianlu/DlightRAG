# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Subject-bound stdio MCP host for the independent Profile Memory package."""

from __future__ import annotations

import argparse
import asyncio
from typing import Annotated, Any

from mcp.server import MCPServer
from mcp.types import ToolAnnotations
from pydantic import Field

from dlightrag_memory import (
    Memory,
    MemoryKind,
    MemoryOperationReceipt,
    MemoryProvenance,
    MemoryRecord,
    MemoryWriteRejectedError,
    __version__,
)
from dlightrag_memory.memory import recall_time
from dlightrag_memory.policy import (
    MEMORY_BODY_LIMIT,
    MEMORY_FUSE_MAX_SOURCES,
    MEMORY_FUSE_MIN_SOURCES,
)
from dlightrag_memory.postgres import PostgresMemoryStore

SERVER_NAME = "dlightrag-memory"


def _listed(records: tuple[MemoryRecord, ...]) -> list[dict[str, str | None]]:
    return [
        {
            "memory_id": r.memory_id,
            "kind": r.kind,
            "body": r.body,
            "written_at_utc": recall_time(r),
            "quote": r.quote,
        }
        for r in records
    ]


async def _lookup(memory: Memory, *, subject: str, query: str) -> dict[str, Any]:
    result = await memory.lookup(owner_id=subject, query=query)
    return {
        "relevant": _listed(result.relevant),
        "standing": _listed(result.standing),
        "standing_omitted": result.standing_omitted,
        "recent": _listed(result.recent),
    }


async def _remember(
    memory: Memory,
    *,
    subject: str,
    kind: MemoryKind,
    body: str,
    supersedes_id: str | None,
    idempotency_key: str,
) -> dict[str, Any]:
    receipt = await memory.remember(
        owner_id=subject,
        kind=kind,
        body=body,
        provenance=_provenance(idempotency_key),
        supersedes_id=supersedes_id,
        idempotency_key=f"mcp:{subject}:{idempotency_key}",
    )
    return _receipt(receipt)


async def _forget(
    memory: Memory,
    *,
    subject: str,
    memory_id: str | None,
    body: str | None,
    idempotency_key: str,
) -> dict[str, Any]:
    receipt = await memory.forget(
        owner_id=subject,
        memory_id=memory_id,
        body=body,
        provenance=_provenance(idempotency_key),
        idempotency_key=f"mcp:{subject}:{idempotency_key}",
    )
    return _receipt(receipt)


async def _fuse(
    memory: Memory,
    *,
    subject: str,
    memory_ids: list[str],
    body: str,
    kind: MemoryKind | None,
    idempotency_key: str,
) -> dict[str, Any]:
    receipt = await memory.fuse(
        owner_id=subject,
        memory_ids=memory_ids,
        body=body,
        kind=kind,
        provenance=_provenance(idempotency_key),
        idempotency_key=f"mcp:{subject}:{idempotency_key}",
    )
    return _receipt(receipt)


async def _undo(
    memory: Memory,
    *,
    subject: str,
    change_id: str,
    idempotency_key: str,
) -> dict[str, Any]:
    receipt = await memory.undo(
        owner_id=subject,
        change_id=change_id,
        provenance=MemoryProvenance(
            origin_kind="undo",
            origin_id=f"mcp:{subject}:{idempotency_key}",
        ),
        idempotency_key=f"mcp:{subject}:{idempotency_key}",
    )
    return _receipt(receipt)


def build_memory_server(memory: Memory, *, subject: str) -> MCPServer:
    """One authorized server with five tools over one bound subject."""
    if not subject.strip():
        raise ValueError("a memory subject is required")
    server = MCPServer(SERVER_NAME, version=__version__, log_level="INFO")

    @server.tool(
        name="memory_lookup",
        description=(
            "Look up the owner's memories relevant to a query, the standing preferences and "
            "experiences, and the newest other memories, with the ids needed before "
            "replacing, forgetting, merging or undoing one. Each memory carries written_at_utc, "
            "when it was last written, to the hour, and a quote where the owner's own words stand "
            "behind it. Context only; never citable."
        ),
        annotations=ToolAnnotations(read_only_hint=True),
    )
    async def memory_lookup(
        query: Annotated[str, Field(min_length=1, description="What to look memories up for.")],
    ) -> dict[str, Any]:
        return await _lookup(memory, subject=subject, query=query)

    @server.tool(
        name="memory_remember",
        description=(
            "Store one durable memory: a preference (what the owner wants of an answer), a "
            "fact (what the owner is like), or an experience (what an agent learned to be "
            "true about its own work). Do not store task state, research claims, citations, "
            "transcripts, credentials, or private keys."
        ),
    )
    async def memory_remember(
        kind: Annotated[MemoryKind, Field(description="preference, fact or experience")],
        body: Annotated[
            str,
            Field(min_length=1, max_length=MEMORY_BODY_LIMIT, description="What to remember."),
        ],
        supersedes_id: Annotated[
            str | None, Field(default=None, description="Active memory id this replaces.")
        ],
        idempotency_key: Annotated[
            str,
            Field(
                min_length=1,
                max_length=255,
                description="Stable mutation key reused verbatim when retrying this operation.",
            ),
        ],
    ) -> dict[str, Any]:
        try:
            return await _remember(
                memory,
                subject=subject,
                kind=kind,
                body=body,
                supersedes_id=supersedes_id,
                idempotency_key=idempotency_key,
            )
        except MemoryWriteRejectedError as exc:
            raise ValueError(exc.public_message) from exc

    @server.tool(
        name="memory_forget",
        description="Idempotently forget one active preference or fact by id or exact body.",
    )
    async def memory_forget(
        memory_id: Annotated[
            str | None, Field(default=None, description="Id of the memory to forget.")
        ],
        body: Annotated[
            str | None, Field(default=None, description="Exact body if the id is unknown.")
        ],
        idempotency_key: Annotated[
            str,
            Field(min_length=1, max_length=255, description="Stable retry key."),
        ],
    ) -> dict[str, Any]:
        try:
            return await _forget(
                memory,
                subject=subject,
                memory_id=memory_id,
                body=body,
                idempotency_key=idempotency_key,
            )
        except MemoryWriteRejectedError as exc:
            raise ValueError(exc.public_message) from exc

    @server.tool(
        name="memory_fuse",
        description=(
            "Replace several active memories that say one thing with one merged sentence the "
            "caller supplies. Name kind when the memories are of different kinds."
        ),
    )
    async def memory_fuse(
        memory_ids: Annotated[
            list[str],
            Field(
                min_length=MEMORY_FUSE_MIN_SOURCES,
                max_length=MEMORY_FUSE_MAX_SOURCES,
                description="Ids of the active memories this one sentence replaces.",
            ),
        ],
        body: Annotated[
            str,
            Field(min_length=1, max_length=MEMORY_BODY_LIMIT, description="The merged memory."),
        ],
        idempotency_key: Annotated[
            str,
            Field(min_length=1, max_length=255, description="Stable retry key."),
        ],
        kind: Annotated[
            MemoryKind | None,
            Field(default=None, description="Kind of the result; required for mixed kinds."),
        ] = None,
    ) -> dict[str, Any]:
        try:
            return await _fuse(
                memory,
                subject=subject,
                memory_ids=memory_ids,
                body=body,
                kind=kind,
                idempotency_key=idempotency_key,
            )
        except MemoryWriteRejectedError as exc:
            raise ValueError(exc.public_message) from exc

    @server.tool(
        name="memory_undo",
        description="Compensate one still-current Memory change by its change id.",
    )
    async def memory_undo(
        change_id: Annotated[str, Field(min_length=1, description="Change id to undo.")],
        idempotency_key: Annotated[
            str,
            Field(min_length=1, max_length=255, description="Stable retry key."),
        ],
    ) -> dict[str, Any]:
        try:
            return await _undo(
                memory,
                subject=subject,
                change_id=change_id,
                idempotency_key=idempotency_key,
            )
        except MemoryWriteRejectedError as exc:
            raise ValueError(exc.public_message) from exc

    return server


def _provenance(idempotency_key: str) -> MemoryProvenance:
    return MemoryProvenance(origin_kind="mcp", origin_id=idempotency_key)


def _receipt(receipt: MemoryOperationReceipt) -> dict[str, Any]:
    return {
        "action": receipt.action,
        "body": receipt.body,
        "change_id": receipt.change_id,
        "kind": receipt.kind,
        "memory_ids": list(receipt.memory_ids),
        "outcome": receipt.outcome,
        "supersedes_id": receipt.supersedes_id,
        "target_change_id": receipt.target_change_id,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="dlightrag-memory-mcp")
    parser.add_argument("--dsn", required=True, help="PostgreSQL connection string")
    parser.add_argument("--subject", required=True, help="Owner subject every tool is bound to")
    return parser.parse_args()


def main() -> None:
    args = _arguments()

    async def run() -> None:
        store = PostgresMemoryStore(dsn=args.dsn)
        await store.initialize()
        server = build_memory_server(Memory(store), subject=args.subject)
        try:
            await server.run_stdio_async()
        finally:
            await store.aclose()

    asyncio.run(run())


__all__ = ["SERVER_NAME", "build_memory_server", "main"]
