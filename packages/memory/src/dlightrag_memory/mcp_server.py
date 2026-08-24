# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Subject-bound stdio MCP host for the independent Profile Memory package."""

from __future__ import annotations

import argparse
import asyncio
from typing import Annotated, Any, Literal

from mcp.server import MCPServer
from mcp.types import ToolAnnotations
from pydantic import Field

from dlightrag_memory import (
    Memory,
    MemoryOperationReceipt,
    MemoryProvenance,
    MemoryWriteRejectedError,
    __version__,
)
from dlightrag_memory.policy import MEMORY_BODY_LIMIT
from dlightrag_memory.postgres import PostgresMemoryStore

SERVER_NAME = "dlightrag-memory"
_MemoryKind = Literal["preference", "fact"]


async def _recall(memory: Memory, *, subject: str, query: str) -> dict[str, Any]:
    result = await memory.recall(owner_id=subject, query=query)
    return {
        "records": [
            {"memory_id": record.memory_id, "kind": record.kind, "body": record.body}
            for record in result.records
        ]
    }


async def _remember(
    memory: Memory,
    *,
    subject: str,
    kind: _MemoryKind,
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
    """One authorized server with four tools over one bound subject."""
    if not subject.strip():
        raise ValueError("a memory subject is required")
    server = MCPServer(SERVER_NAME, version=__version__, log_level="INFO")

    @server.tool(
        name="memory_recall",
        description=(
            "Recall owner preferences and facts relevant to a query, including ids needed "
            "before replacing or forgetting one. Context only; never citable."
        ),
        annotations=ToolAnnotations(read_only_hint=True),
    )
    async def memory_recall(
        query: Annotated[str, Field(min_length=1, description="What to recall memories for.")],
    ) -> dict[str, Any]:
        return await _recall(memory, subject=subject, query=query)

    @server.tool(
        name="memory_remember",
        description=(
            "Store one durable owner preference or fact. Do not store task state, research "
            "claims, citations, transcripts, credentials, or private keys."
        ),
    )
    async def memory_remember(
        kind: Annotated[_MemoryKind, Field(description="preference or fact")],
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
