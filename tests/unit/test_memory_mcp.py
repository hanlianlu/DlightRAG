# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Standalone Memory MCP: subject binding, receipts, and four tools."""

import pytest
from dlightrag_memory import Memory, MemoryProvenance, MemoryWriteRejectedError
from dlightrag_memory.mcp_server import _forget, _recall, _remember, _undo, build_memory_server
from dlightrag_memory.store import InMemoryMemoryStore


async def _memory() -> Memory:
    return Memory(InMemoryMemoryStore())


async def test_recall_returns_bound_subject_records_with_ids() -> None:
    memory = await _memory()
    await memory.remember(
        owner_id="pi-user",
        kind="preference",
        body="No email.",
        provenance=MemoryProvenance(origin_kind="mcp", origin_id="seed"),
        idempotency_key="seed",
    )

    result = await _recall(memory, subject="pi-user", query="email")

    assert result["records"][0]["body"] == "No email."
    assert result["records"][0]["memory_id"]


async def test_remember_writes_with_mcp_provenance_and_replays_receipt() -> None:
    memory = await _memory()
    stored = await _remember(
        memory,
        subject="pi-user",
        kind="fact",
        body="Project uses ruff.",
        supersedes_id=None,
        idempotency_key="write-1",
    )
    replay = await _remember(
        memory,
        subject="pi-user",
        kind="fact",
        body="Project uses ruff.",
        supersedes_id=None,
        idempotency_key="write-1",
    )

    assert stored["outcome"] == "changed"
    assert replay == stored
    (record,) = await memory.list_active(owner_id="pi-user")
    assert record.provenance.origin_kind == "mcp"


async def test_remember_rejects_oversized_bodies() -> None:
    memory = await _memory()
    with pytest.raises(MemoryWriteRejectedError):
        await _remember(
            memory,
            subject="pi-user",
            kind="fact",
            body="x" * 501,
            supersedes_id=None,
            idempotency_key="oversized-1",
        )


async def test_forget_and_undo_return_operation_receipts() -> None:
    memory = await _memory()
    stored = await _remember(
        memory,
        subject="pi-user",
        kind="fact",
        body="Keep me.",
        supersedes_id=None,
        idempotency_key="write-1",
    )
    forgotten = await _forget(
        memory,
        subject="pi-user",
        memory_id=stored["memory_ids"][0],
        body=None,
        idempotency_key="forget-1",
    )
    assert forgotten["outcome"] == "changed"

    undone = await _undo(
        memory,
        subject="pi-user",
        change_id=forgotten["change_id"],
        idempotency_key="undo-1",
    )
    assert undone["outcome"] == "changed"
    assert [row.body for row in await memory.list_active(owner_id="pi-user")] == ["Keep me."]


async def test_build_server_registers_exactly_four_tools() -> None:
    server = build_memory_server(await _memory(), subject="pi-user")
    tools = await server.list_tools()
    assert {tool.name for tool in tools} == {
        "memory_recall",
        "memory_remember",
        "memory_forget",
        "memory_undo",
    }


async def test_build_server_requires_a_subject() -> None:
    with pytest.raises(ValueError, match="subject"):
        build_memory_server(await _memory(), subject="   ")
