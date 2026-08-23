# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""stdio MCP server: subject binding and the three model tools."""

import pytest
from dlightrag_memory import (
    Memory,
    MemoryProvenance,
    MemoryWriteRejectedError,
)
from dlightrag_memory.mcp_server import _forget, _recall, _remember, build_memory_server
from dlightrag_memory.store import InMemoryMemoryStore


async def _memory() -> Memory:
    return Memory(InMemoryMemoryStore())


async def test_recall_returns_bound_subject_records() -> None:
    memory = await _memory()
    await memory.remember(
        owner_id="pi-user",
        kind="preference",
        body="No email.",
        confidence=0.9,
        provenance=MemoryProvenance(run_id="r", session_id="s"),
    )

    result = await _recall(memory, subject="pi-user", query="email")

    assert [record["body"] for record in result["records"]] == ["No email."]


async def test_remember_writes_with_mcp_provenance() -> None:
    memory = await _memory()

    stored = await _remember(
        memory,
        subject="pi-user",
        kind="fact",
        body="Project uses ruff.",
        confidence=1.0,
        supersedes_id=None,
        idempotency_key="write-1",
    )

    assert stored["stored"] is True
    (record,) = await memory.list_active(owner_id="pi-user")
    assert record.provenance.run_id == "mcp:pi-user"

    replay = await _remember(
        memory,
        subject="pi-user",
        kind="fact",
        body="Project uses ruff.",
        confidence=1.0,
        supersedes_id=None,
        idempotency_key="write-1",
    )
    assert replay["memory_id"] == stored["memory_id"]
    assert len(await memory.list_active(owner_id="pi-user")) == 1


async def test_remember_rejects_oversized_bodies() -> None:
    memory = await _memory()

    with pytest.raises(MemoryWriteRejectedError):
        await _remember(
            memory,
            subject="pi-user",
            kind="fact",
            body="x" * 501,
            confidence=1.0,
            supersedes_id=None,
            idempotency_key="oversized-1",
        )


async def test_forget_by_id() -> None:
    memory = await _memory()
    await memory.remember(
        owner_id="pi-user",
        kind="fact",
        body="Keep me.",
        confidence=1.0,
        provenance=MemoryProvenance(run_id="r", session_id="s"),
    )
    (record,) = await memory.list_active(owner_id="pi-user")

    assert await _forget(memory, subject="pi-user", memory_id=record.memory_id, body=None) == (
        "Forgotten."
    )
    assert await memory.list_active(owner_id="pi-user") == ()


async def test_build_server_registers_exactly_three_tools() -> None:
    server = build_memory_server(await _memory(), subject="pi-user")

    tools = await server.list_tools()

    assert {tool.name for tool in tools} == {"memory_recall", "memory_remember", "memory_forget"}


async def test_build_server_requires_a_subject() -> None:
    with pytest.raises(ValueError, match="subject"):
        build_memory_server(await _memory(), subject="   ")
