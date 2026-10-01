# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Standalone Memory MCP: subject binding, its four tools, and pre-storage refusals.

The tools' round trips through the store run against PostgreSQL in
tests/integration/test_memory_pg.py.
"""

import pytest
from dlightrag_memory import Memory, MemoryWriteRejectedError
from dlightrag_memory.mcp_server import _remember, build_memory_server
from dlightrag_memory.postgres import PostgresMemoryStore


async def _unreachable_pool() -> object:
    raise AssertionError("this test must not reach storage")


def _memory() -> Memory:
    return Memory(PostgresMemoryStore(pool_factory=_unreachable_pool))


async def test_remember_rejects_oversized_bodies_before_storage() -> None:
    with pytest.raises(MemoryWriteRejectedError):
        await _remember(
            _memory(),
            subject="pi-user",
            kind="fact",
            body="x" * 501,
            supersedes_id=None,
            idempotency_key="oversized-1",
        )


async def test_build_server_registers_exactly_four_tools() -> None:
    server = build_memory_server(_memory(), subject="pi-user")
    tools = await server.list_tools()
    assert {tool.name for tool in tools} == {
        "memory_recall",
        "memory_remember",
        "memory_forget",
        "memory_undo",
    }


def test_build_server_requires_a_subject() -> None:
    with pytest.raises(ValueError, match="subject"):
        build_memory_server(_memory(), subject="   ")
