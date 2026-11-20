# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Standalone Memory MCP: subject binding, its four tools, and pre-storage refusals.

The tools' round trips through the store run against PostgreSQL in
tests/integration/test_memory_pg.py.
"""

from datetime import UTC, datetime, timedelta, timezone
from typing import Any, cast

import pytest
from dlightrag_memory import (
    Memory,
    MemoryProvenance,
    MemoryRecord,
    MemoryWriteRejectedError,
    RecallResult,
)
from dlightrag_memory.mcp_server import _lookup, _remember, build_memory_server
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


def test_build_server_requires_a_subject() -> None:
    with pytest.raises(ValueError, match="subject"):
        build_memory_server(_memory(), subject="   ")


def _record(
    *, updated_at: datetime | None = None, created_at: datetime | None = None
) -> MemoryRecord:
    return MemoryRecord(
        owner_id="pi-user",
        memory_id="m-1",
        kind="fact",
        body="Lives in Berlin.",
        provenance=MemoryProvenance(origin_kind="mcp", origin_id="seed"),
        created_at=created_at,
        updated_at=updated_at,
    )


class _StubMemory:
    def __init__(self, result: RecallResult) -> None:
        self._result = result

    async def lookup(self, *, owner_id: str, query: str) -> RecallResult:
        return self._result


async def test_lookup_records_carry_when_they_were_last_written_to_the_hour_in_utc() -> None:
    plus_eight = timezone(timedelta(hours=8))
    result = RecallResult(
        relevant=(_record(updated_at=datetime(2026, 10, 18, 1, 5, tzinfo=plus_eight)),),
        standing=(_record(created_at=datetime(2025, 1, 2, 3, 4, tzinfo=UTC)),),
        recent=(_record(),),
    )

    listed = await _lookup(cast(Any, _StubMemory(result)), subject="pi-user", query="home")

    assert listed["relevant"] == [
        {
            "memory_id": "m-1",
            "kind": "fact",
            "body": "Lives in Berlin.",
            "written_at_utc": "2026-10-17 17:00",
            "quote": None,
        }
    ]
    assert listed["standing"][0]["written_at_utc"] == "2025-01-02 03:00"
    # A record with neither timestamp carries the field as null.
    assert listed["recent"][0]["written_at_utc"] is None


async def test_lookup_records_carry_the_owners_words_behind_them_where_there_are_any() -> None:
    quoted = MemoryRecord(
        owner_id="pi-user",
        memory_id="m-2",
        kind="preference",
        body="The owner prefers charts in a dark theme.",
        provenance=MemoryProvenance(origin_kind="session_pass", origin_id="s:1-2"),
        quote="以后图表都用深色主题",
    )
    result = RecallResult(relevant=(quoted,), standing=(_record(),), recent=(quoted,))

    listed = await _lookup(cast(Any, _StubMemory(result)), subject="pi-user", query="charts")

    for section in ("relevant", "recent"):
        assert listed[section][0]["quote"] == "以后图表都用深色主题"
    assert listed["standing"][0]["quote"] is None


async def test_lookup_tool_description_names_written_at_utc() -> None:
    server = build_memory_server(_memory(), subject="pi-user")
    (lookup,) = [tool for tool in await server.list_tools() if tool.name == "memory_lookup"]
    assert "written_at_utc" in (lookup.description or "")
    assert "quote" in (lookup.description or "")
