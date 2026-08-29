# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for PostgreSQL-backed workspace registry."""

from typing import Any

import pytest

from dlightrag.adapters.postgres.corpus.workspaces import PGWorkspaceRegistry


class _Acquire:
    def __init__(self, conn: _Conn) -> None:
        self._conn = conn

    async def __aenter__(self) -> _Conn:
        return self._conn

    async def __aexit__(self, *args: object) -> None:
        return None


class _Tx:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *args: object) -> None:
        return None


class _Pool:
    def __init__(self, conn: _Conn) -> None:
        self._conn = conn

    def acquire(self) -> _Acquire:
        return _Acquire(self._conn)


class _Conn:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.fetched: list[tuple[str, tuple[Any, ...]]] = []
        self.applied: set[tuple[str, str]] = set()
        self.rows: list[dict[str, Any]] = [
            {
                "workspace": "default",
                "display_name": "Default",
                "embedding_model": "voyage-multimodal-3.5",
                "created_at": None,
                "updated_at": None,
            },
            {
                "workspace": "research",
                "display_name": "Research",
                "embedding_model": "voyage-multimodal-3.5",
                "created_at": None,
                "updated_at": None,
            },
        ]

    async def execute(self, query: str, *args: Any) -> None:
        self.executed.append((query, args))
        if query.startswith("INSERT INTO dlightrag_schema_migrations"):
            self.applied.add((str(args[0]), str(args[1])))

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.fetched.append((query, args))
        if "workspace > $1" in query:
            after = str(args[0] or "")
            limit = int(args[-1])
            return sorted(
                (row for row in self.rows if row["workspace"] > after),
                key=lambda row: row["workspace"],
            )[:limit]
        if "dlightrag_schema_migrations" in query and "version" in query:
            scope = str(args[0])
            versions = sorted(
                version for applied_scope, version in self.applied if applied_scope == scope
            )
            return [{"version": version} for version in versions]
        assert "dlightrag_workspace_meta" in query
        return self.rows

    async def fetchval(self, query: str, *args: Any) -> Any:
        if "SELECT EXISTS" in query and "dlightrag_workspace_meta" in query:
            return any(row["workspace"] == str(args[0]) for row in self.rows)
        return True

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        if "dlightrag_workspace_meta" in query:
            workspace = str(args[0])
            return next((row for row in self.rows if row["workspace"] == workspace), None)
        return None

    def transaction(self) -> _Tx:
        return _Tx()


async def test_workspace_registry_initializes_and_migrates_schema() -> None:
    conn = _Conn()
    registry = PGWorkspaceRegistry(pool=_Pool(conn))

    await registry.initialize()

    executed_sql = "\n".join(query for query, _ in conn.executed)
    assert "CREATE TABLE IF NOT EXISTS dlightrag_schema_migrations" in executed_sql
    assert "CREATE TABLE IF NOT EXISTS dlightrag_workspace_meta" in executed_sql
    assert "display_name" in executed_sql
    assert any(
        query.startswith("INSERT INTO dlightrag_schema_migrations")
        and args[:2] == ("workspace_registry", "workspace_meta")
        for query, args in conn.executed
    )


async def test_workspace_registry_upserts_lists_and_deletes() -> None:
    conn = _Conn()
    registry = PGWorkspaceRegistry(pool=_Pool(conn))

    await registry.upsert(
        workspace="new_workspace",
        display_name="New Workspace",
        embedding_model="voyage-multimodal-3.5",
    )
    records = await registry.list()
    await registry.delete("old_workspace")

    assert records == conn.rows
    assert (
        "new_workspace",
        "New Workspace",
        "voyage-multimodal-3.5",
    ) in [args for _, args in conn.executed]
    assert ("old_workspace",) in [args for _, args in conn.executed]


async def test_workspace_registry_exists_uses_one_primary_key_point_lookup() -> None:
    conn = _Conn()
    registry = PGWorkspaceRegistry(pool=_Pool(conn))

    assert await registry.exists("research") is True
    assert await registry.exists("missing") is False


async def test_workspace_registry_rejects_an_empty_workspace() -> None:
    conn = _Conn()
    registry = PGWorkspaceRegistry(pool=_Pool(conn))

    with pytest.raises(ValueError, match="workspace cannot be empty"):
        await registry.exists("  ")


async def test_workspace_registry_list_page_uses_ascending_keyset_without_offset() -> None:
    conn = _Conn()
    registry = PGWorkspaceRegistry(pool=_Pool(conn))

    first = await registry.list_page(after_workspace=None, limit=1)

    assert [item["workspace"] for item in first.items] == ["default"]
    assert first.has_more is True
    assert first.fetched_rows == 2
    page_sql, page_args = conn.fetched[-1]
    assert "ORDER BY workspace ASC" in page_sql
    assert "OFFSET" not in page_sql
    assert int(page_args[-1]) == 2  # limit + 1
    assert str(page_args[0]) == ""  # before-first key

    second = await registry.list_page(after_workspace="default", limit=1)

    assert [item["workspace"] for item in second.items] == ["research"]
    assert second.has_more is False
    assert second.fetched_rows == 1


async def test_workspace_registry_list_page_rejects_invalid_inputs() -> None:
    conn = _Conn()
    registry = PGWorkspaceRegistry(pool=_Pool(conn))

    with pytest.raises(ValueError, match="canonical"):
        await registry.list_page(after_workspace="Finance!", limit=1)
    with pytest.raises(ValueError, match="limit"):
        await registry.list_page(after_workspace=None, limit=0)
    with pytest.raises(ValueError, match="limit"):
        await registry.list_page(after_workspace=None, limit=101)
    with pytest.raises(ValueError, match="workspace cannot be empty"):
        await registry.upsert(
            workspace="  ",
            display_name="Empty",
            embedding_model="voyage-multimodal-3.5",
        )
