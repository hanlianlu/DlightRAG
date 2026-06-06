# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for PostgreSQL-backed workspace registry."""

from __future__ import annotations

from typing import Any

from dlightrag.storage.workspaces import PGWorkspaceRegistry


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

    async def fetch(self, query: str) -> list[dict[str, Any]]:
        assert "dlightrag_workspace_meta" in query
        return self.rows

    async def fetchval(self, query: str, *args: Any) -> Any:
        if "dlightrag_schema_migrations" in query and "version" in query:
            return 1 if (str(args[0]), str(args[1])) in self.applied else None
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
    assert (
        "ALTER TABLE dlightrag_workspace_meta ADD COLUMN IF NOT EXISTS display_name" in executed_sql
    )
    assert any(
        query.startswith("INSERT INTO dlightrag_schema_migrations")
        and args[:2] == ("workspace_registry", "0001_workspace_meta")
        for query, args in conn.executed
    )


async def test_workspace_registry_initialization_canonicalizes_existing_rows() -> None:
    conn = _Conn()
    conn.rows = [
        {
            "workspace": "project-alpha",
            "display_name": "Project Alpha",
            "embedding_model": "voyage-multimodal-3.5",
            "created_at": None,
            "updated_at": None,
        }
    ]
    registry = PGWorkspaceRegistry(pool=_Pool(conn))

    await registry.initialize()

    assert (
        "project_alpha",
        "Project Alpha",
        "voyage-multimodal-3.5",
    ) in [args for _, args in conn.executed]
    assert ("project-alpha",) in [args for _, args in conn.executed]


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


async def test_workspace_registry_normalizes_write_operations() -> None:
    conn = _Conn()
    registry = PGWorkspaceRegistry(pool=_Pool(conn))

    await registry.upsert(
        workspace="Project Alpha",
        display_name="Project Alpha",
        embedding_model="voyage-multimodal-3.5",
    )
    exists = await registry.exists("Project Alpha")
    await registry.delete("Project Alpha")

    assert exists is False
    assert (
        "project_alpha",
        "Project Alpha",
        "voyage-multimodal-3.5",
    ) in [args for _, args in conn.executed]
    assert ("project_alpha",) in [args for _, args in conn.executed]
