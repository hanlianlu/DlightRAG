# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL workspace registry.

The registry is the durable source of truth for workspace existence and
display labels. LightRAG stores own document/KG/vector data per workspace;
this table owns the user-facing workspace list, including empty workspaces.
"""

from __future__ import annotations

from typing import Any

from dlightrag.config import PostgresTarget
from dlightrag.storage.migrations import Migration, apply_migrations, verify_migrations
from dlightrag.utils import normalize_workspace

TABLE = "dlightrag_workspace_meta"

_CREATE = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    workspace       TEXT PRIMARY KEY,
    display_name    TEXT NOT NULL DEFAULT '',
    embedding_model TEXT NOT NULL DEFAULT '',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
)
"""

_MIGRATIONS = (
    f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS display_name TEXT NOT NULL DEFAULT ''",
    f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS embedding_model TEXT NOT NULL DEFAULT ''",
    f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW()",
    f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW()",
    f"UPDATE {TABLE} SET display_name = workspace WHERE display_name = ''",
)

_UPSERT = f"""
INSERT INTO {TABLE} (workspace, display_name, embedding_model)
VALUES ($1, $2, $3)
ON CONFLICT (workspace)
DO UPDATE SET display_name = EXCLUDED.display_name,
              embedding_model = EXCLUDED.embedding_model,
              updated_at = NOW()
"""

_LIST = f"""
SELECT workspace, display_name, embedding_model, created_at, updated_at
FROM {TABLE}
ORDER BY workspace
"""

_DELETE = f"DELETE FROM {TABLE} WHERE workspace = $1"

_EXISTS = f"SELECT 1 FROM {TABLE} WHERE workspace = $1 LIMIT 1"

_CANONICALIZATION_ROWS = f"""
SELECT workspace, display_name, embedding_model
FROM {TABLE}
ORDER BY workspace
"""

_SCHEMA_MIGRATIONS = (
    Migration(
        "0001_workspace_meta",
        "Create and migrate workspace registry",
        (_CREATE, *_MIGRATIONS),
    ),
)


class PGWorkspaceRegistry:
    """Durable workspace registry backed by PostgreSQL."""

    def __init__(self, *, pool: Any = None, target: PostgresTarget | None = None) -> None:
        self._pool = pool
        self._target: PostgresTarget | None = target

    async def _run(self, operation):
        if self._pool is not None:
            async with self._pool.acquire() as conn:
                return await operation(conn)

        from dlightrag.storage.pool import pg_pool

        return await pg_pool.run(operation, target=self._target)

    async def initialize(self, *, read_only: bool = False) -> None:
        """Create or verify the registry table."""

        async def _operation(conn: Any) -> None:
            if read_only:
                exists = await conn.fetchval(
                    "SELECT EXISTS ("
                    "SELECT 1 FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_name = $1"
                    ")",
                    TABLE,
                )
                if not exists:
                    raise RuntimeError(f"{TABLE} is missing; initialize it on the primary first")
                await verify_migrations(
                    conn,
                    scope="workspace_registry",
                    migrations=_SCHEMA_MIGRATIONS,
                )
                return

            await apply_migrations(
                conn,
                scope="workspace_registry",
                migrations=_SCHEMA_MIGRATIONS,
            )
            await _canonicalize_workspace_rows(conn)

        await self._run(_operation)

    async def upsert(
        self,
        *,
        workspace: str,
        display_name: str | None,
        embedding_model: str,
    ) -> None:
        """Insert or update one workspace registry row."""
        workspace_id = _workspace_id(workspace)
        label = (display_name or workspace).strip() or workspace_id

        async def _operation(conn: Any) -> None:
            await conn.execute(_UPSERT, workspace_id, label, embedding_model)

        await self._run(_operation)

    async def list(self) -> list[dict[str, Any]]:
        """Return all registered workspaces."""

        async def _operation(conn: Any) -> list[Any]:
            return await conn.fetch(_LIST)

        rows = await self._run(_operation)
        return [dict(row) for row in rows]

    async def list_workspaces(self) -> list[str]:
        """Return all workspace identifiers."""
        return [row["workspace"] for row in await self.list()]

    async def exists(self, workspace: str) -> bool:
        """Return True when a registry row exists."""
        workspace_id = _workspace_id(workspace)

        async def _operation(conn: Any) -> Any:
            return await conn.fetchrow(_EXISTS, workspace_id)

        row = await self._run(_operation)
        return row is not None

    async def delete(self, workspace: str) -> None:
        """Delete one workspace registry row."""
        workspace_id = _workspace_id(workspace)

        async def _operation(conn: Any) -> None:
            await conn.execute(_DELETE, workspace_id)

        await self._run(_operation)


async def _canonicalize_workspace_rows(conn: Any) -> None:
    rows = await conn.fetch(_CANONICALIZATION_ROWS)
    records = [dict(row) for row in rows]
    existing = {str(row.get("workspace") or "") for row in records}
    for row in records:
        raw_workspace = str(row.get("workspace") or "")
        workspace_id = normalize_workspace(raw_workspace)
        if not workspace_id or workspace_id == raw_workspace:
            continue
        if workspace_id not in existing:
            await conn.execute(
                _UPSERT,
                workspace_id,
                str(row.get("display_name") or raw_workspace),
                str(row.get("embedding_model") or ""),
            )
            existing.add(workspace_id)
        await conn.execute(_DELETE, raw_workspace)


def _workspace_id(workspace: str) -> str:
    workspace_id = normalize_workspace(workspace)
    if not workspace_id:
        raise ValueError("workspace cannot be empty")
    return workspace_id


__all__ = ["PGWorkspaceRegistry"]
