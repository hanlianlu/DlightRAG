# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL workspace registry.

The registry is the durable source of truth for workspace existence and
display labels. LightRAG stores own document/KG/vector data per workspace;
this table owns the user-facing workspace list, including empty workspaces.
"""

from __future__ import annotations

from typing import Any

from dlightrag.config import PostgresTarget

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


class PGWorkspaceRegistry:
    """Durable workspace registry backed by PostgreSQL."""

    def __init__(self, *, pool: Any = None, target: PostgresTarget | None = None) -> None:
        self._pool = pool
        self._target: PostgresTarget | None = target

    async def _get_pool(self) -> Any:
        if self._pool is not None:
            return self._pool

        from dlightrag.storage.pool import pg_pool

        return await pg_pool.get(self._target)

    async def initialize(self, *, read_only: bool = False) -> None:
        """Create or verify the registry table."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
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
                return

            await conn.execute(_CREATE)
            for sql in _MIGRATIONS:
                await conn.execute(sql)

    async def upsert(
        self,
        *,
        workspace: str,
        display_name: str | None,
        embedding_model: str,
    ) -> None:
        """Insert or update one workspace registry row."""
        pool = await self._get_pool()
        label = (display_name or workspace).strip() or workspace
        async with pool.acquire() as conn:
            await conn.execute(_UPSERT, workspace, label, embedding_model)

    async def list(self) -> list[dict[str, Any]]:
        """Return all registered workspaces."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(_LIST)
        return [dict(row) for row in rows]

    async def list_workspaces(self) -> list[str]:
        """Return all workspace identifiers."""
        return [row["workspace"] for row in await self.list()]

    async def exists(self, workspace: str) -> bool:
        """Return True when a registry row exists."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(_EXISTS, workspace)
        return row is not None

    async def delete(self, workspace: str) -> None:
        """Delete one workspace registry row."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(_DELETE, workspace)


__all__ = ["PGWorkspaceRegistry"]
