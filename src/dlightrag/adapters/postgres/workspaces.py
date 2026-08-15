# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL workspace registry.

The registry is the durable source of truth for workspace existence and
display labels. LightRAG stores own document/KG/vector data per workspace;
this table owns the user-facing workspace list, including empty workspaces.
"""

from typing import Any

from dlightrag_rag.ports import CorpusSchemaError

from dlightrag.adapters.postgres._migrations import (
    Migration,
    TableRequirement,
    apply_migrations,
    verify_migrations,
)

_CREATE = """
CREATE TABLE IF NOT EXISTS dlightrag_workspace_meta (
    workspace       TEXT PRIMARY KEY,
    display_name    TEXT NOT NULL DEFAULT '',
    embedding_model TEXT NOT NULL DEFAULT '',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
)
"""

_UPSERT = """
INSERT INTO dlightrag_workspace_meta (workspace, display_name, embedding_model)
VALUES ($1, $2, $3)
ON CONFLICT (workspace)
DO UPDATE SET display_name = EXCLUDED.display_name,
              embedding_model = EXCLUDED.embedding_model,
              updated_at = NOW()
"""

_LIST = """
SELECT workspace, display_name, embedding_model, created_at, updated_at
FROM dlightrag_workspace_meta
ORDER BY workspace
"""

_DELETE = "DELETE FROM dlightrag_workspace_meta WHERE workspace = $1"

_SCHEMA_MIGRATIONS = (
    Migration(
        "0001_workspace_meta",
        "Create and migrate workspace registry",
        (_CREATE,),
    ),
)

_SCHEMA_TABLES = (
    TableRequirement(
        name="dlightrag_workspace_meta",
        columns=("workspace", "display_name", "embedding_model", "created_at", "updated_at"),
        primary_key=("workspace",),
    ),
)


class PGWorkspaceRegistry:
    """Durable workspace registry backed by PostgreSQL."""

    def __init__(self, *, pool: Any = None) -> None:
        self._pool = pool

    async def _run(self, operation):
        if self._pool is not None:
            async with self._pool.acquire() as conn:
                return await operation(conn)

        from dlightrag.adapters.postgres._pool import pg_pool

        return await pg_pool.run(operation)

    async def initialize(self, *, validate_only: bool = False) -> None:
        """Create/migrate the registry table, or validate it (reader)."""

        async def _operation(conn: Any) -> None:
            if validate_only:
                await verify_migrations(
                    conn,
                    scope="workspace_registry",
                    migrations=_SCHEMA_MIGRATIONS,
                    tables=_SCHEMA_TABLES,
                    schema_error=CorpusSchemaError,
                )
                return
            await apply_migrations(
                conn,
                scope="workspace_registry",
                migrations=_SCHEMA_MIGRATIONS,
                schema_error=CorpusSchemaError,
            )

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

    async def delete(self, workspace: str) -> bool:
        """Delete one workspace registry row."""
        workspace_id = _workspace_id(workspace)

        async def _operation(conn: Any) -> bool:
            result = await conn.execute(_DELETE, workspace_id)
            return result != "DELETE 0"

        return await self._run(_operation)


def _workspace_id(workspace: str) -> str:
    workspace_id = str(workspace).strip()
    if not workspace_id:
        raise ValueError("workspace cannot be empty")
    return workspace_id


__all__ = ["PGWorkspaceRegistry"]
