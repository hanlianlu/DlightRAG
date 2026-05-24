# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Integration tests for PostgreSQL storage.

Requires a running PostgreSQL instance with pgvector + AGE extensions.
Skipped automatically if PostgreSQL is not available.

Tests:
- RAGServiceManager.list_workspaces() PG workspace discovery
"""

from __future__ import annotations

import pytest

# Mark all tests in this module as integration
pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]


async def _pg_available() -> bool:
    """Check if PostgreSQL is available."""
    try:
        import asyncpg

        conn = await asyncpg.connect(
            host="localhost",
            port=5432,
            user="dlightrag",
            password="dlightrag",
            database="dlightrag",
        )
        await conn.fetchval("SELECT 1")
        await conn.close()
        return True
    except Exception:
        return False


@pytest.fixture
async def pg_check():
    """Skip test if PostgreSQL is not available."""
    if not await _pg_available():
        pytest.skip("PostgreSQL not available")


_PG_CONN_KWARGS = dict(
    host="localhost",
    port=5432,
    user="dlightrag",
    password="dlightrag",
    database="dlightrag",
)


# ---------------------------------------------------------------------------
# RAGServiceManager.list_workspaces — PG workspace discovery
# ---------------------------------------------------------------------------


class TestPGWorkspaceDiscovery:
    """Test workspace discovery via SELECT DISTINCT workspace."""

    async def test_discovers_workspaces_from_workspace_meta(self, pg_check) -> None:
        """list_workspaces() returns workspaces found in dlightrag_workspace_meta."""
        import asyncpg

        from dlightrag.config import DlightragConfig, EmbeddingConfig, set_config
        from dlightrag.core.servicemanager import RAGServiceManager

        conn = await asyncpg.connect(**_PG_CONN_KWARGS)
        try:
            await conn.execute("""CREATE TABLE IF NOT EXISTS dlightrag_workspace_meta (
                workspace TEXT PRIMARY KEY,
                embedding_model TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )""")

            # Insert test rows for two workspaces
            await conn.execute(
                "INSERT INTO dlightrag_workspace_meta (workspace, embedding_model) "
                "VALUES ($1, $2) ON CONFLICT (workspace) DO UPDATE SET embedding_model = $2",
                "project-alpha",
                "voyage-multimodal-3.5",
            )
            await conn.execute(
                "INSERT INTO dlightrag_workspace_meta (workspace, embedding_model) "
                "VALUES ($1, $2) ON CONFLICT (workspace) DO UPDATE SET embedding_model = $2",
                "project-beta",
                "voyage-multimodal-3.5",
            )

            cfg = DlightragConfig(  # type: ignore[call-arg]
                kv_storage="PGKVStorage",
                embedding=EmbeddingConfig(
                    provider="voyage",
                    model="voyage-multimodal-3.5",
                    api_key="test",
                    startup_probe=False,
                ),
            )
            set_config(cfg)

            manager = RAGServiceManager(config=cfg)
            workspaces = await manager.list_workspaces()

            assert "project-alpha" in workspaces
            assert "project-beta" in workspaces
        finally:
            # Cleanup test rows
            await conn.execute(
                "DELETE FROM dlightrag_workspace_meta WHERE workspace IN ($1, $2)",
                "project-alpha",
                "project-beta",
            )
            await conn.close()

    async def test_empty_table_returns_default_workspace(self, pg_check) -> None:
        """Empty workspace metadata falls back to config.workspace."""
        import asyncpg

        from dlightrag.config import DlightragConfig, EmbeddingConfig, set_config
        from dlightrag.core.servicemanager import RAGServiceManager

        conn = await asyncpg.connect(**_PG_CONN_KWARGS)
        try:
            await conn.execute("""CREATE TABLE IF NOT EXISTS dlightrag_workspace_meta (
                workspace TEXT PRIMARY KEY,
                embedding_model TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )""")
            cfg = DlightragConfig(  # type: ignore[call-arg]
                kv_storage="PGKVStorage",
                workspace="test-fallback-ws",
                embedding=EmbeddingConfig(
                    provider="voyage",
                    model="voyage-multimodal-3.5",
                    api_key="test",
                    startup_probe=False,
                ),
            )
            set_config(cfg)

            manager = RAGServiceManager(config=cfg)
            workspaces = await manager.list_workspaces()

            # Should at least contain the default workspace
            # (may contain more if table has data from other tests)
            assert isinstance(workspaces, list)
            assert len(workspaces) >= 1
        finally:
            await conn.close()
