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

_TEST_WORKSPACE_ALPHA = "test_pg_storage_alpha"
_TEST_WORKSPACE_BETA = "test_pg_storage_beta"
_TEST_WORKSPACES = (_TEST_WORKSPACE_ALPHA, _TEST_WORKSPACE_BETA)


async def _delete_test_workspaces(conn) -> None:  # noqa: ANN001
    """Remove integration-test registry rows from the shared local database."""
    await conn.execute(
        "DELETE FROM dlightrag_workspace_meta WHERE workspace = ANY($1::text[])",
        list(_TEST_WORKSPACES),
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
            await _delete_test_workspaces(conn)
            await conn.execute("""CREATE TABLE IF NOT EXISTS dlightrag_workspace_meta (
                workspace TEXT PRIMARY KEY,
                embedding_model TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )""")

            # Insert test rows for two workspaces.
            await conn.execute(
                "INSERT INTO dlightrag_workspace_meta (workspace, embedding_model) "
                "VALUES ($1, $2) ON CONFLICT (workspace) DO UPDATE SET embedding_model = $2",
                _TEST_WORKSPACE_ALPHA,
                "voyage-multimodal-3.5",
            )
            await conn.execute(
                "INSERT INTO dlightrag_workspace_meta (workspace, embedding_model) "
                "VALUES ($1, $2) ON CONFLICT (workspace) DO UPDATE SET embedding_model = $2",
                _TEST_WORKSPACE_BETA,
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

            assert _TEST_WORKSPACE_ALPHA in workspaces
            assert _TEST_WORKSPACE_BETA in workspaces
        finally:
            await _delete_test_workspaces(conn)
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
            assert "test_fallback_ws" in workspaces
        finally:
            await conn.execute(
                "DELETE FROM dlightrag_workspace_meta WHERE workspace = $1",
                "test_fallback_ws",
            )
            await conn.close()
