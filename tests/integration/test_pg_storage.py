# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Integration tests for PostgreSQL storage.

Requires a running PostgreSQL instance with pgvector + AGE extensions.
Skipped automatically if PostgreSQL is not available.

Tests:
- CorpusAdmin.list_workspaces() PG workspace discovery
"""

from types import SimpleNamespace
from typing import Any, cast

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


async def _open_workspace_registry() -> tuple[Any, Any]:
    import asyncpg

    from dlightrag.adapters.postgres.corpus.workspaces import PGWorkspaceRegistry

    pool = await asyncpg.create_pool(
        host=str(_PG_CONN_KWARGS["host"]),
        port=int(_PG_CONN_KWARGS["port"]),
        user=str(_PG_CONN_KWARGS["user"]),
        password=str(_PG_CONN_KWARGS["password"]),
        database=str(_PG_CONN_KWARGS["database"]),
        min_size=1,
        max_size=1,
    )
    registry = PGWorkspaceRegistry(pool=pool)
    await registry.initialize()
    return pool, registry


async def _delete_test_workspaces(registry: Any, *extra_workspaces: str) -> None:
    """Remove integration-test registry rows from the shared local database."""
    for workspace in (*_TEST_WORKSPACES, *extra_workspaces):
        await registry.delete(workspace)


def _corpus_admin(config: Any) -> Any:
    from dlightrag.adapters.postgres.corpus.corpus import build_pg_corpus_backend
    from dlightrag.application.corpus_admin import CorpusAdmin
    from dlightrag.application.settings import corpus_admin_settings

    backend = build_pg_corpus_backend(config)
    return CorpusAdmin(
        settings=corpus_admin_settings(config),
        pool=cast(Any, SimpleNamespace()),
        maintenance=backend.maintenance,
        ingest_jobs=cast(Any, SimpleNamespace()),
        file_panel=cast(Any, SimpleNamespace()),
        source_download_for=cast(Any, lambda _workspace: SimpleNamespace()),
    )


# ---------------------------------------------------------------------------
# CorpusAdmin.list_workspaces - PG workspace discovery
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("pg_check")
class TestPGWorkspaceDiscovery:
    """Test workspace discovery via SELECT DISTINCT workspace."""

    async def test_discovers_workspaces_from_workspace_meta(self) -> None:
        """list_workspaces() returns workspaces found in dlightrag_workspace_meta."""
        from dlightrag.adapters.postgres.core._pool import pg_pool
        from dlightrag.application.config import DlightragConfig, set_config
        from dlightrag.engine.ai.settings import EmbeddingSettings

        pool, registry = await _open_workspace_registry()
        try:
            await _delete_test_workspaces(registry)
            await registry.upsert(
                workspace=_TEST_WORKSPACE_ALPHA,
                display_name="Test PG Storage Alpha",
                embedding_model="voyage-multimodal-3.5",
            )
            await registry.upsert(
                workspace=_TEST_WORKSPACE_BETA,
                display_name="Test PG Storage Beta",
                embedding_model="voyage-multimodal-3.5",
            )

            cfg = DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
                models={
                    "embedding": EmbeddingSettings(
                        provider="voyage",
                        model="voyage-multimodal-3.5",
                        api_key="test",
                        startup_probe=False,
                    ),
                },
            )
            set_config(cfg)

            pg_pool.bind(cfg)
            corpora = _corpus_admin(cfg)
            try:
                workspaces = await corpora.list_workspaces()

                assert _TEST_WORKSPACE_ALPHA in workspaces
                assert _TEST_WORKSPACE_BETA in workspaces
            finally:
                await pg_pool.close()
        finally:
            await _delete_test_workspaces(registry)
            await pool.close()

    async def test_empty_table_returns_default_workspace(self) -> None:
        """Empty workspace metadata falls back to config.deployment.workspace."""
        from dlightrag.adapters.postgres.core._pool import pg_pool
        from dlightrag.application.config import DlightragConfig, set_config
        from dlightrag.engine.ai.settings import EmbeddingSettings

        pool, registry = await _open_workspace_registry()
        try:
            await _delete_test_workspaces(registry, "test-fallback-ws")
            cfg = DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
                deployment={
                    "workspace": "test-fallback-ws",
                },
                models={
                    "embedding": EmbeddingSettings(
                        provider="voyage",
                        model="voyage-multimodal-3.5",
                        api_key="test",
                        startup_probe=False,
                    ),
                },
            )
            set_config(cfg)

            pg_pool.bind(cfg)
            corpora = _corpus_admin(cfg)
            try:
                workspaces = await corpora.list_workspaces()

                # Should at least contain the default workspace
                # (may contain more if table has data from other tests)
                assert isinstance(workspaces, list)
                assert len(workspaces) >= 1
                assert "test_fallback_ws" in workspaces
            finally:
                await pg_pool.close()
        finally:
            await _delete_test_workspaces(registry, "test-fallback-ws")
            await pool.close()
