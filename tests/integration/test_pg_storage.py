# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Integration tests for PostgreSQL storage.

Requires a running PostgreSQL instance with pgvector + AGE extensions.
Skipped automatically if PostgreSQL is not available.

Tests:
- CorpusAdmin.list_workspaces() PG workspace discovery
"""

import datetime
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
        metadata_search=cast(Any, SimpleNamespace()),
        source_download_for=cast(Any, lambda _workspace: SimpleNamespace()),
    )


# ---------------------------------------------------------------------------
# File panel - bounded mixed-direction keyset traversal
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("pg_check")
async def test_file_panel_traverses_null_and_timestamp_groups_without_gaps() -> None:
    import asyncpg

    from dlightrag.adapters.postgres.corpus.file_panel import PGFilePanelStore
    from dlightrag.application.corpus_admin import (
        FilePanelCursor,
        FilePanelPageRequest,
    )

    workspace = "test_pg_file_panel"
    other_workspace = "test_pg_file_panel_other"
    pool = await asyncpg.create_pool(
        host=str(_PG_CONN_KWARGS["host"]),
        port=int(_PG_CONN_KWARGS["port"]),
        user=str(_PG_CONN_KWARGS["user"]),
        password=str(_PG_CONN_KWARGS["password"]),
        database=str(_PG_CONN_KWARGS["database"]),
        min_size=1,
        max_size=1,
    )
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS LIGHTRAG_DOC_STATUS (
                    workspace varchar(255) NOT NULL,
                    id varchar(255) NOT NULL,
                    status varchar(64),
                    file_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT LIGHTRAG_DOC_STATUS_PK PRIMARY KEY (workspace, id)
                )
                """
            )
            await conn.execute(
                "DELETE FROM LIGHTRAG_DOC_STATUS WHERE workspace = ANY($1::varchar[])",
                [workspace, other_workspace],
            )
            timestamp = datetime.datetime(2026, 3, 4, 5, 6, 7, 123456)
            rows = [
                (workspace, "null-a", "processed", "/null-a", None),
                (workspace, "null-b", "processed", "/null-b", None),
                (workspace, "same-a", "processed", "/same-a", timestamp),
                (workspace, "same-b", "processed", "/same-b", timestamp),
                (
                    workspace,
                    "older",
                    "processed",
                    "/older",
                    timestamp - datetime.timedelta(days=1),
                ),
                (workspace, "ignored", "pending", "/ignored", timestamp),
                (other_workspace, "foreign", "processed", "/foreign", timestamp),
            ]
            await conn.executemany(
                """
                INSERT INTO LIGHTRAG_DOC_STATUS (
                    workspace, id, status, file_path, updated_at
                ) VALUES ($1, $2, $3, $4, $5)
                """,
                rows,
            )

        store = PGFilePanelStore(pool=pool)
        await store.ensure_page_index()
        cursor: FilePanelCursor | None = None
        observed: list[str] = []
        while True:
            page = await store.list_processed_files(
                workspace,
                page=FilePanelPageRequest(limit=2, cursor=cursor),
            )
            assert len(page.items) <= 2
            assert page.fetched_rows <= 3
            observed.extend(item.doc_id for item in page.items)
            if not page.has_more:
                break
            assert page.items
            last = page.items[-1]
            cursor = FilePanelCursor(
                workspace=workspace,
                updated_at=last.updated_at,
                doc_id=last.doc_id,
            )

        assert observed == ["null-a", "null-b", "same-a", "same-b", "older"]
        assert len(observed) == len(set(observed))
        async with pool.acquire() as conn:
            indexdef = await conn.fetchval(
                "SELECT indexdef FROM pg_indexes WHERE indexname = $1",
                "idx_dlightrag_file_panel_processed_updated_id",
            )
            assert indexdef is not None
            normalized = " ".join(str(indexdef).split()).lower()
            assert "workspace, updated_at desc, id" in normalized
            assert "where" in normalized and "status" in normalized and "processed" in normalized
    finally:
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM LIGHTRAG_DOC_STATUS WHERE workspace = ANY($1::varchar[])",
                [workspace, other_workspace],
            )
        await pool.close()


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


# ---------------------------------------------------------------------------
# Metadata search - bounded doc_id keyset traversal
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("pg_check")
async def test_metadata_search_traverses_contains_fallback_without_gaps() -> None:
    import asyncpg

    from dlightrag.adapters.postgres.corpus.pg_metadata_search import PGMetadataSearchStore
    from dlightrag.application.corpus_admin import (
        MetadataSearchCursor,
        MetadataSearchPageRequest,
    )
    from dlightrag.engine.rag.retrieval import MetadataFilter

    workspace = "test_pg_metadata_search"
    other_workspace = "test_pg_metadata_search_other"
    pool = await asyncpg.create_pool(
        host=str(_PG_CONN_KWARGS["host"]),
        port=int(_PG_CONN_KWARGS["port"]),
        user=str(_PG_CONN_KWARGS["user"]),
        password=str(_PG_CONN_KWARGS["password"]),
        database=str(_PG_CONN_KWARGS["database"]),
        min_size=1,
        max_size=1,
    )
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dlightrag_doc_metadata (
                    workspace VARCHAR(255) NOT NULL,
                    doc_id VARCHAR(255) NOT NULL,
                    filename VARCHAR(512),
                    filename_stem VARCHAR(512),
                    PRIMARY KEY (workspace, doc_id)
                )
                """
            )
            await conn.execute(
                "DELETE FROM dlightrag_doc_metadata WHERE workspace = ANY($1::text[])",
                [workspace, other_workspace],
            )
            rows: list[tuple[str, str, str, str]] = []
            # 120 contains-only matches: no filename or stem equals the filter.
            for index in range(120):
                rows.append(
                    (
                        workspace,
                        f"doc-{index:03d}",
                        f"Quarterly Report draft {index}.pdf",
                        f"Quarterly Report draft {index}",
                    )
                )
            # Three exact stem matches; the widened fallback must not run.
            for index in range(3):
                rows.append((workspace, f"exact-{index:03d}", "Exact Doc.pdf", "Exact Doc"))
            # Contains-only matches for the exact filter.
            for index in range(5):
                rows.append(
                    (workspace, f"copy-{index:03d}", "Exact Doc copy.pdf", "Exact Doc copy")
                )
            for index in range(7):
                rows.append(
                    (
                        other_workspace,
                        f"foreign-{index:03d}",
                        "Quarterly Report draft.pdf",
                        "Quarterly Report draft",
                    )
                )
            await conn.executemany(
                """
                INSERT INTO dlightrag_doc_metadata (workspace, doc_id, filename, filename_stem)
                VALUES ($1, $2, $3, $4)
                """,
                rows,
            )

        store = PGMetadataSearchStore(pool=pool)
        filters = MetadataFilter(filename="Quarterly Report")
        cursor: MetadataSearchCursor | None = None
        observed: list[str] = []
        while True:
            page = await store.search_metadata_page(
                workspace,
                filters,
                page=MetadataSearchPageRequest(limit=40, cursor=cursor),
            )
            assert len(page.document_ids) <= 40
            assert page.fetched_rows <= 41
            observed.extend(page.document_ids)
            if not page.has_more:
                break
            assert page.document_ids
            cursor = MetadataSearchCursor(
                workspace=workspace,
                after_doc_id=page.document_ids[-1],
                mode=page.mode,
            )
        assert len(observed) == 120
        assert len(set(observed)) == 120
        assert page.mode == "contains"

        # A filter with exact matches stays exact: the contains-only rows must
        # never enter the traversal.
        exact_page = await store.search_metadata_page(
            workspace,
            MetadataFilter(filename="Exact Doc"),
            page=MetadataSearchPageRequest(limit=50),
        )
        assert exact_page.mode == "exact"
        assert exact_page.has_more is False
        assert exact_page.fetched_rows == 3
        assert sorted(exact_page.document_ids) == ["exact-000", "exact-001", "exact-002"]
    finally:
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM dlightrag_doc_metadata WHERE workspace = ANY($1::text[])",
                [workspace, other_workspace],
            )
        await pool.close()


# ---------------------------------------------------------------------------
# Child roster - bounded newest-first keyset traversal
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("pg_check")
async def test_child_roster_traverses_newest_first_with_timestamp_ties() -> None:
    import uuid

    import asyncpg

    from dlightrag.adapters.postgres.answer.answer_runs import PGAnswerRunStore
    from dlightrag.application.answer_runs import (
        ChildRosterCursor,
        ChildRosterPageRequest,
    )

    owner = "test_pg_child_roster"
    run_id = str(uuid.uuid4())
    pool = await asyncpg.create_pool(
        host=str(_PG_CONN_KWARGS["host"]),
        port=int(_PG_CONN_KWARGS["port"]),
        user=str(_PG_CONN_KWARGS["user"]),
        password=str(_PG_CONN_KWARGS["password"]),
        database=str(_PG_CONN_KWARGS["database"]),
        min_size=1,
        max_size=1,
    )
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dlightrag_answer_child_sessions (
                    owner_id TEXT NOT NULL,
                    run_id UUID NOT NULL,
                    child_session_id UUID NOT NULL,
                    parent_session_id UUID NOT NULL,
                    parent_call_id TEXT NOT NULL,
                    parent_intent_id UUID,
                    status TEXT NOT NULL,
                    summary TEXT,
                    objective TEXT,
                    context_mode TEXT,
                    model_role TEXT,
                    tools_json JSONB,
                    usage_json JSONB,
                    depth INTEGER NOT NULL,
                    context_snapshot_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    plan_json JSONB,
                    budget_json JSONB,
                    host_state_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    lease_owner TEXT,
                    lease_expires_at TIMESTAMPTZ,
                    fencing_epoch BIGINT NOT NULL DEFAULT 0,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (owner_id, run_id, child_session_id)
                )
                """
            )
            await conn.execute(
                "DELETE FROM dlightrag_answer_child_sessions WHERE owner_id = $1",
                owner,
            )
            # The local development database already owns the real schema with
            # its run foreign key; insert one minimal parent run for the FK.
            await conn.execute(
                "DELETE FROM dlightrag_answer_runs WHERE owner_id = $1 AND run_id = $2::uuid",
                owner,
                run_id,
            )
            await conn.execute(
                """
                INSERT INTO dlightrag_answer_runs (
                    owner_id, run_id, request_fingerprint, prepared_input_json, status
                ) VALUES ($1, $2::uuid, 'child-roster-test', '{}'::jsonb, 'queued')
                """,
                owner,
                run_id,
            )
            # 120 children across three timestamp groups so the newest-first
            # traversal must break ties on child_session_id DESC. The page
            # limit of 30 is deliberately not a divisor of 40, so every
            # continuation lands inside a same-timestamp group and the
            # equality-tie branch (created_at = cursor AND child_session_id <
            # cursor) is exercised on a real page boundary.
            base = datetime.datetime(2026, 3, 4, 5, 6, 7, tzinfo=datetime.UTC)
            rows: list[tuple[Any, ...]] = []
            for index in range(120):
                child_id = uuid.uuid4()
                timestamp = base + datetime.timedelta(days=index // 40)
                rows.append(
                    (
                        owner,
                        run_id,
                        child_id,
                        uuid.uuid4(),
                        f"call-{index}",
                        None,
                        "succeeded",
                        None,
                        f"objective {index}",
                        None,
                        "query",
                        None,
                        None,
                        1,
                        "{}",
                        None,
                        None,
                        "{}",
                        None,
                        None,
                        0,
                        timestamp,
                        timestamp,
                    )
                )
            await conn.executemany(
                """
                INSERT INTO dlightrag_answer_child_sessions (
                    owner_id, run_id, child_session_id, parent_session_id, parent_call_id,
                    parent_intent_id, status, summary, objective, context_mode, model_role,
                    tools_json, usage_json, depth, context_snapshot_json, plan_json,
                    budget_json, host_state_json, lease_owner, lease_expires_at,
                    fencing_epoch, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
                          $15, $16, $17, $18, $19, $20, $21, $22, $23)
                """,
                rows,
            )

        store = PGAnswerRunStore(pool=pool)
        cursor: ChildRosterCursor | None = None
        observed: list[str] = []
        tie_continuations = 0
        previous_last_created_at = None
        while True:
            page = await store.list_child_sessions_page(
                owner_id=owner,
                run_id=run_id,
                page=ChildRosterPageRequest(limit=30, cursor=cursor),
            )
            assert len(page.children) <= 30
            assert page.fetched_rows <= 31
            if previous_last_created_at is not None and page.children:
                first_created_at = page.children[0]["created_at"]
                if first_created_at == previous_last_created_at:
                    # The continuation resumed inside the previous page's
                    # timestamp group: the equality-tie branch fired.
                    tie_continuations += 1
            observed.extend(str(row["child_session_id"]) for row in page.children)
            if not page.has_more:
                break
            assert page.children
            last = page.children[-1]
            previous_last_created_at = last["created_at"]
            cursor = ChildRosterCursor(
                run_id=uuid.UUID(run_id),
                created_at=last["created_at"],
                child_session_id=uuid.UUID(str(last["child_session_id"])),
            )

        assert len(observed) == 120
        assert len(set(observed)) == 120
        # Every one of the three continuations resumes inside a same-timestamp
        # group, so the equality-tie predicate must have fired at least once.
        assert tie_continuations == 3

        # The traversal order is newest-first with id ties descending: group 2
        # (latest timestamps) before group 1, then group 0.
        expected = [
            str(child_id)
            for child_id, _timestamp in sorted(
                ((row[2], row[21]) for row in rows),
                key=lambda pair: (pair[1], pair[0]),
                reverse=True,
            )
        ]
        assert observed == expected

        # A foreign owner sees nothing through the same bounded store path.
        foreign = await store.list_child_sessions_page(
            owner_id="someone-else",
            run_id=run_id,
            page=ChildRosterPageRequest(limit=10),
        )
        assert foreign.children == ()
        assert foreign.has_more is False
    finally:
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM dlightrag_answer_child_sessions WHERE owner_id = $1",
                owner,
            )
            await conn.execute(
                "DELETE FROM dlightrag_answer_runs WHERE owner_id = $1 AND run_id = $2::uuid",
                owner,
                run_id,
            )
        await pool.close()
