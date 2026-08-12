# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unit tests for the read-only replica reader role."""

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from lightrag.kg.pgtable_impl import PGTableGraphStorage

from dlightrag.config import DlightragConfig, EmbeddingConfig


def _config(*, service_role: str = "writer", **overrides) -> DlightragConfig:
    return cast(Any, DlightragConfig)(
        _env_file=None,
        service_role=service_role,
        embedding=EmbeddingConfig(
            provider="voyage", model="m", api_key="k", dim=8, startup_probe=False
        ),
        **overrides,
    )


class TestServiceRoleConfig:
    def test_defaults_to_writer(self) -> None:
        cfg = _config()
        assert cfg.service_role == "writer"
        assert cfg.is_reader is False

    def test_reader_predicate(self) -> None:
        assert _config(service_role="reader").is_reader is True

    def test_env_spelling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DLIGHTRAG_SERVICE_ROLE", "reader")
        cfg = cast(Any, DlightragConfig)(
            _env_file=None,
            embedding=EmbeddingConfig(
                provider="voyage", model="m", api_key="k", dim=8, startup_probe=False
            ),
        )
        assert cfg.is_reader is True

    def test_invalid_role_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _config(service_role="replica")

    def test_require_writer_allows_writer(self) -> None:
        _config().require_writer("ingest")  # no raise

    def test_require_writer_rejects_reader(self) -> None:
        with pytest.raises(PermissionError):
            _config(service_role="reader").require_writer("ingest")


class TestReaderPoolSessionModes:
    """A reader is corpus-read-only: only the LightRAG pool runs read-only sessions."""

    def test_reader_domain_pool_stays_writable(self) -> None:
        cfg = _config(service_role="reader")
        assert "default_transaction_read_only" not in cfg.domain_pool_server_settings()
        assert "server_settings" not in cfg.pg_connection_kwargs()

    def test_reader_corpus_pool_is_read_only(self) -> None:
        cfg = _config(service_role="reader")
        assert cfg.lightrag_pool_server_settings()["default_transaction_read_only"] == "on"
        assert "default_transaction_read_only=on" in cfg.postgres_server_settings_env_value()

    def test_writer_has_no_read_only_guc_on_either_pool(self) -> None:
        cfg = _config()
        assert "default_transaction_read_only" not in cfg.domain_pool_server_settings()
        assert "default_transaction_read_only" not in cfg.lightrag_pool_server_settings()
        assert "server_settings" not in cfg.pg_connection_kwargs()

    def test_reader_corpus_read_only_cannot_be_overridden_by_session_setting(self) -> None:
        cfg = _config(
            service_role="reader",
            postgres_session_settings={"default_transaction_read_only": "off"},
        )
        # Reader invariant is applied last and wins on the corpus pool.
        assert cfg.lightrag_pool_server_settings()["default_transaction_read_only"] == "on"


class TestPgPoolBinding:
    def test_bind_same_signature_ok(self) -> None:
        from dlightrag.storage.pool import PGPool

        pool = PGPool()
        pool.bind(_config())
        pool.bind(_config())  # identical signature -> no raise

    def test_bind_incompatible_role_raises(self) -> None:
        from dlightrag.storage.pool import PGPool

        pool = PGPool()
        pool.bind(_config())
        with pytest.raises(RuntimeError):
            pool.bind(_config(service_role="reader"))

    def test_bind_incompatible_endpoint_raises(self) -> None:
        from dlightrag.storage.pool import PGPool

        pool = PGPool()
        pool.bind(_config())
        with pytest.raises(RuntimeError):
            pool.bind(_config(postgres_host="other-host"))

    async def test_close_clears_binding(self) -> None:
        from dlightrag.storage.pool import PGPool

        pool = PGPool()
        pool.bind(_config())
        await pool.close()
        pool.bind(_config(service_role="reader"))  # rebinding after close is allowed


_MANAGER_WRITE_CALLS = [
    ("astart_ingest_job", ("ws", None), {}),
    ("aget_ingest_job", ("job-1",), {}),
    ("ajoin_ingest_job", ("job-1",), {}),
    ("acancel_ingest_job", ("job-1",), {}),
    ("acreate_workspace", ("ws",), {}),
    ("areset", (), {}),
]


@pytest.mark.parametrize(("method", "args", "kwargs"), _MANAGER_WRITE_CALLS)
async def test_manager_write_guards_reject_reader(method, args, kwargs) -> None:
    from dlightrag.core.servicemanager import RAGServiceManager

    manager = object.__new__(RAGServiceManager)
    manager._config = _config(service_role="reader")
    with pytest.raises(PermissionError):
        await getattr(manager, method)(*args, **kwargs)


_SERVICE_WRITE_CALLS = [
    ("areset", (), {}),
    ("aupdate_metadata", ("doc-1", {}), {}),
    ("adelete_files", (), {}),
    ("aingest", ("local",), {}),
    ("aingest_source", (None,), {}),
    ("aretry_failed_docs", (), {}),
    ("_upsert_workspace_meta", (), {}),
]


@pytest.mark.parametrize(("method", "args", "kwargs"), _SERVICE_WRITE_CALLS)
async def test_service_write_guards_reject_reader(method, args, kwargs) -> None:
    from dlightrag.core.service import RAGService

    service = object.__new__(RAGService)
    service.config = _config(service_role="reader")
    with pytest.raises(PermissionError):
        await getattr(service, method)(*args, **kwargs)


# ---------------------------------------------------------------------------
# Validation-only reader startup
# ---------------------------------------------------------------------------


class _LedgerConn:
    """Fake connection that answers migration-ledger reads and records writes."""

    def __init__(self, applied: set[tuple[str, str]], *, ledger_exists: bool = True) -> None:
        self.applied = applied
        self.ledger_exists = ledger_exists
        self.executed: list[str] = []

    async def fetchval(self, sql: str, *args: Any) -> Any:
        if "to_regclass" in sql:
            return self.ledger_exists
        raise AssertionError(f"unexpected fetchval: {sql}")

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, str]]:
        scope = str(args[0])
        return [
            {"version": version}
            for applied_scope, version in sorted(self.applied)
            if applied_scope == scope
        ]

    async def execute(self, sql: str, *args: Any) -> str:
        self.executed.append(sql)
        return "OK"


def _required_domain_scopes() -> list[tuple[str, tuple[Any, ...], Any]]:
    from dlightrag.storage import answer_runs, pg_metadata_index, web_conversations, workspaces

    return [
        (
            "workspace_registry",
            workspaces._SCHEMA_MIGRATIONS,
            workspaces.PGWorkspaceRegistry,
        ),
        (
            "doc_metadata",
            pg_metadata_index._SCHEMA_MIGRATIONS,
            pg_metadata_index.PGMetadataIndex,
        ),
        (
            "answer_runs",
            answer_runs.ANSWER_RUN_MIGRATIONS,
            answer_runs.PGAnswerRunStore,
        ),
        (
            "web_conversations",
            web_conversations.WEB_CONVERSATION_MIGRATIONS,
            web_conversations.PGWebConversationStore,
        ),
    ]


@contextmanager
def _domain_pool_routed_to(conn: _LedgerConn) -> Iterator[None]:
    """Route every domain-store operation at ``conn`` without a real pool."""
    from dlightrag.storage.pool import pg_pool

    async def _run(operation: Any) -> Any:
        return await operation(conn)

    with (
        patch.object(pg_pool, "run", _run),
        patch.object(pg_pool, "run_once", _run),
    ):
        yield


@pytest.mark.parametrize("scope_index", range(4))
async def test_reader_startup_validates_domain_schema_without_ddl(scope_index: int) -> None:
    scope, migrations, store_cls = _required_domain_scopes()[scope_index]
    conn = _LedgerConn({(scope, migration.version) for migration in migrations})

    with _domain_pool_routed_to(conn):
        await store_cls().initialize(validate_only=True)

    assert conn.executed == []


@pytest.mark.parametrize("scope_index", range(4))
async def test_reader_startup_fails_on_incompatible_domain_schema(scope_index: int) -> None:
    from dlightrag.storage.migrations import SchemaValidationError

    scope, _migrations, store_cls = _required_domain_scopes()[scope_index]
    conn = _LedgerConn(set())

    with _domain_pool_routed_to(conn), pytest.raises(SchemaValidationError, match=scope):
        await store_cls().initialize(validate_only=True)

    assert conn.executed == []


@pytest.mark.parametrize("scope_index", range(4))
async def test_reader_startup_fails_when_the_migration_ledger_is_absent(scope_index: int) -> None:
    from dlightrag.storage.migrations import SchemaValidationError

    _scope, _migrations, store_cls = _required_domain_scopes()[scope_index]
    conn = _LedgerConn(set(), ledger_exists=False)

    with (
        _domain_pool_routed_to(conn),
        pytest.raises(SchemaValidationError, match="dlightrag_schema_migrations"),
    ):
        await store_cls().initialize(validate_only=True)

    assert conn.executed == []


async def test_reader_workspace_registry_schema_failure_stops_startup() -> None:
    from dlightrag.core.servicemanager import RAGServiceManager
    from dlightrag.storage.migrations import SchemaValidationError

    manager = object.__new__(RAGServiceManager)
    manager._config = _config(service_role="reader")
    manager._startup_warnings = []

    with pytest.raises(SchemaValidationError):
        await _initialize_registry(manager, SchemaValidationError("workspace_registry"))

    assert manager._startup_warnings == []


async def test_transient_workspace_registry_failure_only_warns() -> None:
    from dlightrag.core.servicemanager import RAGServiceManager

    manager = object.__new__(RAGServiceManager)
    manager._config = _config()
    manager._startup_warnings = []

    await _initialize_registry(manager, RuntimeError("transient"))

    assert manager._startup_warnings == ["Workspace registry unavailable"]


async def _initialize_registry(manager: Any, failure: Exception) -> None:
    import dlightrag.storage.workspaces as workspaces_module

    class _FailingRegistry:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def initialize(self, *, validate_only: bool = False) -> None:
            raise failure

    original = workspaces_module.PGWorkspaceRegistry
    workspaces_module.PGWorkspaceRegistry = _FailingRegistry  # type: ignore[misc]
    try:
        await manager._initialize_workspace_registry()
    finally:
        workspaces_module.PGWorkspaceRegistry = original  # type: ignore[misc]


@pytest.mark.parametrize("failing_step", ["_initialize_workspace_registry", "_get_service"])
async def test_reader_startup_never_degrades_past_a_schema_failure(
    failing_step: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dlightrag.observability as observability
    from dlightrag.core.servicemanager import RAGServiceManager
    from dlightrag.storage.migrations import SchemaValidationError
    from dlightrag.storage.pool import pg_pool

    monkeypatch.setattr(observability, "init_tracing", lambda _config: None)
    monkeypatch.setattr(pg_pool, "bind", lambda _config: None)
    monkeypatch.setattr(RAGServiceManager, "_initialize_workspace_registry", AsyncMock())
    monkeypatch.setattr(RAGServiceManager, "_probe_role_image_capabilities", AsyncMock())
    monkeypatch.setattr(RAGServiceManager, "_get_retrieval_planner", lambda self: None)
    monkeypatch.setattr(RAGServiceManager, "_get_service", AsyncMock())
    recovery = AsyncMock()
    monkeypatch.setattr(RAGServiceManager, "_start_ingest_job_recovery", recovery)
    monkeypatch.setattr(
        RAGServiceManager,
        failing_step,
        AsyncMock(side_effect=SchemaValidationError("doc_metadata is missing versions")),
    )
    closed = AsyncMock()
    monkeypatch.setattr(RAGServiceManager, "aclose", closed)

    with pytest.raises(SchemaValidationError, match="doc_metadata"):
        await RAGServiceManager.acreate(config=_config(service_role="reader"))

    closed.assert_awaited_once_with()
    recovery.assert_not_awaited()


async def test_reader_answer_run_store_starts_in_validation_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dlightrag.storage.answer_runs as answer_runs_module
    from dlightrag.core.servicemanager import RAGServiceManager

    seen: list[bool] = []

    class _Store:
        async def initialize(self, *, validate_only: bool = False) -> None:
            seen.append(validate_only)

    monkeypatch.setattr(answer_runs_module, "PGAnswerRunStore", _Store, raising=False)

    manager = object.__new__(RAGServiceManager)
    manager._config = _config(service_role="reader")
    manager._answer_run_store = None
    manager._answer_store_lock = asyncio.Lock()

    await manager._get_answer_run_store()

    assert seen == [True]


async def test_reader_serves_web_routes() -> None:
    import dlightrag.config as config_module
    from dlightrag.api.server import create_app

    original = config_module._config
    config_module._config = _config(service_role="reader")
    try:
        app = create_app()
    finally:
        config_module._config = original

    assert "/web/answer" in set(app.openapi()["paths"])
    assert app.state.web_conversation_service is not None


async def test_reader_does_not_recover_ingest_jobs() -> None:
    from dlightrag.core.servicemanager import RAGServiceManager

    manager = object.__new__(RAGServiceManager)
    manager._config = _config(service_role="reader")
    manager._startup_warnings = []
    recovery = AsyncMock()
    manager._ingest_jobs = cast(Any, SimpleNamespace(start_recovery=recovery))

    await manager._start_ingest_job_recovery()

    recovery.assert_not_awaited()


async def test_writer_recovers_ingest_jobs() -> None:
    from dlightrag.core.servicemanager import RAGServiceManager

    manager = object.__new__(RAGServiceManager)
    manager._config = _config()
    manager._startup_warnings = []
    recovery = AsyncMock()
    manager._ingest_jobs = cast(Any, SimpleNamespace(start_recovery=recovery))

    await manager._start_ingest_job_recovery()

    recovery.assert_awaited_once_with()


class TestReadOnlyAdapter:
    def test_overrides_initdb_without_ddl_bootstrap(self) -> None:
        from lightrag.kg.postgres_impl import PostgreSQLDB

        from dlightrag.storage.lightrag_readonly import ReadOnlyPostgreSQLDB

        # The reader adapter must override initdb so reconnect never re-enters
        # LightRAG's extension/table/graph bootstrap.
        assert ReadOnlyPostgreSQLDB.initdb is not PostgreSQLDB.initdb
        # It inherits LightRAG's reconnect/retry machinery unchanged.
        assert ReadOnlyPostgreSQLDB._ensure_pool is PostgreSQLDB._ensure_pool
        assert ReadOnlyPostgreSQLDB._run_with_retry is PostgreSQLDB._run_with_retry

    async def test_attach_entry_point_initializes_binds_schema_and_status(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dlightrag.storage.lightrag_readonly as readonly

        class FakeConn:
            def __init__(self) -> None:
                self.fetchval_calls: list[tuple[str, tuple[Any, ...]]] = []

            async def fetchval(self, sql: str, *args: Any) -> Any:
                self.fetchval_calls.append((sql, args))
                if sql == "SHOW transaction_read_only":
                    return "on"
                if "SELECT 1 FROM " in sql:
                    return 1
                raise AssertionError(f"unexpected SQL: {sql}")

        class FakeAcquire:
            def __init__(self, conn: FakeConn) -> None:
                self._conn = conn

            async def __aenter__(self) -> FakeConn:
                return self._conn

            async def __aexit__(self, *_exc: object) -> bool:
                return False

        class FakePool:
            def __init__(self, conn: FakeConn) -> None:
                self._conn = conn

            def acquire(self) -> FakeAcquire:
                return FakeAcquire(self._conn)

        class FakeDB:
            def __init__(self, db_config: dict[str, Any]) -> None:
                self.db_config = db_config
                self.workspace = "db-workspace"
                self.pool = FakePool(conn)
                self.initdb = AsyncMock()

        conn = FakeConn()
        set_workspace_calls: list[str | None] = []
        init_pipeline_status = AsyncMock()
        manager = SimpleNamespace(
            _lock=asyncio.Lock(),
            _instances={"db": None, "ref_count": 0, "vector_signature": None},
            get_config=lambda *, vector_storage=None: {
                "database": "db",
                "vector_storage": vector_storage,
            },
            _build_vector_signature=lambda config, vector_storage: {
                "database": config["database"],
                "vector_storage": vector_storage,
            },
        )
        manager._assert_compatible_vector_signature = lambda signature: None
        monkeypatch.setattr(readonly, "ClientManager", manager, raising=False)
        monkeypatch.setattr(readonly, "ReadOnlyPostgreSQLDB", FakeDB, raising=False)
        monkeypatch.setattr(
            readonly,
            "namespace_to_table_name",
            lambda namespace: {"full_docs": "LIGHTRAG_DOC_FULL"}.get(namespace),
            raising=False,
        )
        monkeypatch.setattr(readonly, "get_default_workspace", lambda: None, raising=False)
        monkeypatch.setattr(
            readonly,
            "set_default_workspace",
            lambda workspace=None: set_workspace_calls.append(workspace),
            raising=False,
        )
        monkeypatch.setattr(
            readonly,
            "initialize_pipeline_status",
            init_pipeline_status,
            raising=False,
        )
        monkeypatch.setattr(
            readonly,
            "StoragesStatus",
            SimpleNamespace(INITIALIZED="initialized"),
            raising=False,
        )

        graph = PGTableGraphStorage.__new__(PGTableGraphStorage)
        graph.db = None
        graph.workspace = ""
        full_docs = SimpleNamespace(db=None, workspace=None, namespace="full_docs", table_name=None)
        chunks_vdb = SimpleNamespace(
            db=None,
            workspace=None,
            namespace=None,
            table_name="LIGHTRAG_DOC_CHUNKS",
        )
        llm_cache = SimpleNamespace(
            db=None, workspace="kept-workspace", namespace=None, table_name=None
        )
        lightrag = SimpleNamespace(
            workspace="reader-workspace",
            full_docs=full_docs,
            text_chunks=None,
            full_entities=None,
            full_relations=None,
            entity_chunks=None,
            relation_chunks=None,
            entities_vdb=None,
            relationships_vdb=None,
            chunks_vdb=chunks_vdb,
            chunk_entity_relation_graph=graph,
            llm_response_cache=llm_cache,
            doc_status=None,
        )

        await readonly.attach_lightrag_storages_read_only(
            lightrag, config=_config(service_role="reader")
        )

        db = manager._instances["db"]
        assert isinstance(db, FakeDB)
        db.initdb.assert_awaited_once()
        assert manager._instances["ref_count"] == 4
        assert manager._instances["vector_signature"] == {
            "database": "db",
            "vector_storage": _config(service_role="reader").vector_storage,
        }
        assert full_docs.db is db
        assert chunks_vdb.db is db
        assert llm_cache.db is db
        assert full_docs.workspace == "db-workspace"
        assert chunks_vdb.workspace == "db-workspace"
        assert graph.workspace == "db-workspace"
        init_pipeline_status.assert_awaited_once_with(workspace="reader-workspace")
        assert set_workspace_calls == ["reader-workspace"]
        assert lightrag._storages_status == "initialized"
        assert lightrag._owning_loop is asyncio.get_running_loop()
        assert any(sql == "SHOW transaction_read_only" for sql, _ in conn.fetchval_calls)
        probed = " ".join(sql for sql, _ in conn.fetchval_calls if "SELECT 1 FROM " in sql)
        assert "LIGHTRAG_DOC_FULL" in probed
        assert "lightrag_graph_nodes" in probed
        assert "lightrag_graph_edges" in probed

    async def test_attach_entry_point_reuses_db_and_preserves_signature_checks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dlightrag.storage.lightrag_readonly as readonly

        class FakeConn:
            async def fetchval(self, sql: str, *args: Any) -> Any:
                if sql == "SHOW transaction_read_only":
                    return "on"
                if "SELECT 1 FROM " in sql:
                    return 1
                raise AssertionError(f"unexpected SQL: {sql}")

        class FakeAcquire:
            def __init__(self, conn: FakeConn) -> None:
                self._conn = conn

            async def __aenter__(self) -> FakeConn:
                return self._conn

            async def __aexit__(self, *_exc: object) -> bool:
                return False

        existing_db = SimpleNamespace(
            pool=SimpleNamespace(acquire=lambda: FakeAcquire(FakeConn())), workspace=""
        )
        seen_signatures: list[dict[str, Any]] = []
        manager = SimpleNamespace(
            _lock=asyncio.Lock(),
            _instances={"db": existing_db, "ref_count": 7, "vector_signature": {"database": "db"}},
            get_config=lambda *, vector_storage=None: {"database": "db"},
            _build_vector_signature=lambda config, vector_storage: {"database": config["database"]},
            _assert_compatible_vector_signature=lambda signature: seen_signatures.append(signature),
        )
        monkeypatch.setattr(readonly, "ClientManager", manager, raising=False)
        monkeypatch.setattr(
            readonly,
            "namespace_to_table_name",
            lambda namespace: "LIGHTRAG_DOC_FULL",
            raising=False,
        )
        monkeypatch.setattr(readonly, "get_default_workspace", lambda: "already-set", raising=False)
        monkeypatch.setattr(readonly, "set_default_workspace", AsyncMock(), raising=False)
        monkeypatch.setattr(
            readonly,
            "initialize_pipeline_status",
            AsyncMock(),
            raising=False,
        )
        monkeypatch.setattr(
            readonly,
            "StoragesStatus",
            SimpleNamespace(INITIALIZED="initialized"),
            raising=False,
        )

        lightrag = SimpleNamespace(
            workspace="reader-workspace",
            full_docs=SimpleNamespace(
                db=None, workspace=None, namespace="full_docs", table_name=None
            ),
            text_chunks=None,
            full_entities=None,
            full_relations=None,
            entity_chunks=None,
            relation_chunks=None,
            entities_vdb=None,
            relationships_vdb=None,
            chunks_vdb=None,
            chunk_entity_relation_graph=None,
            llm_response_cache=None,
            doc_status=None,
        )

        await readonly.attach_lightrag_storages_read_only(
            lightrag, config=_config(service_role="reader")
        )

        assert manager._instances["db"] is existing_db
        assert manager._instances["ref_count"] == 8
        assert seen_signatures == [{"database": "db"}]

    async def test_attach_entry_point_releases_db_when_schema_verification_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dlightrag.storage.lightrag_readonly as readonly

        class FakeConn:
            async def fetchval(self, sql: str, *args: Any) -> Any:
                if sql == "SHOW transaction_read_only":
                    return "off"
                raise AssertionError(f"unexpected SQL after read-only failure: {sql}")

        class FakeAcquire:
            async def __aenter__(self) -> FakeConn:
                return FakeConn()

            async def __aexit__(self, *_exc: object) -> bool:
                return False

        existing_db = SimpleNamespace(
            pool=SimpleNamespace(acquire=FakeAcquire),
            workspace="",
        )
        instances = {
            "db": existing_db,
            "ref_count": 7,
            "vector_signature": {"database": "db"},
        }

        async def release_client(db: object) -> None:
            assert db is existing_db
            instances["ref_count"] -= 1

        manager = SimpleNamespace(
            _lock=asyncio.Lock(),
            _instances=instances,
            get_config=lambda *, vector_storage=None: {"database": "db"},
            _build_vector_signature=lambda config, vector_storage: {"database": config["database"]},
            _assert_compatible_vector_signature=lambda signature: None,
            release_client=AsyncMock(side_effect=release_client),
        )
        monkeypatch.setattr(readonly, "ClientManager", manager, raising=False)

        lightrag = SimpleNamespace(
            workspace="reader-workspace",
            full_docs=SimpleNamespace(
                db=None, workspace=None, namespace="full_docs", table_name="LIGHTRAG_DOC_FULL"
            ),
            text_chunks=None,
            full_entities=None,
            full_relations=None,
            entity_chunks=None,
            relation_chunks=None,
            entities_vdb=None,
            relationships_vdb=None,
            chunks_vdb=None,
            chunk_entity_relation_graph=None,
            llm_response_cache=None,
            doc_status=None,
        )

        with pytest.raises(RuntimeError, match="not read-only"):
            await readonly.attach_lightrag_storages_read_only(
                lightrag,
                config=_config(service_role="reader"),
            )

        assert instances["ref_count"] == 7

    async def test_read_only_db_rollback_finishes_before_propagating_cancellation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dlightrag.storage.lightrag_readonly as readonly

        close_started = asyncio.Event()
        allow_close = asyncio.Event()

        async def close_pool() -> None:
            close_started.set()
            await allow_close.wait()

        pool = SimpleNamespace(close=AsyncMock(side_effect=close_pool))
        db = SimpleNamespace(pool=pool)
        instances = {"db": db, "ref_count": 3, "vector_signature": {"database": "db"}}

        async def release_client(released_db: object) -> None:
            assert released_db is db
            async with manager._lock:
                instances["ref_count"] -= 1
                if instances["ref_count"] == 0:
                    await pool.close()
                    instances["db"] = None
                    instances["vector_signature"] = None

        manager = SimpleNamespace(
            _lock=asyncio.Lock(),
            _instances=instances,
            release_client=AsyncMock(side_effect=release_client),
        )
        monkeypatch.setattr(readonly, "ClientManager", manager, raising=False)

        rollback = asyncio.create_task(readonly._release_read_only_db(db, reference_count=3))
        await close_started.wait()
        rollback.cancel()
        allow_close.set()

        with pytest.raises(asyncio.CancelledError):
            await rollback

        assert instances == {"db": None, "ref_count": 0, "vector_signature": None}
        pool.close.assert_awaited_once_with()

    async def test_attach_entry_point_signature_mismatch_keeps_refcount(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dlightrag.storage.lightrag_readonly as readonly

        class FakeConn:
            async def fetchval(self, sql: str, *args: Any) -> Any:
                raise AssertionError(f"schema should not be queried after signature failure: {sql}")

        class FakeAcquire:
            def __init__(self, conn: FakeConn) -> None:
                self._conn = conn

            async def __aenter__(self) -> FakeConn:
                return self._conn

            async def __aexit__(self, *_exc: object) -> bool:
                return False

        existing_db = SimpleNamespace(
            pool=SimpleNamespace(acquire=lambda: FakeAcquire(FakeConn())), workspace=""
        )
        manager = SimpleNamespace(
            _lock=asyncio.Lock(),
            _instances={"db": existing_db, "ref_count": 5, "vector_signature": {"database": "db"}},
            get_config=lambda *, vector_storage=None: {"database": "db"},
            _build_vector_signature=lambda config, vector_storage: {"database": config["database"]},
            _assert_compatible_vector_signature=lambda signature: (_ for _ in ()).throw(
                RuntimeError("vector mismatch")
            ),
        )
        monkeypatch.setattr(readonly, "ClientManager", manager, raising=False)

        lightrag = SimpleNamespace(
            workspace="reader-workspace",
            full_docs=SimpleNamespace(
                db=None, workspace=None, namespace="full_docs", table_name=None
            ),
            text_chunks=None,
            full_entities=None,
            full_relations=None,
            entity_chunks=None,
            relation_chunks=None,
            entities_vdb=None,
            relationships_vdb=None,
            chunks_vdb=None,
            chunk_entity_relation_graph=None,
            llm_response_cache=None,
            doc_status=None,
        )

        with pytest.raises(RuntimeError, match="vector mismatch"):
            await readonly.attach_lightrag_storages_read_only(
                lightrag,
                config=_config(service_role="reader"),
            )

        assert manager._instances["ref_count"] == 5
