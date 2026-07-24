# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unit tests for the read-only replica reader role."""

from typing import Any, cast

import pytest

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


class TestReaderReadOnlyGuc:
    def test_reader_forces_read_only_on_both_pools(self) -> None:
        cfg = _config(service_role="reader")
        assert cfg.postgres_server_settings_dict()["default_transaction_read_only"] == "on"
        assert cfg.pg_connection_kwargs()["server_settings"] == {
            "default_transaction_read_only": "on"
        }

    def test_writer_has_no_read_only_guc(self) -> None:
        cfg = _config()
        assert "default_transaction_read_only" not in cfg.postgres_server_settings_dict()
        assert "server_settings" not in cfg.pg_connection_kwargs()

    def test_reader_read_only_cannot_be_overridden_by_session_setting(self) -> None:
        cfg = _config(
            service_role="reader",
            postgres_session_settings={"default_transaction_read_only": "off"},
        )
        # Reader invariant is applied last and wins.
        assert cfg.postgres_server_settings_dict()["default_transaction_read_only"] == "on"


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
