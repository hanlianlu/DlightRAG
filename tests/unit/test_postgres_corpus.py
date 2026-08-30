# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Focused behavior for the PostgreSQL corpus composition adapter."""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from dlightrag.adapters.postgres.corpus import corpus as corpus_module
from dlightrag.adapters.postgres.corpus.corpus import (
    PGCorpusCoordination,
    PGCorpusMaintenanceStore,
    PGCorpusRuntimeBinder,
    build_pg_corpus_backend,
)
from dlightrag.application.config import DlightragConfig
from dlightrag.engine.rag.retrieval.bm25 import BM25Profile
from tests.config_helpers import clone_config, mutate_config


class _Connection:
    def __init__(self, *, max_connections: str) -> None:
        self._max_connections = max_connections

    async def fetchval(self, query: str) -> str:
        assert query == "SHOW max_connections"
        return self._max_connections


async def test_connection_budget_warning_is_owned_by_coordination(
    test_config: DlightragConfig,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("WEB_CONCURRENCY", "2")
    coordination = PGCorpusCoordination(
        connection_kwargs=test_config.pg_connection_kwargs(),
        workspace=test_config.deployment.workspace,
        reader=False,
        require_halfvec=False,
        required_extensions=(),
        lightrag_pool_max_size=16,
        domain_pool_max_size=10,
        acquire_timeout=test_config.storage.postgres.acquire_timeout,
    )

    with caplog.at_level(logging.WARNING, logger="dlightrag.adapters.postgres.corpus.corpus"):
        await coordination._log_connection_budget(_Connection(max_connections="50"))

    assert "PostgreSQL connection budget is tight" in caplog.text
    assert "estimated_pool_connections=52" in caplog.text


def test_backend_factory_applies_lightrag_environment_on_create(
    test_config: DlightragConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    apply_backend = MagicMock()
    apply_sidecar = MagicMock()
    apply_runtime = MagicMock()
    monkeypatch.setattr(
        DlightragConfig,
        "apply_lightrag_backend_env",
        lambda _self, *, force=False: apply_backend(force=force),
    )
    monkeypatch.setattr(
        DlightragConfig,
        "apply_lightrag_sidecar_env",
        lambda _self: apply_sidecar(),
    )
    monkeypatch.setattr(
        DlightragConfig,
        "apply_lightrag_runtime_env",
        lambda _self, *, force=False: apply_runtime(force=force),
    )

    backend = build_pg_corpus_backend(test_config)

    assert backend.workspace_id == test_config.deployment.workspace
    apply_backend.assert_called_once_with(force=True)
    apply_sidecar.assert_called_once_with()
    apply_runtime.assert_called_once_with(force=True)


def test_retrieval_partition_specs_cover_vector_filter_and_ann_contract() -> None:
    lightrag = SimpleNamespace(
        chunks_vdb=SimpleNamespace(
            table_name="lightrag_vdb_chunks_8",
            db=SimpleNamespace(vector_index_type="HNSW"),
        )
    )

    chunks, vectors = corpus_module.lightrag_retrieval_table_specs(lightrag)

    assert chunks.name.lower() == "lightrag_doc_chunks"
    assert chunks.primary_key == ("workspace", "id")
    assert vectors.name == "lightrag_vdb_chunks_8"
    assert "full_doc_id" in vectors.required_columns
    assert vectors.required_index_markers == ("USING hnsw",)


@pytest.mark.parametrize("is_reader", [False, True])
async def test_runtime_binder_composes_workspace_stores(
    test_config: DlightragConfig,
    monkeypatch: pytest.MonkeyPatch,
    is_reader: bool,
) -> None:
    config = clone_config(test_config)
    mutate_config(config, "deployment.service_role", "reader" if is_reader else "writer")
    mutate_config(config, "corpus.retrieval.bm25_enabled", True)
    metadata = SimpleNamespace(initialize=AsyncMock())
    chunks = object()
    vectors = SimpleNamespace(ensure_document_scope_index=AsyncMock())
    file_panel = SimpleNamespace(ensure_page_index=AsyncMock())
    bm25 = object()
    profiles = (BM25Profile(name="en", text_config="english", languages=("en",)),)
    metadata_constructor = MagicMock(return_value=metadata)
    chunk_constructor = MagicMock(return_value=chunks)
    vector_constructor = MagicMock(return_value=vectors)
    file_panel_constructor = MagicMock(return_value=file_panel)
    create_bm25 = AsyncMock(return_value=bm25)
    foundation = SimpleNamespace(ensure_tables=AsyncMock(), verify_tables=AsyncMock())
    foundation_constructor = MagicMock(return_value=foundation)
    guard = SimpleNamespace(
        verify_surface=MagicMock(),
        verify_read_only_attach_contract=MagicMock(),
        verify_all=AsyncMock(),
    )
    guard_constructor = MagicMock(return_value=guard)
    attach_read_only = AsyncMock()
    monkeypatch.setattr(corpus_module, "PGMetadataIndex", metadata_constructor)
    monkeypatch.setattr(corpus_module, "PGCorpusChunkStore", chunk_constructor)
    monkeypatch.setattr(corpus_module, "PGFilteredVectorSearch", vector_constructor)
    monkeypatch.setattr(corpus_module, "PGFilePanelStore", file_panel_constructor)
    monkeypatch.setattr(corpus_module, "PGPartitionFoundation", foundation_constructor)
    monkeypatch.setattr(corpus_module, "profiles_from_config", MagicMock(return_value=profiles))
    monkeypatch.setattr(corpus_module, "create_postgres_bm25", create_bm25)
    monkeypatch.setattr(corpus_module, "PGLightRAGContractGuard", guard_constructor)
    monkeypatch.setattr(corpus_module, "attach_lightrag_storages_read_only", attach_read_only)
    chunks_vdb = object()
    lightrag = SimpleNamespace(chunks_vdb=chunks_vdb, initialize_storages=AsyncMock())

    stores = await PGCorpusRuntimeBinder(config).attach(lightrag)

    metadata_constructor.assert_called_once_with(workspace=config.deployment.workspace)
    guard_constructor.assert_called_once_with(lightrag)
    guard.verify_surface.assert_called_once_with()
    guard.verify_all.assert_awaited_once_with()
    if is_reader:
        foundation.ensure_tables.assert_not_awaited()
        foundation.verify_tables.assert_awaited_once_with(
            specs=corpus_module.lightrag_retrieval_table_specs(lightrag)
        )
        file_panel_constructor.assert_not_called()
        guard.verify_read_only_attach_contract.assert_called_once_with()
        attach_read_only.assert_awaited_once_with(lightrag, config=config)
        lightrag.initialize_storages.assert_not_awaited()
    else:
        foundation.verify_tables.assert_not_awaited()
        foundation.ensure_tables.assert_awaited_once_with(
            specs=corpus_module.lightrag_retrieval_table_specs(lightrag)
        )
        file_panel_constructor.assert_called_once_with()
        file_panel.ensure_page_index.assert_awaited_once_with()
        guard.verify_read_only_attach_contract.assert_not_called()
        attach_read_only.assert_not_awaited()
        lightrag.initialize_storages.assert_awaited_once_with()
    metadata.initialize.assert_awaited_once_with(validate_only=is_reader)
    chunk_constructor.assert_called_once_with(lightrag)
    vector_constructor.assert_called_once_with(
        chunks_vdb,
        exact_threshold=config.corpus.retrieval.metadata_filter_exact_vector_threshold,
    )
    if is_reader:
        vectors.ensure_document_scope_index.assert_not_awaited()
    else:
        vectors.ensure_document_scope_index.assert_awaited_once_with()
    create_bm25.assert_awaited_once_with(config, profiles=profiles)
    assert stores.metadata_index is metadata
    assert stores.chunks is chunks
    assert stores.filtered_vectors is vectors
    assert stores.bm25 is bm25
    assert stores.bm25_languages == ("en",)


@pytest.mark.parametrize("validate_only", [False, True])
async def test_maintenance_initializes_registry_and_promotion_job_schemas(
    validate_only: bool,
) -> None:
    registry = SimpleNamespace(initialize=AsyncMock())
    promotion_jobs = SimpleNamespace(initialize=AsyncMock())
    store = PGCorpusMaintenanceStore(
        {},
        workspace_registry=registry,  # pyright: ignore[reportArgumentType]
        promotion_jobs=promotion_jobs,  # pyright: ignore[reportArgumentType]
    )

    await store.initialize(validate_only=validate_only)

    registry.initialize.assert_awaited_once_with(validate_only=validate_only)
    promotion_jobs.initialize.assert_awaited_once_with(validate_only=validate_only)


async def test_runtime_binder_rejects_missing_postgres_chunk_backend(
    test_config: DlightragConfig,
) -> None:
    lightrag = SimpleNamespace(
        chunks_vdb=None,
        text_chunks=None,
        full_docs=None,
        doc_status=None,
        initialize_storages=AsyncMock(),
        finalize_storages=AsyncMock(),
        aquery_data=AsyncMock(),
        apipeline_enqueue_documents=AsyncMock(),
        apipeline_process_enqueue_documents=AsyncMock(),
    )

    with pytest.raises(RuntimeError, match="chunks_vdb missing"):
        await PGCorpusRuntimeBinder(test_config).attach(lightrag)

    lightrag.initialize_storages.assert_awaited_once_with()
