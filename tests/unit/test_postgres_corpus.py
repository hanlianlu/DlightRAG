# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Focused behavior for the PostgreSQL corpus composition adapter."""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from dlightrag_rag.retrieval.bm25 import BM25Profile

from dlightrag.adapters.postgres import corpus as corpus_module
from dlightrag.adapters.postgres.corpus import (
    PGCorpusBackendFactory,
    PGCorpusCoordination,
    PGCorpusRuntimeBinder,
)
from dlightrag.config import DlightragConfig


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
        workspace=test_config.workspace,
        reader=False,
        require_halfvec=False,
        required_extensions=(),
        lightrag_pool_max_size=16,
        domain_pool_max_size=10,
        acquire_timeout=test_config.postgres_acquire_timeout,
    )

    with caplog.at_level(logging.WARNING, logger="dlightrag.adapters.postgres.corpus"):
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

    factory = PGCorpusBackendFactory(test_config)

    apply_backend.assert_not_called()
    apply_sidecar.assert_not_called()
    apply_runtime.assert_not_called()

    factory.create()

    apply_backend.assert_called_once_with(force=True)
    apply_sidecar.assert_called_once_with()
    apply_runtime.assert_called_once_with(force=True)


@pytest.mark.parametrize("is_reader", [False, True])
async def test_runtime_binder_composes_workspace_stores(
    test_config: DlightragConfig,
    monkeypatch: pytest.MonkeyPatch,
    is_reader: bool,
) -> None:
    config = test_config.model_copy(
        update={"service_role": "reader" if is_reader else "writer", "bm25_enabled": True}
    )
    metadata = SimpleNamespace(initialize=AsyncMock())
    chunks = object()
    vectors = SimpleNamespace(ensure_document_scope_index=AsyncMock())
    bm25 = object()
    profiles = (BM25Profile(name="en", text_config="english", languages=("en",)),)
    metadata_constructor = MagicMock(return_value=metadata)
    chunk_constructor = MagicMock(return_value=chunks)
    vector_constructor = MagicMock(return_value=vectors)
    create_bm25 = AsyncMock(return_value=bm25)
    monkeypatch.setattr(corpus_module, "PGMetadataIndex", metadata_constructor)
    monkeypatch.setattr(corpus_module, "PGCorpusChunkStore", chunk_constructor)
    monkeypatch.setattr(corpus_module, "PGFilteredVectorSearch", vector_constructor)
    monkeypatch.setattr(corpus_module, "profiles_from_config", MagicMock(return_value=profiles))
    monkeypatch.setattr(corpus_module, "create_postgres_bm25", create_bm25)
    chunks_vdb = object()
    lightrag = SimpleNamespace(chunks_vdb=chunks_vdb)

    stores = await PGCorpusRuntimeBinder(config).bind(lightrag)

    metadata_constructor.assert_called_once_with(workspace=config.workspace)
    metadata.initialize.assert_awaited_once_with(validate_only=is_reader)
    chunk_constructor.assert_called_once_with(lightrag)
    vector_constructor.assert_called_once_with(
        chunks_vdb,
        exact_threshold=config.metadata_filter_exact_vector_threshold,
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
