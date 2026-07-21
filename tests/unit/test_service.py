# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for RAGService facade (core/service.py)."""

import logging
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from lightrag.constants import DEFAULT_COSINE_THRESHOLD, DEFAULT_MM_IMAGE_MIN_PIXEL

from dlightrag.config import DlightragConfig
from dlightrag.core.document_embedding import RobustDocumentEmbedder
from dlightrag.core.ingestion.engine import PreparedIngestFile, UnifiedIngestionEngine
from dlightrag.core.service import RAGService, RemoteIngestWindowProgress
from dlightrag.sourcing.base import AsyncDataSource, SourceDocument


class _FakePostgresConn:
    def __init__(self, *, max_connections: str = "80") -> None:
        self.max_connections = max_connections

    async def fetchval(self, sql: str, *args) -> str:
        if sql == "SHOW max_connections":
            return self.max_connections
        raise AssertionError(f"unexpected SQL: {sql!r}")


# ---------------------------------------------------------------------------
# TestRAGServiceAingest
# ---------------------------------------------------------------------------


class TestRAGServiceAingest:
    """Test ingestion logic -- replace defaults, azure lifecycle."""

    def _make_initialized_service(self, config: DlightragConfig) -> tuple[RAGService, MagicMock]:
        service = RAGService(config=config)
        service._initialized = True
        ingestion = MagicMock()
        ingestion.aingest_file = AsyncMock(return_value={"status": "success"})
        ingestion.aingest_files = AsyncMock(
            return_value={"processed": 1, "errors": [], "results": [{"status": "success"}]}
        )
        service._ingestion_engine = ingestion
        return service, ingestion

    @patch("dlightrag.storage.workspaces.PGWorkspaceRegistry")
    async def test_workspace_meta_upsert_uses_canonical_workspace_id(
        self,
        mock_registry_cls: MagicMock,
        test_config: DlightragConfig,
    ) -> None:
        registry = MagicMock()
        registry.initialize = AsyncMock()
        registry.upsert = AsyncMock()
        mock_registry_cls.return_value = registry
        config = test_config.model_copy(update={"workspace": "test-fallback-ws"})
        service = RAGService(config=config)

        await service._upsert_workspace_meta()

        registry.upsert.assert_awaited_once_with(
            workspace="test_fallback_ws",
            display_name="test-fallback-ws",
            embedding_model=config.embedding.model,
        )

    async def test_aingest_not_initialized_raises(self, test_config: DlightragConfig) -> None:
        service = RAGService(config=test_config)
        with pytest.raises(RuntimeError, match="not initialized"):
            await service.aingest(source_type="local", path="/tmp/f.pdf")

    async def test_aingest_replace_default_from_config(
        self, test_config: DlightragConfig, tmp_path: Path
    ) -> None:
        test_config.ingestion_replace_default = True
        fake_pdf = tmp_path / "file.pdf"
        fake_pdf.write_bytes(b"%PDF-fake")
        service, ingestion = self._make_initialized_service(test_config)
        await service.aingest(source_type="local", path=str(fake_pdf))
        call_kwargs = ingestion.aingest_file.call_args
        assert call_kwargs.kwargs["replace"] is True

    async def test_aingest_replace_explicit_overrides_config(
        self, test_config: DlightragConfig, tmp_path: Path
    ) -> None:
        test_config.ingestion_replace_default = True
        fake_pdf = tmp_path / "file.pdf"
        fake_pdf.write_bytes(b"%PDF-fake")
        service, ingestion = self._make_initialized_service(test_config)
        await service.aingest(source_type="local", path=str(fake_pdf), replace=False)
        call_kwargs = ingestion.aingest_file.call_args
        assert call_kwargs.kwargs["replace"] is False

    # -- Azure blob lifecycle --

    async def test_aingest_azure_defaults_prefix_when_neither_set(
        self, test_config: DlightragConfig
    ) -> None:
        """When neither blob_path nor prefix provided, prefix defaults to '' (entire container)."""
        service, _ = self._make_initialized_service(test_config)
        mock_source = AsyncMock()
        seen_prefixes: list[str | None] = []

        async def _aiter_documents(prefix: str | None = None):
            seen_prefixes.append(prefix)
            for item in ():
                yield item

        mock_source.aiter_documents = _aiter_documents
        await service.aingest(source_type="azure_blob", source=mock_source, container_name="c")
        assert seen_prefixes == [""]

    async def test_aingest_azure_calls_aclose(self, test_config: DlightragConfig) -> None:
        """source.aclose() is called after successful ingestion."""
        service, _ = self._make_initialized_service(test_config)
        mock_source = AsyncMock()

        async def _aiter_documents(prefix: str | None = None):
            for item in ():
                yield item

        mock_source.aiter_documents = _aiter_documents
        await service.aingest(source_type="azure_blob", source=mock_source, container_name="c")
        mock_source.aclose.assert_awaited_once()

    async def test_aingest_azure_calls_aclose_on_error(self, test_config: DlightragConfig) -> None:
        """source.aclose() is called even when ingestion raises."""
        service, ingestion = self._make_initialized_service(test_config)
        ingestion.aingest_files = AsyncMock(side_effect=RuntimeError("ingestion failed"))
        mock_source = AsyncMock()

        async def _aiter_documents(prefix: str | None = None):
            yield SourceDocument(key="f.pdf")

        mock_source.aiter_documents = _aiter_documents
        mock_source.amaterialize_document = AsyncMock(
            side_effect=lambda _document, destination: destination.write_bytes(b"%PDF")
        )
        with pytest.raises(RuntimeError, match="ingestion failed"):
            await service.aingest(source_type="azure_blob", source=mock_source, container_name="c")
        mock_source.aclose.assert_awaited_once()


# ---------------------------------------------------------------------------
# TestRAGServiceClose
# ---------------------------------------------------------------------------


class TestRAGServiceClose:
    """Test cleanup logic."""

    async def test_close_handles_errors(self, test_config: DlightragConfig) -> None:
        service = RAGService(config=test_config)
        service._initialized = True
        service._lightrag = MagicMock()
        service._lightrag.finalize_storages = AsyncMock(side_effect=RuntimeError("cleanup failed"))

        # Should not raise
        await service.aclose()

    async def test_close_only_closes_underlying_multimodal_embedder(
        self, test_config: DlightragConfig
    ) -> None:
        service = RAGService(config=test_config)
        service._initialized = True
        service._multimodal_embedder = AsyncMock()
        service._document_embedder = MagicMock(spec=RobustDocumentEmbedder)

        await service.aclose()

        service._multimodal_embedder.aclose.assert_awaited_once()
        assert service._document_embedder.mock_calls == []

    async def test_close_shuts_down_lightrag_role_worker_pools(
        self, test_config: DlightragConfig
    ) -> None:
        service = RAGService(config=test_config)
        service._initialized = True
        query_func = MagicMock()
        query_func.shutdown = AsyncMock()
        extract_func = MagicMock()
        extract_func.shutdown = AsyncMock()
        embedding_func = MagicMock()
        embedding_func.shutdown = AsyncMock()
        lightrag = MagicMock()
        lightrag.embedding_func = MagicMock(func=embedding_func)
        lightrag.llm_model_func = None
        lightrag.rerank_model_func = None
        lightrag.role_llm_funcs = {
            "query": query_func,
            "extract": extract_func,
        }
        lightrag.finalize_storages = AsyncMock()
        service._lightrag = lightrag

        await service.aclose()

        embedding_func.shutdown.assert_awaited_once()
        query_func.shutdown.assert_awaited_once()
        extract_func.shutdown.assert_awaited_once()

    async def test_warmup_uses_lightrag_query_param(self, test_config: DlightragConfig) -> None:
        service = RAGService(config=test_config)
        service._lightrag = MagicMock()
        service._lightrag.aquery = AsyncMock()

        await service._warmup_lightrag_workers()

        await_args = service._lightrag.aquery.await_args
        assert await_args is not None
        kwargs = await_args.kwargs
        assert kwargs["param"].mode == "naive"
        assert kwargs["param"].enable_rerank is False


# ---------------------------------------------------------------------------
# TestRAGServiceRetrieve
# ---------------------------------------------------------------------------


class TestRAGServiceRetrieve:
    """Test aretrieve delegation to RetrievalEngine."""

    def _make_retrieval_service(self, config: DlightragConfig) -> tuple[RAGService, MagicMock]:
        from dlightrag.core.retrieval.protocols import RetrievalResult

        service = RAGService(config=config)
        service._initialized = True
        orchestrator = MagicMock()
        orchestrator.aretrieve = AsyncMock(return_value=RetrievalResult(contexts={"chunks": []}))
        service._retrieval_orchestrator = orchestrator
        service._lightrag_stores = MagicMock()
        return service, orchestrator

    async def test_aretrieve_delegates_to_orchestrator(self, test_config):
        service, orchestrator = self._make_retrieval_service(test_config)
        await service.aretrieve("test query")
        orchestrator.aretrieve.assert_awaited_once()

    async def test_aretrieve_forwards_caller_bm25_query(self, test_config):
        service, orchestrator = self._make_retrieval_service(test_config)
        await service.aretrieve("test query", bm25_query="alpha beta")
        call_kwargs = orchestrator.aretrieve.call_args.kwargs
        assert call_kwargs["bm25_query"] == "alpha beta"

    async def test_aretrieve_caller_bm25_query_overrides_plan(self, test_config):
        from types import SimpleNamespace

        service, orchestrator = self._make_retrieval_service(test_config)
        plan = SimpleNamespace(
            metadata_filter=None, metadata_filter_source=None, bm25_query="plan terms"
        )
        await service.aretrieve("test query", bm25_query="caller terms", _plan=plan)
        call_kwargs = orchestrator.aretrieve.call_args.kwargs
        assert call_kwargs["bm25_query"] == "caller terms"

    async def test_aretrieve_blank_bm25_query_falls_back_to_plan(self, test_config):
        from types import SimpleNamespace

        service, orchestrator = self._make_retrieval_service(test_config)
        plan = SimpleNamespace(
            metadata_filter=None, metadata_filter_source=None, bm25_query="plan terms"
        )
        await service.aretrieve("test query", bm25_query="   ", _plan=plan)
        call_kwargs = orchestrator.aretrieve.call_args.kwargs
        assert call_kwargs["bm25_query"] == "plan terms"

    async def test_aretrieve_without_bm25_query_or_plan_passes_none(self, test_config):
        service, orchestrator = self._make_retrieval_service(test_config)
        await service.aretrieve("test query")
        call_kwargs = orchestrator.aretrieve.call_args.kwargs
        assert call_kwargs["bm25_query"] is None

    async def test_aretrieve_reranks_after_hydrating_fused_chunks(
        self, test_config, monkeypatch: pytest.MonkeyPatch
    ):
        from dlightrag.core.retrieval.protocols import RetrievalResult

        service, orchestrator = self._make_retrieval_service(test_config)
        orchestrator.aretrieve.return_value = RetrievalResult(
            contexts={
                "chunks": [
                    {"chunk_id": "semantic-a", "content": "semantic"},
                    {"chunk_id": "bm25-visual", "content": "visual text"},
                ],
                "entities": [],
                "relationships": [],
            }
        )

        async def hydrate(_stores, chunks, *, include_image_data=True):
            chunks[1]["image_data"] = "hydrated-image"

        async def rerank_func(query: str, chunks: list[dict], top_k: int) -> list[dict]:
            assert query == "test query"
            assert top_k == 3
            assert chunks[1]["image_data"] == "hydrated-image"
            return [chunks[1], chunks[0]]

        service._rerank_func = rerank_func
        monkeypatch.setattr(
            "dlightrag.core.retrieval.provenance.hydrate_lightrag_chunk_provenance",
            hydrate,
        )

        result = await service.aretrieve("test query", chunk_top_k=3)

        assert [c["chunk_id"] for c in result.contexts["chunks"]] == ["bm25-visual", "semantic-a"]
        assert result.trace["reranked_chunk_count"] == 2

    async def test_aretrieve_caps_fused_chunks_when_rerank_disabled(
        self, test_config, monkeypatch: pytest.MonkeyPatch
    ):
        from dlightrag.core.retrieval.protocols import RetrievalResult

        service, orchestrator = self._make_retrieval_service(test_config)
        orchestrator.aretrieve.return_value = RetrievalResult(
            contexts={
                "chunks": [{"chunk_id": f"c{i}", "content": f"c{i}"} for i in range(5)],
                "entities": [],
                "relationships": [],
            }
        )

        async def hydrate(_stores, chunks, *, include_image_data=True):
            return None

        service._rerank_func = None  # reranker disabled
        monkeypatch.setattr(
            "dlightrag.core.retrieval.provenance.hydrate_lightrag_chunk_provenance",
            hydrate,
        )

        result = await service.aretrieve("test query", chunk_top_k=3)

        # Budget is honored even without a reranker: only the top-3 fused chunks
        # survive instead of leaking the full semantic∪BM25 union.
        assert [c["chunk_id"] for c in result.contexts["chunks"]] == ["c0", "c1", "c2"]
        assert result.trace["reranked_chunk_count"] == 3

    async def test_aretrieve_rerank_failure_keeps_hydrated_rrf_top_k(
        self, test_config, monkeypatch: pytest.MonkeyPatch
    ):
        from dlightrag.core.retrieval.protocols import RetrievalResult

        service, orchestrator = self._make_retrieval_service(test_config)
        orchestrator.aretrieve.return_value = RetrievalResult(
            contexts={
                "chunks": [{"chunk_id": f"c{i}", "content": f"c{i}"} for i in range(5)],
                "entities": [],
                "relationships": [],
            }
        )

        async def hydrate(_stores, chunks, *, include_image_data=True):
            for chunk in chunks:
                chunk["page_idx"] = 7

        async def fail_rerank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
            assert query == "test query"
            assert top_k == 3
            assert all(chunk["page_idx"] == 7 for chunk in chunks)
            raise RuntimeError("provider unavailable")

        service._rerank_func = fail_rerank
        monkeypatch.setattr(
            "dlightrag.core.retrieval.provenance.hydrate_lightrag_chunk_provenance",
            hydrate,
        )

        result = await service.aretrieve("test query", chunk_top_k=3)

        assert [chunk["chunk_id"] for chunk in result.contexts["chunks"]] == [
            "c0",
            "c1",
            "c2",
        ]
        assert result.trace["reranked_chunk_count"] == 3
        assert result.trace["rerank_error"] == "RuntimeError"

    async def test_aretrieve_skips_chunks_hydrated_by_backend(
        self, test_config, monkeypatch: pytest.MonkeyPatch
    ):
        from dlightrag.core.retrieval.protocols import RetrievalResult

        service, orchestrator = self._make_retrieval_service(test_config)
        orchestrator.aretrieve.return_value = RetrievalResult(
            contexts={
                "chunks": [
                    {
                        "chunk_id": "semantic-a",
                        "content": "semantic",
                        "full_doc_id": "doc-a",
                    },
                    {"chunk_id": "bm25-b", "content": "lexical", "full_doc_id": "doc-b"},
                ],
                "entities": [],
                "relationships": [],
            },
            trace={"provenance_hydrated_chunk_ids": ["semantic-a"]},
        )
        seen: list[list[str]] = []

        async def hydrate(_stores, chunks, *, include_image_data=True):
            seen.append([chunk["chunk_id"] for chunk in chunks])
            chunks[0]["page_idx"] = 7

        monkeypatch.setattr(
            "dlightrag.core.retrieval.provenance.hydrate_lightrag_chunk_provenance",
            hydrate,
        )

        result = await service.aretrieve("test query")

        assert seen == [["bm25-b"]]
        assert result.contexts["chunks"][1]["page_idx"] == 7

    async def test_aretrieve_defers_image_hydration_for_text_reranker(
        self, test_config, monkeypatch: pytest.MonkeyPatch
    ):
        """Text reranker: image bytes are read only for chunks that survive rerank."""
        from dlightrag.core.retrieval.protocols import RetrievalResult

        service, orchestrator = self._make_retrieval_service(test_config)
        service._rerank_consumes_images = False  # text-only reranker
        orchestrator.aretrieve.return_value = RetrievalResult(
            contexts={
                "chunks": [
                    {"chunk_id": "keep", "content": "keep"},
                    {"chunk_id": "drop", "content": "drop"},
                ],
                "entities": [],
                "relationships": [],
            }
        )

        calls: list[tuple[list[str], bool]] = []

        async def hydrate(_stores, chunks, *, include_image_data=True):
            calls.append(([c["chunk_id"] for c in chunks], include_image_data))
            if include_image_data:
                for c in chunks:
                    c["image_data"] = f"img-{c['chunk_id']}"

        async def rerank_func(query: str, chunks: list[dict], top_k: int) -> list[dict]:
            # Image bytes are still deferred at rerank time for a text reranker.
            assert "image_data" not in chunks[0]
            return [c for c in chunks if c["chunk_id"] == "keep"]

        service._rerank_func = rerank_func
        monkeypatch.setattr(
            "dlightrag.core.retrieval.provenance.hydrate_lightrag_chunk_provenance",
            hydrate,
        )

        result = await service.aretrieve("test query", chunk_top_k=3)

        # Pass 1 hydrates metadata only (no image bytes) for the union; pass 2
        # reads image bytes for the single surviving chunk.
        assert calls[0] == (["keep", "drop"], False)
        assert calls[1] == (["keep"], True)
        survivors = result.contexts["chunks"]
        assert [c["chunk_id"] for c in survivors] == ["keep"]
        assert survivors[0]["image_data"] == "img-keep"

    async def test_aretrieve_not_initialized_raises(self, test_config):
        service = RAGService(config=test_config)
        with pytest.raises(RuntimeError, match="not initialized"):
            await service.aretrieve("query")


# ---------------------------------------------------------------------------
# TestRAGServiceFileManagement
# ---------------------------------------------------------------------------


class TestRAGServiceFileManagement:
    """Test alist_ingested_files and adelete_files delegation."""

    async def test_alist_not_initialized_raises(self, test_config):
        service = RAGService(config=test_config)
        with pytest.raises(RuntimeError, match="not initialized"):
            await service.alist_ingested_files()

    async def test_adelete_not_initialized_raises(self, test_config):
        service = RAGService(config=test_config)
        with pytest.raises(RuntimeError, match="not initialized"):
            await service.adelete_files(filenames=["a.pdf"])

    async def test_alist_reads_lightrag_doc_status(self, test_config):
        service = RAGService(config=test_config)
        service._initialized = True
        service._lightrag_stores = MagicMock()
        service._lightrag_stores.docs_by_status = AsyncMock(
            return_value={"d1": MagicMock(file_path="/tmp/a.pdf", updated_at="now")}
        )
        result = await service.alist_ingested_files()
        assert result == [
            {
                "doc_id": "d1",
                "file_path": "/tmp/a.pdf",
                "status": "processed",
                "updated_at": "now",
            }
        ]
        service._lightrag_stores.docs_by_status.assert_awaited_once()

    async def test_adelete_uses_cascade_pipeline(self, test_config):
        service = RAGService(config=test_config)
        service._initialized = True
        service._lightrag = MagicMock()
        service._lightrag.adelete_by_doc_id = AsyncMock()
        service._lightrag.doc_status = MagicMock()
        service._lightrag.doc_status.get_doc_by_file_path = AsyncMock(return_value=None)
        service._lightrag.doc_status.get_docs_by_status = AsyncMock(
            return_value={"d1": MagicMock(file_path="/tmp/a.pdf")}
        )
        result = await service.adelete_files(filenames=["a.pdf"])
        assert result[0]["status"] == "deleted"
        assert result[0]["docs_deleted"] == 1
        service._lightrag.adelete_by_doc_id.assert_awaited_once()


# ---------------------------------------------------------------------------
# TestBuildVectorDbKwargs
# ---------------------------------------------------------------------------


class TestBuildVectorDbKwargs:
    """Test _build_vector_db_kwargs passthrough."""

    def test_default_uses_lightrag_cosine_threshold(self, test_config: DlightragConfig) -> None:
        result = RAGService._build_vector_db_kwargs(test_config)
        assert result == {"cosine_better_than_threshold": DEFAULT_COSINE_THRESHOLD}

    def test_passthrough_merges_kwargs(self, test_config: DlightragConfig) -> None:
        test_config.vector_db_kwargs = {
            "index_type": "HNSW_SQ",
            "sq_type": "SQ8",
            "hnsw_m": 32,
        }
        result = RAGService._build_vector_db_kwargs(test_config)
        assert result["cosine_better_than_threshold"] == DEFAULT_COSINE_THRESHOLD
        assert result["index_type"] == "HNSW_SQ"
        assert result["sq_type"] == "SQ8"
        assert result["hnsw_m"] == 32

    def test_passthrough_overrides_default(self, test_config: DlightragConfig) -> None:
        test_config.vector_db_kwargs = {"cosine_better_than_threshold": 0.5}
        result = RAGService._build_vector_db_kwargs(test_config)
        assert result["cosine_better_than_threshold"] == 0.5


# ---------------------------------------------------------------------------
# TestRequiredPostgresExtensions
# ---------------------------------------------------------------------------


class TestRequiredPostgresExtensions:
    """Test DlightRAG-owned PostgreSQL extension bootstrap policy."""

    def test_bm25_defaults_require_textsearch_and_jieba(self, test_config: DlightragConfig) -> None:
        test_config.bm25_enabled = True

        assert RAGService._required_postgres_extensions(test_config) == (
            "pg_textsearch",
            "pg_jieba",
        )

    def test_bm25_disabled_requires_no_extra_extensions(self, test_config: DlightragConfig) -> None:
        test_config.bm25_enabled = False

        assert RAGService._required_postgres_extensions(test_config) == ()


class TestPostgresConcurrencySanity:
    """Test startup connection-budget warnings."""

    async def test_warns_when_pool_budget_exceeds_postgres_headroom(
        self,
        test_config: DlightragConfig,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setenv("WEB_CONCURRENCY", "2")
        test_config.postgres_lightrag_pool_max_size = 16
        test_config.postgres_pool_max_size = 10
        service = RAGService(config=test_config)

        with caplog.at_level(logging.WARNING, logger="dlightrag.core.service"):
            await service._check_postgres_concurrency_sanity(
                _FakePostgresConn(max_connections="50")
            )

        assert "PostgreSQL connection budget is tight" in caplog.text
        assert "estimated_pool_connections=52" in caplog.text


# ---------------------------------------------------------------------------
# TestBuildAddonParams
# ---------------------------------------------------------------------------


class TestBuildAddonParams:
    """Test LightRAG 1.5 addon_params contract."""

    def test_uses_lightrag_15_entity_guidance(self, test_config: DlightragConfig) -> None:
        test_config.kg_entity_types = ["Product", "Technology", "Organization"]

        result = RAGService._build_addon_params(test_config)

        assert result["language"] == "English"
        assert "entity_types" not in result
        assert result["entity_types_guidance"] == (
            "Prioritize domain entities in these categories: Product, Technology, Organization."
        )

    def test_addon_params_include_extraction_prompt_profile(
        self, test_config: DlightragConfig
    ) -> None:
        test_config.extraction.language = "Chinese"
        test_config.extraction.entity_type_prompt_file = "domain-entities.yaml"
        test_config.kg_entity_types = []

        result = RAGService._build_addon_params(test_config)

        assert result == {
            "language": "Chinese",
            "entity_type_prompt_file": "domain-entities.yaml",
            "chunker": {"paragraph_semantic": {"chunk_token_size": 1024}},
        }


class TestBuildRetrievalBackend:
    """Test DlightRAG retrieval backend configuration wiring."""

    def test_passes_query_budget_config(self, test_config: DlightragConfig) -> None:
        test_config.direct_visual_top_k = 9
        test_config.max_entity_tokens = 111
        test_config.max_relation_tokens = 222
        test_config.max_total_tokens = 333

        backend = RAGService._build_retrieval_backend(
            test_config,
            lightrag=MagicMock(),
            stores=MagicMock(),
            embedder=MagicMock(),
        )

        assert backend._direct_visual_top_k == 9
        assert backend._max_entity_tokens == 111
        assert backend._max_relation_tokens == 222
        assert backend._max_total_tokens == 333


class TestDirectImageEmbeddingCapability:
    """Test image embedding capability resolution for the direct-visual leg."""

    async def test_text_only_embedder_disables_direct_image_embedding_without_probe(self) -> None:
        embedder = MagicMock()
        embedder.supports_images = False
        embedder.probe_image_embedding = AsyncMock()

        enabled = await RAGService._resolve_direct_image_embedding_enabled(
            embedder,
            startup_probe=True,
            require_image_support=False,
        )

        assert enabled is False
        embedder.probe_image_embedding.assert_not_awaited()

    async def test_failed_startup_probe_disables_direct_image_embedding(self) -> None:
        embedder = MagicMock()
        embedder.supports_images = True
        embedder.probe_image_embedding = AsyncMock(side_effect=RuntimeError("image rejected"))

        enabled = await RAGService._resolve_direct_image_embedding_enabled(
            embedder,
            startup_probe=True,
            require_image_support=False,
        )

        assert enabled is False
        embedder.probe_image_embedding.assert_awaited_once()

    async def test_required_multimodal_rejects_text_only_embedder(self) -> None:
        embedder = MagicMock()
        embedder.supports_images = False
        embedder.probe_image_embedding = AsyncMock()

        with pytest.raises(
            ValueError,
            match="input_modality='multimodal'.*does not support image inputs",
        ):
            await RAGService._resolve_direct_image_embedding_enabled(
                embedder,
                startup_probe=True,
                require_image_support=True,
            )

        embedder.probe_image_embedding.assert_not_awaited()

    async def test_required_multimodal_probe_failure_is_fatal(self) -> None:
        embedder = MagicMock()
        embedder.supports_images = True
        embedder.probe_image_embedding = AsyncMock(side_effect=RuntimeError("image rejected"))

        with pytest.raises(
            ValueError,
            match="input_modality='multimodal'.*startup probe failed",
        ):
            await RAGService._resolve_direct_image_embedding_enabled(
                embedder,
                startup_probe=True,
                require_image_support=True,
            )

    async def test_disabled_probe_trusts_resolved_multimodal_capability(self) -> None:
        embedder = MagicMock()
        embedder.supports_images = True
        embedder.probe_image_embedding = AsyncMock()

        enabled = await RAGService._resolve_direct_image_embedding_enabled(
            embedder,
            startup_probe=False,
            require_image_support=True,
        )

        assert enabled is True
        embedder.probe_image_embedding.assert_not_awaited()

    @pytest.mark.parametrize(
        ("configured_minimum", "expected_minimum"),
        [(None, DEFAULT_MM_IMAGE_MIN_PIXEL), (64, 64)],
    )
    def test_document_embedder_factory_uses_shared_runtime_limits(
        self,
        test_config: DlightragConfig,
        configured_minimum: int | None,
        expected_minimum: int,
    ) -> None:
        test_config.embedding.dim = 7
        test_config.embedding_func_max_async = 5
        test_config.parser_sidecars.vlm.min_image_pixel = configured_minimum
        embedder = MagicMock()
        expected = MagicMock()

        with patch(
            "dlightrag.core.document_embedding.RobustDocumentEmbedder",
            return_value=expected,
        ) as constructor:
            result = RAGService._build_document_embedder(
                test_config,
                embedder,
                image_enabled=True,
            )

        assert result is expected
        constructor.assert_called_once_with(
            embedder=embedder,
            image_enabled=True,
            dimension=7,
            min_image_pixel=expected_minimum,
            batch_size=8,
            max_concurrency=5,
        )


# ---------------------------------------------------------------------------
# TestRAGServiceLightRAGMainPath
# ---------------------------------------------------------------------------


class TestRAGServiceLightRAGMainPath:
    """Test LightRAG-main path behavior in RAGService."""

    async def test_aingest_azure_blob_single(self, test_config: DlightragConfig) -> None:
        """Downloads one blob into an ephemeral parser item and stores remote metadata."""
        service = RAGService(config=test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_file = AsyncMock()
        seen_items: list[PreparedIngestFile] = []

        async def _ingest(items: list[PreparedIngestFile], **_: object) -> dict[str, object]:
            seen_items.extend(items)
            assert len(items) == 1
            assert items[0].parser_path.exists()
            return {
                "processed": 1,
                "errors": [],
                "results": [{"doc_id": "d1", "page_count": 2}],
            }

        service._ingestion_engine.aingest_files = AsyncMock(side_effect=_ingest)

        mock_source = AsyncMock()
        mock_source.amaterialize_document = AsyncMock(
            side_effect=lambda _doc_id, destination: destination.write_bytes(b"%PDF-fake-content")
        )

        result = await service.aingest(
            source_type="azure_blob",
            container_name="test-container",
            blob_path="docs/report.pdf",
            source=mock_source,
        )

        service._ingestion_engine.aingest_file.assert_not_awaited()
        service._ingestion_engine.aingest_files.assert_awaited_once()
        assert result["doc_id"] == "d1"
        item = seen_items[0]
        assert item.source_uri == "azure://test-container/docs/report.pdf"
        assert item.download_locator == "azure://test-container/docs/report.pdf"
        assert item.display_filename == "report.pdf"
        assert item.parser_path.suffix == ".pdf"
        assert "report" in item.parser_path.name
        assert not item.parser_path.exists()
        assert not (test_config.working_dir_path / "sources").exists()

    async def test_aingest_unified_azure_blob_batch(self, test_config: DlightragConfig) -> None:
        """Prefix ingest calls the engine once with a prepared batch."""
        service = RAGService(config=test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_file = AsyncMock()
        seen_items: list[PreparedIngestFile] = []

        async def _ingest(items: list[PreparedIngestFile], **_: object) -> dict[str, object]:
            seen_items.extend(items)
            assert all(item.parser_path.exists() for item in items)
            return {
                "processed": len(items),
                "errors": [],
                "results": [{"doc_id": item.display_filename} for item in items],
            }

        service._ingestion_engine.aingest_files = AsyncMock(side_effect=_ingest)

        mock_source = AsyncMock()

        async def _aiter_documents(prefix: str | None = None):
            assert prefix == "docs/"
            yield SourceDocument(key="team-a/report.pdf")
            yield SourceDocument(key="team-b/report.pdf")

        mock_source.aiter_documents = _aiter_documents
        mock_source.amaterialize_document = AsyncMock(
            side_effect=lambda _document, destination: destination.write_bytes(b"%PDF-fake")
        )

        result = await service.aingest(
            source_type="azure_blob",
            container_name="c",
            prefix="docs/",
            source=mock_source,
        )

        service._ingestion_engine.aingest_file.assert_not_awaited()
        service._ingestion_engine.aingest_files.assert_awaited_once()
        assert result["processed"] == 2
        assert [item.source_uri for item in seen_items] == [
            "azure://c/team-a/report.pdf",
            "azure://c/team-b/report.pdf",
        ]
        assert [item.download_locator for item in seen_items] == [
            "azure://c/team-a/report.pdf",
            "azure://c/team-b/report.pdf",
        ]
        assert [item.display_filename for item in seen_items] == ["report.pdf", "report.pdf"]
        assert len({item.parser_path.name for item in seen_items}) == 2
        assert all(item.parser_path.suffix == ".pdf" for item in seen_items)
        assert all(not item.parser_path.exists() for item in seen_items)

    async def test_aingest_unified_s3(self, test_config: DlightragConfig) -> None:
        """S3 single-object ingest uses the same remote batch path."""
        service = RAGService(config=test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_file = AsyncMock()
        service._ingestion_engine.aingest_files = AsyncMock(
            return_value={
                "processed": 1,
                "errors": [],
                "results": [{"doc_id": "d1", "status": "success"}],
            }
        )

        mock_source = AsyncMock()
        mock_source.amaterialize_document = AsyncMock(
            side_effect=lambda _doc_id, destination: destination.write_bytes(b"%PDF-fake")
        )
        mock_source.aclose = AsyncMock()

        with patch("dlightrag.sourcing.aws_s3.S3DataSource", return_value=mock_source):
            result = await service.aingest(
                source_type="s3",
                bucket="my-bucket",
                s3_key="docs/report.pdf",
                replace=True,
            )

        service._ingestion_engine.aingest_file.assert_not_awaited()
        service._ingestion_engine.aingest_files.assert_awaited_once()
        assert result["status"] == "success"

    async def test_aingest_s3_prefix_streams_keys_without_materializing_list(
        self, test_config: DlightragConfig
    ) -> None:
        """S3 prefix ingest consumes streaming keys instead of alist_documents()."""

        class StreamingS3Source:
            def __init__(self) -> None:
                self.loaded: list[str] = []

            async def alist_documents(self, prefix: str | None = None) -> list[SourceDocument]:
                raise AssertionError("prefix ingest should stream keys")

            async def aiter_documents(self, prefix: str | None = None):
                assert prefix == "docs/"
                yield SourceDocument(key="docs/a.pdf")
                yield SourceDocument(key="docs/b.pdf")

            async def amaterialize_document(
                self, document: SourceDocument, destination: Path
            ) -> None:
                self.loaded.append(document.key)
                destination.write_bytes(b"%PDF-fake")

            async def aclose(self) -> None:
                return None

        service = RAGService(config=test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_files = AsyncMock(
            return_value={
                "processed": 2,
                "errors": [],
                "results": [{"doc_id": "d1"}, {"doc_id": "d2"}],
            }
        )
        source = StreamingS3Source()

        result = await service.aingest(
            source_type="s3",
            bucket="my-bucket",
            prefix="docs/",
            source=source,
        )

        assert result["processed"] == 2
        assert source.loaded == ["docs/a.pdf", "docs/b.pdf"]

    async def test_aingest_s3_prefix_resume_skips_completed_windows(
        self, test_config: DlightragConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Recovered remote jobs resume from the next unfinished source window."""
        monkeypatch.setattr("dlightrag.core.service._REMOTE_INGEST_BATCH_SIZE", 2)

        class StreamingS3Source:
            def __init__(self) -> None:
                self.loaded: list[str] = []

            async def aiter_documents(self, prefix: str | None = None):
                assert prefix == "docs/"
                for name in ("a", "b", "c", "d", "e"):
                    yield SourceDocument(key=f"docs/{name}.pdf")

            async def amaterialize_document(
                self, document: SourceDocument, destination: Path
            ) -> None:
                self.loaded.append(document.key)
                destination.write_bytes(b"%PDF-fake")

            async def aclose(self) -> None:
                return None

        service = RAGService(config=test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_files = AsyncMock(
            side_effect=lambda items, **_: {
                "processed": len(items),
                "errors": [],
                "results": [{"doc_id": item.display_filename} for item in items],
            }
        )
        source = StreamingS3Source()

        result = await service.aingest(
            source_type="s3",
            bucket="my-bucket",
            prefix="docs/",
            source=source,
            _resume_from_window=2,
        )

        assert result["processed"] == 1
        assert source.loaded == ["docs/e.pdf"]
        service._ingestion_engine.aingest_files.assert_awaited_once()

    async def test_aingest_s3_prefix_progress_includes_download_errors(
        self, test_config: DlightragConfig
    ) -> None:
        class PartiallyFailingS3Source:
            async def aiter_documents(self, prefix: str | None = None):
                yield SourceDocument(key="docs/a.pdf")
                yield SourceDocument(key="docs/b.pdf")

            async def amaterialize_document(
                self, document: SourceDocument, destination: Path
            ) -> None:
                if document.key == "docs/b.pdf":
                    raise RuntimeError("download failed")
                destination.write_bytes(b"%PDF-fake")

            async def aclose(self) -> None:
                return None

        service = RAGService(config=test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_files = AsyncMock(
            return_value={"processed": 1, "errors": [], "results": [{"doc_id": "d1"}]}
        )
        progress_events: list[RemoteIngestWindowProgress] = []

        async def _record_progress(progress: RemoteIngestWindowProgress) -> None:
            progress_events.append(progress)

        result = await service.aingest(
            source_type="s3",
            bucket="my-bucket",
            prefix="docs/",
            source=PartiallyFailingS3Source(),
            _progress_callback=_record_progress,
        )

        assert result["processed"] == 1
        assert result["errors"] == ["b.pdf: remote materialization failed"]
        assert len(progress_events) == 1
        progress = progress_events[0]
        assert progress.total_delta == 2
        assert progress.processed_delta == 1
        assert progress.failed_delta == 1
        assert progress.errors == ("b.pdf: remote materialization failed",)

    @pytest.mark.parametrize(
        ("source_type", "source_uri"),
        [
            ("s3", "s3://my-bucket/docs/report.pdf"),
            ("azure_blob", "azure://my-container/docs/report.pdf"),
            ("url", "https://example.com/docs/report.pdf?token=secret"),
        ],
    )
    async def test_remote_source_retention_keeps_workspace_file_and_explicit_contract(
        self,
        test_config: DlightragConfig,
        source_type: str,
        source_uri: str,
    ) -> None:
        class RemoteSource(AsyncDataSource):
            async def aiter_documents(self, prefix: str | None = None):
                yield SourceDocument(key="asset-123", display_filename="report.pdf")

            async def amaterialize_document(
                self, document: SourceDocument, destination: Path
            ) -> None:
                destination.write_bytes(b"%PDF-retained")

        service = RAGService(config=test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        seen_items: list[PreparedIngestFile] = []

        async def _ingest(items: list[PreparedIngestFile], **_: object) -> dict[str, object]:
            seen_items.extend(items)
            assert items[0].parser_path.exists()
            return {"processed": 1, "errors": [], "results": [{"doc_id": "d1"}]}

        service._ingestion_engine.aingest_files = AsyncMock(side_effect=_ingest)

        result = await service.aingest_source(
            RemoteSource(),
            source_type=source_type,
            source_uri_for_key=lambda _key: source_uri,
            retain_source_file=True,
        )

        assert result["processed"] == 1
        item = seen_items[0]
        assert item.parser_path.exists()
        assert item.parser_path.read_bytes() == b"%PDF-retained"
        assert item.source_uri == source_uri
        assert item.download_locator == str(item.parser_path)
        assert item.display_filename == "report.pdf"
        assert item.parser_path.suffix == ".pdf"
        assert item.source_uri != item.download_locator
        assert item.parser_path.is_relative_to(
            test_config.input_dir_path / test_config.workspace / "__remote_sources__" / source_type
        )
        assert "__remote_ingest__" not in str(item.parser_path)

    async def test_remote_source_retention_call_override_can_disable_config(
        self, test_config: DlightragConfig
    ) -> None:
        test_config.retain_remote_source_files = True

        class RemoteSource(AsyncDataSource):
            async def aiter_documents(self, prefix: str | None = None):
                yield SourceDocument(key="docs/report.pdf")

            async def amaterialize_document(
                self, document: SourceDocument, destination: Path
            ) -> None:
                destination.write_bytes(b"%PDF-transient")

        service = RAGService(config=test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        seen_items: list[PreparedIngestFile] = []

        async def _ingest(items: list[PreparedIngestFile], **_: object) -> dict[str, object]:
            seen_items.extend(items)
            assert items[0].parser_path.exists()
            return {"processed": 1, "errors": [], "results": [{"doc_id": "d1"}]}

        service._ingestion_engine.aingest_files = AsyncMock(side_effect=_ingest)

        result = await service.aingest_source(
            RemoteSource(),
            source_type="s3",
            source_uri_for_key=lambda _key: "s3://my-bucket/docs/report.pdf",
            download_uri_for_key=lambda _key: "s3://my-bucket/docs/report.pdf",
            retain_source_file=False,
        )

        assert result["processed"] == 1
        item = seen_items[0]
        assert item.source_uri == "s3://my-bucket/docs/report.pdf"
        assert item.download_locator == "s3://my-bucket/docs/report.pdf"
        assert not item.parser_path.exists()

    async def test_aingest_source_accepts_sdk_async_data_source(
        self, test_config: DlightragConfig
    ) -> None:
        """SDK callers can ingest custom connector output without pretending it is S3."""

        class BynderSource(AsyncDataSource):
            def __init__(self) -> None:
                self.loaded: list[str] = []
                self.closed = False

            async def aiter_documents(
                self, prefix: str | None = None
            ) -> AsyncIterator[SourceDocument]:
                assert prefix == "approved/"
                yield SourceDocument(key="asset-123/report.pdf")

            async def amaterialize_document(
                self, document: SourceDocument, destination: Path
            ) -> None:
                self.loaded.append(document.key)
                destination.write_bytes(b"%PDF-fake")

            async def aclose(self) -> None:
                self.closed = True

        service = RAGService(config=test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        seen_items: list[PreparedIngestFile] = []

        async def _ingest(items: list[PreparedIngestFile], **_: object) -> dict[str, object]:
            seen_items.extend(items)
            return {
                "processed": len(items),
                "errors": [],
                "results": [{"doc_id": item.display_filename} for item in items],
            }

        service._ingestion_engine.aingest_files = AsyncMock(side_effect=_ingest)
        source = BynderSource()

        result = await service.aingest_source(
            source,
            source_type="bynder",
            prefix="approved/",
            source_uri_for_key=lambda key: f"bynder://assets/{key}",
            download_uri_for_key=lambda key: f"https://cdn.example.com/assets/{Path(key).name}",
            title="Approved asset",
            metadata={"collection": "marketing"},
        )

        assert result["processed"] == 1
        assert source.loaded == ["asset-123/report.pdf"]
        assert source.closed is True
        assert seen_items[0].source_uri == "bynder://assets/asset-123/report.pdf"
        assert seen_items[0].download_locator == ("https://cdn.example.com/assets/report.pdf")
        assert seen_items[0].display_filename == "report.pdf"
        assert seen_items[0].parser_path.suffix == ".pdf"

    async def test_custom_non_retained_source_without_download_uri_fails_before_materialize(
        self, test_config: DlightragConfig
    ) -> None:
        materialized = False

        class BynderSource(AsyncDataSource):
            async def aiter_documents(self, prefix: str | None = None):
                yield SourceDocument(key="asset/report.pdf", source_uri="bynder://asset/1")

            async def amaterialize_document(
                self, document: SourceDocument, destination: Path
            ) -> None:
                nonlocal materialized
                materialized = True

        service = RAGService(config=test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_files = AsyncMock(
            return_value={"processed": 1, "errors": [], "results": []}
        )

        result = await service.aingest_source(
            BynderSource(),
            source_type="bynder",
            retain_source_file=False,
        )

        assert result["processed"] == 0
        assert result["errors"] == [
            "retain_source_file=false requires a durable download_uri for source report.pdf; "
            "provide download_uri/download_uri_for_key or enable retain_source_file"
        ]
        assert materialized is False

    async def test_signed_url_requires_retention_or_explicit_download_uri(
        self, test_config: DlightragConfig
    ) -> None:
        class SignedUrlSource(AsyncDataSource):
            async def aiter_documents(self, prefix: str | None = None):
                yield SourceDocument(
                    key="report.pdf",
                    source_uri="https://fetch.example.com/report.pdf?token=secret",
                )

            async def amaterialize_document(
                self, document: SourceDocument, destination: Path
            ) -> None:
                raise AssertionError("materialization must not run")

        service = RAGService(config=test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_files = AsyncMock(
            return_value={"processed": 1, "errors": [], "results": []}
        )

        result = await service.aingest_source(
            SignedUrlSource(),
            source_type="url",
            retain_source_file=False,
        )

        assert result["processed"] == 0
        assert "durable download_uri" in result["errors"][0]
        assert "token=secret" not in result["errors"][0]

    async def test_custom_non_retained_source_accepts_separate_download_uri(
        self, test_config: DlightragConfig
    ) -> None:
        class BynderSource(AsyncDataSource):
            async def aiter_documents(self, prefix: str | None = None):
                yield SourceDocument(
                    key="asset/report.pdf",
                    source_uri="bynder://asset/1",
                    download_uri="https://cdn.example.com/assets/1.pdf",
                )

            async def amaterialize_document(
                self, document: SourceDocument, destination: Path
            ) -> None:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"%PDF-fake")

        service = RAGService(config=test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_files = AsyncMock(
            return_value={"processed": 1, "errors": [], "results": []}
        )

        result = await service.aingest_source(
            BynderSource(),
            source_type="bynder",
            download_uri_for_key=lambda _key: "s3://fallback/report.pdf",
            retain_source_file=False,
        )

        assert result["processed"] == 1
        await_args = service._ingestion_engine.aingest_files.await_args
        assert await_args is not None
        prepared = await_args.args[0][0]
        assert prepared.source_uri == "bynder://asset/1"
        assert prepared.download_locator == "https://cdn.example.com/assets/1.pdf"

    async def test_invalid_download_uri_fails_before_materialize_and_logs_safely(
        self,
        test_config: DlightragConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        materialized = False

        class BynderSource(AsyncDataSource):
            async def aiter_documents(self, prefix: str | None = None):
                yield SourceDocument(
                    key="asset/report.pdf",
                    source_uri="bynder://asset/1",
                    download_uri="https://cdn.example.com/assets/1.pdf?token=secret",
                )

            async def amaterialize_document(
                self, document: SourceDocument, destination: Path
            ) -> None:
                nonlocal materialized
                materialized = True

        service = RAGService(config=test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()

        with caplog.at_level(logging.INFO, logger="dlightrag.core.service"):
            result = await service.aingest_source(
                BynderSource(),
                source_type="bynder",
                retain_source_file=False,
            )

        assert result["processed"] == 0
        assert result["errors"] == [
            "invalid durable download_uri for source report.pdf; "
            "provide a supported durable URI or enable retain_source_file"
        ]
        assert materialized is False
        outcome = next(
            record
            for record in caplog.records
            if record.message == "source_download_locator_outcome"
        )
        outcome_fields = vars(outcome)
        assert outcome_fields["outcome"] == "unsupported"
        assert outcome_fields["locator_kind"] == "https"
        assert outcome_fields["source_filename"] == "report.pdf"
        assert "token=secret" not in caplog.text

    async def test_materialization_failure_does_not_expose_remote_exception(
        self,
        test_config: DlightragConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        signed_url = "https://fetch.example.com/report.pdf?token=secret"

        class BynderSource(AsyncDataSource):
            async def aiter_documents(self, prefix: str | None = None):
                yield SourceDocument(
                    key=signed_url,
                    source_uri="bynder://asset/1",
                    download_uri="https://cdn.example.com/assets/1.pdf",
                )

            async def amaterialize_document(
                self, document: SourceDocument, destination: Path
            ) -> None:
                raise ValueError(f"download failed from {signed_url}")

        service = RAGService(config=test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()

        with caplog.at_level(logging.WARNING):
            result = await service.aingest_source(
                BynderSource(), source_type="bynder", retain_source_file=False
            )

        assert result["errors"] == ["report.pdf: remote materialization failed"]
        assert "token=secret" not in caplog.text
        assert signed_url not in caplog.text

    async def test_source_uri_callback_failure_is_sanitized_before_bounded_map(
        self,
        test_config: DlightragConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        materialized = False
        secret = "resolver_token=secret"

        class BynderSource(AsyncDataSource):
            async def aiter_documents(self, prefix: str | None = None):
                yield SourceDocument(key="asset/report.pdf")

            async def amaterialize_document(
                self, document: SourceDocument, destination: Path
            ) -> None:
                nonlocal materialized
                materialized = True

        def source_uri_for_key(_key: str) -> str:
            raise RuntimeError(f"resolver failed: https://fetch.example.com/?{secret}")

        service = RAGService(config=test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()

        with caplog.at_level(logging.WARNING):
            result = await service.aingest_source(
                BynderSource(),
                source_type="bynder",
                source_uri_for_key=source_uri_for_key,
                retain_source_file=False,
            )

        assert result["errors"] == ["report.pdf: source_uri resolution failed"]
        assert materialized is False
        assert secret not in caplog.text
        assert all(
            secret not in str(value) for record in caplog.records for value in vars(record).values()
        )

    async def test_remote_replace_uses_only_exact_download_locator_matches(
        self, test_config: DlightragConfig
    ) -> None:
        class BynderSource(AsyncDataSource):
            def __init__(self, asset_id: str) -> None:
                self.asset_id = asset_id

            async def aiter_documents(self, prefix: str | None = None):
                yield SourceDocument(
                    key="asset/report.pdf",
                    source_uri=f"bynder://asset/{self.asset_id}",
                    download_uri=(f"https://cdn.example.com/assets/{self.asset_id}/report.pdf"),
                    display_filename="report.pdf",
                )

            async def amaterialize_document(
                self, document: SourceDocument, destination: Path
            ) -> None:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"%PDF-fake")

        service = RAGService(config=test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_files = AsyncMock(
            return_value={"processed": 1, "errors": [], "results": []}
        )
        service._lightrag = MagicMock()
        service._lightrag.adelete_by_doc_id = AsyncMock(
            return_value=SimpleNamespace(status="success")
        )
        service._lightrag.doc_status = MagicMock()
        service._lightrag.doc_status.get_doc_by_file_path = AsyncMock(
            side_effect=AssertionError("remote replace must not use fuzzy file lookup")
        )
        service._lightrag.doc_status.get_docs_by_status = AsyncMock(
            side_effect=AssertionError("remote replace must not scan same-basename docs")
        )
        service._metadata_index = AsyncMock()

        async def find_exact(locator: str) -> list[str]:
            if locator == "https://cdn.example.com/assets/1/report.pdf":
                return ["doc-exact"]
            return []

        service._metadata_index.find_by_download_locator.side_effect = find_exact

        await service.aingest_source(
            BynderSource("1"),
            source_type="bynder",
            replace=True,
            retain_source_file=False,
        )
        await service.aingest_source(
            BynderSource("2"),
            source_type="bynder",
            replace=True,
            retain_source_file=False,
        )

        locator_calls = service._metadata_index.find_by_download_locator.await_args_list
        assert locator_calls[0] == call("https://cdn.example.com/assets/1/report.pdf")
        assert locator_calls[2] == call("https://cdn.example.com/assets/2/report.pdf")
        assert "__remote_sources__" in locator_calls[1].args[0]
        assert "__remote_sources__" in locator_calls[3].args[0]
        service._lightrag.adelete_by_doc_id.assert_awaited_once_with(
            "doc-exact", delete_llm_cache=True
        )
        service._metadata_index.delete.assert_awaited_once_with("doc-exact")
        service._lightrag.doc_status.get_doc_by_file_path.assert_not_awaited()
        service._lightrag.doc_status.get_docs_by_status.assert_not_awaited()

    async def test_aingest_source_accepts_per_document_metadata(
        self, test_config: DlightragConfig
    ) -> None:
        class BynderSource(AsyncDataSource):
            async def aiter_documents(
                self, prefix: str | None = None
            ) -> AsyncIterator[SourceDocument]:
                assert prefix == "approved/"
                yield SourceDocument(
                    key="asset-123/report.pdf",
                    source_uri="bynder://assets/asset-123",
                    download_uri="https://cdn.example.com/assets/asset-123.pdf",
                    display_filename="report.pdf",
                    title="Asset report",
                    metadata={"department": "Legal", "asset_id": "asset-123"},
                )

            async def amaterialize_document(
                self, document: SourceDocument, destination: Path
            ) -> None:
                assert document.key == "asset-123/report.pdf"
                destination.write_bytes(b"%PDF-fake")

        service = RAGService(config=test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        seen_items: list[PreparedIngestFile] = []

        async def _ingest(items: list[PreparedIngestFile], **kwargs: object) -> dict[str, object]:
            seen_items.extend(items)
            assert kwargs["metadata"] == {"source_system": "bynder", "department": "Marketing"}
            return {"processed": len(items), "errors": [], "results": []}

        service._ingestion_engine.aingest_files = AsyncMock(side_effect=_ingest)

        result = await service.aingest_source(
            BynderSource(),
            source_type="bynder",
            prefix="approved/",
            metadata={"source_system": "bynder", "department": "Marketing"},
            metadata_policy="validate",
        )

        assert result["processed"] == 1
        assert seen_items[0].source_uri == "bynder://assets/asset-123"
        assert seen_items[0].download_locator == ("https://cdn.example.com/assets/asset-123.pdf")
        assert seen_items[0].display_filename == "report.pdf"
        assert seen_items[0].title == "Asset report"
        assert seen_items[0].metadata == {"department": "Legal", "asset_id": "asset-123"}
        assert seen_items[0].metadata_policy is None

    async def test_remote_parser_path_uses_display_filename_extension(
        self, test_config: DlightragConfig
    ) -> None:
        class BynderSource(AsyncDataSource):
            async def aiter_documents(self, prefix: str | None = None):
                yield SourceDocument(
                    key="asset-123",
                    source_uri="bynder://assets/asset-123",
                    download_uri="https://cdn.example.com/assets/asset-123",
                    display_filename="report.pdf",
                )

            async def amaterialize_document(
                self, document: SourceDocument, destination: Path
            ) -> None:
                destination.write_bytes(b"%PDF-fake")

        service = RAGService(config=test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        seen_items: list[PreparedIngestFile] = []

        async def _ingest(items: list[PreparedIngestFile], **_: object) -> dict[str, object]:
            seen_items.extend(items)
            return {"processed": 1, "errors": [], "results": [{"doc_id": "doc-new"}]}

        service._ingestion_engine.aingest_files = AsyncMock(side_effect=_ingest)

        result = await service.aingest_source(
            BynderSource(), source_type="bynder", retain_source_file=False
        )

        assert result["processed"] == 1
        assert seen_items[0].parser_path.suffix == ".pdf"
        assert seen_items[0].display_filename == "report.pdf"

    async def test_aingest_local_manifest_preserves_workspace_relative_path(
        self, test_config: DlightragConfig
    ) -> None:
        input_root = test_config.input_dir_path / test_config.workspace
        source = input_root / "docs" / "report.pdf"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"%PDF-fake")
        service = RAGService(config=test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        seen_items: list[PreparedIngestFile] = []

        async def _ingest(items: list[PreparedIngestFile], **_: object) -> dict[str, object]:
            seen_items.extend(items)
            return {"processed": len(items), "errors": [], "results": []}

        service._ingestion_engine.aingest_files = AsyncMock(side_effect=_ingest)

        result = await service.aingest(
            source_type="local",
            documents=[{"path": str(source), "metadata": {"asset_id": "local-a"}}],
        )

        assert result["processed"] == 1
        assert seen_items[0].parser_path == source
        assert seen_items[0].source_uri == f"local://{test_config.workspace}/docs/report.pdf"
        assert seen_items[0].download_locator == str(source)
        assert seen_items[0].metadata == {"asset_id": "local-a"}

    async def test_aingest_source_accepts_sync_close(self, test_config: DlightragConfig) -> None:
        class SyncCloseSource(AsyncDataSource):
            def __init__(self) -> None:
                self.closed = False

            async def aiter_documents(
                self, prefix: str | None = None
            ) -> AsyncIterator[SourceDocument]:
                yield SourceDocument(key="asset-123/report.pdf")

            async def amaterialize_document(
                self, document: SourceDocument, destination: Path
            ) -> None:
                destination.write_bytes(b"%PDF-fake")

            def aclose(self) -> None:
                self.closed = True

        service = RAGService(config=test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_files = AsyncMock(
            return_value={"processed": 1, "errors": [], "results": [{"doc_id": "d1"}]}
        )
        source = SyncCloseSource()

        result = await service.aingest_source(
            source,
            source_type="bynder",
            source_uri_for_key=lambda key: f"bynder://assets/{key}",
            download_uri_for_key=lambda _key: "https://cdn.example.com/assets/report.pdf",
        )

        assert result["processed"] == 1
        assert source.closed is True

    async def test_aingest_url_uses_url_data_source(self, test_config: DlightragConfig) -> None:
        """REST/MCP URL jobs enter the same remote ingest pipeline."""
        test_config.url_ingest_private_host_allowlist = ["*.corp.example"]
        service = RAGService(config=test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_files = AsyncMock(
            return_value={
                "processed": 1,
                "errors": [],
                "results": [{"doc_id": "d1", "status": "success"}],
            }
        )

        mock_source = MagicMock()

        async def _aiter_documents(prefix: str | None = None):
            assert prefix is None
            yield SourceDocument(
                key="getting-started.html",
                source_uri="https://api.bynder.com/docs/getting-started",
            )

        mock_source.aiter_documents = _aiter_documents
        mock_source.amaterialize_document = AsyncMock(
            side_effect=lambda _document, destination: destination.write_bytes(b"<html></html>")
        )
        mock_source.source_uri_for_key = lambda key: "https://api.bynder.com/docs/getting-started"
        mock_source.download_uri_for_key = lambda key: "https://api.bynder.com/docs/getting-started"
        mock_source.aclose = AsyncMock()

        with patch("dlightrag.sourcing.url.URLDataSource", return_value=mock_source) as cls:
            result = await service.aingest(
                source_type="url",
                url="https://api.bynder.com/docs/getting-started",
                filename="getting-started.html",
            )

        cls.assert_called_once_with(
            urls=["https://api.bynder.com/docs/getting-started"],
            filename="getting-started.html",
            max_download_bytes=test_config.url_ingest_max_bytes,
            allow_private_hosts=["*.corp.example"],
        )
        assert result["status"] == "success"
        await_args = service._ingestion_engine.aingest_files.await_args
        assert await_args is not None
        call_items = await_args.args[0]
        assert call_items[0].source_uri == "https://api.bynder.com/docs/getting-started"
        assert call_items[0].download_locator == ("https://api.bynder.com/docs/getting-started")
        assert call_items[0].display_filename == "getting-started.html"
        mock_source.aclose.assert_awaited_once()

    @pytest.mark.parametrize(
        ("request_fields", "constructor_fields"),
        [
            (
                {
                    "url": "https://fetch.example.com/report.pdf?token=secret",
                    "download_uri": "https://cdn.example.com/report.pdf",
                },
                {"download_uri": "https://cdn.example.com/report.pdf"},
            ),
            (
                {
                    "urls": [
                        "https://fetch.example.com/a.pdf?token=one",
                        "https://fetch.example.com/b.pdf?token=two",
                    ],
                    "download_uris": [
                        "https://cdn.example.com/a.pdf",
                        "https://cdn.example.com/b.pdf",
                    ],
                },
                {
                    "download_uris": [
                        "https://cdn.example.com/a.pdf",
                        "https://cdn.example.com/b.pdf",
                    ]
                },
            ),
        ],
    )
    async def test_aingest_url_forwards_download_shortcuts_and_callbacks(
        self,
        test_config: DlightragConfig,
        request_fields: dict[str, object],
        constructor_fields: dict[str, object],
    ) -> None:
        service = RAGService(config=test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service.aingest_source = AsyncMock(  # type: ignore[method-assign]
            return_value={"processed": 1, "errors": [], "results": []}
        )
        mock_source = MagicMock()
        mock_source.source_uri_for_key = MagicMock()
        mock_source.download_uri_for_key = MagicMock()

        with patch("dlightrag.sourcing.url.URLDataSource", return_value=mock_source) as cls:
            await service.aingest(source_type="url", **request_fields)

        constructor_call = cls.call_args
        assert constructor_call is not None
        for field, value in constructor_fields.items():
            assert constructor_call.kwargs[field] == value
        delegated_call = service.aingest_source.await_args
        assert delegated_call is not None
        delegated = delegated_call.kwargs
        assert delegated["source_uri_for_key"] is mock_source.source_uri_for_key
        assert delegated["download_uri_for_key"] is mock_source.download_uri_for_key

    async def test_aingest_unified_blob_batch_failure_leaves_no_temp_dirs(
        self, test_config: DlightragConfig
    ) -> None:
        """Remote batch failures do not create obsolete temp directories."""
        service = RAGService(config=test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_files = AsyncMock(
            side_effect=RuntimeError("render failed")
        )

        mock_source = AsyncMock()
        mock_source.amaterialize_document = AsyncMock(
            side_effect=lambda _doc_id, destination: destination.write_bytes(b"%PDF-fake")
        )

        with pytest.raises(RuntimeError, match="render failed"):
            await service.aingest(
                source_type="azure_blob",
                container_name="c",
                blob_path="f.pdf",
                source=mock_source,
            )

        # Verify no temp dirs remain
        import os

        temp_base = test_config.temp_dir
        if temp_base.exists():
            assert len(os.listdir(temp_base)) == 0
        assert not (test_config.working_dir_path / "sources").exists()

    async def test_aingest_unified_delegates_to_engine(
        self, test_config: DlightragConfig, tmp_path: Path
    ) -> None:
        """Local ingestion stages parser sources before delegating to the unified engine."""
        fake_pdf = tmp_path / "f.pdf"
        fake_pdf.write_bytes(b"%PDF-fake")

        service = RAGService(config=test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_file = AsyncMock(
            return_value={"doc_id": "d1", "page_count": 3, "file_path": str(fake_pdf)}
        )

        result = await service.aingest(source_type="local", path=str(fake_pdf))
        service._ingestion_engine.aingest_file.assert_awaited_once()
        staged = test_config.input_dir_path / test_config.workspace / "f.pdf"
        assert service._ingestion_engine.aingest_file.call_args.args[0] == staged
        assert service._ingestion_engine.aingest_file.call_args.kwargs["source_uri"] == (
            f"local://{test_config.workspace}/f.pdf"
        )
        assert service._ingestion_engine.aingest_file.call_args.kwargs["download_locator"] == str(
            staged
        )
        assert staged.read_bytes() == b"%PDF-fake"
        assert result["doc_id"] == "d1"
        assert result["page_count"] == 3

    async def test_aingest_local_directory_uses_batch_pipeline(
        self, test_config: DlightragConfig, tmp_path: Path
    ) -> None:
        """Local directory ingestion batches contained files into LightRAG's staged pipeline."""
        docs_dir = tmp_path / "docs"
        nested_dir = docs_dir / "nested"
        upload_tmp_dir = docs_dir / "__uploads__" / "batch"
        nested_dir.mkdir(parents=True)
        upload_tmp_dir.mkdir(parents=True)
        pdf = docs_dir / "b.pdf"
        docx = docs_dir / "a.docx"
        pptx = nested_dir / "c.pptx"
        for path in (pdf, docx, pptx):
            path.write_bytes(b"fake")
        (upload_tmp_dir / "stale.pdf").write_bytes(b"stale")

        service = RAGService(config=test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_file = AsyncMock()
        service._ingestion_engine.aingest_files = AsyncMock(
            side_effect=lambda paths, **kwargs: {
                "processed": len(paths),
                "errors": [],
                "results": [
                    {"doc_id": item.parser_path.name, "file_path": str(item.parser_path)}
                    for item in paths
                ],
            }
        )

        result = await service.aingest(source_type="local", path=str(docs_dir))

        assert result["processed"] == 3
        assert [item["doc_id"] for item in result["results"]] == ["a.docx", "b.pdf", "c.pptx"]
        staged_root = test_config.input_dir_path / test_config.workspace
        service._ingestion_engine.aingest_file.assert_not_awaited()
        service._ingestion_engine.aingest_files.assert_awaited_once()
        await_args = service._ingestion_engine.aingest_files.await_args
        assert await_args is not None
        items = list(await_args.args[0])
        assert [item.parser_path for item in items] == [
            staged_root / "a.docx",
            staged_root / "b.pdf",
            staged_root / "nested" / "c.pptx",
        ]
        assert [item.source_uri for item in items] == [
            f"local://{test_config.workspace}/a.docx",
            f"local://{test_config.workspace}/b.pdf",
            f"local://{test_config.workspace}/nested/c.pptx",
        ]
        assert [item.download_locator for item in items] == [
            str(staged_root / "a.docx"),
            str(staged_root / "b.pdf"),
            str(staged_root / "nested" / "c.pptx"),
        ]
        assert (staged_root / "a.docx").read_bytes() == b"fake"
        assert (staged_root / "b.pdf").read_bytes() == b"fake"
        assert (staged_root / "nested" / "c.pptx").read_bytes() == b"fake"

    async def test_aingest_local_directory_offloads_scan_and_staging(
        self,
        test_config: DlightragConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import dlightrag.core.service as service_module

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "a.pdf").write_bytes(b"a")
        (docs_dir / "b.pdf").write_bytes(b"b")
        calls = []

        async def fake_to_thread(func, *args, **kwargs):
            calls.append(func)
            return func(*args, **kwargs)

        monkeypatch.setattr(service_module.asyncio, "to_thread", fake_to_thread)
        service = RAGService(config=test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_file = AsyncMock()
        service._ingestion_engine.aingest_files = AsyncMock(
            return_value={"processed": 2, "errors": [], "results": []}
        )

        await service.aingest(source_type="local", path=str(docs_dir))

        assert service_module.iter_ingestable_files in calls
        assert calls.count(service_module.stage_input_file) == 2

    async def test_aingest_explicit_upload_batch_directory_is_ingestable(
        self, test_config: DlightragConfig, tmp_path: Path
    ) -> None:
        """Web upload batches live under __uploads__ and must still be ingestable."""
        upload_dir = tmp_path / "docs" / "__uploads__" / "batch"
        upload_dir.mkdir(parents=True)
        pdf = upload_dir / "uploaded.pdf"
        pdf.write_bytes(b"%PDF-fake")

        service = RAGService(config=test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_file = AsyncMock()
        service._ingestion_engine.aingest_files = AsyncMock(
            return_value={"processed": 1, "errors": [], "results": []}
        )

        result = await service.aingest(source_type="local", path=str(upload_dir))

        assert result["processed"] == 1
        staged_root = test_config.input_dir_path / test_config.workspace
        service._ingestion_engine.aingest_files.assert_awaited_once()
        await_args = service._ingestion_engine.aingest_files.await_args
        assert await_args is not None
        item = await_args.args[0][0]
        assert item.parser_path == staged_root / "uploaded.pdf"
        assert item.source_uri == f"local://{test_config.workspace}/uploaded.pdf"
        assert item.download_locator == str(staged_root / "uploaded.pdf")
        assert (staged_root / "uploaded.pdf").read_bytes() == b"%PDF-fake"

    async def test_aingest_replace_purges_existing_doc_before_ingest(
        self, test_config: DlightragConfig, tmp_path: Path
    ) -> None:
        """replace=True removes the previous doc through the cascade before ingesting."""
        fake_pdf = tmp_path / "f.pdf"
        fake_pdf.write_bytes(b"%PDF-fake")
        events: list[str] = []

        async def delete_doc(*args: object, **kwargs: object) -> None:
            events.append("delete")

        async def ingest_file(*args: object, **kwargs: object) -> dict[str, object]:
            events.append("ingest")
            return {"doc_id": "new-doc", "page_count": 1, "file_path": str(fake_pdf)}

        service = RAGService(config=test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_file = AsyncMock(side_effect=ingest_file)
        service._lightrag = MagicMock()
        service._lightrag.adelete_by_doc_id = AsyncMock(side_effect=delete_doc)
        service._lightrag.doc_status = MagicMock()
        # Real LightRAG get_doc_by_file_path strips 'id' — only returns metadata
        service._lightrag.doc_status.get_doc_by_file_path = AsyncMock(
            return_value={"file_path": str(fake_pdf)}
        )
        service._lightrag.doc_status.get_docs_by_status = AsyncMock(
            return_value={"old-doc": SimpleNamespace(file_path=str(fake_pdf))}
        )
        service._metadata_index = MagicMock()
        service._metadata_index.get = AsyncMock(return_value={"page_count": 0})
        service._metadata_index.delete = AsyncMock()

        result = await service.aingest(source_type="local", path=str(fake_pdf), replace=True)

        assert events == ["delete", "ingest"]
        assert result["doc_id"] == "new-doc"
        service._lightrag.adelete_by_doc_id.assert_awaited_once_with(
            "old-doc", delete_llm_cache=True
        )
        service._metadata_index.delete.assert_awaited_once_with("old-doc")

    async def test_aretrieve_unified_delegates(self, test_config: DlightragConfig) -> None:
        """aretrieve delegates directly to the retrieval orchestrator."""
        from dlightrag.core.retrieval.protocols import RetrievalResult

        expected = RetrievalResult(
            answer=None,
            contexts={"chunks": []},
        )

        service = RAGService(config=test_config)
        service._initialized = True
        service._retrieval_orchestrator = MagicMock()
        service._retrieval_orchestrator.aretrieve = AsyncMock(return_value=expected)

        result = await service.aretrieve("test query")
        service._retrieval_orchestrator.aretrieve.assert_awaited_once()
        assert isinstance(result, RetrievalResult)
        assert result.answer is None
        assert result.contexts == {"chunks": []}

    async def test_aretrieve_normalizes_custom_metadata_filters(
        self, test_config: DlightragConfig
    ) -> None:
        from dlightrag.core.retrieval.metadata_fields import MetadataFieldRegistry
        from dlightrag.core.retrieval.models import MetadataFilter
        from dlightrag.core.retrieval.protocols import RetrievalResult

        service = RAGService(config=test_config)
        service._initialized = True
        service._backend = MagicMock()
        service._lightrag = None
        service._metadata_registry = MetadataFieldRegistry.from_config(
            {"department": {"type": "string", "filter_ops": ["exact"]}}
        )
        service._retrieval_orchestrator = MagicMock()
        service._retrieval_orchestrator.aretrieve = AsyncMock(
            return_value=RetrievalResult(contexts={"chunks": []})
        )

        await service.aretrieve(
            "test query",
            filters=MetadataFilter(custom={"department": " Finance "}),
        )

        await_args = service._retrieval_orchestrator.aretrieve.await_args
        assert await_args is not None
        call_kwargs = await_args.kwargs
        assert call_kwargs["metadata_filter"].custom == {"department": "finance"}

    async def test_metadata_search_normalizes_custom_metadata_filters(
        self, test_config: DlightragConfig
    ) -> None:
        from dlightrag.core.retrieval.metadata_fields import MetadataFieldRegistry
        from dlightrag.core.retrieval.models import MetadataFilter

        service = RAGService(config=test_config)
        service._metadata_registry = MetadataFieldRegistry.from_config(
            {"department": {"type": "string", "filter_ops": ["exact"]}}
        )
        service._metadata_index = AsyncMock()
        service._metadata_index.query = AsyncMock(return_value=["doc-1"])

        result = await service.asearch_metadata(MetadataFilter(custom={"department": " Finance "}))

        assert result == ["doc-1"]
        sent_filter = service._metadata_index.query.await_args.args[0]
        assert sent_filter.custom == {"department": "finance"}

    async def test_metadata_enrichment_uses_full_doc_id_without_path_fallback(
        self, test_config: DlightragConfig
    ) -> None:
        from dlightrag.core.retrieval.protocols import RetrievalResult

        service = RAGService(config=test_config)
        service._metadata_index = AsyncMock()
        service._metadata_index.find_by_file_path = AsyncMock()
        service._metadata_index.find_by_filename = AsyncMock()
        service._metadata_index.get = AsyncMock()
        service._metadata_index.get_many = AsyncMock(
            return_value={
                "doc-right": {
                    "department": "finance",
                    "ingest_strategy": "lightrag_sidecar_unified",
                    "parse_engine": "mineru",
                    "filename": "report.pdf",
                    "metadata_json": {"large": True},
                    "process_options": {"ocr": True},
                }
            }
        )
        result = RetrievalResult(
            contexts={
                "chunks": [
                    {
                        "chunk_id": "chunk-1",
                        "full_doc_id": "doc-right",
                        "file_path": "/inputs/default/finance/report.pdf",
                    },
                    {
                        "chunk_id": "chunk-2",
                        "file_path": "/inputs/default/finance/report.pdf",
                    },
                ]
            }
        )

        await service._enrich_chunks_with_metadata(result)

        assert result.contexts["chunks"][0]["metadata"] == {
            "department": "finance",
            "source_file_name": "report.pdf",
        }
        assert "metadata" not in result.contexts["chunks"][1]
        service._metadata_index.get_many.assert_awaited_once_with(["doc-right"])
        service._metadata_index.get.assert_not_awaited()
        service._metadata_index.find_by_file_path.assert_not_awaited()
        service._metadata_index.find_by_filename.assert_not_awaited()

    async def test_metadata_enrichment_surfaces_distinct_source_contract(
        self, test_config: DlightragConfig
    ) -> None:
        from dlightrag.core.retrieval.protocols import RetrievalResult

        service = RAGService(config=test_config)
        service._metadata_index = AsyncMock()
        service._metadata_index.get_many = AsyncMock(
            return_value={
                "doc-remote": {
                    "filename": "report.pdf",
                    "file_path": "/deleted/__remote_ingest__/report.pdf",
                    "source_uri": "bynder://asset/1",
                    "download_locator": "https://cdn.example.com/assets/1.pdf",
                }
            }
        )
        result = RetrievalResult(
            contexts={
                "chunks": [
                    {
                        "chunk_id": "chunk-1",
                        "full_doc_id": "doc-remote",
                        "file_path": "report__a1b2c3d4e5f6.pdf",
                    }
                ]
            }
        )

        await service._enrich_chunks_with_metadata(result)

        meta = result.contexts["chunks"][0]["metadata"]
        assert meta == {
            "source_uri": "bynder://asset/1",
            "source_download_locator": "https://cdn.example.com/assets/1.pdf",
            "source_file_name": "report.pdf",
        }
        assert "source_file_path" not in meta
        assert "download_locator" not in meta

    async def test_get_metadata_hides_internal_paths_and_locator(
        self, test_config: DlightragConfig
    ) -> None:
        service = RAGService(config=test_config)
        service._metadata_index = AsyncMock()
        service._metadata_index.get.return_value = {
            "workspace": "finance",
            "doc_id": "doc-1",
            "filename": "report.pdf",
            "file_path": "/srv/dlightrag/inputs/finance/report.pdf",
            "source_uri": "bynder://asset/1",
            "download_locator": "https://cdn.example.com/assets/1.pdf",
        }

        result = await service.aget_metadata("doc-1")

        assert result == {
            "filename": "report.pdf",
            "source_uri": "bynder://asset/1",
        }

    async def test_retry_failed_doc_uses_metadata_locator_not_deleted_parser_path(
        self, test_config: DlightragConfig
    ) -> None:
        service = RAGService(config=test_config)
        deleted_parser_path = "/srv/dlightrag/inputs/finance/__remote_ingest__/url/batch/report.pdf"
        events: list[str] = []
        service.alist_failed_docs = AsyncMock(
            return_value=[
                {
                    "doc_id": "doc-failed",
                    "file_path": deleted_parser_path,
                    "error": "parser failed",
                }
            ]
        )
        service._metadata_index = AsyncMock()

        async def get_metadata(doc_id: str) -> dict[str, str]:
            assert doc_id == "doc-failed"
            events.append("metadata")
            return {
                "filename": "/private/reports/report.pdf",
                "source_uri": "bynder://asset/1",
                "download_locator": "https://cdn.example.com/assets/1.pdf",
            }

        service._metadata_index.get = AsyncMock(side_effect=get_metadata)
        service._lightrag = MagicMock()

        async def delete_failed(doc_id: str, *, delete_llm_cache: bool) -> SimpleNamespace:
            assert doc_id == "doc-failed"
            assert delete_llm_cache is True
            assert events == ["metadata", "ingest"]
            events.append("delete")
            return SimpleNamespace(status="success")

        service._lightrag.adelete_by_doc_id = AsyncMock(side_effect=delete_failed)

        async def retry_locator(
            source_uri: str, download_locator: str, filename: str
        ) -> dict[str, str]:
            assert events == ["metadata"]
            assert source_uri == "bynder://asset/1"
            assert download_locator == "https://cdn.example.com/assets/1.pdf"
            assert filename == "report.pdf"
            assert deleted_parser_path not in (source_uri, download_locator)
            events.append("ingest")
            return {"doc_id": "doc-replacement", "status": "success"}

        service._aingest_download_locator = AsyncMock(side_effect=retry_locator)  # type: ignore[attr-defined]
        service.aingest = AsyncMock(side_effect=AssertionError("must not parse doc_status path"))

        result = await service.aretry_failed_docs()

        assert events == ["metadata", "ingest", "delete"]
        assert result["retried"] == 1
        assert result["succeeded"] == 1
        assert result["failed"] == 0

    async def test_retry_enqueue_failure_preserves_original_failed_document(
        self, test_config: DlightragConfig, tmp_path: Path
    ) -> None:
        from lightrag.utils import compute_mdhash_id
        from lightrag.utils_pipeline import normalize_document_file_path

        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.4")
        original_doc_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
        original_status = {
            "status": "failed",
            "chunks_list": [],
            "content_hash": None,
            "content_summary": "parser failed",
            "error_msg": "parser failed",
        }
        original_metadata = {
            "filename": "report.pdf",
            "source_uri": "local://default/report.pdf",
            "download_locator": str(source),
        }
        statuses = {original_doc_id: dict(original_status)}
        full_docs = {original_doc_id: {"sidecar_location": None}}
        metadata_records: dict[str, dict[str, object]] = {original_doc_id: dict(original_metadata)}

        stores = AsyncMock()
        stores.get_doc_status.side_effect = lambda doc_id: statuses.get(doc_id)
        stores.get_full_doc.side_effect = lambda doc_id: full_docs.get(doc_id)

        metadata_index = AsyncMock()
        metadata_index.get.side_effect = lambda doc_id: metadata_records.get(doc_id)

        async def upsert_metadata(doc_id: str, record: dict[str, object]) -> None:
            metadata_records[doc_id] = dict(record)

        async def delete_metadata(doc_id: str) -> None:
            metadata_records.pop(doc_id, None)

        metadata_index.upsert.side_effect = upsert_metadata
        metadata_index.delete.side_effect = delete_metadata

        lightrag = MagicMock()

        async def delete_document(doc_id: str, *, delete_llm_cache: bool) -> SimpleNamespace:
            assert delete_llm_cache is True
            statuses.pop(doc_id, None)
            full_docs.pop(doc_id, None)
            return SimpleNamespace(status="success")

        lightrag.adelete_by_doc_id = AsyncMock(side_effect=delete_document)
        lightrag.apipeline_enqueue_documents = AsyncMock(side_effect=RuntimeError("enqueue failed"))
        lightrag.apipeline_process_enqueue_documents = AsyncMock()

        service = RAGService(config=test_config)
        service._initialized = True
        service._lightrag = lightrag
        service._metadata_index = metadata_index
        document_embedder = AsyncMock()
        document_embedder.image_enabled = False
        document_embedder.dimension = test_config.embedding.dim
        service._ingestion_engine = UnifiedIngestionEngine(
            lightrag=lightrag,
            stores=stores,
            metadata_index=metadata_index,
            document_embedder=document_embedder,
            workspace=test_config.workspace,
            parser_rules=test_config.parser.rules,
            chunk_options=test_config.parser.chunk_options,
        )
        service.alist_failed_docs = AsyncMock(
            return_value=[
                {
                    "doc_id": original_doc_id,
                    "file_path": "report.pdf",
                    "error": "parser failed",
                }
            ]
        )

        result = await service.aretry_failed_docs()

        assert result["failed"] == 1
        assert statuses[original_doc_id] == original_status
        assert metadata_records[original_doc_id] == original_metadata
        lightrag.adelete_by_doc_id.assert_not_awaited()

    async def test_remote_retry_uses_unique_parser_basename_and_original_metadata(
        self, test_config: DlightragConfig
    ) -> None:
        service = RAGService(config=test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        seen_items: list[PreparedIngestFile] = []

        async def ingest(items: list[PreparedIngestFile], **_: object) -> dict[str, object]:
            seen_items.extend(items)
            return {"processed": 1, "errors": [], "results": [{"doc_id": "doc-new"}]}

        service._ingestion_engine.aingest_files = AsyncMock(side_effect=ingest)
        source = MagicMock()

        async def iter_documents(prefix: str | None = None):
            assert prefix is None
            yield SourceDocument(
                key="report.pdf",
                source_uri="bynder://asset/1",
                download_uri="https://cdn.example.com/assets/1",
                display_filename="report.pdf",
            )

        source.aiter_documents = iter_documents
        source.source_uri_for_key = lambda _key: "bynder://asset/1"
        source.download_uri_for_key = lambda _key: "https://cdn.example.com/assets/1"
        source.amaterialize_document = AsyncMock(
            side_effect=lambda _document, destination: destination.write_bytes(b"%PDF-1.4")
        )
        source.aclose = AsyncMock()

        with patch("dlightrag.sourcing.url.URLDataSource", return_value=source):
            result = await service._aingest_download_locator(  # type: ignore[attr-defined]
                "bynder://asset/1",
                "https://cdn.example.com/assets/1",
                "report.pdf",
            )

        assert result["doc_id"] == "doc-new"
        assert len(seen_items) == 1
        item = seen_items[0]
        assert "__retry_" in item.parser_path.stem
        assert item.parser_path.suffix == ".pdf"
        assert item.display_filename == "report.pdf"
        assert item.source_uri == "bynder://asset/1"
        assert item.download_locator == "https://cdn.example.com/assets/1"

    @pytest.mark.parametrize(
        "metadata",
        [
            None,
            {"download_locator": "https://cdn.example.com/assets/1.pdf"},
            {"source_uri": "bynder://asset/1"},
            {
                "source_uri": "bynder://asset/1",
                "download_locator": "https://cdn.example.com/assets/1.pdf",
            },
        ],
    )
    async def test_retry_failed_doc_requires_complete_metadata_contract(
        self,
        test_config: DlightragConfig,
        metadata: dict[str, str] | None,
    ) -> None:
        service = RAGService(config=test_config)
        service.alist_failed_docs = AsyncMock(
            return_value=[
                {
                    "doc_id": "doc-failed",
                    "file_path": "https://legacy.example.com/report.pdf",
                    "error": "parser failed",
                }
            ]
        )
        service._metadata_index = AsyncMock()
        service._metadata_index.get.return_value = metadata
        service._lightrag = MagicMock()
        service._lightrag.adelete_by_doc_id = AsyncMock()
        service._aingest_download_locator = AsyncMock()  # type: ignore[attr-defined]
        service.aingest = AsyncMock()

        result = await service.aretry_failed_docs()

        assert result["failed"] == 1
        service._lightrag.adelete_by_doc_id.assert_not_awaited()
        service._aingest_download_locator.assert_not_awaited()  # type: ignore[attr-defined]
        service.aingest.assert_not_awaited()

    async def test_retry_failed_doc_rejects_non_raising_ingest_failure(
        self, test_config: DlightragConfig
    ) -> None:
        service = RAGService(config=test_config)
        service.alist_failed_docs = AsyncMock(
            return_value=[
                {
                    "doc_id": "doc-failed",
                    "file_path": "/deleted/__remote_ingest__/report.pdf",
                    "error": "parser failed",
                }
            ]
        )
        service._metadata_index = AsyncMock()
        service._metadata_index.get.return_value = {
            "filename": "report.pdf",
            "source_uri": "bynder://asset/1",
            "download_locator": "https://cdn.example.com/assets/1.pdf",
        }
        service._lightrag = MagicMock()
        service._lightrag.adelete_by_doc_id = AsyncMock()
        service._aingest_download_locator = AsyncMock(  # type: ignore[attr-defined]
            return_value={
                "processed": 0,
                "errors": ["report.pdf: remote materialization failed"],
                "results": [],
            }
        )

        result = await service.aretry_failed_docs()

        assert result["retried"] == 1
        assert result["succeeded"] == 0
        assert result["failed"] == 1
        assert result["failed_docs"] == [
            {"doc_id": "doc-failed", "reason": "retry ingestion failed"}
        ]
        service._lightrag.adelete_by_doc_id.assert_not_awaited()
        service._metadata_index.delete.assert_not_awaited()

    async def test_retry_failed_doc_hides_metadata_load_exception(
        self, test_config: DlightragConfig, caplog: pytest.LogCaptureFixture
    ) -> None:
        service = RAGService(config=test_config)
        service.alist_failed_docs = AsyncMock(
            return_value=[
                {
                    "doc_id": "doc-failed",
                    "file_path": "/deleted/__remote_ingest__/report.pdf",
                    "error": "parser failed",
                }
            ]
        )
        service._metadata_index = AsyncMock()
        service._metadata_index.get.side_effect = RuntimeError(
            "database failed for https://private?token=secret"
        )
        service._lightrag = MagicMock()
        service._lightrag.adelete_by_doc_id = AsyncMock()

        result = await service.aretry_failed_docs()

        assert result["failed_docs"] == [
            {"doc_id": "doc-failed", "reason": "source metadata unavailable"}
        ]
        service._lightrag.adelete_by_doc_id.assert_not_awaited()
        assert "token=secret" not in caplog.text

    async def test_retry_failed_doc_preserves_original_when_ingest_raises(
        self, test_config: DlightragConfig, caplog: pytest.LogCaptureFixture
    ) -> None:
        service = RAGService(config=test_config)
        service.alist_failed_docs = AsyncMock(
            return_value=[
                {
                    "doc_id": "doc-failed",
                    "file_path": "/deleted/__remote_ingest__/report.pdf",
                    "error": "parser failed",
                }
            ]
        )
        service._metadata_index = AsyncMock()
        service._metadata_index.get.return_value = {
            "filename": "report.pdf",
            "source_uri": "bynder://asset/1",
            "download_locator": "https://cdn.example.com/assets/1.pdf",
        }
        service._lightrag = MagicMock()
        service._lightrag.adelete_by_doc_id = AsyncMock()
        service._aingest_download_locator = AsyncMock(  # type: ignore[attr-defined]
            side_effect=RuntimeError(
                "download failed for https://signed.example.com/report?token=secret"
            )
        )

        result = await service.aretry_failed_docs()

        assert result["failed_docs"] == [
            {
                "doc_id": "doc-failed",
                "file_path": "/deleted/__remote_ingest__/report.pdf",
                "reason": "retry ingestion failed",
            }
        ]
        service._lightrag.adelete_by_doc_id.assert_not_awaited()
        service._metadata_index.delete.assert_not_awaited()
        assert "token=secret" not in caplog.text

    async def test_retry_failed_doc_requires_returned_document_id(
        self, test_config: DlightragConfig
    ) -> None:
        service = RAGService(config=test_config)
        service.alist_failed_docs = AsyncMock(
            return_value=[
                {
                    "doc_id": "doc-failed",
                    "file_path": "/deleted/__remote_ingest__/report.pdf",
                    "error": "parser failed",
                }
            ]
        )
        service._metadata_index = AsyncMock()
        service._metadata_index.get.return_value = {
            "filename": "report.pdf",
            "source_uri": "bynder://asset/1",
            "download_locator": "https://cdn.example.com/assets/1.pdf",
        }
        service._lightrag = MagicMock()
        service._lightrag.adelete_by_doc_id = AsyncMock()
        service._aingest_download_locator = AsyncMock(  # type: ignore[attr-defined]
            return_value={"processed": 1, "errors": [], "results": []}
        )

        result = await service.aretry_failed_docs()

        assert result["succeeded"] == 0
        assert result["failed"] == 1
        service._lightrag.adelete_by_doc_id.assert_not_awaited()
        service._metadata_index.delete.assert_not_awaited()

    async def test_retry_failed_doc_deletes_old_metadata_for_new_batch_doc_id(
        self, test_config: DlightragConfig
    ) -> None:
        service = RAGService(config=test_config)
        service.alist_failed_docs = AsyncMock(
            return_value=[
                {
                    "doc_id": "doc-old",
                    "file_path": "/deleted/__remote_ingest__/report.pdf",
                    "error": "parser failed",
                }
            ]
        )
        service._metadata_index = AsyncMock()
        service._metadata_index.get.return_value = {
            "filename": "report.pdf",
            "source_uri": "bynder://asset/1",
            "download_locator": "s3://documents/assets/1.pdf",
        }
        events: list[str] = []

        async def delete_old_metadata(doc_id: str) -> None:
            assert doc_id == "doc-old"
            events.append("metadata-delete")

        service._metadata_index.delete.side_effect = delete_old_metadata
        service._lightrag = MagicMock()

        async def delete_old_document(*_args: object, **_kwargs: object) -> SimpleNamespace:
            events.append("lightrag-delete")
            return SimpleNamespace(status="success")

        service._lightrag.adelete_by_doc_id = AsyncMock(side_effect=delete_old_document)

        async def retry_locator(*_args: str) -> dict[str, object]:
            events.append("ingest")
            return {
                "processed": 1,
                "errors": [],
                "results": [{"doc_id": "doc-new", "source_kind": "document"}],
            }

        service._aingest_download_locator = AsyncMock(  # type: ignore[attr-defined]
            side_effect=retry_locator
        )

        result = await service.aretry_failed_docs()

        assert result["succeeded"] == 1
        assert events == ["ingest", "lightrag-delete", "metadata-delete"]
        service._lightrag.adelete_by_doc_id.assert_awaited_once_with(
            "doc-old", delete_llm_cache=True
        )
        service._metadata_index.delete.assert_awaited_once_with("doc-old")

    async def test_retry_failed_doc_preserves_metadata_when_old_cleanup_fails(
        self, test_config: DlightragConfig, caplog: pytest.LogCaptureFixture
    ) -> None:
        service = RAGService(config=test_config)
        service.alist_failed_docs = AsyncMock(
            return_value=[
                {
                    "doc_id": "doc-old",
                    "file_path": "/deleted/__remote_ingest__/report.pdf",
                    "error": "parser failed",
                }
            ]
        )
        service._metadata_index = AsyncMock()
        service._metadata_index.get.return_value = {
            "filename": "report.pdf",
            "source_uri": "bynder://asset/1",
            "download_locator": "s3://documents/assets/1.pdf",
        }
        service._lightrag = MagicMock()

        async def delete_document(doc_id: str, *, delete_llm_cache: bool) -> SimpleNamespace:
            assert delete_llm_cache is True
            if doc_id == "doc-old":
                return SimpleNamespace(
                    status="fail",
                    message="cleanup failed for s3://private?token=secret",
                )
            assert doc_id == "doc-new"
            return SimpleNamespace(status="success")

        service._lightrag.adelete_by_doc_id = AsyncMock(side_effect=delete_document)
        service._aingest_download_locator = AsyncMock(  # type: ignore[attr-defined]
            return_value={
                "processed": 1,
                "errors": [],
                "results": [{"doc_id": "doc-new", "source_kind": "document"}],
            }
        )

        result = await service.aretry_failed_docs()

        assert result["succeeded"] == 0
        assert result["failed"] == 1
        assert service._lightrag.adelete_by_doc_id.await_args_list == [
            call("doc-old", delete_llm_cache=True),
            call("doc-new", delete_llm_cache=True),
        ]
        service._metadata_index.delete.assert_awaited_once_with("doc-new")
        assert "token=secret" not in caplog.text

    async def test_retry_keeps_replacement_when_old_metadata_cleanup_fails(
        self, test_config: DlightragConfig, caplog: pytest.LogCaptureFixture
    ) -> None:
        service = RAGService(config=test_config)
        service.alist_failed_docs = AsyncMock(
            return_value=[
                {
                    "doc_id": "doc-old",
                    "file_path": "/deleted/__remote_ingest__/report.pdf",
                    "error": "parser failed",
                }
            ]
        )
        service._metadata_index = AsyncMock()
        service._metadata_index.get.return_value = {
            "filename": "report.pdf",
            "source_uri": "bynder://asset/1",
            "download_locator": "s3://documents/assets/1.pdf",
        }
        service._metadata_index.delete.side_effect = RuntimeError(
            "metadata unavailable token=secret"
        )
        service._lightrag = MagicMock()
        service._lightrag.adelete_by_doc_id = AsyncMock(
            return_value=SimpleNamespace(status="success")
        )
        service._aingest_download_locator = AsyncMock(  # type: ignore[attr-defined]
            return_value={
                "processed": 1,
                "errors": [],
                "results": [{"doc_id": "doc-new", "source_kind": "document"}],
            }
        )

        result = await service.aretry_failed_docs()

        assert result["succeeded"] == 1
        assert result["failed"] == 0
        service._lightrag.adelete_by_doc_id.assert_awaited_once_with(
            "doc-old", delete_llm_cache=True
        )
        service._metadata_index.delete.assert_awaited_once_with("doc-old")
        assert "token=secret" not in caplog.text

    @pytest.mark.parametrize(
        "deletion_result",
        [
            None,
            {},
            {"status": "unknown"},
            SimpleNamespace(status="not_found"),
        ],
    )
    async def test_retry_failed_doc_requires_positive_old_cleanup_status(
        self,
        test_config: DlightragConfig,
        deletion_result: object,
    ) -> None:
        service = RAGService(config=test_config)
        service.alist_failed_docs = AsyncMock(
            return_value=[
                {
                    "doc_id": "doc-old",
                    "file_path": "/deleted/__remote_ingest__/report.pdf",
                    "error": "parser failed",
                }
            ]
        )
        service._metadata_index = AsyncMock()
        service._metadata_index.get.return_value = {
            "filename": "report.pdf",
            "source_uri": "bynder://asset/1",
            "download_locator": "s3://documents/assets/1.pdf",
        }
        service._lightrag = MagicMock()

        async def delete_document(doc_id: str, *, delete_llm_cache: bool) -> object:
            assert delete_llm_cache is True
            if doc_id == "doc-old":
                return deletion_result
            assert doc_id == "doc-new"
            return SimpleNamespace(status="success")

        service._lightrag.adelete_by_doc_id = AsyncMock(side_effect=delete_document)
        service._aingest_download_locator = AsyncMock(  # type: ignore[attr-defined]
            return_value={
                "processed": 1,
                "errors": [],
                "results": [{"doc_id": "doc-new", "source_kind": "document"}],
            }
        )

        result = await service.aretry_failed_docs()

        assert result["succeeded"] == 0
        assert result["failed"] == 1
        service._metadata_index.delete.assert_awaited_once_with("doc-new")

    async def test_retry_failed_doc_keeps_old_metadata_for_same_single_doc_id(
        self, test_config: DlightragConfig
    ) -> None:
        service = RAGService(config=test_config)
        service.alist_failed_docs = AsyncMock(
            return_value=[
                {
                    "doc_id": "doc-same",
                    "file_path": "/inputs/default/report.pdf",
                    "error": "parser failed",
                }
            ]
        )
        service._metadata_index = AsyncMock()
        service._metadata_index.get.return_value = {
            "filename": "report.pdf",
            "source_uri": "local://default/report.pdf",
            "download_locator": "/inputs/default/report.pdf",
        }
        service._lightrag = MagicMock()
        service._lightrag.adelete_by_doc_id = AsyncMock()
        service._validate_retry_source_contract = MagicMock(  # type: ignore[method-assign]
            return_value=("local", {"path": "/inputs/default/report.pdf"})
        )
        service._aingest_download_locator = AsyncMock(  # type: ignore[attr-defined]
            return_value={"doc_id": "doc-same", "source_kind": "document"}
        )

        result = await service.aretry_failed_docs()

        assert result["succeeded"] == 1
        service._lightrag.adelete_by_doc_id.assert_not_awaited()
        service._metadata_index.delete.assert_not_awaited()

    @pytest.mark.parametrize(
        ("download_locator", "source_type", "source_kwargs", "document_field"),
        [
            (
                "https://cdn.example.com/assets/1.pdf",
                "url",
                {},
                ("url", "https://cdn.example.com/assets/1.pdf"),
            ),
            (
                "s3://documents/assets/1.pdf",
                "s3",
                {"bucket": "documents"},
                ("key", "assets/1.pdf"),
            ),
            (
                "azure://documents/assets/1.pdf",
                "azure_blob",
                {"container_name": "documents"},
                ("key", "assets/1.pdf"),
            ),
        ],
    )
    async def test_download_locator_dispatch_preserves_remote_source_identity(
        self,
        test_config: DlightragConfig,
        download_locator: str,
        source_type: str,
        source_kwargs: dict[str, str],
        document_field: tuple[str, str],
    ) -> None:
        from dlightrag.core.client_contracts import IngestDocument

        service = RAGService(config=test_config)
        service._initialized = True
        service.aingest = AsyncMock(return_value={"status": "success"})

        result = await service._aingest_download_locator(  # type: ignore[attr-defined]
            "bynder://asset/1",
            download_locator,
            "report.pdf",
        )

        assert result == {"status": "success"}
        call = service.aingest.await_args
        assert call is not None
        assert call.args == (source_type,)
        assert call.kwargs["replace"] is False
        for key, value in source_kwargs.items():
            assert call.kwargs[key] == value
        assert len(call.kwargs["documents"]) == 1
        document = call.kwargs["documents"][0]
        assert isinstance(document, IngestDocument)
        assert document.source_uri == "bynder://asset/1"
        assert document.filename == "report.pdf"
        assert getattr(document, document_field[0]) == document_field[1]
        if source_type == "url":
            assert document.download_uri == download_locator

    async def test_download_locator_dispatch_preserves_local_source_identity(
        self, test_config: DlightragConfig, tmp_path: Path
    ) -> None:
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.4")
        service = RAGService(config=test_config)
        service._ingestion_engine = AsyncMock()
        service._ingestion_engine.aingest_files.return_value = {
            "processed": 1,
            "errors": [],
            "results": [{"status": "success"}],
        }

        result = await service._aingest_download_locator(  # type: ignore[attr-defined]
            "local://legacy/abcdef/report.pdf",
            str(source),
            "report.pdf",
        )

        assert result == {"status": "success"}
        service._ingestion_engine.aingest_file.assert_not_awaited()
        service._ingestion_engine.aingest_files.assert_awaited_once()
        items = service._ingestion_engine.aingest_files.await_args.args[0]
        assert len(items) == 1
        item = items[0]
        assert item.source_uri == "local://legacy/abcdef/report.pdf"
        assert item.download_locator == str(source)
        assert item.display_filename == "report.pdf"
        assert "__retry_" in item.parser_path.stem
        assert item.parser_path.suffix == ".pdf"
        assert not item.parser_path.exists()

    async def test_download_locator_dispatch_rejects_invalid_remote_locator(
        self, test_config: DlightragConfig
    ) -> None:
        service = RAGService(config=test_config)
        service.aingest = AsyncMock()

        with pytest.raises(ValueError, match="durable download_uri"):
            await service._aingest_download_locator(  # type: ignore[attr-defined]
                "bynder://asset/1",
                "https://cdn.example.com/assets/1.pdf?token=secret",
                "report.pdf",
            )

        service.aingest.assert_not_awaited()

    async def test_close_lightrag_main_cleanup(self, test_config: DlightragConfig) -> None:
        """close() finalizes LightRAG storages."""
        service = RAGService(config=test_config)
        service._initialized = True
        service._lightrag = AsyncMock()

        await service.aclose()

        service._lightrag.finalize_storages.assert_awaited_once()

    async def test_adelete_files_dry_run_reports_matches_without_mutating(
        self, test_config: DlightragConfig
    ) -> None:
        service = RAGService(config=test_config)
        service._initialized = True
        service._lightrag = MagicMock()
        service._lightrag.adelete_by_doc_id = AsyncMock()
        service._lightrag.doc_status = MagicMock()
        service._lightrag.doc_status.get_doc_by_file_path = AsyncMock(return_value=None)
        service._lightrag.doc_status.get_docs_by_status = AsyncMock(
            return_value={"d1": MagicMock(file_path="/tmp/a.pdf")}
        )
        service._metadata_index = MagicMock()
        service._metadata_index.delete = AsyncMock()

        results = await service.adelete_files(filenames=["a.pdf"], dry_run=True)

        assert results == [
            {
                "identifier": "a.pdf",
                "status": "would_delete",
                "dry_run": True,
                "docs_deleted": 0,
                "errors": [],
                "matched_doc_ids": ["d1"],
                "matched_file_paths": ["/tmp/a.pdf"],
                "sources_used": ["doc_status"],
            }
        ]
        service._lightrag.adelete_by_doc_id.assert_not_awaited()
        service._metadata_index.delete.assert_not_awaited()

    async def test_adelete_files_unified_not_found(self, test_config: DlightragConfig) -> None:
        """Deletion returns not_found when doc is not in doc_status or metadata."""
        service = RAGService(config=test_config)
        service._initialized = True
        service._lightrag = MagicMock()
        service._lightrag.doc_status = MagicMock()
        service._lightrag.doc_status.get_doc_by_file_path = AsyncMock(return_value=None)
        service._lightrag.doc_status.get_docs_by_status = AsyncMock(return_value={})

        results = await service.adelete_files(filenames=["nonexistent.pdf"])

        assert results[0]["status"] == "not_found"
