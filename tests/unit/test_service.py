# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for RAGService facade (core/service.py)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dlightrag.config import DlightragConfig
from dlightrag.core.ingestion.engine import PreparedIngestFile
from dlightrag.core.service import RAGService, RemoteIngestWindowProgress
from dlightrag.sourcing.base import AsyncDataSource


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

    def _make_initialized_service(self, config: DlightragConfig) -> RAGService:
        service = RAGService(config=config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_file = AsyncMock(return_value={"status": "success"})
        service._ingestion_engine.aingest_files = AsyncMock(
            return_value={"processed": 1, "errors": [], "results": [{"status": "success"}]}
        )
        return service

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
        service = self._make_initialized_service(test_config)
        await service.aingest(source_type="local", path=str(fake_pdf))
        call_kwargs = service._ingestion_engine.aingest_file.call_args
        assert call_kwargs.kwargs["replace"] is True

    async def test_aingest_replace_explicit_overrides_config(
        self, test_config: DlightragConfig, tmp_path: Path
    ) -> None:
        test_config.ingestion_replace_default = True
        fake_pdf = tmp_path / "file.pdf"
        fake_pdf.write_bytes(b"%PDF-fake")
        service = self._make_initialized_service(test_config)
        await service.aingest(source_type="local", path=str(fake_pdf), replace=False)
        call_kwargs = service._ingestion_engine.aingest_file.call_args
        assert call_kwargs.kwargs["replace"] is False

    # -- Azure blob lifecycle --

    async def test_aingest_azure_defaults_prefix_when_neither_set(
        self, test_config: DlightragConfig
    ) -> None:
        """When neither blob_path nor prefix provided, prefix defaults to '' (entire container)."""
        service = self._make_initialized_service(test_config)
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
        service = self._make_initialized_service(test_config)
        mock_source = AsyncMock()

        async def _aiter_documents(prefix: str | None = None):
            for item in ():
                yield item

        mock_source.aiter_documents = _aiter_documents
        await service.aingest(source_type="azure_blob", source=mock_source, container_name="c")
        mock_source.aclose.assert_awaited_once()

    async def test_aingest_azure_calls_aclose_on_error(self, test_config: DlightragConfig) -> None:
        """source.aclose() is called even when ingestion raises."""
        service = self._make_initialized_service(test_config)
        service._ingestion_engine.aingest_files = AsyncMock(
            side_effect=RuntimeError("ingestion failed")
        )
        mock_source = AsyncMock()

        async def _aiter_documents(prefix: str | None = None):
            yield "f.pdf"

        mock_source.aiter_documents = _aiter_documents
        mock_source.amaterialize_document = AsyncMock(
            side_effect=lambda _doc_id, destination: destination.write_bytes(b"%PDF")
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
        await service.close()

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

        await service.close()

        embedding_func.shutdown.assert_awaited_once()
        query_func.shutdown.assert_awaited_once()
        extract_func.shutdown.assert_awaited_once()


# ---------------------------------------------------------------------------
# TestRAGServiceRetrieve
# ---------------------------------------------------------------------------


class TestRAGServiceRetrieve:
    """Test aretrieve delegation to RetrievalEngine."""

    def _make_retrieval_service(self, config: DlightragConfig) -> RAGService:
        from dlightrag.core.retrieval.protocols import RetrievalResult

        service = RAGService(config=config)
        service._initialized = True
        service._retrieval_orchestrator = MagicMock()
        service._retrieval_orchestrator.aretrieve = AsyncMock(
            return_value=RetrievalResult(contexts={"chunks": []})
        )
        return service

    async def test_aretrieve_delegates_to_orchestrator(self, test_config):
        service = self._make_retrieval_service(test_config)
        await service.aretrieve("test query")
        service._retrieval_orchestrator.aretrieve.assert_awaited_once()

    async def test_aretrieve_passes_multimodal_content(self, test_config):
        service = self._make_retrieval_service(test_config)
        mc = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]
        await service.aretrieve("test query", multimodal_content=mc)
        call_kwargs = service._retrieval_orchestrator.aretrieve.call_args.kwargs
        assert call_kwargs["multimodal_content"] == mc

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
        service._lightrag = MagicMock()
        service._lightrag.doc_status = MagicMock()
        service._lightrag.doc_status.get_docs_by_status = AsyncMock(
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
        service._lightrag.doc_status.get_docs_by_status.assert_awaited_once()

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

    def test_default_has_cosine_threshold(self, test_config: DlightragConfig) -> None:
        result = RAGService._build_vector_db_kwargs(test_config)
        assert result == {"cosine_better_than_threshold": 0.3}

    def test_passthrough_merges_kwargs(self, test_config: DlightragConfig) -> None:
        test_config.vector_db_kwargs = {
            "index_type": "HNSW_SQ",
            "sq_type": "SQ8",
            "hnsw_m": 32,
        }
        result = RAGService._build_vector_db_kwargs(test_config)
        assert result["cosine_better_than_threshold"] == 0.3
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
            embedder=MagicMock(),
            rerank_func=MagicMock(),
        )

        assert backend._direct_visual_top_k == 9
        assert backend._max_entity_tokens == 111
        assert backend._max_relation_tokens == 222
        assert backend._max_total_tokens == 333


class TestDirectImageEmbeddingCapability:
    """Test direct image embedding capability resolution."""

    async def test_text_only_embedder_disables_direct_image_embedding_without_probe(self) -> None:
        embedder = MagicMock()
        embedder.supports_images = False
        embedder.probe_image_embedding = AsyncMock()

        enabled = await RAGService._resolve_direct_image_embedding_enabled(
            embedder,
            startup_probe=True,
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
        )

        assert enabled is False
        embedder.probe_image_embedding.assert_awaited_once()


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
        assert item.metadata_path == "azure://test-container/docs/report.pdf"
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
            yield "team-a/report.pdf"
            yield "team-b/report.pdf"

        mock_source.aiter_documents = _aiter_documents
        mock_source.amaterialize_document = AsyncMock(
            side_effect=lambda _doc_id, destination: destination.write_bytes(b"%PDF-fake")
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
        assert [item.metadata_path for item in seen_items] == [
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
                source_type="s3", bucket="my-bucket", key="docs/report.pdf"
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

            async def alist_documents(self, prefix: str | None = None) -> list[str]:
                raise AssertionError("prefix ingest should stream keys")

            async def aiter_documents(self, prefix: str | None = None):
                assert prefix == "docs/"
                yield "docs/a.pdf"
                yield "docs/b.pdf"

            async def amaterialize_document(self, doc_id: str, destination: Path) -> None:
                self.loaded.append(doc_id)
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
                    yield f"docs/{name}.pdf"

            async def amaterialize_document(self, doc_id: str, destination: Path) -> None:
                self.loaded.append(doc_id)
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
                yield "docs/a.pdf"
                yield "docs/b.pdf"

            async def amaterialize_document(self, doc_id: str, destination: Path) -> None:
                if doc_id == "docs/b.pdf":
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
        assert result["errors"] == ["s3://my-bucket/docs/b.pdf: download failed"]
        assert len(progress_events) == 1
        progress = progress_events[0]
        assert progress.total_delta == 2
        assert progress.processed_delta == 1
        assert progress.failed_delta == 1
        assert progress.errors == ("s3://my-bucket/docs/b.pdf: download failed",)

    @pytest.mark.parametrize(
        ("source_type", "source_uri"),
        [
            ("s3", "s3://my-bucket/docs/report.pdf"),
            ("azure_blob", "azure://my-container/docs/report.pdf"),
            ("url", "https://example.com/docs/report.pdf"),
        ],
    )
    async def test_remote_source_retention_keeps_workspace_file_and_metadata_path(
        self,
        test_config: DlightragConfig,
        source_type: str,
        source_uri: str,
    ) -> None:
        class RemoteSource(AsyncDataSource):
            async def aiter_documents(self, prefix: str | None = None):
                yield "docs/report.pdf"

            async def amaterialize_document(self, doc_id: str, destination: Path) -> None:
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
        assert item.metadata_path == str(item.parser_path)
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
                yield "docs/report.pdf"

            async def amaterialize_document(self, doc_id: str, destination: Path) -> None:
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
            retain_source_file=False,
        )

        assert result["processed"] == 1
        item = seen_items[0]
        assert item.metadata_path == "s3://my-bucket/docs/report.pdf"
        assert not item.parser_path.exists()

    async def test_aingest_source_accepts_sdk_async_data_source(
        self, test_config: DlightragConfig
    ) -> None:
        """SDK callers can ingest custom connector output without pretending it is S3."""

        class BynderSource(AsyncDataSource):
            def __init__(self) -> None:
                self.loaded: list[str] = []
                self.closed = False

            async def aiter_documents(self, prefix: str | None = None) -> AsyncIterator[str]:
                assert prefix == "approved/"
                yield "asset-123/report.pdf"

            async def amaterialize_document(self, doc_id: str, destination: Path) -> None:
                self.loaded.append(doc_id)
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
            title="Approved asset",
            metadata={"collection": "marketing"},
        )

        assert result["processed"] == 1
        assert source.loaded == ["asset-123/report.pdf"]
        assert source.closed is True
        assert seen_items[0].metadata_path == "bynder://assets/asset-123/report.pdf"
        assert seen_items[0].display_filename == "report.pdf"
        assert seen_items[0].parser_path.suffix == ".pdf"

    async def test_aingest_url_uses_url_data_source(self, test_config: DlightragConfig) -> None:
        """REST/MCP URL jobs enter the same remote ingest pipeline."""
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
            yield "getting-started.html"

        mock_source.aiter_documents = _aiter_documents
        mock_source.amaterialize_document = AsyncMock(
            side_effect=lambda _doc_id, destination: destination.write_bytes(b"<html></html>")
        )
        mock_source.source_uri_for_key = lambda key: "https://api.bynder.com/docs/getting-started"
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
        )
        assert result["status"] == "success"
        await_args = service._ingestion_engine.aingest_files.await_args
        assert await_args is not None
        call_items = await_args.args[0]
        assert call_items[0].metadata_path == "https://api.bynder.com/docs/getting-started"
        assert call_items[0].display_filename == "getting-started.html"
        mock_source.aclose.assert_awaited_once()

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
                "results": [{"doc_id": path.name, "file_path": str(path)} for path in paths],
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
        assert list(await_args.args[0]) == [
            staged_root / "a.docx",
            staged_root / "b.pdf",
            staged_root / "nested" / "c.pptx",
        ]
        assert (staged_root / "a.docx").read_bytes() == b"fake"
        assert (staged_root / "b.pdf").read_bytes() == b"fake"
        assert (staged_root / "nested" / "c.pptx").read_bytes() == b"fake"

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
        assert list(await_args.args[0]) == [staged_root / "uploaded.pdf"]
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

    async def test_metadata_enrichment_prefers_exact_file_path_over_basename(
        self, test_config: DlightragConfig
    ) -> None:
        from dlightrag.core.retrieval.protocols import RetrievalResult

        service = RAGService(config=test_config)
        service._metadata_index = AsyncMock()
        service._metadata_index.find_by_file_path = AsyncMock(return_value=["doc-right"])
        service._metadata_index.find_by_filename = AsyncMock(return_value=["doc-wrong"])
        service._metadata_index.get = AsyncMock(
            side_effect=lambda doc_id: {
                "doc-right": {"department": "finance", "filename": "report.pdf"},
                "doc-wrong": {"department": "legal", "filename": "report.pdf"},
            }.get(doc_id)
        )
        result = RetrievalResult(
            contexts={
                "chunks": [
                    {
                        "chunk_id": "chunk-1",
                        "file_path": "/inputs/default/finance/report.pdf",
                    }
                ]
            }
        )

        await service._enrich_chunks_with_metadata(result)

        chunk = result.contexts["chunks"][0]
        assert chunk["metadata"] == {"department": "finance"}
        service._metadata_index.find_by_file_path.assert_awaited_once_with(
            "/inputs/default/finance/report.pdf"
        )
        service._metadata_index.find_by_filename.assert_not_awaited()

    async def test_close_lightrag_main_cleanup(self, test_config: DlightragConfig) -> None:
        """close() finalizes LightRAG storages."""
        service = RAGService(config=test_config)
        service._initialized = True
        service._lightrag = AsyncMock()

        await service.close()

        service._lightrag.finalize_storages.assert_awaited_once()

    # -- File deletion --

    async def test_adelete_files_unified(self, test_config: DlightragConfig) -> None:
        """Deletion removes LightRAG data and metadata index entries."""
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
        service._metadata_index.get = AsyncMock(return_value={"page_count": 3})
        service._metadata_index.delete = AsyncMock()

        results = await service.adelete_files(filenames=["a.pdf"])

        assert len(results) == 1
        assert results[0]["status"] == "deleted"
        service._lightrag.adelete_by_doc_id.assert_awaited_once()
        service._metadata_index.delete.assert_awaited_once_with("d1")

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

    async def test_adelete_files_unified_continues_after_layer1_failure(
        self, test_config: DlightragConfig
    ) -> None:
        """Metadata cleanup still runs when LightRAG deletion fails."""
        service = RAGService(config=test_config)
        service._initialized = True
        service._lightrag = MagicMock()
        service._lightrag.adelete_by_doc_id = AsyncMock(side_effect=RuntimeError("LightRAG down"))
        service._lightrag.doc_status = MagicMock()
        service._lightrag.doc_status.get_doc_by_file_path = AsyncMock(return_value=None)
        service._lightrag.doc_status.get_docs_by_status = AsyncMock(
            return_value={"d1": MagicMock(file_path="/tmp/a.pdf")}
        )
        service._metadata_index = MagicMock()
        service._metadata_index.get = AsyncMock(return_value={"page_count": 2})
        service._metadata_index.delete = AsyncMock()

        results = await service.adelete_files(filenames=["a.pdf"])

        assert results[0]["status"] == "deleted_with_errors"
        service._metadata_index.delete.assert_awaited_once_with("d1")
