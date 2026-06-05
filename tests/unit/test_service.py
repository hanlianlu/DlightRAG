# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for RAGService facade (core/service.py)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dlightrag.config import DlightragConfig
from dlightrag.core.service import RAGService, _parse_postgres_server_settings

# ---------------------------------------------------------------------------
# TestRAGServiceAingest
# ---------------------------------------------------------------------------


def test_parse_postgres_server_settings_decodes_query_string() -> None:
    assert _parse_postgres_server_settings(
        "hnsw.ef_search=384&application_name=dlightrag+api&statement_timeout=60000"
    ) == {
        "hnsw.ef_search": "384",
        "application_name": "dlightrag api",
        "statement_timeout": "60000",
    }


class TestRAGServiceAingest:
    """Test ingestion logic -- replace defaults, azure lifecycle."""

    def _make_initialized_service(self, config: DlightragConfig) -> RAGService:
        service = RAGService(config=config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_file = AsyncMock(return_value={"status": "success"})
        service.retrieval = MagicMock()
        return service

    async def test_aingest_not_initialized_raises(self, test_config: DlightragConfig) -> None:
        service = RAGService(config=test_config)
        with pytest.raises(RuntimeError, match="not initialized"):
            await service.aingest(source_type="local", path="/tmp/f.pdf")

    async def test_aingest_query_role_rejected(self, test_config: DlightragConfig) -> None:
        test_config.runtime_role = "query"
        service = self._make_initialized_service(test_config)
        with pytest.raises(PermissionError, match="runtime_role='query'"):
            await service.aingest(source_type="local", path="/tmp/file.pdf")

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
        mock_source.alist_documents = AsyncMock(return_value=[])
        await service.aingest(source_type="azure_blob", source=mock_source, container_name="c")
        mock_source.alist_documents.assert_awaited_once_with(prefix="")

    async def test_aingest_azure_calls_aclose(self, test_config: DlightragConfig) -> None:
        """source.aclose() is called after successful ingestion."""
        service = self._make_initialized_service(test_config)
        mock_source = AsyncMock()
        mock_source.alist_documents = AsyncMock(return_value=[])
        await service.aingest(source_type="azure_blob", source=mock_source, container_name="c")
        mock_source.aclose.assert_awaited_once()

    async def test_aingest_azure_calls_aclose_on_error(self, test_config: DlightragConfig) -> None:
        """source.aclose() is called even when ingestion raises."""
        service = self._make_initialized_service(test_config)
        service._ingestion_engine.aingest_file = AsyncMock(
            side_effect=RuntimeError("ingestion failed")
        )
        mock_source = AsyncMock()
        mock_source.alist_documents = AsyncMock(return_value=["f.pdf"])
        mock_source.aload_document = AsyncMock(return_value=b"%PDF")
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
        service.ingestion = None
        service.retrieval = None

        # Should not raise
        await service.close()


# ---------------------------------------------------------------------------
# TestRAGServiceRetrieve
# ---------------------------------------------------------------------------


class TestRAGServiceRetrieve:
    """Test aretrieve delegation to RetrievalEngine."""

    def _make_retrieval_service(self, config: DlightragConfig) -> RAGService:
        service = RAGService(config=config)
        service._initialized = True
        service.retrieval = MagicMock()
        service.retrieval.aretrieve = AsyncMock(return_value=MagicMock())
        service.ingestion = MagicMock()
        return service

    async def test_aretrieve_delegates_to_retrieval(self, test_config):
        service = self._make_retrieval_service(test_config)
        await service.aretrieve("test query")
        service.retrieval.aretrieve.assert_awaited_once()

    async def test_aretrieve_passes_multimodal_content(self, test_config):
        service = self._make_retrieval_service(test_config)
        mc = [{"type": "image"}]
        await service.aretrieve("test query", multimodal_content=mc)
        call_kwargs = service.retrieval.aretrieve.call_args.kwargs
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

    async def test_adelete_query_role_rejected(self, test_config):
        test_config.runtime_role = "query"
        service = RAGService(config=test_config)
        service._initialized = True
        with pytest.raises(PermissionError, match="runtime_role='query'"):
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
# TestBuildAddonParams
# ---------------------------------------------------------------------------


class TestBuildAddonParams:
    """Test LightRAG addon_params compatibility."""

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


# ---------------------------------------------------------------------------
# TestRAGServiceLightRAGMainPath
# ---------------------------------------------------------------------------


class TestRAGServiceLightRAGMainPath:
    """Test LightRAG-main path behavior in RAGService."""

    async def test_aingest_azure_blob_single(self, test_config: DlightragConfig) -> None:
        """Downloads a single blob and ingests through the unified engine."""
        service = RAGService(config=test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_file = AsyncMock(
            return_value={"doc_id": "d1", "page_count": 2, "file_path": "/tmp/report.pdf"}
        )

        mock_source = AsyncMock()
        mock_source.aload_document = AsyncMock(return_value=b"%PDF-fake-content")

        result = await service.aingest(
            source_type="azure_blob",
            container_name="test-container",
            blob_path="docs/report.pdf",
            source=mock_source,
        )

        service._ingestion_engine.aingest_file.assert_awaited_once()
        assert result["doc_id"] == "d1"

    async def test_aingest_unified_azure_blob_batch(self, test_config: DlightragConfig) -> None:
        """Batch-ingests blobs by prefix."""
        service = RAGService(config=test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_file = AsyncMock(
            return_value={"doc_id": "d1", "page_count": 1, "file_path": "/tmp/f.pdf"}
        )

        mock_source = AsyncMock()
        mock_source.alist_documents = AsyncMock(return_value=["a.pdf", "b.pdf"])
        mock_source.aload_document = AsyncMock(return_value=b"%PDF-fake")

        result = await service.aingest(
            source_type="azure_blob",
            container_name="c",
            prefix="docs/",
            source=mock_source,
        )

        assert service._ingestion_engine.aingest_file.await_count == 2
        assert result["processed"] == 2

    async def test_aingest_unified_s3(self, test_config: DlightragConfig) -> None:
        """S3 downloads an object and ingests via engine.aingest."""
        service = RAGService(config=test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_file = AsyncMock(
            return_value={"doc_id": "d1", "status": "success"}
        )

        mock_source = AsyncMock()
        mock_source.aload_document = AsyncMock(return_value=b"%PDF-fake")
        mock_source.aclose = AsyncMock()

        with patch("dlightrag.sourcing.aws_s3.S3DataSource", return_value=mock_source):
            result = await service.aingest(
                source_type="s3", bucket="my-bucket", key="docs/report.pdf"
            )

        service._ingestion_engine.aingest_file.assert_awaited_once()
        assert result["status"] == "success"

    async def test_aingest_unified_blob_temp_cleanup(self, test_config: DlightragConfig) -> None:
        """Temp directory is cleaned up even if unified.aingest fails."""
        service = RAGService(config=test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_file = AsyncMock(
            side_effect=RuntimeError("render failed")
        )

        mock_source = AsyncMock()
        mock_source.aload_document = AsyncMock(return_value=b"%PDF-fake")

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

    async def test_aingest_local_directory_delegates_each_file(
        self, test_config: DlightragConfig, tmp_path: Path
    ) -> None:
        """Local directory ingestion sends contained files, not the directory, to parser routing."""
        docs_dir = tmp_path / "docs"
        nested_dir = docs_dir / "nested"
        nested_dir.mkdir(parents=True)
        pdf = docs_dir / "b.pdf"
        docx = docs_dir / "a.docx"
        pptx = nested_dir / "c.pptx"
        for path in (pdf, docx, pptx):
            path.write_bytes(b"fake")

        service = RAGService(config=test_config)
        service._initialized = True
        service._ingestion_engine = MagicMock()
        service._ingestion_engine.aingest_file = AsyncMock(
            side_effect=lambda path, **kwargs: {"doc_id": path.name, "file_path": str(path)}
        )

        result = await service.aingest(source_type="local", path=str(docs_dir))

        assert result["processed"] == 3
        assert [item["doc_id"] for item in result["results"]] == ["a.docx", "b.pdf", "c.pptx"]
        staged_root = test_config.input_dir_path / test_config.workspace
        assert [
            call.args[0] for call in service._ingestion_engine.aingest_file.await_args_list
        ] == [
            staged_root / "a.docx",
            staged_root / "b.pdf",
            staged_root / "c.pptx",
        ]
        assert (staged_root / "a.docx").read_bytes() == b"fake"
        assert (staged_root / "b.pdf").read_bytes() == b"fake"
        assert (staged_root / "c.pptx").read_bytes() == b"fake"

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
        """aretrieve delegates directly to the retrieval backend."""
        from dlightrag.core.retrieval.protocols import RetrievalResult

        expected = RetrievalResult(
            answer=None,
            contexts={"chunks": []},
        )

        service = RAGService(config=test_config)
        service._initialized = True
        service._backend = MagicMock()
        service._backend.aretrieve = AsyncMock(return_value=expected)

        result = await service.aretrieve("test query")
        service._backend.aretrieve.assert_awaited_once()
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

        call_kwargs = service._retrieval_orchestrator.aretrieve.await_args.kwargs
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

        result = await service.asearch_metadata(
            MetadataFilter(custom={"department": " Finance "})
        )

        assert result == ["doc-1"]
        sent_filter = service._metadata_index.query.await_args.args[0]
        assert sent_filter.custom == {"department": "finance"}

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
