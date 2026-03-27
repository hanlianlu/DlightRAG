# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for UnifiedRepresentEngine orchestrator."""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from dlightrag.core.retrieval.protocols import RetrievalResult
from dlightrag.unifiedrepresent.engine import UnifiedRepresentEngine
from dlightrag.unifiedrepresent.renderer import RenderResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _async_gen_from_list(items: list) -> AsyncIterator:
    """Create an async generator that yields items from a list."""
    for item in items:
        yield item


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config() -> MagicMock:
    config = MagicMock()
    config.page_render_dpi = 250
    # Nested embedding config
    config.embedding.model = "test-model"
    config.embedding.dim = 1024
    config.embedding.base_url = "http://localhost:8000/v1"
    config.embedding.api_key = "test-key"
    config.embedding_func_max_async = 8
    config.kg_entity_types = ["Person", "Organization"]
    # Nested rerank config
    config.rerank.enabled = False
    config.default_mode = "mix"
    config.top_k = 60
    config.chunk_top_k = 10
    # Batch ingestion config
    config.ingestion_batch_pages = 20
    return config


def _make_lightrag() -> MagicMock:
    lightrag = MagicMock()
    lightrag.full_docs = MagicMock()
    lightrag.full_docs.upsert = AsyncMock()
    lightrag.text_chunks = MagicMock()
    lightrag.text_chunks.upsert = AsyncMock()
    lightrag.chunks_vdb = MagicMock()
    lightrag.chunks_vdb.upsert = AsyncMock()
    lightrag.chunks_vdb.embedding_func = MagicMock()
    lightrag.doc_status = MagicMock()
    lightrag.doc_status.upsert = AsyncMock()
    return lightrag


# ---------------------------------------------------------------------------
# TestUnifiedEngineInit
# ---------------------------------------------------------------------------


class TestUpsertWithVisualVectors:
    """Test _upsert_with_visual_vectors embedding swap logic."""

    @patch("dlightrag.unifiedrepresent.engine.VisualRetriever")
    @patch("dlightrag.unifiedrepresent.engine.EntityExtractor")
    @patch("dlightrag.unifiedrepresent.engine.VisualEmbedder")
    @patch("dlightrag.unifiedrepresent.engine.PageRenderer")
    async def test_upsert_swaps_embedding_and_restores(
        self,
        _renderer: MagicMock,
        _embedder: MagicMock,
        _extractor: MagicMock,
        _retriever: MagicMock,
    ) -> None:
        config = _make_config()
        lightrag = _make_lightrag()
        original_func = lightrag.chunks_vdb.embedding_func

        engine = UnifiedRepresentEngine(
            lightrag=lightrag,
            visual_chunks=MagicMock(),
            config=config,
        )

        chunks_data = {
            "chunk-001": {"content": "page one text", "full_doc_id": "doc-1"},
            "chunk-002": {"content": "page two text", "full_doc_id": "doc-1"},
        }
        vectors = [[0.1] * 1024, [0.2] * 1024]

        await engine._upsert_with_visual_vectors(chunks_data, vectors)

        lightrag.chunks_vdb.upsert.assert_awaited_once_with(chunks_data)
        # Original embedding func should be restored
        assert lightrag.chunks_vdb.embedding_func is original_func

    @patch("dlightrag.unifiedrepresent.engine.VisualRetriever")
    @patch("dlightrag.unifiedrepresent.engine.EntityExtractor")
    @patch("dlightrag.unifiedrepresent.engine.VisualEmbedder")
    @patch("dlightrag.unifiedrepresent.engine.PageRenderer")
    async def test_upsert_restores_on_error(
        self,
        _renderer: MagicMock,
        _embedder: MagicMock,
        _extractor: MagicMock,
        _retriever: MagicMock,
    ) -> None:
        config = _make_config()
        lightrag = _make_lightrag()
        original_func = lightrag.chunks_vdb.embedding_func
        lightrag.chunks_vdb.upsert = AsyncMock(side_effect=RuntimeError("boom"))

        engine = UnifiedRepresentEngine(
            lightrag=lightrag,
            visual_chunks=MagicMock(),
            config=config,
        )

        chunks_data = {"chunk-001": {"content": "text", "full_doc_id": "d1"}}
        vectors = [[0.1] * 1024]

        with pytest.raises(RuntimeError, match="boom"):
            await engine._upsert_with_visual_vectors(chunks_data, vectors)

        # Embedding func restored despite error
        assert lightrag.chunks_vdb.embedding_func is original_func


# ---------------------------------------------------------------------------
# TestAingest
# ---------------------------------------------------------------------------


class TestAingest:
    """Test aingest() batch pipeline end-to-end with mocked sub-components."""

    @patch("dlightrag.unifiedrepresent.engine.VisualRetriever")
    @patch("dlightrag.unifiedrepresent.engine.EntityExtractor")
    @patch("dlightrag.unifiedrepresent.engine.VisualEmbedder")
    @patch("dlightrag.unifiedrepresent.engine.PageRenderer")
    async def test_aingest_two_pages(
        self,
        _renderer_cls: MagicMock,
        _embedder_cls: MagicMock,
        _extractor_cls: MagicMock,
        _retriever_cls: MagicMock,
    ) -> None:
        config = _make_config()
        lightrag = _make_lightrag()
        visual_chunks = MagicMock()
        visual_chunks.upsert = AsyncMock()

        engine = UnifiedRepresentEngine(
            lightrag=lightrag,
            visual_chunks=visual_chunks,
            config=config,
        )

        # Mock renderer: render_file_batched yields one batch with 2 pages
        img_0 = Image.new("RGB", (100, 100), "white")
        img_1 = Image.new("RGB", (100, 100), "blue")
        render_result = RenderResult(
            pages=[(0, img_0), (1, img_1)],
            metadata={
                "title": "Test Doc",
                "author": "Tester",
                "creation_date": "2025-01-01",
                "page_count": 2,
                "original_format": "pdf",
            },
        )
        engine.renderer.render_file_batched = MagicMock(
            return_value=_async_gen_from_list([render_result])
        )

        # Mock embedder: returns (2, 1024) vectors
        vectors = [[0.0] * 1024, [0.0] * 1024]
        engine.embedder.embed_pages = AsyncMock(return_value=vectors)

        # Mock extractor: returns page_infos
        page_infos = [
            {
                "chunk_id": "chunk-aaa",
                "page_index": 0,
                "content": "Page one description",
            },
            {
                "chunk_id": "chunk-bbb",
                "page_index": 1,
                "content": "Page two description",
            },
        ]
        engine.extractor.extract_from_pages = AsyncMock(return_value=page_infos)

        # Mock _upsert_with_visual_vectors
        engine._upsert_with_visual_vectors = AsyncMock()

        result = await engine.aingest("/fake/doc.pdf", doc_id="doc-test")

        # Verify return value
        assert result["doc_id"] == "doc-test"
        assert result["page_count"] == 2
        assert result["file_path"] == "/fake/doc.pdf"

        # Verify full_docs.upsert called
        lightrag.full_docs.upsert.assert_awaited_once()
        upsert_arg = lightrag.full_docs.upsert.call_args[0][0]
        assert "doc-test" in upsert_arg

        # Verify text_chunks.upsert called with correct data
        lightrag.text_chunks.upsert.assert_awaited_once()
        tc_arg = lightrag.text_chunks.upsert.call_args[0][0]
        assert "chunk-aaa" in tc_arg
        assert "chunk-bbb" in tc_arg
        assert tc_arg["chunk-aaa"]["source_type"] == "unified_represent"
        assert tc_arg["chunk-bbb"]["chunk_order_index"] == 1

        # Verify visual_chunks.upsert called with image data
        visual_chunks.upsert.assert_awaited_once()
        vc_arg = visual_chunks.upsert.call_args[0][0]
        assert "chunk-aaa" in vc_arg
        assert "image_data" in vc_arg["chunk-aaa"]
        assert vc_arg["chunk-aaa"]["full_doc_id"] == "doc-test"
        assert vc_arg["chunk-aaa"]["file_path"] == "/fake/doc.pdf"
        assert vc_arg["chunk-aaa"]["doc_title"] == "Test Doc"
        assert "content" not in vc_arg["chunk-aaa"]

        # Verify _upsert_with_visual_vectors called
        engine._upsert_with_visual_vectors.assert_awaited_once()


# ---------------------------------------------------------------------------
# TestAingestMultipleBatches
# ---------------------------------------------------------------------------


class TestAingestMultipleBatches:
    """Test aingest() with multiple batches from render_file_batched."""

    @patch("dlightrag.unifiedrepresent.engine.VisualRetriever")
    @patch("dlightrag.unifiedrepresent.engine.EntityExtractor")
    @patch("dlightrag.unifiedrepresent.engine.VisualEmbedder")
    @patch("dlightrag.unifiedrepresent.engine.PageRenderer")
    async def test_aingest_two_batches(
        self,
        _renderer_cls: MagicMock,
        _embedder_cls: MagicMock,
        _extractor_cls: MagicMock,
        _retriever_cls: MagicMock,
    ) -> None:
        config = _make_config()
        lightrag = _make_lightrag()
        visual_chunks = MagicMock()
        visual_chunks.upsert = AsyncMock()

        engine = UnifiedRepresentEngine(
            lightrag=lightrag,
            visual_chunks=visual_chunks,
            config=config,
        )

        # Two batches: batch 1 has pages 0,1; batch 2 has page 2
        batch_1 = RenderResult(
            pages=[(0, Image.new("RGB", (10, 10), "white")), (1, Image.new("RGB", (10, 10)))],
            metadata={"title": "Doc", "page_count": 3, "original_format": "pdf"},
        )
        batch_2 = RenderResult(
            pages=[(2, Image.new("RGB", (10, 10), "blue"))],
            metadata={"title": "Doc", "page_count": 3, "original_format": "pdf"},
        )
        engine.renderer.render_file_batched = MagicMock(
            return_value=_async_gen_from_list([batch_1, batch_2])
        )

        # Embedder: returns correct-sized vectors per batch
        engine.embedder.embed_pages = AsyncMock(
            side_effect=[
                [[0.0] * 1024, [0.0] * 1024],
                [[0.0] * 1024],
            ]
        )

        # Extractor: returns page_infos per batch
        engine.extractor.extract_from_pages = AsyncMock(
            side_effect=[
                [
                    {"chunk_id": "chunk-aaa", "page_index": 0, "content": "Page one"},
                    {"chunk_id": "chunk-bbb", "page_index": 1, "content": "Page two"},
                ],
                [
                    {"chunk_id": "chunk-ccc", "page_index": 2, "content": "Page three"},
                ],
            ]
        )
        engine._upsert_with_visual_vectors = AsyncMock()

        result = await engine.aingest("/fake/doc.pdf", doc_id="doc-test")

        # Total page count accumulates across batches
        assert result["page_count"] == 3

        # text_chunks.upsert called twice (once per batch)
        assert lightrag.text_chunks.upsert.await_count == 2

        # visual_chunks.upsert called twice (once per batch)
        assert visual_chunks.upsert.await_count == 2

        # _upsert_with_visual_vectors called twice (once per batch)
        assert engine._upsert_with_visual_vectors.await_count == 2

        # full_docs written once at the end
        lightrag.full_docs.upsert.assert_awaited_once()
        fd_arg = lightrag.full_docs.upsert.call_args[0][0]
        assert fd_arg["doc-test"]["page_count"] == 3


# ---------------------------------------------------------------------------
# TestAingestEmptyFile
# ---------------------------------------------------------------------------


class TestAingestEmptyFile:
    """Test aingest raises ValueError when renderer yields 0 pages."""

    @patch("dlightrag.unifiedrepresent.engine.VisualRetriever")
    @patch("dlightrag.unifiedrepresent.engine.EntityExtractor")
    @patch("dlightrag.unifiedrepresent.engine.VisualEmbedder")
    @patch("dlightrag.unifiedrepresent.engine.PageRenderer")
    async def test_empty_render_raises(
        self,
        _renderer_cls: MagicMock,
        _embedder_cls: MagicMock,
        _extractor_cls: MagicMock,
        _retriever_cls: MagicMock,
    ) -> None:
        config = _make_config()
        lightrag = _make_lightrag()

        engine = UnifiedRepresentEngine(
            lightrag=lightrag,
            visual_chunks=MagicMock(),
            config=config,
        )

        # Empty render: async generator that yields nothing
        engine.renderer.render_file_batched = MagicMock(return_value=_async_gen_from_list([]))

        with pytest.raises(ValueError, match="No pages rendered"):
            await engine.aingest("/fake/empty.pdf")


# ---------------------------------------------------------------------------
# TestAretrieve
# ---------------------------------------------------------------------------


class TestAretrieve:
    """Test aretrieve delegates to retriever.retrieve."""

    @patch("dlightrag.unifiedrepresent.engine.VisualRetriever")
    @patch("dlightrag.unifiedrepresent.engine.EntityExtractor")
    @patch("dlightrag.unifiedrepresent.engine.VisualEmbedder")
    @patch("dlightrag.unifiedrepresent.engine.PageRenderer")
    async def test_delegates_to_retriever(
        self,
        _renderer_cls: MagicMock,
        _embedder_cls: MagicMock,
        _extractor_cls: MagicMock,
        _retriever_cls: MagicMock,
    ) -> None:
        config = _make_config()
        lightrag = _make_lightrag()

        engine = UnifiedRepresentEngine(
            lightrag=lightrag,
            visual_chunks=MagicMock(),
            config=config,
        )

        expected = {
            "contexts": {"entities": [], "relationships": [], "chunks": []},
        }
        engine.retriever.retrieve = AsyncMock(return_value=expected)

        result = await engine.aretrieve("test query")

        engine.retriever.retrieve.assert_awaited_once_with(
            query="test query",
            mode="mix",
            top_k=60,
            chunk_top_k=10,
            images=None,
        )
        assert isinstance(result, RetrievalResult)
        assert result.answer is None
        assert result.contexts == expected.get("contexts", {})


# ---------------------------------------------------------------------------
# TestAingestDocStatus
# ---------------------------------------------------------------------------


class TestAingestDocStatus:
    """Test aingest writes doc_status for cross-backend deletion support."""

    @patch("dlightrag.unifiedrepresent.engine.VisualRetriever")
    @patch("dlightrag.unifiedrepresent.engine.EntityExtractor")
    @patch("dlightrag.unifiedrepresent.engine.VisualEmbedder")
    @patch("dlightrag.unifiedrepresent.engine.PageRenderer")
    async def test_aingest_writes_doc_status(
        self,
        _renderer_cls: MagicMock,
        _embedder_cls: MagicMock,
        _extractor_cls: MagicMock,
        _retriever_cls: MagicMock,
    ) -> None:
        from lightrag.base import DocStatus

        config = _make_config()
        lightrag = _make_lightrag()
        lightrag.doc_status = MagicMock()
        lightrag.doc_status.upsert = AsyncMock()
        visual_chunks = MagicMock()
        visual_chunks.upsert = AsyncMock()

        engine = UnifiedRepresentEngine(
            lightrag=lightrag,
            visual_chunks=visual_chunks,
            config=config,
        )

        # Mock renderer: yields one batch with 2 pages
        img_0 = Image.new("RGB", (100, 100), "white")
        img_1 = Image.new("RGB", (100, 100), "blue")
        render_result = RenderResult(
            pages=[(0, img_0), (1, img_1)],
            metadata={"title": "Test", "page_count": 2},
        )
        engine.renderer.render_file_batched = MagicMock(
            return_value=_async_gen_from_list([render_result])
        )
        engine.embedder.embed_pages = AsyncMock(return_value=[[0.0] * 1024, [0.0] * 1024])
        engine.extractor.extract_from_pages = AsyncMock(
            return_value=[
                {"chunk_id": "chunk-aaa", "page_index": 0, "content": "Page one"},
                {"chunk_id": "chunk-bbb", "page_index": 1, "content": "Page two"},
            ]
        )
        engine._upsert_with_visual_vectors = AsyncMock()

        await engine.aingest("/fake/doc.pdf", doc_id="doc-test")

        # Verify doc_status.upsert called twice: PROCESSING then PROCESSED
        assert lightrag.doc_status.upsert.await_count == 2

        # First call: PROCESSING checkpoint
        first_arg = lightrag.doc_status.upsert.call_args_list[0][0][0]
        assert first_arg["doc-test"]["status"] == DocStatus.PROCESSING
        assert first_arg["doc-test"]["chunks_list"] == []

        # Second call: PROCESSED with chunk list
        final_arg = lightrag.doc_status.upsert.call_args_list[1][0][0]
        entry = final_arg["doc-test"]
        assert entry["status"] == DocStatus.PROCESSED
        assert entry["chunks_count"] == 2
        assert entry["chunks_list"] == ["chunk-aaa", "chunk-bbb"]
        assert entry["file_path"] == "/fake/doc.pdf"
        assert "created_at" in entry
        assert "updated_at" in entry


# ---------------------------------------------------------------------------
# TestProtocolCompliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    """Test that UnifiedRepresentEngine satisfies RetrievalBackend Protocol."""

    @patch("dlightrag.unifiedrepresent.engine.VisualRetriever")
    @patch("dlightrag.unifiedrepresent.engine.EntityExtractor")
    @patch("dlightrag.unifiedrepresent.engine.VisualEmbedder")
    @patch("dlightrag.unifiedrepresent.engine.PageRenderer")
    async def test_aretrieve_returns_retrieval_result(self, _r, _em, _ex, _ret):
        config = _make_config()
        lightrag = _make_lightrag()
        engine = UnifiedRepresentEngine(
            lightrag=lightrag,
            visual_chunks=MagicMock(),
            config=config,
        )
        engine.retriever.retrieve = AsyncMock(
            return_value={
                "contexts": {"chunks": []},
            }
        )
        result = await engine.aretrieve("test query")
        assert isinstance(result, RetrievalResult)
        assert result.answer is None
        assert result.contexts == {"chunks": []}

    @patch("dlightrag.unifiedrepresent.engine.VisualRetriever")
    @patch("dlightrag.unifiedrepresent.engine.EntityExtractor")
    @patch("dlightrag.unifiedrepresent.engine.VisualEmbedder")
    @patch("dlightrag.unifiedrepresent.engine.PageRenderer")
    async def test_accepts_kwargs(self, _r, _em, _ex, _ret):
        config = _make_config()
        lightrag = _make_lightrag()
        engine = UnifiedRepresentEngine(
            lightrag=lightrag,
            visual_chunks=MagicMock(),
            config=config,
        )
        engine.retriever.retrieve = AsyncMock(return_value={"contexts": {}})
        # Should not raise — **kwargs absorbs multimodal_content
        result = await engine.aretrieve("q", multimodal_content=[{"type": "image"}])
        assert isinstance(result, RetrievalResult)
