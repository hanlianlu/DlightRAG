"""Tests for VisualRetriever — output schema and data preservation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from dlightrag.unifiedrepresent.retriever import VisualRetriever


def _make_retriever() -> VisualRetriever:
    lightrag = MagicMock()
    visual_chunks = MagicMock()
    config = MagicMock()
    config.default_mode = "mix"
    config.top_k = 60
    config.chunk_top_k = 10
    return VisualRetriever(
        lightrag=lightrag,
        visual_chunks=visual_chunks,
        config=config,
    )


class TestTextRetrieveOutputSchema:
    """Test that _text_retrieve preserves reference_id, chunk_id, file_path."""

    async def test_chunks_have_required_fields(self):
        retriever = _make_retriever()

        retriever.lightrag.aquery_data = AsyncMock(return_value={
            "data": {
                "entities": [],
                "relationships": [],
                "chunks": [
                    {
                        "chunk_id": "chunk-abc",
                        "reference_id": "1",
                        "content": "Page one text",
                        "file_path": "/data/report.pdf",
                    },
                    {
                        "chunk_id": "chunk-def",
                        "reference_id": "1",
                        "content": "Page two text",
                        "file_path": "/data/report.pdf",
                    },
                    {
                        "chunk_id": "chunk-ghi",
                        "reference_id": "2",
                        "content": "Analysis content",
                        "file_path": "/data/analysis.xlsx",
                    },
                ],
            },
        })

        retriever.visual_chunks.get_by_ids = AsyncMock(return_value=[
            {
                "full_doc_id": "doc-1",
                "file_path": "/data/report.pdf",
                "page_index": 0,
                "image_data": "base64data1",
                "doc_title": "Report",
            },
            {
                "full_doc_id": "doc-1",
                "file_path": "/data/report.pdf",
                "page_index": 1,
                "image_data": "base64data2",
                "doc_title": "Report",
            },
            {
                "full_doc_id": "doc-2",
                "file_path": "/data/analysis.xlsx",
                "page_index": 0,
                "image_data": "base64data3",
                "doc_title": "Analysis",
            },
        ])

        result = await retriever._text_retrieve("test query", top_k=60, chunk_top_k=10)

        chunks = result["contexts"]["chunks"]
        assert len(chunks) == 3

        for chunk in chunks:
            assert "reference_id" in chunk, "Missing reference_id"
            assert "chunk_id" in chunk, "Missing chunk_id"
            assert "file_path" in chunk, "Missing file_path"
            assert "content" in chunk, "Missing content"
            assert "page_idx" in chunk, "Missing page_idx"

        ref_ids = {c["reference_id"] for c in chunks}
        assert ref_ids == {"1", "2"}, f"Expected doc-level reference_ids, got {ref_ids}"

        chunk_ids = {c["chunk_id"] for c in chunks}
        assert "chunk-abc" in chunk_ids

    async def test_page_idx_is_one_based(self):
        """page_idx in output should be 1-based (0→1, 1→2)."""
        retriever = _make_retriever()

        retriever.lightrag.aquery_data = AsyncMock(return_value={
            "data": {
                "entities": [],
                "relationships": [],
                "chunks": [
                    {
                        "chunk_id": "chunk-abc",
                        "reference_id": "1",
                        "content": "text",
                        "file_path": "/data/doc.pdf",
                    },
                ],
            },
        })

        retriever.visual_chunks.get_by_ids = AsyncMock(return_value=[
            {
                "full_doc_id": "doc-1",
                "file_path": "/data/doc.pdf",
                "page_index": 0,
                "image_data": "base64",
                "doc_title": "Doc",
            },
        ])

        result = await retriever._text_retrieve("query", top_k=60, chunk_top_k=10)
        chunk = result["contexts"]["chunks"][0]
        assert chunk["page_idx"] == 1, "page_idx should be 1-based"


class TestQueryByVisualEmbedding:
    """Test query_by_visual_embedding output schema."""

    async def test_output_has_required_fields(self):
        retriever = _make_retriever()
        retriever.embedder = MagicMock()

        import numpy as np

        retriever.embedder.embed_pages = AsyncMock(
            return_value=np.array([[0.1] * 1024], dtype=np.float32)
        )

        retriever.lightrag.chunks_vdb.query = AsyncMock(return_value=[
            {
                "id": "chunk-xyz",
                "content": "page content",
                "chunk_order_index": 2,
                "distance": 0.85,
                "file_path": "/data/doc.pdf",
            },
        ])

        img_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        with patch("dlightrag.unifiedrepresent.retriever.Image") as mock_image:
            mock_pil = MagicMock()
            mock_image.open.return_value = mock_pil

            results = await retriever.query_by_visual_embedding([img_bytes], top_k=5)

        assert len(results) == 1
        chunk = results[0]
        assert "chunk_id" in chunk
        assert "page_idx" in chunk
        assert "content" in chunk
        assert chunk["chunk_id"] == "chunk-xyz"
        assert chunk["page_idx"] == 3  # 0-based 2 -> 1-based 3
