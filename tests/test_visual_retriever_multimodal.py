"""Tests for VisualRetriever multimodal query capabilities."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from dlightrag.unifiedrepresent.retriever import VisualRetriever


def _make_retriever(*, embedder=None, chunks_vdb=None):
    """Create a VisualRetriever with mocked dependencies."""
    lightrag = MagicMock()
    if chunks_vdb is not None:
        lightrag.chunks_vdb = chunks_vdb
    else:
        lightrag.chunks_vdb = AsyncMock()

    visual_chunks = AsyncMock()
    config = MagicMock()

    retriever = VisualRetriever(
        lightrag=lightrag,
        visual_chunks=visual_chunks,
        config=config,
    )
    if embedder is not None:
        retriever.embedder = embedder
    return retriever


@pytest.mark.asyncio
async def test_query_by_visual_embedding_single_image():
    """Single image produces embedding query against chunks_vdb."""
    mock_embedder = AsyncMock()
    mock_embedder.embed_pages = AsyncMock(
        return_value=np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    )

    mock_vdb = AsyncMock()
    mock_vdb.query = AsyncMock(
        return_value=[
            {"id": "chunk-1", "distance": 0.9},
            {"id": "chunk-2", "distance": 0.8},
        ]
    )

    retriever = _make_retriever(embedder=mock_embedder, chunks_vdb=mock_vdb)

    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (1, 1)).save(buf, format="PNG")
    fake_png = buf.getvalue()

    results = await retriever.query_by_visual_embedding([fake_png], top_k=5)

    mock_embedder.embed_pages.assert_called_once()
    mock_vdb.query.assert_called_once()
    call_kwargs = mock_vdb.query.call_args
    assert call_kwargs.kwargs.get("query_embedding") is not None
    assert len(results) == 2
    # Verify normalized schema matches text chunk format
    assert results[0]["chunk_id"] == "chunk-1"
    assert results[0]["relevance_score"] == 0.9
    assert "content" in results[0]
    assert "page_idx" in results[0]


@pytest.mark.asyncio
async def test_query_by_visual_embedding_multiple_images_round_robin():
    """Multiple images produce round-robin merged results."""
    call_count = 0

    async def mock_embed(images):
        return np.array([[0.1, 0.2, 0.3]] * len(images), dtype=np.float32)

    mock_embedder = AsyncMock()
    mock_embedder.embed_pages = mock_embed

    async def mock_query(query="", top_k=5, query_embedding=None):
        nonlocal call_count
        call_count += 1
        return [{"id": f"chunk-{call_count}-{i}", "distance": 0.9 - i * 0.1} for i in range(2)]

    mock_vdb = AsyncMock()
    mock_vdb.query = mock_query

    retriever = _make_retriever(embedder=mock_embedder, chunks_vdb=mock_vdb)

    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (1, 1)).save(buf, format="PNG")
    fake_png = buf.getvalue()

    results = await retriever.query_by_visual_embedding([fake_png, fake_png], top_k=5)

    assert call_count == 2
    assert len(results) == 4


@pytest.mark.asyncio
async def test_query_by_visual_embedding_empty_images():
    """Empty images list returns empty results."""
    retriever = _make_retriever()
    results = await retriever.query_by_visual_embedding([], top_k=5)
    assert results == []
