# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the direct image->image retrieval leg."""

import base64
import io
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from PIL import Image

from dlightrag.rag.retrieval.visual import DirectVisualRetriever


def _image_block(*, size: tuple[int, int] = (2, 2), mode: str = "RGB") -> dict[str, Any]:
    buf = io.BytesIO()
    Image.new(mode, size, "white").save(buf, format="PNG")
    payload = base64.b64encode(buf.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{payload}"}}


def _stores(vector_results: Any) -> MagicMock:
    stores = MagicMock()
    stores.chunks_vdb = MagicMock()
    if isinstance(vector_results, list) and vector_results and isinstance(vector_results[0], list):
        stores.chunks_vdb.query = AsyncMock(side_effect=vector_results)
    else:
        stores.chunks_vdb.query = AsyncMock(return_value=list(vector_results))
    return stores


async def test_visual_leg_embeds_and_searches() -> None:
    stores = _stores([{"id": "img1", "content": "visual", "file_path": "a.pdf", "distance": 0.12}])
    embedder = MagicMock()
    embedder.embed_query_images = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

    retriever = DirectVisualRetriever(embedder=embedder, stores=stores, top_k=2)
    chunks = await retriever.search([_image_block()])

    embedder.embed_query_images.assert_awaited_once()
    assert stores.chunks_vdb.query.await_args.kwargs["top_k"] == 2
    assert [c["chunk_id"] for c in chunks] == ["img1"]


async def test_visual_leg_batches_query_images_in_one_embedding_request() -> None:
    stores = _stores(
        [
            [{"id": "img-a", "content": "a", "file_path": "a", "distance": 0.2}],
            [{"id": "img-b", "content": "b", "file_path": "b", "distance": 0.1}],
        ]
    )
    embedder = MagicMock()
    embedder.embed_query_images = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])

    retriever = DirectVisualRetriever(embedder=embedder, stores=stores, top_k=5)
    chunks = await retriever.search([_image_block(), _image_block()])

    assert embedder.embed_query_images.await_count == 1
    assert stores.chunks_vdb.query.await_count == 2
    assert [c["chunk_id"] for c in chunks] == ["img-b", "img-a"]


async def test_visual_leg_dedup_keeps_closest_distance() -> None:
    stores = _stores(
        [
            [{"id": "dup", "content": "far", "file_path": "a", "distance": 0.9}],
            [{"id": "dup", "content": "near", "file_path": "a", "distance": 0.1}],
        ]
    )
    embedder = MagicMock()
    embedder.embed_query_images = AsyncMock(return_value=[[0.1], [0.2]])

    retriever = DirectVisualRetriever(embedder=embedder, stores=stores, top_k=5)
    chunks = await retriever.search([_image_block(), _image_block()])

    assert len(chunks) == 1
    assert chunks[0]["relevance_score"] == 0.1


async def test_visual_leg_degrades_to_empty_when_embedding_fails() -> None:
    stores = _stores([])
    embedder = MagicMock()
    embedder.embed_query_images = AsyncMock(side_effect=RuntimeError("provider down"))

    retriever = DirectVisualRetriever(embedder=embedder, stores=stores, top_k=5)

    assert await retriever.search([_image_block()]) == []
    stores.chunks_vdb.query.assert_not_awaited()


async def test_visual_leg_rejects_images_above_decode_pixel_ceiling() -> None:
    stores = _stores([])
    embedder = MagicMock()
    embedder.embed_query_images = AsyncMock(return_value=[[0.1]])
    retriever = DirectVisualRetriever(embedder=embedder, stores=stores, top_k=5)

    assert await retriever.search([_image_block(size=(8_000, 5_001), mode="1")]) == []
    embedder.embed_query_images.assert_not_awaited()


async def test_visual_leg_disabled_by_zero_top_k() -> None:
    stores = _stores([])
    embedder = MagicMock()
    embedder.embed_query_images = AsyncMock()

    retriever = DirectVisualRetriever(embedder=embedder, stores=stores, top_k=0)

    assert await retriever.search([_image_block()]) == []
    embedder.embed_query_images.assert_not_awaited()
