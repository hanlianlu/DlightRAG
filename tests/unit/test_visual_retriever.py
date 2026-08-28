# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the direct image->image retrieval leg."""

import base64
import io
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from PIL import Image

from dlightrag.engine.rag.retrieval.visual import (
    DirectVisualRetriever,
    PreparedVisualQuery,
    VisualEmbeddingDomain,
)


def _image_block(*, size: tuple[int, int] = (2, 2), mode: str = "RGB") -> dict[str, Any]:
    buf = io.BytesIO()
    Image.new(mode, size, "white").save(buf, format="PNG")
    payload = base64.b64encode(buf.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{payload}"}}


def _embedder(vectors: Any = None, *, error: Exception | None = None) -> MagicMock:
    embedder = MagicMock()
    embedder.model = "visual-model"
    embedder.dim = 3
    embedder.input_modality = "multimodal"
    embedder.provider = "test-provider"
    embedder.request_url = "https://embed.example.test/v1/images"
    if error is not None:
        embedder.embed_query_images = AsyncMock(side_effect=error)
    else:
        embedder.embed_query_images = AsyncMock(return_value=vectors)
    return embedder


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
    embedder = _embedder([[0.1, 0.2, 0.3]])

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
    embedder = _embedder([[0.1, 0.2], [0.3, 0.4]])
    embedder.dim = 2

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
    embedder = _embedder([[0.1], [0.2]])
    embedder.dim = 1

    retriever = DirectVisualRetriever(embedder=embedder, stores=stores, top_k=5)
    chunks = await retriever.search([_image_block(), _image_block()])

    assert len(chunks) == 1
    assert chunks[0]["relevance_score"] == 0.1


async def test_visual_leg_degrades_to_empty_when_embedding_fails() -> None:
    stores = _stores([])
    embedder = _embedder(error=RuntimeError("provider down"))

    retriever = DirectVisualRetriever(embedder=embedder, stores=stores, top_k=5)

    assert await retriever.search([_image_block()]) == []
    stores.chunks_vdb.query.assert_not_awaited()


async def test_visual_leg_rejects_images_above_decode_pixel_ceiling() -> None:
    stores = _stores([])
    embedder = _embedder([[0.1]])
    embedder.dim = 1
    retriever = DirectVisualRetriever(embedder=embedder, stores=stores, top_k=5)

    assert await retriever.search([_image_block(size=(8_000, 5_001), mode="1")]) == []
    embedder.embed_query_images.assert_not_awaited()


async def test_visual_leg_disabled_by_zero_top_k() -> None:
    stores = _stores([])
    embedder = _embedder()

    retriever = DirectVisualRetriever(embedder=embedder, stores=stores, top_k=0)

    assert retriever.embedding_domain is None
    assert await retriever.search([_image_block()]) == []
    embedder.embed_query_images.assert_not_awaited()


async def test_prepared_query_is_immutable_and_search_does_not_reembed() -> None:
    stores = _stores([{"id": "img1", "distance": 0.1}])
    embedder = _embedder([[0.1, 0.2, 0.3]])
    retriever = DirectVisualRetriever(embedder=embedder, stores=stores, top_k=2)

    prepared = await retriever.prepare([_image_block()])

    assert prepared is not None
    assert prepared.vectors == ((0.1, 0.2, 0.3),)
    assert await retriever.search_prepared(prepared)
    embedder.embed_query_images.assert_awaited_once()


async def test_prepared_query_domain_mismatch_degrades_without_wrong_vdb_query() -> None:
    stores = _stores([])
    embedder = _embedder([[0.1, 0.2, 0.3]])
    retriever = DirectVisualRetriever(embedder=embedder, stores=stores, top_k=2)
    wrong_domain = VisualEmbeddingDomain(
        provider="other-provider",
        model="visual-model",
        endpoint_fingerprint=None,
        dimension=3,
        input_modality="multimodal",
    )
    prepared = PreparedVisualQuery(domain=wrong_domain, vectors=((0.1, 0.2, 0.3),))

    assert await retriever.search_prepared(prepared) == []
    stores.chunks_vdb.query.assert_not_awaited()
    embedder.embed_query_images.assert_not_awaited()
