# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the LightRAG mix retrieval backend."""

from __future__ import annotations

import base64
import io
from unittest.mock import AsyncMock, MagicMock

from PIL import Image

from dlightrag.core.retrieval.lightrag_backend import LightRAGMixBackend


def _image_payload() -> dict[str, str]:
    image = Image.new("RGB", (2, 2), "white")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return {"type": "image", "data": base64.b64encode(buf.getvalue()).decode("ascii")}


async def test_backend_always_queries_lightrag_mix() -> None:
    lightrag = MagicMock()
    lightrag.aquery_data = AsyncMock(
        return_value={
            "data": {
                "chunks": [{"id": "txt1", "content": "alpha", "file_path": "/docs/a.pdf"}],
                "entities": [{"entity_name": "Alpha"}],
                "relationships": [],
            }
        }
    )
    lightrag.text_chunks = MagicMock()
    visual_chunks = MagicMock()
    visual_chunks.get_by_ids = AsyncMock(
        return_value=[{"image_data": "page-bytes", "page_index": 1, "file_path": "/docs/a.pdf"}]
    )

    backend = LightRAGMixBackend(lightrag=lightrag, visual_chunks=visual_chunks)
    result = await backend.aretrieve("question", mode="mix", top_k=5)

    param = lightrag.aquery_data.await_args.kwargs["param"]
    assert param.mode == "mix"
    assert result.contexts["entities"] == [{"entity_name": "Alpha"}]
    assert result.contexts["chunks"][0]["chunk_id"] == "txt1"
    assert result.contexts["chunks"][0]["image_data"] == "page-bytes"
    assert result.contexts["chunks"][0]["page_idx"] == 2


async def test_backend_embeds_query_images_directly() -> None:
    lightrag = MagicMock()
    lightrag.aquery_data = AsyncMock(
        return_value={"data": {"chunks": [], "entities": [], "relationships": []}}
    )
    lightrag.text_chunks = MagicMock()
    lightrag.chunks_vdb = MagicMock()
    lightrag.chunks_vdb.query = AsyncMock(
        return_value=[
            {
                "id": "img1",
                "content": "visual match",
                "file_path": "/docs/img.png",
                "distance": 0.12,
            }
        ]
    )

    embedder = MagicMock()
    embedder.embed_query_images = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    visual_chunks = MagicMock()
    visual_chunks.get_by_ids = AsyncMock(
        return_value=[{"image_data": "image-bytes", "page_index": 0, "file_path": "/docs/img.png"}]
    )

    backend = LightRAGMixBackend(
        lightrag=lightrag,
        visual_chunks=visual_chunks,
        embedder=embedder,
    )
    result = await backend.aretrieve("find this", multimodal_content=[_image_payload()])

    embedder.embed_query_images.assert_awaited_once()
    lightrag.chunks_vdb.query.assert_awaited_once()
    assert lightrag.chunks_vdb.query.await_args.kwargs["query_embedding"] == [0.1, 0.2, 0.3]
    assert result.contexts["chunks"][0]["chunk_id"] == "img1"
    assert result.contexts["chunks"][0]["image_data"] == "image-bytes"
