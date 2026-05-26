# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the LightRAG mix retrieval backend."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from PIL import Image

from dlightrag.core.retrieval.lightrag_backend import LightRAGMixBackend


def _image_payload() -> dict[str, str]:
    image = Image.new("RGB", (2, 2), "white")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return {"type": "image", "data": base64.b64encode(buf.getvalue()).decode("ascii")}


def _write_image(path: Path) -> None:
    Image.new("RGB", (2, 2), "white").save(path)


async def test_backend_always_queries_lightrag_mix() -> None:
    lightrag = MagicMock()
    lightrag.aquery_data = AsyncMock(
        return_value={
            "data": {
                "chunks": [{"id": "txt1", "content": "alpha", "file_path": "/docs/a.pdf"}],
                "entities": [{"entity_name": "Alpha"}],
                "relationships": [],
                "references": [{"reference_id": "3", "file_path": "/docs/a.pdf"}],
            }
        }
    )
    lightrag.text_chunks = MagicMock()
    lightrag.text_chunks.get_by_ids = AsyncMock(return_value=[None])

    backend = LightRAGMixBackend(lightrag=lightrag)
    result = await backend.aretrieve("question", mode="mix", top_k=5)

    param = lightrag.aquery_data.await_args.kwargs["param"]
    assert param.mode == "mix"
    assert result.contexts["entities"] == [{"entity_name": "Alpha"}]
    assert result.contexts["chunks"][0]["chunk_id"] == "txt1"
    assert result.contexts["chunks"][0]["reference_id"] == "3"
    assert result.contexts["chunks"][0].get("page_idx") is None


async def test_backend_forwards_chunk_top_k_to_lightrag_query_param() -> None:
    lightrag = MagicMock()
    lightrag.aquery_data = AsyncMock(
        return_value={"data": {"chunks": [], "entities": [], "relationships": []}}
    )
    lightrag.text_chunks = MagicMock()
    lightrag.text_chunks.get_by_ids = AsyncMock(return_value=[])

    backend = LightRAGMixBackend(lightrag=lightrag)
    await backend.aretrieve("question", top_k=60, chunk_top_k=30)

    param = lightrag.aquery_data.await_args.kwargs["param"]
    assert param.top_k == 60
    assert param.chunk_top_k == 30


async def test_backend_hydrates_image_chunks_from_lightrag_text_chunks(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    _write_image(image_path)
    lightrag = MagicMock()
    lightrag.aquery_data = AsyncMock(
        return_value={
            "data": {
                "chunks": [{"id": "img1", "content": "visual", "file_path": str(image_path)}],
                "entities": [],
                "relationships": [],
            }
        }
    )
    lightrag.text_chunks = MagicMock()
    lightrag.text_chunks.get_by_ids = AsyncMock(
        return_value=[
            {
                "id": "img1",
                "content": "visual",
                "file_path": str(image_path),
                "sidecar": {"page_index": 2},
            }
        ]
    )

    backend = LightRAGMixBackend(lightrag=lightrag)
    result = await backend.aretrieve("question")

    chunk = result.contexts["chunks"][0]
    assert chunk["chunk_id"] == "img1"
    assert chunk["image_data"]
    assert chunk["page_idx"] == 3


async def test_backend_hydrates_text_chunk_page_from_lightrag_block_sidecar(
    tmp_path: Path,
) -> None:
    parsed_dir = tmp_path / "sample.parsed"
    parsed_dir.mkdir()
    (parsed_dir / "sample.blocks.jsonl").write_text(
        json.dumps(
            {
                "type": "content",
                "blockid": "block-1",
                "content": "body",
                "positions": [{"type": "bbox", "anchor": 4, "range": [1, 2, 3, 4]}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    lightrag = MagicMock()
    lightrag.aquery_data = AsyncMock(
        return_value={
            "data": {
                "chunks": [{"id": "txt1", "content": "alpha", "file_path": "/docs/a.pdf"}],
                "entities": [],
                "relationships": [],
            }
        }
    )
    lightrag.text_chunks = MagicMock()
    lightrag.text_chunks.get_by_ids = AsyncMock(
        return_value=[
            {
                "id": "txt1",
                "content": "alpha",
                "file_path": "/docs/a.pdf",
                "full_doc_id": "doc-1",
                "sidecar": {
                    "type": "block",
                    "id": "block-1",
                    "refs": [{"type": "block", "id": "block-1"}],
                },
            }
        ]
    )
    lightrag.full_docs = MagicMock()
    lightrag.full_docs.get_by_id = AsyncMock(return_value={"sidecar_location": parsed_dir.as_uri()})

    backend = LightRAGMixBackend(lightrag=lightrag)
    result = await backend.aretrieve("question")

    assert result.contexts["chunks"][0]["page_idx"] == 5


async def test_backend_embeds_query_images_directly(tmp_path: Path) -> None:
    image_path = tmp_path / "img.png"
    _write_image(image_path)
    lightrag = MagicMock()
    lightrag.aquery_data = AsyncMock(
        return_value={"data": {"chunks": [], "entities": [], "relationships": []}}
    )
    lightrag.text_chunks = MagicMock()
    lightrag.text_chunks.get_by_ids = AsyncMock(return_value=[None])
    lightrag.chunks_vdb = MagicMock()
    lightrag.chunks_vdb.query = AsyncMock(
        return_value=[
            {
                "id": "img1",
                "content": "visual match",
                "file_path": str(image_path),
                "distance": 0.12,
            }
        ]
    )

    embedder = MagicMock()
    embedder.embed_query_images = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

    backend = LightRAGMixBackend(
        lightrag=lightrag,
        embedder=embedder,
    )
    result = await backend.aretrieve("find this", multimodal_content=[_image_payload()])

    embedder.embed_query_images.assert_awaited_once()
    lightrag.chunks_vdb.query.assert_awaited_once()
    assert lightrag.chunks_vdb.query.await_args.kwargs["query_embedding"] == [0.1, 0.2, 0.3]
    assert result.contexts["chunks"][0]["chunk_id"] == "img1"


async def test_backend_hydrates_v150rc3_drawing_sidecar_from_drawings_json(
    tmp_path: Path,
) -> None:
    """v1.5.0rc3 visual chunks carry sidecar={type,id,refs} with no path field.
    The image path must be resolved from drawings.json in the parsed artifact dir."""
    parsed_dir = tmp_path / "sample.parsed"
    assets_dir = parsed_dir / "sample.blocks.assets"
    assets_dir.mkdir(parents=True)
    image_path = assets_dir / "img-0001.png"
    _write_image(image_path)

    (parsed_dir / "sample.drawings.json").write_text(
        json.dumps(
            {
                "drawings": {
                    "im-hash-0001": {
                        "id": "im-hash-0001",
                        "path": "sample.blocks.assets/img-0001.png",
                        "format": "png",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    lightrag = MagicMock()
    lightrag.aquery_data = AsyncMock(
        return_value={
            "data": {
                "chunks": [{"id": "mm1", "content": "visual", "file_path": "/docs/report.pdf"}],
                "entities": [],
                "relationships": [],
            }
        }
    )
    lightrag.text_chunks = MagicMock()
    lightrag.text_chunks.get_by_ids = AsyncMock(
        return_value=[
            {
                "id": "mm1",
                "content": "visual",
                "file_path": "/docs/report.pdf",
                "full_doc_id": "doc-1",
                "sidecar": {
                    "type": "drawing",
                    "id": "im-hash-0001",
                    "refs": [{"type": "drawing", "id": "im-hash-0001"}],
                },
            }
        ]
    )
    lightrag.full_docs = MagicMock()
    lightrag.full_docs.get_by_id = AsyncMock(return_value={"sidecar_location": parsed_dir.as_uri()})

    backend = LightRAGMixBackend(lightrag=lightrag)
    result = await backend.aretrieve("question")

    chunk = result.contexts["chunks"][0]
    assert chunk["chunk_id"] == "mm1"
    assert chunk["image_data"]
    assert chunk["image_mime_type"] == "image/png"
    # file_path should be remapped from the sidecar asset path to the document path
    assert chunk["file_path"] == "/docs/report.pdf"
    assert result.contexts["chunks"][0]["image_data"]
