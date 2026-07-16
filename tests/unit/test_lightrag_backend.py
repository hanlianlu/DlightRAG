# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the LightRAG mix retrieval backend."""

import base64
import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from PIL import Image

from dlightrag.core.retrieval.lightrag_backend import LightRAGMixBackend


def _image_block() -> dict[str, Any]:
    buf = io.BytesIO()
    Image.new("RGB", (2, 2), "white").save(buf, format="PNG")
    payload = base64.b64encode(buf.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{payload}"}}


def _write_image(path: Path) -> None:
    Image.new("RGB", (2, 2), "white").save(path)


def _stores(
    *,
    raw_chunks: list[dict[str, Any] | None] | None = None,
    full_doc: dict[str, Any] | None = None,
    vector_results: list[dict[str, Any]] | None = None,
) -> MagicMock:
    stores = MagicMock()
    stores.context_chunks_by_ids = AsyncMock(return_value=[])
    stores.get_text_chunks = AsyncMock(return_value=list(raw_chunks or []))
    stores.get_full_doc = AsyncMock(return_value=full_doc)
    stores.chunks_vdb = MagicMock()
    stores.chunks_vdb.query = AsyncMock(return_value=list(vector_results or []))
    return stores


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
    stores = _stores(raw_chunks=[None])

    backend = LightRAGMixBackend(lightrag=lightrag, stores=stores)
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
    stores = _stores()

    backend = LightRAGMixBackend(lightrag=lightrag, stores=stores)
    await backend.aretrieve("question", top_k=60, chunk_top_k=30)

    param = lightrag.aquery_data.await_args.kwargs["param"]
    assert param.top_k == 60
    assert param.chunk_top_k == 30


async def test_backend_forwards_query_token_caps_to_lightrag_query_param() -> None:
    lightrag = MagicMock()
    lightrag.aquery_data = AsyncMock(
        return_value={"data": {"chunks": [], "entities": [], "relationships": []}}
    )
    stores = _stores()

    backend = LightRAGMixBackend(
        lightrag=lightrag,
        stores=stores,
        max_entity_tokens=111,
        max_relation_tokens=222,
        max_total_tokens=333,
    )
    await backend.aretrieve("question", top_k=60, chunk_top_k=30)

    param = lightrag.aquery_data.await_args.kwargs["param"]
    assert param.max_entity_tokens == 111
    assert param.max_relation_tokens == 222
    assert param.max_total_tokens == 333


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
    stores = _stores(
        raw_chunks=[
            {
                "id": "img1",
                "content": "visual",
                "file_path": str(image_path),
                "sidecar": {"page_index": 2},
            }
        ]
    )

    backend = LightRAGMixBackend(lightrag=lightrag, stores=stores)
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
    stores = _stores(
        raw_chunks=[
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
        ],
        full_doc={"sidecar_location": parsed_dir.as_uri()},
    )

    backend = LightRAGMixBackend(lightrag=lightrag, stores=stores)
    result = await backend.aretrieve("question")

    assert result.contexts["chunks"][0]["page_idx"] == 5
    assert result.contexts["chunks"][0]["full_doc_id"] == "doc-1"


async def test_backend_hydrates_multimodal_chunk_page_from_sidecar_item(
    tmp_path: Path,
) -> None:
    """Table/drawing/equation chunks reference a modality item id, not a block;
    provenance is recovered via the item's ``blockid`` in ``*.tables.json``."""
    parsed_dir = tmp_path / "sample.parsed"
    parsed_dir.mkdir()
    (parsed_dir / "sample.blocks.jsonl").write_text(
        json.dumps(
            {
                "type": "content",
                "blockid": "block-9",
                "content": "<table>…</table>",
                "positions": [{"type": "bbox", "anchor": 4, "range": [1, 2, 3, 4]}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (parsed_dir / "sample.tables.json").write_text(
        json.dumps({"version": "1.0", "tables": {"tb-1": {"id": "tb-1", "blockid": "block-9"}}}),
        encoding="utf-8",
    )

    lightrag = MagicMock()
    lightrag.aquery_data = AsyncMock(
        return_value={
            "data": {
                "chunks": [
                    {"id": "doc-1-mm-table-000", "content": "[Table Name] X", "file_path": "/d.pdf"}
                ],
                "entities": [],
                "relationships": [],
            }
        }
    )
    stores = _stores(
        raw_chunks=[
            {
                "id": "doc-1-mm-table-000",
                "content": "[Table Name] X",
                "file_path": "/d.pdf",
                "full_doc_id": "doc-1",
                "sidecar": {
                    "type": "table",
                    "id": "tb-1",
                    "refs": [{"type": "table", "id": "tb-1"}],
                },
            }
        ],
        full_doc={"sidecar_location": parsed_dir.as_uri()},
    )

    backend = LightRAGMixBackend(lightrag=lightrag, stores=stores)
    result = await backend.aretrieve("question")

    chunk = result.contexts["chunks"][0]
    assert chunk["page_idx"] == 5
    assert chunk["bbox"] == {"page_index": 4, "range": [1, 2, 3, 4]}


async def test_backend_hydrates_v150_drawing_sidecar_from_drawings_json(
    tmp_path: Path,
) -> None:
    """LightRAG 1.5 visual chunks carry sidecar={type,id,refs} with no path field.
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
                        "img_path": "sample.blocks.assets/img-0001.png",
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
    stores = _stores(
        raw_chunks=[
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
        ],
        full_doc={"sidecar_location": parsed_dir.as_uri()},
    )

    backend = LightRAGMixBackend(lightrag=lightrag, stores=stores)
    result = await backend.aretrieve("question")

    chunk = result.contexts["chunks"][0]
    assert chunk["chunk_id"] == "mm1"
    assert chunk["image_data"]
    assert chunk["image_mime_type"] == "image/png"
    # file_path should be remapped from the sidecar asset path to the document path
    assert chunk["file_path"] == "/docs/report.pdf"
    assert result.contexts["chunks"][0]["image_data"]


async def test_backend_direct_visual_leg_embeds_and_searches() -> None:
    lightrag = MagicMock()
    lightrag.aquery_data = AsyncMock(
        return_value={"data": {"chunks": [], "entities": [], "relationships": []}}
    )
    stores = _stores(
        raw_chunks=[None],
        vector_results=[
            {"id": "img1", "content": "visual match", "file_path": "a.pdf", "distance": 0.12}
        ],
    )
    embedder = MagicMock()
    embedder.embed_query_images = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

    backend = LightRAGMixBackend(lightrag=lightrag, stores=stores, embedder=embedder)
    result = await backend.aretrieve("find this", query_image_blocks=[_image_block()])

    embedder.embed_query_images.assert_awaited_once()
    stores.chunks_vdb.query.assert_awaited_once()
    assert result.contexts["chunks"][0]["chunk_id"] == "img1"
    assert result.trace["direct_visual_chunk_count"] == 1


async def test_backend_batches_query_image_embeddings_in_one_request() -> None:
    lightrag = MagicMock()
    lightrag.aquery_data = AsyncMock(
        return_value={"data": {"chunks": [], "entities": [], "relationships": []}}
    )
    stores = _stores(raw_chunks=[None, None])
    stores.chunks_vdb.query = AsyncMock(
        side_effect=[
            [{"id": "img-a", "content": "a", "file_path": "a", "distance": 0.2}],
            [{"id": "img-b", "content": "b", "file_path": "b", "distance": 0.1}],
        ]
    )
    embedder = MagicMock()
    embedder.embed_query_images = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])

    backend = LightRAGMixBackend(lightrag=lightrag, stores=stores, embedder=embedder)
    result = await backend.aretrieve(
        "find these", query_image_blocks=[_image_block(), _image_block()]
    )

    # Both query images are embedded in ONE batched request; each vector searches.
    embedder.embed_query_images.assert_awaited_once()
    assert embedder.embed_query_images.await_count == 1
    assert stores.chunks_vdb.query.await_count == 2
    assert [c["chunk_id"] for c in result.contexts["chunks"][:2]] == ["img-b", "img-a"]


async def test_backend_query_image_dedup_keeps_closest_distance() -> None:
    lightrag = MagicMock()
    lightrag.aquery_data = AsyncMock(
        return_value={"data": {"chunks": [], "entities": [], "relationships": []}}
    )
    stores = _stores(raw_chunks=[None])
    # Two query-image vectors retrieve the same chunk at different distances;
    # the merge keeps the CLOSEST, not whichever arrived last.
    stores.chunks_vdb.query = AsyncMock(
        side_effect=[
            [{"id": "dup", "content": "far", "file_path": "a", "distance": 0.9}],
            [{"id": "dup", "content": "near", "file_path": "a", "distance": 0.1}],
        ]
    )
    embedder = MagicMock()
    embedder.embed_query_images = AsyncMock(return_value=[[0.1], [0.2]])

    backend = LightRAGMixBackend(lightrag=lightrag, stores=stores, embedder=embedder)
    result = await backend.aretrieve("q", query_image_blocks=[_image_block(), _image_block()])

    visual = [c for c in result.contexts["chunks"] if c["chunk_id"] == "dup"]
    assert len(visual) == 1
    assert visual[0]["relevance_score"] == 0.1


async def test_backend_skips_direct_visual_without_embedder() -> None:
    lightrag = MagicMock()
    lightrag.aquery_data = AsyncMock(
        return_value={"data": {"chunks": [], "entities": [], "relationships": []}}
    )
    stores = _stores()

    backend = LightRAGMixBackend(lightrag=lightrag, stores=stores, embedder=None)
    result = await backend.aretrieve("find this", query_image_blocks=[_image_block()])

    assert result.trace["direct_visual_chunk_count"] == 0
    stores.chunks_vdb.query.assert_not_awaited()


async def test_backend_uses_dedicated_direct_visual_top_k() -> None:
    lightrag = MagicMock()
    lightrag.aquery_data = AsyncMock(
        return_value={"data": {"chunks": [], "entities": [], "relationships": []}}
    )
    stores = _stores(
        raw_chunks=[None],
        vector_results=[{"id": "img1", "content": "visual", "file_path": "a", "distance": 0.12}],
    )
    embedder = MagicMock()
    embedder.embed_query_images = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

    backend = LightRAGMixBackend(
        lightrag=lightrag, stores=stores, embedder=embedder, direct_visual_top_k=2
    )
    result = await backend.aretrieve(
        "find this", chunk_top_k=9, query_image_blocks=[_image_block()]
    )

    assert stores.chunks_vdb.query.await_args.kwargs["top_k"] == 2
    assert result.contexts["chunks"][0]["chunk_id"] == "img1"


async def test_backend_rejects_drawing_sidecar_image_path_outside_artifact_dir(
    tmp_path: Path,
) -> None:
    parsed_dir = tmp_path / "sample.parsed"
    parsed_dir.mkdir()
    outside = tmp_path / "outside.png"
    _write_image(outside)

    (parsed_dir / "sample.drawings.json").write_text(
        json.dumps(
            {
                "drawings": {
                    "im-hash-0001": {
                        "id": "im-hash-0001",
                        "img_path": "../outside.png",
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
    stores = _stores(
        raw_chunks=[
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
        ],
        full_doc={"sidecar_location": parsed_dir.as_uri()},
    )

    backend = LightRAGMixBackend(lightrag=lightrag, stores=stores)
    result = await backend.aretrieve("question")

    assert result.contexts["chunks"][0]["image_data"] is None
