# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the LightRAG mix retrieval backend."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from dlightrag.rag.retrieval.lightrag_backend import LightRAGMixBackend
from dlightrag.rag.retrieval.provenance import hydrate_lightrag_chunk_provenance


def _write_image(path: Path) -> None:
    Image.new("RGB", (2, 2), "white").save(path)


def _stores(
    *,
    raw_chunks: list[dict[str, Any] | None] | None = None,
    full_doc: dict[str, Any] | None = None,
) -> MagicMock:
    stores = MagicMock()
    stores.context_chunks_by_ids = AsyncMock(return_value=[])
    stores.get_text_chunks = AsyncMock(return_value=list(raw_chunks or []))
    stores.get_full_doc = AsyncMock(return_value=full_doc)
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
    assert param.only_need_context is False
    assert param.include_references is False
    assert result.contexts["entities"] == [{"entity_name": "Alpha"}]
    assert result.contexts["chunks"][0]["chunk_id"] == "txt1"
    assert result.contexts["chunks"][0]["reference_id"] == "3"
    assert result.contexts["chunks"][0].get("page_number") is None
    stores.get_text_chunks.assert_not_awaited()


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


async def test_provenance_hydrates_image_chunks_from_lightrag_text_chunks(tmp_path: Path) -> None:
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
                "sidecar": {"page_number": 3},
            }
        ]
    )

    backend = LightRAGMixBackend(lightrag=lightrag, stores=stores)
    result = await backend.aretrieve("question")
    await hydrate_lightrag_chunk_provenance(stores, result.contexts["chunks"])

    chunk = result.contexts["chunks"][0]
    assert chunk["chunk_id"] == "img1"
    assert chunk["image_data"]
    assert chunk["page_number"] == 3


async def test_provenance_hydrates_text_chunk_page_from_lightrag_block_sidecar(
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
                "positions": [{"type": "bbox", "anchor": 1, "range": [1, 2, 3, 4]}],
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
    await hydrate_lightrag_chunk_provenance(stores, result.contexts["chunks"])

    assert result.contexts["chunks"][0]["page_number"] == 1
    assert "page_idx" not in result.contexts["chunks"][0]
    assert "bbox" not in result.contexts["chunks"][0]
    assert result.contexts["chunks"][0]["full_doc_id"] == "doc-1"


async def test_sidecar_provenance_index_loading_runs_off_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio
    import threading

    parsed_dir = tmp_path / "sample.parsed"
    parsed_dir.mkdir()
    (parsed_dir / "sample.blocks.jsonl").write_text(
        json.dumps(
            {
                "type": "content",
                "blockid": "block-1",
                "content": "body",
                "positions": [{"type": "bbox", "anchor": 1, "range": [1, 2, 3, 4]}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    loop_thread = threading.get_ident()
    load_threads: list[int] = []
    real_to_thread = asyncio.to_thread

    async def to_thread(func, *args, **kwargs):  # noqa: ANN001, ANN202
        def observed():
            if getattr(func, "__name__", "") == "load_block_provenance_index":
                load_threads.append(threading.get_ident())
            return func(*args, **kwargs)

        return await real_to_thread(observed)

    monkeypatch.setattr(asyncio, "to_thread", to_thread)
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

    chunks = [{"chunk_id": "txt1", "content": "alpha", "file_path": "/docs/a.pdf"}]
    await hydrate_lightrag_chunk_provenance(stores, chunks, include_image_data=False)

    assert chunks[0]["page_number"] == 1
    assert load_threads and loop_thread not in load_threads


async def test_provenance_hydrates_multimodal_chunk_page_from_sidecar_item(
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
                "positions": [{"type": "bbox", "anchor": "1", "range": [1, 2, 3, 4]}],
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
    await hydrate_lightrag_chunk_provenance(stores, result.contexts["chunks"])

    chunk = result.contexts["chunks"][0]
    assert chunk["page_number"] == 1
    assert "page_idx" not in chunk
    assert "bbox" not in chunk


async def test_provenance_hydrates_v150_drawing_sidecar_from_drawings_json(
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
    await hydrate_lightrag_chunk_provenance(stores, result.contexts["chunks"])

    chunk = result.contexts["chunks"][0]
    assert chunk["chunk_id"] == "mm1"
    assert chunk["image_data"]
    assert chunk["image_mime_type"] == "image/png"
    # file_path should be remapped from the sidecar asset path to the document path
    assert chunk["file_path"] == "/docs/report.pdf"
    assert result.contexts["chunks"][0]["image_data"]


async def test_provenance_rejects_drawing_sidecar_image_path_outside_artifact_dir(
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
    await hydrate_lightrag_chunk_provenance(stores, result.contexts["chunks"])

    assert result.contexts["chunks"][0]["image_data"] is None
