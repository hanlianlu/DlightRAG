# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for direct image chunk construction."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from PIL import Image

from dlightrag.core.ingestion.direct_image import (
    build_direct_image_chunk,
    direct_image_chunk_id,
    native_image_ref,
)
from dlightrag.core.ingestion.lightrag_sidecar import LightRAGSidecarRef


def test_direct_image_chunk_id_is_deterministic() -> None:
    ref = LightRAGSidecarRef(sidecar_type="drawing", sidecar_id="fig-1")

    assert direct_image_chunk_id("default", "doc-1", ref) == "default:doc-1:sidecar:drawing:fig-1"


async def test_build_direct_image_chunk_embeds_original_image(tmp_path) -> None:
    path = tmp_path / "fig.png"
    Image.new("RGB", (1, 1), "white").save(path)
    ref = LightRAGSidecarRef(sidecar_type="drawing", sidecar_id="fig-1", asset_path=path)
    embedder = AsyncMock()
    embedder.embed_index_images.return_value = [[0.1, 0.2, 0.3]]

    chunk_id, row, vector = await build_direct_image_chunk(
        workspace="default",
        full_doc_id="doc-1",
        ref=ref,
        embedder=embedder,
        text_content="figure caption",
    )

    assert chunk_id == "default:doc-1:sidecar:drawing:fig-1"
    assert row["content"] == "figure caption"
    assert row["sidecar"] == {
        "type": "drawing",
        "id": "fig-1",
        "path": str(path),
    }
    assert vector == [0.1, 0.2, 0.3]
    embedder.embed_index_images.assert_awaited_once()


async def test_build_direct_image_chunk_keeps_sidecar_provenance(tmp_path) -> None:
    path = tmp_path / "fig.png"
    Image.new("RGB", (1, 1), "white").save(path)
    ref = LightRAGSidecarRef(
        sidecar_type="drawing",
        sidecar_id="fig-1",
        asset_path=path,
        page_index=2,
        bbox={"page_index": 2, "range": [1, 2, 3, 4]},
        block_id="block-1",
    )
    embedder = AsyncMock()
    embedder.embed_index_images.return_value = [[0.1, 0.2, 0.3]]

    _, row, _ = await build_direct_image_chunk(
        workspace="default",
        full_doc_id="doc-1",
        ref=ref,
        embedder=embedder,
        text_content="figure caption",
    )

    assert row["sidecar"] == {
        "type": "drawing",
        "id": "fig-1",
        "path": str(path),
        "page_index": 2,
        "bbox": {"page_index": 2, "range": [1, 2, 3, 4]},
        "block_id": "block-1",
    }


async def test_build_direct_image_chunk_requires_asset_path() -> None:
    ref = LightRAGSidecarRef(sidecar_type="drawing", sidecar_id="fig-1")

    with pytest.raises(ValueError, match="asset_path"):
        await build_direct_image_chunk(
            workspace="default",
            full_doc_id="doc-1",
            ref=ref,
            embedder=AsyncMock(),
            text_content="figure",
        )


def test_native_image_ref_uses_file_stem(tmp_path) -> None:
    path = tmp_path / "original.png"

    ref = native_image_ref(path)

    assert ref.sidecar_type == "native_image"
    assert ref.sidecar_id == "original"
    assert ref.asset_path == path.resolve()
