# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for visual semantic projection documents."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from dlightrag.core.ingestion.lightrag_sidecar import LightRAGSidecarRef
from dlightrag.core.ingestion.visual_semantics import (
    build_visual_semantic_projection,
    visual_semantic_doc_id,
)


def test_visual_semantic_doc_id_is_deterministic() -> None:
    ref = LightRAGSidecarRef(sidecar_type="native_image", sidecar_id="image-1")

    assert (
        visual_semantic_doc_id("default", "doc-1", ref)
        == "default:doc-1:visual-semantic:native_image:image-1"
    )


async def test_visual_semantic_projection_contains_source_and_description(tmp_path) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(b"png")
    ref = LightRAGSidecarRef(
        sidecar_type="native_image",
        sidecar_id="image-1",
        asset_path=path,
    )
    vlm_func = AsyncMock(return_value="A diagram with two labeled nodes.")

    doc_id, text = await build_visual_semantic_projection(
        workspace="default",
        source_doc_id="doc-1",
        ref=ref,
        vlm_func=vlm_func,
    )

    assert doc_id == "default:doc-1:visual-semantic:native_image:image-1"
    assert "Source image id: image-1" in text
    assert "Source image type: native_image" in text
    assert "A diagram with two labeled nodes." in text
    vlm_func.assert_awaited_once()


async def test_visual_semantic_projection_requires_asset_path() -> None:
    ref = LightRAGSidecarRef(sidecar_type="native_image", sidecar_id="image-1")

    with pytest.raises(ValueError, match="asset_path"):
        await build_visual_semantic_projection(
            workspace="default",
            source_doc_id="doc-1",
            ref=ref,
            vlm_func=AsyncMock(),
        )
