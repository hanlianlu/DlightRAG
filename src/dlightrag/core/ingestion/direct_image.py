# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Direct image embedding chunk creation for native and extracted images."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from dlightrag.core.ingestion.lightrag_sidecar import LightRAGSidecarRef


def direct_image_chunk_id(workspace: str, full_doc_id: str, ref: LightRAGSidecarRef) -> str:
    """Return deterministic direct-image chunk id."""
    return f"{workspace}:{full_doc_id}:sidecar:{ref.sidecar_type}:{ref.sidecar_id}"


async def build_direct_image_chunk(
    *,
    workspace: str,
    full_doc_id: str,
    ref: LightRAGSidecarRef,
    embedder: Any,
    text_content: str,
) -> tuple[str, dict[str, Any], list[float]]:
    """Build one direct-image chunk row and vector."""
    if ref.asset_path is None:
        raise ValueError(f"{ref.sidecar_type}:{ref.sidecar_id} has no asset_path")
    image = Image.open(ref.asset_path).convert("RGB")
    try:
        vector = (await embedder.embed_index_images([image]))[0]
    finally:
        image.close()
    chunk_id = direct_image_chunk_id(workspace, full_doc_id, ref)
    row = {
        "full_doc_id": full_doc_id,
        "tokens": 0,
        "chunk_order_index": 0,
        "content": text_content,
        "file_path": str(ref.asset_path),
        "heading": {},
        "sidecar": {"type": ref.sidecar_type, "id": ref.sidecar_id},
        "llm_cache_list": [],
    }
    return chunk_id, row, vector


def native_image_ref(path: Path) -> LightRAGSidecarRef:
    """Return a sidecar-like ref for a native image file."""
    return LightRAGSidecarRef(
        sidecar_type="native_image",
        sidecar_id=path.stem,
        asset_path=path.resolve(),
        page_number=None,
        bbox=None,
        payload={"source_path": str(path)},
    )
