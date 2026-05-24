# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Visual semantic text projections for native image KG insertion."""

from __future__ import annotations

from typing import Any

from dlightrag.core.ingestion.lightrag_sidecar import LightRAGSidecarRef


def visual_semantic_doc_id(workspace: str, source_doc_id: str, ref: LightRAGSidecarRef) -> str:
    """Return deterministic LightRAG semantic projection doc id."""
    return f"{workspace}:{source_doc_id}:visual-semantic:{ref.sidecar_type}:{ref.sidecar_id}"


async def build_visual_semantic_projection(
    *,
    workspace: str,
    source_doc_id: str,
    ref: LightRAGSidecarRef,
    vlm_func: Any,
) -> tuple[str, str]:
    """Return deterministic LightRAG text input for native/fallback image KG."""
    if ref.asset_path is None:
        raise ValueError(f"{ref.sidecar_type}:{ref.sidecar_id} has no asset_path")
    doc_id = visual_semantic_doc_id(workspace, source_doc_id, ref)
    description = await vlm_func(
        prompt="Describe visible entities, relationships, text, scene attributes, and uncertainty.",
        image_path=str(ref.asset_path),
    )
    text = (
        "[Visual Semantic Projection]\n"
        f"Source image id: {ref.sidecar_id}\n"
        f"Source image type: {ref.sidecar_type}\n"
        f"Description:\n{description}"
    )
    return doc_id, text
