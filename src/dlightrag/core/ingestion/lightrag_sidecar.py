# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Resolve LightRAG multimodal sidecar assets for vector overrides."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from dlightrag.core.sidecar_provenance import resolve_sidecar_asset_path


@dataclass(frozen=True)
class LightRAGDrawingAsset:
    """A LightRAG drawing multimodal chunk and its source image file."""

    chunk_id: str
    sidecar_id: str
    image_path: Path


def collect_lightrag_drawing_assets(
    artifact_dir: Path,
    *,
    doc_id: str,
) -> list[LightRAGDrawingAsset]:
    """Return successful LightRAG drawing mm chunks with their source image files.

    Mirrors LightRAG 1.5's ``_build_mm_chunks_from_sidecars`` contract:
    only the first ``*.blocks.jsonl``-anchored ``*.drawings.json`` sidecar is
    considered, chunk ids are ``{doc_id}-mm-drawing-{local_idx:03d}``, and
    skipped or missing VLM analysis items do not produce chunks.
    """
    drawings_path = _drawings_path_for_blocks(artifact_dir)
    if drawings_path is None or not drawings_path.exists():
        return []

    try:
        payload = json.loads(drawings_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(payload, dict):
        return []
    drawings = payload.get("drawings")
    if not isinstance(drawings, dict):
        return []

    assets: list[LightRAGDrawingAsset] = []
    for local_idx, (item_id, item) in enumerate(drawings.items()):
        if not isinstance(item, dict):
            continue
        analysis = item.get("llm_analyze_result")
        if not isinstance(analysis, dict) or analysis.get("status") != "success":
            continue
        image_path = _resolve_drawing_image_path(artifact_dir, item)
        if image_path is None:
            continue
        assets.append(
            LightRAGDrawingAsset(
                chunk_id=f"{doc_id}-mm-drawing-{local_idx:03d}",
                sidecar_id=str(item_id),
                image_path=image_path,
            )
        )
    return assets


def _drawings_path_for_blocks(artifact_dir: Path) -> Path | None:
    block_paths = sorted(artifact_dir.glob("*.blocks.jsonl"))
    if not block_paths:
        return None
    block_path = block_paths[0]
    return Path(str(block_path)[: -len(".blocks.jsonl")] + ".drawings.json")


def _resolve_drawing_image_path(artifact_dir: Path, item: dict[str, object]) -> Path | None:
    raw_path = item.get("path") or item.get("img_path") or item.get("image_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    return resolve_sidecar_asset_path(artifact_dir, raw_path)
