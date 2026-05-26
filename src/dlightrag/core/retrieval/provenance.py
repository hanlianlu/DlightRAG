# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Hydrate retrieved LightRAG chunks with display provenance."""

from __future__ import annotations

import base64
import inspect
import logging
from pathlib import Path
from typing import Any

from dlightrag.core.sidecar_provenance import (
    BlockProvenance,
    block_ids_from_sidecar,
    explicit_item_bbox,
    explicit_item_page_index,
    first_provenance_for_blocks,
    load_block_provenance_index,
    sidecar_dir_from_location,
)
from dlightrag.utils.images import detect_image_mime_type

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


async def hydrate_lightrag_chunk_provenance(lightrag: Any, chunks: list[dict[str, Any]]) -> None:
    """Hydrate image bytes and page labels from LightRAG chunk/full_doc sidecars."""
    if not chunks:
        return

    chunk_ids = [c["chunk_id"] for c in chunks]
    raw_chunks = await _fetch_raw_chunks(lightrag, chunk_ids)
    full_doc_cache: dict[str, dict[str, Any] | None] = {}
    block_index_cache: dict[Path, dict[str, BlockProvenance]] = {}

    for chunk, raw in zip(chunks, raw_chunks, strict=False):
        raw_chunk = raw if isinstance(raw, dict) else {}
        _merge_raw_chunk_fields(chunk, raw_chunk)

        sidecar = _chunk_sidecar(chunk, raw_chunk)
        _hydrate_page_index_direct(chunk, sidecar)
        _hydrate_bbox_direct(chunk, sidecar)
        if chunk.get("page_idx") is None:
            provenance = await _provenance_from_block_sidecar(
                lightrag,
                sidecar=sidecar,
                chunk=chunk,
                raw_chunk=raw_chunk,
                full_doc_cache=full_doc_cache,
                block_index_cache=block_index_cache,
            )
            if provenance is not None:
                if provenance.page_index is not None:
                    chunk["page_idx"] = provenance.page_index + 1
                if provenance.bbox is not None and chunk.get("bbox") is None:
                    chunk["bbox"] = provenance.bbox

        _hydrate_image_data(chunk, sidecar)

        # Sidecar image chunks have asset paths as file_path (e.g., .blocks.assets/hash.jpg).
        # Remap to the parent document's file_path so citation grouping works correctly.
        if chunk.get("image_data") and _is_sidecar_asset_path(str(chunk.get("file_path", ""))):
            doc_id = chunk.get("full_doc_id")
            if doc_id:
                full_doc = await _fetch_full_doc(lightrag, doc_id, full_doc_cache)
                if full_doc and full_doc.get("file_path"):
                    chunk["file_path"] = full_doc["file_path"]


async def _fetch_raw_chunks(lightrag: Any, chunk_ids: list[str]) -> list[Any]:
    try:
        return await lightrag.text_chunks.get_by_ids(chunk_ids)
    except Exception:
        logger.debug("LightRAG text chunk hydration failed", exc_info=True)
        return [None for _ in chunk_ids]


def _merge_raw_chunk_fields(chunk: dict[str, Any], raw_chunk: dict[str, Any]) -> None:
    if not raw_chunk:
        return
    if not chunk.get("file_path"):
        chunk["file_path"] = raw_chunk.get("file_path", "")
    if not chunk.get("full_doc_id") and raw_chunk.get("full_doc_id"):
        chunk["full_doc_id"] = raw_chunk["full_doc_id"]


def _chunk_sidecar(chunk: dict[str, Any], raw_chunk: dict[str, Any]) -> dict[str, Any]:
    raw_sidecar = raw_chunk.get("sidecar")
    if isinstance(raw_sidecar, dict):
        return raw_sidecar
    chunk_sidecar = chunk.get("sidecar")
    return chunk_sidecar if isinstance(chunk_sidecar, dict) else {}


def _hydrate_page_index_direct(chunk: dict[str, Any], sidecar: dict[str, Any]) -> None:
    page_index = explicit_item_page_index(sidecar)
    if page_index is not None:
        chunk["page_idx"] = page_index + 1


def _hydrate_bbox_direct(chunk: dict[str, Any], sidecar: dict[str, Any]) -> None:
    bbox = explicit_item_bbox(sidecar)
    if bbox is not None:
        chunk["bbox"] = bbox


async def _provenance_from_block_sidecar(
    lightrag: Any,
    *,
    sidecar: dict[str, Any],
    chunk: dict[str, Any],
    raw_chunk: dict[str, Any],
    full_doc_cache: dict[str, dict[str, Any] | None],
    block_index_cache: dict[Path, dict[str, BlockProvenance]],
) -> BlockProvenance | None:
    block_ids = block_ids_from_sidecar(sidecar)
    if not block_ids:
        return None

    artifact_dir = await _artifact_dir_for_chunk(
        lightrag,
        chunk=chunk,
        raw_chunk=raw_chunk,
        full_doc_cache=full_doc_cache,
    )
    if artifact_dir is None or not artifact_dir.exists():
        return None

    cache_key = artifact_dir.resolve()
    if cache_key not in block_index_cache:
        block_index_cache[cache_key] = load_block_provenance_index(cache_key)
    return first_provenance_for_blocks(block_ids, block_index_cache[cache_key])


async def _artifact_dir_for_chunk(
    lightrag: Any,
    *,
    chunk: dict[str, Any],
    raw_chunk: dict[str, Any],
    full_doc_cache: dict[str, dict[str, Any] | None],
) -> Path | None:
    location = raw_chunk.get("sidecar_location") or chunk.get("sidecar_location")
    if isinstance(location, str):
        return sidecar_dir_from_location(location)

    doc_id = raw_chunk.get("full_doc_id") or chunk.get("full_doc_id")
    if not isinstance(doc_id, str) or not doc_id:
        return None

    full_doc = await _fetch_full_doc(lightrag, doc_id, full_doc_cache)
    if not isinstance(full_doc, dict):
        return None
    location = full_doc.get("sidecar_location")
    return sidecar_dir_from_location(location if isinstance(location, str) else None)


async def _fetch_full_doc(
    lightrag: Any,
    doc_id: str,
    cache: dict[str, dict[str, Any] | None],
) -> dict[str, Any] | None:
    if doc_id in cache:
        return cache[doc_id]

    store = getattr(lightrag, "full_docs", None)
    if store is None:
        cache[doc_id] = None
        return None

    try:
        result = store.get_by_id(doc_id)
        if inspect.isawaitable(result):
            result = await result
    except Exception:
        logger.debug("LightRAG full_doc provenance lookup failed", exc_info=True)
        result = None

    cache[doc_id] = result if isinstance(result, dict) else None
    return cache[doc_id]


_SIDECAR_ASSETS_MARKER = ".blocks.assets/"


def _is_sidecar_asset_path(file_path: str) -> bool:
    return _SIDECAR_ASSETS_MARKER in file_path


def _hydrate_image_data(chunk: dict[str, Any], sidecar: dict[str, Any]) -> None:
    if chunk.get("image_data"):
        return  # Already hydrated
    image_path = sidecar.get("path") or chunk.get("file_path")
    if not isinstance(image_path, str):
        return
    path = Path(image_path)
    if path.suffix.lower() not in _IMAGE_SUFFIXES or not path.exists():
        return
    chunk["image_data"] = base64.b64encode(path.read_bytes()).decode("ascii")
    chunk["image_mime_type"] = detect_image_mime_type(path)
