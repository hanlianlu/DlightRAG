# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Hydrate retrieved LightRAG chunks with display provenance."""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
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
    resolve_sidecar_asset_path,
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

        await _hydrate_image_data(
            chunk,
            sidecar,
            lightrag=lightrag,
            raw_chunk=raw_chunk,
            full_doc_cache=full_doc_cache,
        )

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


def _page_index_from_filename(stem: str) -> int | None:
    """Extract zero-based page index from a parser-generated filename stem.

    Handles patterns like ``page_1``, ``page-01``, ``p2``, ``page3_drawings``.
    """
    import re

    m = re.search(r"(?:^|[_-])p(?:age)?[_-]?(\d+)", stem, re.IGNORECASE)
    if m is None:
        return None
    return max(int(m.group(1)) - 1, 0)


def _load_sidecar_drawing_path(
    artifact_dir: Path, drawing_id: str, *, page_idx: int | None = None
) -> str | None:
    """Resolve a sidecar drawing's image path from ``*.drawings.json``.

    When *page_idx* is given, prefers the candidate whose page matches the
    chunk's page.  Parser-generated drawing IDs are often page-local
    (``im-0``, ``im-1``, …), so the same ID can appear in every page's
    drawings file.  First-match would return the wrong image for any page
    after the first.
    """
    candidates: list[tuple[int | None, str]] = []  # (item_page_idx, path)
    for drawings_path in sorted(artifact_dir.glob("*.drawings.json")):
        try:
            data = json.loads(drawings_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        drawings = data.get("drawings")
        if not isinstance(drawings, dict):
            continue
        item = drawings.get(drawing_id)
        if isinstance(item, dict):
            rel_path = _drawing_asset_path(item)
            if isinstance(rel_path, str):
                candidate = resolve_sidecar_asset_path(artifact_dir, rel_path)
                if candidate is not None:
                    item_page = explicit_item_page_index(item)
                    if item_page is None:
                        item_page = _page_index_from_filename(drawings_path.stem)
                    candidates.append((item_page, str(candidate)))
    if not candidates:
        return None
    if page_idx is not None:
        for item_page, path in candidates:
            if item_page == page_idx:
                return path
    return candidates[0][1]


def _drawing_asset_path(item: dict[str, Any]) -> str | None:
    raw = item.get("path") or item.get("img_path") or item.get("image_path")
    return raw if isinstance(raw, str) and raw.strip() else None


async def _hydrate_image_data(
    chunk: dict[str, Any],
    sidecar: dict[str, Any],
    *,
    lightrag: Any,
    raw_chunk: dict[str, Any],
    full_doc_cache: dict[str, dict[str, Any] | None],
) -> None:
    if chunk.get("image_data"):
        return  # Already hydrated

    image_path: str | None = sidecar.get("path")  # DlightRAG direct-image chunks

    # LightRAG 1.5 visual chunks: sidecar has type/id/refs but no path.
    # Resolve the image path from drawings.json in the parsed artifact directory.
    if not isinstance(image_path, str) and sidecar.get("type") == "drawing":
        drawing_id = sidecar.get("id")
        if isinstance(drawing_id, str):
            artifact_dir = await _artifact_dir_for_chunk(
                lightrag,
                chunk=chunk,
                raw_chunk=raw_chunk,
                full_doc_cache=full_doc_cache,
            )
            if artifact_dir is not None:
                image_path = await asyncio.to_thread(
                    _load_sidecar_drawing_path,
                    artifact_dir,
                    drawing_id,
                    page_idx=chunk.get("page_idx"),
                )

    if not isinstance(image_path, str):
        image_path = chunk.get("file_path")
    if not isinstance(image_path, str):
        return
    payload = await asyncio.to_thread(_image_payload_from_path, Path(image_path))
    if payload is None:
        return
    chunk["image_data"], chunk["image_mime_type"] = payload


def _image_payload_from_path(path: Path) -> tuple[str, str] | None:
    if path.suffix.lower() not in _IMAGE_SUFFIXES or not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode("ascii"), detect_image_mime_type(path)
