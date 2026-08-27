# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Hydrate retrieved LightRAG chunks with display provenance."""

import asyncio
import base64
import inspect
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lightrag.utils_pipeline import resolve_sidecar_uri

from dlightrag.engine.ai.media import detect_image_mime_type
from dlightrag.engine.rag.corpus.ingestion.sidecar_provenance import (
    BlockProvenance,
    block_ids_from_multimodal_item,
    block_ids_from_sidecar,
    explicit_item_page_number,
    first_provenance_for_blocks,
    is_multimodal_sidecar,
    load_block_provenance_index,
    resolve_sidecar_asset_path,
)

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(slots=True)
class ProvenanceCache:
    """Lookups shared across hydration passes over the same candidate set."""

    full_docs: dict[str, dict[str, Any] | None] = field(default_factory=dict)
    block_indexes: dict[Path, dict[str, BlockProvenance]] = field(default_factory=dict)
    drawings: dict[Path, list[tuple[Path, dict[str, Any]]]] = field(default_factory=dict)
    merged_chunk_ids: set[str] = field(default_factory=set)


async def hydrate_lightrag_chunk_provenance(
    stores: Any,
    chunks: list[dict[str, Any]],
    *,
    include_image_data: bool = True,
    cache: ProvenanceCache | None = None,
) -> None:
    """Hydrate page numbers and, when ``include_image_data``, image bytes.

    ``include_image_data=False`` resolves the cheap page metadata but skips the
    expensive base64 image read, so a caller can defer
    image hydration until after rerank truncation for a text-only reranker.
    Passing the same *cache* to both passes skips the second chunk-row fetch,
    since the first pass already merged those fields onto the chunks.
    """
    if not chunks:
        return

    cache = cache if cache is not None else ProvenanceCache()
    chunk_ids = [c["chunk_id"] for c in chunks]
    pending = [cid for cid in chunk_ids if cid not in cache.merged_chunk_ids]
    fetched = (
        dict(zip(pending, await _fetch_raw_chunks(stores, pending), strict=False))
        if pending
        else {}
    )
    full_doc_cache = cache.full_docs
    block_index_cache = cache.block_indexes

    for chunk in chunks:
        raw = fetched.get(chunk["chunk_id"])
        raw_chunk = raw if isinstance(raw, dict) else {}
        _merge_raw_chunk_fields(chunk, raw_chunk)
        cache.merged_chunk_ids.add(chunk["chunk_id"])

        sidecar = _chunk_sidecar(chunk, raw_chunk)
        _hydrate_page_number_direct(chunk, sidecar)
        if chunk.get("page_number") is None:
            provenance = await _provenance_from_block_sidecar(
                stores,
                sidecar=sidecar,
                chunk=chunk,
                raw_chunk=raw_chunk,
                full_doc_cache=full_doc_cache,
                block_index_cache=block_index_cache,
            )
            if provenance is not None:
                if provenance.page_number is not None:
                    chunk["page_number"] = provenance.page_number

        if include_image_data:
            await _hydrate_image_data(
                chunk,
                sidecar,
                stores=stores,
                raw_chunk=raw_chunk,
                full_doc_cache=full_doc_cache,
                drawings_cache=cache.drawings,
            )

        # Sidecar image chunks have asset paths as file_path (e.g., .blocks.assets/hash.jpg).
        # Remap to the parent document's file_path so citation grouping works correctly.
        if chunk.get("image_data") and _is_sidecar_asset_path(str(chunk.get("file_path", ""))):
            doc_id = chunk.get("full_doc_id")
            if doc_id:
                full_doc = await _fetch_full_doc(stores, doc_id, full_doc_cache)
                if full_doc and full_doc.get("file_path"):
                    chunk["file_path"] = full_doc["file_path"]


async def _fetch_raw_chunks(stores: Any, chunk_ids: list[str]) -> list[Any]:
    try:
        return await stores.get_text_chunks(chunk_ids)
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
    for key in ("sidecar", "sidecar_location"):
        if not chunk.get(key) and raw_chunk.get(key) is not None:
            chunk[key] = raw_chunk[key]


def _chunk_sidecar(chunk: dict[str, Any], raw_chunk: dict[str, Any]) -> dict[str, Any]:
    raw_sidecar = raw_chunk.get("sidecar")
    if isinstance(raw_sidecar, dict):
        return raw_sidecar
    chunk_sidecar = chunk.get("sidecar")
    return chunk_sidecar if isinstance(chunk_sidecar, dict) else {}


def _hydrate_page_number_direct(chunk: dict[str, Any], sidecar: dict[str, Any]) -> None:
    page_number = explicit_item_page_number(sidecar)
    if page_number is not None:
        chunk["page_number"] = page_number


async def _provenance_from_block_sidecar(
    stores: Any,
    *,
    sidecar: dict[str, Any],
    chunk: dict[str, Any],
    raw_chunk: dict[str, Any],
    full_doc_cache: dict[str, dict[str, Any] | None],
    block_index_cache: dict[Path, dict[str, BlockProvenance]],
) -> BlockProvenance | None:
    block_ids = block_ids_from_sidecar(sidecar)
    # Multimodal chunks (table/drawing/equation) reference their own modality
    # item id, not a block, so ``block_ids`` is empty here. Their source block
    # is recoverable from the modality sidecar file, resolved below once the
    # artifact dir is known.
    needs_item_lookup = not block_ids and is_multimodal_sidecar(sidecar)
    if not block_ids and not needs_item_lookup:
        return None

    artifact_dir = await _artifact_dir_for_chunk(
        stores,
        chunk=chunk,
        raw_chunk=raw_chunk,
        full_doc_cache=full_doc_cache,
    )
    if artifact_dir is None or not artifact_dir.exists():
        return None

    if needs_item_lookup:
        block_ids = await asyncio.to_thread(block_ids_from_multimodal_item, artifact_dir, sidecar)
        if not block_ids:
            return None

    cache_key = artifact_dir.resolve()
    if cache_key not in block_index_cache:
        block_index_cache[cache_key] = await asyncio.to_thread(
            load_block_provenance_index, cache_key
        )
    return first_provenance_for_blocks(block_ids, block_index_cache[cache_key])


async def _artifact_dir_for_chunk(
    stores: Any,
    *,
    chunk: dict[str, Any],
    raw_chunk: dict[str, Any],
    full_doc_cache: dict[str, dict[str, Any] | None],
) -> Path | None:
    location = raw_chunk.get("sidecar_location") or chunk.get("sidecar_location")
    if isinstance(location, str):
        return resolve_sidecar_uri(location)

    doc_id = raw_chunk.get("full_doc_id") or chunk.get("full_doc_id")
    if not isinstance(doc_id, str) or not doc_id:
        return None

    full_doc = await _fetch_full_doc(stores, doc_id, full_doc_cache)
    if not isinstance(full_doc, dict):
        return None
    location = full_doc.get("sidecar_location")
    return resolve_sidecar_uri(location if isinstance(location, str) else None)


async def _fetch_full_doc(
    stores: Any,
    doc_id: str,
    cache: dict[str, dict[str, Any] | None],
) -> dict[str, Any] | None:
    if doc_id in cache:
        return cache[doc_id]

    try:
        result = stores.get_full_doc(doc_id)
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


def _page_number_from_filename(stem: str) -> int | None:
    """Extract a 1-based page number from a parser-generated filename stem.

    Handles patterns like ``page_1``, ``page-01``, ``p2``, ``page3_drawings``.
    """
    import re

    m = re.search(r"(?:^|[_-])p(?:age)?[_-]?(\d+)", stem, re.IGNORECASE)
    if m is None:
        return None
    page_number = int(m.group(1))
    return page_number if page_number >= 1 else None


def _load_drawings_files(artifact_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    files: list[tuple[Path, dict[str, Any]]] = []
    for drawings_path in sorted(artifact_dir.glob("*.drawings.json")):
        try:
            data = json.loads(drawings_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError, OSError:
            continue
        if not isinstance(data, dict):
            continue
        drawings = data.get("drawings")
        if isinstance(drawings, dict):
            files.append((drawings_path, drawings))
    return files


def _load_sidecar_drawing_path(
    artifact_dir: Path,
    drawing_id: str,
    *,
    page_number: int | None = None,
    drawings_cache: dict[Path, list[tuple[Path, dict[str, Any]]]],
) -> str | None:
    """Resolve a sidecar drawing's image path from ``*.drawings.json``.

    When *page_number* is given, prefers the candidate whose page matches the
    chunk's page.  Parser-generated drawing IDs are often page-local
    (``im-0``, ``im-1``, …), so the same ID can appear in every page's
    drawings file.  First-match would return the wrong image for any page
    after the first.
    """
    cache_key = artifact_dir.resolve()
    if cache_key not in drawings_cache:
        drawings_cache[cache_key] = _load_drawings_files(artifact_dir)

    candidates: list[tuple[int | None, str]] = []
    for drawings_path, drawings in drawings_cache[cache_key]:
        item = drawings.get(drawing_id)
        if isinstance(item, dict):
            rel_path = _drawing_asset_path(item)
            if isinstance(rel_path, str):
                candidate = resolve_sidecar_asset_path(artifact_dir, rel_path)
                if candidate is not None:
                    item_page = explicit_item_page_number(item)
                    if item_page is None:
                        item_page = _page_number_from_filename(drawings_path.stem)
                    candidates.append((item_page, str(candidate)))
    if not candidates:
        return None
    if page_number is not None:
        for item_page, path in candidates:
            if item_page == page_number:
                return path
    return candidates[0][1]


def _drawing_asset_path(item: dict[str, Any]) -> str | None:
    raw = item.get("path") or item.get("img_path") or item.get("image_path")
    return raw if isinstance(raw, str) and raw.strip() else None


async def _hydrate_image_data(
    chunk: dict[str, Any],
    sidecar: dict[str, Any],
    *,
    stores: Any,
    raw_chunk: dict[str, Any],
    full_doc_cache: dict[str, dict[str, Any] | None],
    drawings_cache: dict[Path, list[tuple[Path, dict[str, Any]]]],
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
                stores,
                chunk=chunk,
                raw_chunk=raw_chunk,
                full_doc_cache=full_doc_cache,
            )
            if artifact_dir is not None:
                image_path = await asyncio.to_thread(
                    _load_sidecar_drawing_path,
                    artifact_dir,
                    drawing_id,
                    page_number=chunk.get("page_number"),
                    drawings_cache=drawings_cache,
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
