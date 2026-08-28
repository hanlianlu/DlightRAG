# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Hydrate retrieved LightRAG chunks with display provenance."""

import asyncio
import base64
import inspect
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lightrag.utils_pipeline import resolve_sidecar_uri

from dlightrag.engine.ai.media import detect_image_mime_type
from dlightrag.engine.rag.corpus.ingestion.sidecar_provenance import (
    SidecarArtifactIndex,
    block_ids_from_sidecar,
    explicit_item_page_number,
    first_provenance_for_blocks,
    is_multimodal_sidecar,
)

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
_FILESYSTEM_CONCURRENCY = 8
_SIDECAR_ASSETS_MARKER = ".blocks.assets/"


@dataclass(slots=True)
class ProvenanceCache:
    """Request-local lookups shared across candidate hydration passes."""

    raw_chunks: dict[str, dict[str, Any] | None] = field(default_factory=dict)
    full_docs: dict[str, dict[str, Any] | None] = field(default_factory=dict)
    artifact_indexes: dict[Path, SidecarArtifactIndex | None] = field(default_factory=dict)
    image_paths: dict[str, Path | None] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _ChunkHydrationPlan:
    chunk: dict[str, Any]
    sidecar: dict[str, Any]
    artifact_dir: Path | None
    full_doc: dict[str, Any] | None


async def hydrate_lightrag_chunk_provenance(
    stores: Any,
    chunks: list[dict[str, Any]],
    *,
    include_image_data: bool = True,
    cache: ProvenanceCache | None = None,
) -> None:
    """Hydrate page numbers and optional image bytes without changing chunk order.

    Reusing *cache* for the cheap pre-rerank and payload post-rerank passes avoids
    refetching chunk/full-document rows and reparsing artifact directories. The
    cheap pass resolves image paths but never reads image bytes.
    """
    if not chunks:
        return

    cache = cache if cache is not None else ProvenanceCache()
    await _prime_raw_chunk_cache(stores, chunks, cache)
    await _merge_raw_chunks(chunks, cache)
    await _prime_full_doc_cache(stores, chunks, cache)

    plans = [_plan_for_chunk(chunk, cache) for chunk in chunks]
    artifact_dirs = list(
        dict.fromkeys(plan.artifact_dir for plan in plans if plan.artifact_dir is not None)
    )
    await _prime_artifact_index_cache(artifact_dirs, cache)

    async def _hydrate(plan: _ChunkHydrationPlan) -> None:
        await _hydrate_chunk(plan, include_image_data=include_image_data, cache=cache)

    await _run_bounded(plans, _hydrate)


async def _prime_raw_chunk_cache(
    stores: Any,
    chunks: list[dict[str, Any]],
    cache: ProvenanceCache,
) -> None:
    pending = list(
        dict.fromkeys(
            chunk["chunk_id"] for chunk in chunks if chunk["chunk_id"] not in cache.raw_chunks
        )
    )
    if not pending:
        return
    fetched = await _fetch_raw_chunks(stores, pending)
    for index, chunk_id in enumerate(pending):
        raw = fetched[index] if index < len(fetched) else None
        cache.raw_chunks[chunk_id] = raw if isinstance(raw, dict) else None


async def _merge_raw_chunks(
    chunks: list[dict[str, Any]],
    cache: ProvenanceCache,
) -> None:
    async def _merge(chunk: dict[str, Any]) -> None:
        chunk_id = chunk["chunk_id"]
        raw_chunk = cache.raw_chunks.get(chunk_id) or {}
        _merge_raw_chunk_fields(chunk, raw_chunk)

    await _run_bounded(chunks, _merge)


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


async def _prime_full_doc_cache(
    stores: Any,
    chunks: list[dict[str, Any]],
    cache: ProvenanceCache,
) -> None:
    pending: list[str] = []
    for chunk in chunks:
        raw_chunk = cache.raw_chunks.get(chunk["chunk_id"]) or {}
        doc_id = _full_doc_id(chunk, raw_chunk)
        if doc_id is not None and doc_id not in cache.full_docs and doc_id not in pending:
            pending.append(doc_id)
    if not pending:
        return

    try:
        result = stores.get_full_docs(pending)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, list):
            raise TypeError("LightRAG get_full_docs() did not return a list")
    except Exception:
        logger.debug(
            "LightRAG full_doc batch provenance lookup failed; falling back to singular reads",
            exc_info=True,
        )
        result = await _fallback_full_doc_reads(stores, pending)

    for index, doc_id in enumerate(pending):
        full_doc = result[index] if index < len(result) else None
        cache.full_docs[doc_id] = full_doc if isinstance(full_doc, dict) else None


async def _fallback_full_doc_reads(
    stores: Any,
    doc_ids: list[str],
) -> list[dict[str, Any] | None]:
    async def _fetch(doc_id: str) -> dict[str, Any] | None:
        try:
            result = stores.get_full_doc(doc_id)
            if inspect.isawaitable(result):
                result = await result
        except Exception:
            logger.debug("LightRAG full_doc provenance lookup failed", exc_info=True)
            return None
        return result if isinstance(result, dict) else None

    return await _run_bounded(doc_ids, _fetch)


def _full_doc_id(chunk: dict[str, Any], raw_chunk: dict[str, Any]) -> str | None:
    doc_id = raw_chunk.get("full_doc_id") or chunk.get("full_doc_id")
    return doc_id if isinstance(doc_id, str) and doc_id else None


def _plan_for_chunk(chunk: dict[str, Any], cache: ProvenanceCache) -> _ChunkHydrationPlan:
    raw_chunk = cache.raw_chunks.get(chunk["chunk_id"]) or {}
    sidecar = _chunk_sidecar(chunk, raw_chunk)
    doc_id = _full_doc_id(chunk, raw_chunk)
    full_doc = cache.full_docs.get(doc_id) if doc_id is not None else None
    artifact_dir = _artifact_dir_for_chunk(chunk, raw_chunk, full_doc)
    return _ChunkHydrationPlan(
        chunk=chunk,
        sidecar=sidecar,
        artifact_dir=artifact_dir,
        full_doc=full_doc,
    )


def _chunk_sidecar(chunk: dict[str, Any], raw_chunk: dict[str, Any]) -> dict[str, Any]:
    raw_sidecar = raw_chunk.get("sidecar")
    if isinstance(raw_sidecar, dict):
        return raw_sidecar
    chunk_sidecar = chunk.get("sidecar")
    return chunk_sidecar if isinstance(chunk_sidecar, dict) else {}


def _artifact_dir_for_chunk(
    chunk: dict[str, Any],
    raw_chunk: dict[str, Any],
    full_doc: dict[str, Any] | None,
) -> Path | None:
    location = raw_chunk.get("sidecar_location") or chunk.get("sidecar_location")
    if not isinstance(location, str) and isinstance(full_doc, dict):
        location = full_doc.get("sidecar_location")
    try:
        artifact_dir = resolve_sidecar_uri(location if isinstance(location, str) else None)
        return artifact_dir.resolve() if artifact_dir is not None else None
    except Exception:
        logger.debug("LightRAG sidecar location resolution failed", exc_info=True)
        return None


async def _prime_artifact_index_cache(
    artifact_dirs: list[Path],
    cache: ProvenanceCache,
) -> None:
    pending = [path for path in artifact_dirs if path not in cache.artifact_indexes]

    async def _load(path: Path) -> SidecarArtifactIndex | None:
        try:
            return await asyncio.to_thread(_load_sidecar_artifact_index, path)
        except Exception:
            logger.debug("LightRAG sidecar artifact index loading failed", exc_info=True)
            return None

    indexes = await _run_bounded(pending, _load)
    cache.artifact_indexes.update(zip(pending, indexes, strict=True))


def _load_sidecar_artifact_index(artifact_dir: Path) -> SidecarArtifactIndex:
    """Filesystem seam for one complete artifact-directory parse."""
    return SidecarArtifactIndex.load(artifact_dir)


async def _hydrate_chunk(
    plan: _ChunkHydrationPlan,
    *,
    include_image_data: bool,
    cache: ProvenanceCache,
) -> None:
    chunk = plan.chunk
    _hydrate_page_number_direct(chunk, plan.sidecar)
    artifact_index = (
        cache.artifact_indexes.get(plan.artifact_dir) if plan.artifact_dir is not None else None
    )
    if chunk.get("page_number") is None and artifact_index is not None:
        block_ids = block_ids_from_sidecar(plan.sidecar)
        if not block_ids and is_multimodal_sidecar(plan.sidecar):
            block_ids = artifact_index.block_ids_for_multimodal_item(plan.sidecar)
        provenance = first_provenance_for_blocks(block_ids, artifact_index.block_provenance)
        if provenance is not None and provenance.page_number is not None:
            chunk["page_number"] = provenance.page_number

    chunk_id = chunk["chunk_id"]
    if chunk_id not in cache.image_paths:
        cache.image_paths[chunk_id] = _resolve_image_path(plan, artifact_index)

    if include_image_data and not chunk.get("image_data"):
        image_path = cache.image_paths[chunk_id]
        if image_path is not None:
            payload = await _read_image_payload(image_path)
            if payload is not None:
                chunk["image_data"], chunk["image_mime_type"] = payload

    # Canonical source identity is independent of whether image bytes were read.
    if _is_sidecar_asset_path(str(chunk.get("file_path", ""))):
        if plan.full_doc and plan.full_doc.get("file_path"):
            chunk["file_path"] = plan.full_doc["file_path"]


def _hydrate_page_number_direct(chunk: dict[str, Any], sidecar: dict[str, Any]) -> None:
    page_number = explicit_item_page_number(sidecar)
    if page_number is not None:
        chunk["page_number"] = page_number


def _resolve_image_path(
    plan: _ChunkHydrationPlan,
    artifact_index: SidecarArtifactIndex | None,
) -> Path | None:
    chunk = plan.chunk
    if chunk.get("image_data"):
        return None

    image_path: str | Path | None = plan.sidecar.get("path")
    if not isinstance(image_path, str) and plan.sidecar.get("type") == "drawing":
        drawing_id = plan.sidecar.get("id")
        if isinstance(drawing_id, str) and artifact_index is not None:
            image_path = artifact_index.drawing_asset_path(
                drawing_id,
                page_number=chunk.get("page_number"),
            )

    if not isinstance(image_path, str | Path):
        image_path = chunk.get("file_path")
    if not isinstance(image_path, str | Path):
        return None
    path = Path(image_path)
    return path if path.suffix.lower() in _IMAGE_SUFFIXES else None


async def _read_image_payload(path: Path) -> tuple[str, str] | None:
    try:
        return await asyncio.to_thread(_image_payload_from_path, path)
    except Exception:
        logger.debug("LightRAG provenance image read failed", exc_info=True)
        return None


def _is_sidecar_asset_path(file_path: str) -> bool:
    return _SIDECAR_ASSETS_MARKER in file_path


def _image_payload_from_path(path: Path) -> tuple[str, str] | None:
    if path.suffix.lower() not in _IMAGE_SUFFIXES or not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode("ascii"), detect_image_mime_type(path)


async def _run_bounded[T, R](
    items: Sequence[T],
    worker: Callable[[T], Awaitable[R]],
) -> list[R]:
    """Run at most the private filesystem cap tasks at once, preserving order."""

    async def _invoke(item: T) -> R:
        return await worker(item)

    results: list[R] = []
    for start in range(0, len(items), _FILESYSTEM_CONCURRENCY):
        batch = items[start : start + _FILESYSTEM_CONCURRENCY]
        async with asyncio.TaskGroup() as task_group:
            tasks = [task_group.create_task(_invoke(item)) for item in batch]
        results.extend(task.result() for task in tasks)
    return results
