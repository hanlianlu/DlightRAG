# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""LightRAG deletion helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from inspect import isawaitable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# _compute_mdhash_id is imported lazily in cascade_delete to avoid a hard
# dependency on lightrag at module import time.


@dataclass
class DeletionContext:
    """Aggregated deletion context from available data sources."""

    identifier: str  # Original filename/path requested for deletion
    doc_ids: set[str] = field(default_factory=set)
    file_paths: set[str] = field(default_factory=set)
    sources_used: list[str] = field(default_factory=list)  # For audit trail


async def collect_deletion_context(
    identifier: str,
    lightrag: Any = None,
    metadata_index: Any = None,
) -> DeletionContext:
    """Find doc_id(s) for a file using LightRAG doc_status and metadata index.

    Strategy order:
    1. LightRAG doc_status - storage-agnostic lookup by file_path
    2. Metadata index - filename-based lookup (PGMetadataIndex)
    """
    ctx = DeletionContext(identifier=identifier)
    basename = Path(identifier).name
    stem = Path(identifier).stem

    # Strategy 1: LightRAG doc_status lookup (storage-agnostic)
    if lightrag and hasattr(lightrag, "doc_status"):
        doc_status = lightrag.doc_status
        try:
            # Try exact path match first
            result = await doc_status.get_doc_by_file_path(identifier)

            # get_doc_by_file_path may return a dict without 'id' field
            # (LightRAG strips it).  Always extract file_path from it if present.
            if isinstance(result, dict):
                fp = result.get("file_path", "")
                if fp:
                    ctx.file_paths.add(fp)

            # get_doc_by_file_path omits the primary-key id, so we must
            # scan docs to find the doc_id.  This runs whenever doc_ids is
            # still empty — for both relative and absolute identifiers.
            # We also always scan FAILED docs to collect DUPLICATE records
            # that share the same file_path, so the receipt is cleaned up
            # alongside the original document.
            if not ctx.doc_ids:
                from lightrag.base import DocStatus

                all_docs = await doc_status.get_docs_by_status(DocStatus.PROCESSED)
                for d_id, doc_info in all_docs.items():
                    fp = getattr(doc_info, "file_path", "") or ""
                    stored_name = Path(fp).name
                    stored_stem = Path(fp).stem

                    if fp == identifier or stored_name == basename or stored_stem == stem:
                        ctx.doc_ids.add(d_id)
                        if fp:
                            ctx.file_paths.add(fp)

            # Collect DUPLICATE/FAILED receipts for the same file so they
            # are removed together with the original document.
            from lightrag.base import DocStatus

            failed_docs = await doc_status.get_docs_by_status(DocStatus.FAILED)
            for d_id, doc_info in failed_docs.items():
                fp = getattr(doc_info, "file_path", "") or ""
                stored_name = Path(fp).name
                stored_stem = Path(fp).stem

                if fp == identifier or stored_name == basename or stored_stem == stem:
                    ctx.doc_ids.add(d_id)
                    if fp:
                        ctx.file_paths.add(fp)

            if ctx.doc_ids or ctx.file_paths:
                ctx.sources_used.append("doc_status")

        except Exception as e:
            logger.warning(f"LightRAG doc_status lookup failed for {identifier}: {e}")

    # Strategy 2: Metadata index lookup (PGMetadataIndex by filename)
    if metadata_index and not ctx.doc_ids:
        try:
            doc_ids: list[str] = []
            find_by_file_path = getattr(metadata_index, "find_by_file_path", None)
            if callable(find_by_file_path):
                maybe_doc_ids = find_by_file_path(identifier)
                if isawaitable(maybe_doc_ids):
                    maybe_doc_ids = await maybe_doc_ids
                if isinstance(maybe_doc_ids, list):
                    doc_ids = [str(doc_id) for doc_id in maybe_doc_ids]
            if not doc_ids:
                doc_ids = await metadata_index.find_by_filename(basename)
            for d_id in doc_ids:
                ctx.doc_ids.add(d_id)
            if doc_ids and lightrag and hasattr(lightrag, "doc_status"):
                try:
                    await _add_doc_status_file_paths_for_doc_ids(ctx, lightrag.doc_status)
                except Exception as exc:
                    logger.debug(
                        "Best-effort doc_status file path hydration failed: %s",
                        exc,
                        exc_info=True,
                    )
            if doc_ids:
                ctx.sources_used.append("metadata_index")
        except Exception as e:
            logger.warning(f"Metadata index lookup failed for {identifier}: {e}")

    logger.info(
        f"Deletion context for {identifier}: "
        f"doc_ids={len(ctx.doc_ids)}, "
        f"file_paths={len(ctx.file_paths)}, sources={ctx.sources_used}"
    )

    return ctx


async def cascade_delete(
    ctx: DeletionContext,
    lightrag: Any,
    metadata_index: Any | None = None,
) -> dict[str, Any]:
    """Cascade deletion with per-layer fault isolation.

    Each layer is wrapped in try/except so failures in one layer don't
    prevent cleanup in subsequent layers.

    Layers:
        1. LightRAG cross-backend cleanup (adelete_by_doc_id)
        2. DlightRAG metadata index entries
    """
    stats: dict[str, Any] = {"docs_deleted": 0, "errors": []}

    for doc_id in ctx.doc_ids:
        # Layer 1: LightRAG (full_docs, doc_status, text_chunks, chunks_vdb, KG)
        try:
            await lightrag.adelete_by_doc_id(doc_id, delete_llm_cache=True)
            stats["docs_deleted"] += 1
        except Exception as exc:
            stats["errors"].append(f"Layer 1 LightRAG ({doc_id}): {exc}")
            logger.warning("cascade_delete Layer 1 failed for %s: %s", doc_id, exc)

        # Layer 2: DlightRAG metadata index
        if metadata_index is not None:
            try:
                await metadata_index.delete(doc_id)
            except Exception as exc:
                stats["errors"].append(f"Layer 2 metadata ({doc_id}): {exc}")
                logger.warning("cascade_delete Layer 2 failed for %s: %s", doc_id, exc)

    return stats


def remove_deleted_files(file_paths: set[str], input_dir: str) -> int:
    """Delete physical files and parsed artifact directories from disk.

    Handles the full LightRAG parser artifact layout:

    - Source files in ``input_dir/``
    - Parsed artifacts under ``input_dir/__parsed__/``:
      ``<name>.parsed/``, ``<name>.mineru_raw/``, ``<name>.docling_raw/``
    - Collision-suffixed variants (``<name>_001.parsed/``, etc.)

    Best-effort — failures are logged but never raised, so a missing file
    on disk does not block the DB-level deletion from succeeding.

    Args:
        file_paths: Absolute paths to ingested files (from LightRAG doc_status).
        input_dir: The workspace's input directory (parent of the source files).

    Returns:
        Number of files/directories removed.
    """
    import re
    import shutil

    from lightrag.constants import PARSED_ARTIFACT_DIR_SUFFIXES, PARSED_DIR_NAME

    removed = 0
    input_root = Path(input_dir)
    default_parsed_root = input_root / PARSED_DIR_NAME
    _collision_re = re.compile(r"_\d{3}$")

    for fp in file_paths:
        if _is_remote_source_path(fp):
            continue
        path = Path(fp)
        filename = path.name
        stem = path.stem
        source_root = _source_root_for_stored_path(path, input_root)
        parsed_roots = [source_root / PARSED_DIR_NAME]
        if default_parsed_root not in parsed_roots:
            parsed_roots.append(default_parsed_root)

        # 1. Remove the source file (may be in input_dir/ or moved into
        #    __parsed__/ by LightRAG after ingest).
        source_candidates = [source_root / filename]
        if path.is_absolute():
            source_candidates.insert(0, path)
        source_candidates.extend(parsed_root / filename for parsed_root in parsed_roots)
        for candidate in dict.fromkeys(source_candidates):
            try:
                if candidate.exists() and candidate.is_file():
                    candidate.unlink()
                    removed += 1
            except OSError:
                logger.debug("Failed to remove source file: %s", candidate, exc_info=True)

        # 2. Remove parsed artifact directories under __parsed__/.
        #    LightRAG creates:  <stem>.parsed/, <stem>.mineru_raw/,
        #    <stem>.docling_raw/, plus collision-suffixed variants
        #    (<stem>_001.parsed/, etc.).
        for parsed_root in parsed_roots:
            try:
                if not parsed_root.exists() or not parsed_root.is_dir():
                    continue
                for entry in sorted(parsed_root.iterdir()):
                    if not entry.is_dir():
                        continue
                    entry_name = entry.name
                    for suffix in PARSED_ARTIFACT_DIR_SUFFIXES:
                        expected = f"{stem}{suffix}"
                        if entry_name == expected or (
                            entry_name.endswith(suffix)
                            and _collision_re.sub("", entry_name[: -len(suffix)]) == stem
                        ):
                            shutil.rmtree(entry, ignore_errors=True)
                            removed += 1
                            break
            except OSError:
                logger.debug("Failed to scan parsed dir: %s", parsed_root, exc_info=True)

    return removed


async def _add_doc_status_file_paths_for_doc_ids(ctx: DeletionContext, doc_status: Any) -> None:
    from lightrag.base import DocStatus

    for status in (DocStatus.PROCESSED, DocStatus.FAILED):
        maybe_docs = doc_status.get_docs_by_status(status)
        if isawaitable(maybe_docs):
            maybe_docs = await maybe_docs
        if not isinstance(maybe_docs, dict):
            continue
        docs = maybe_docs
        for doc_id in ctx.doc_ids:
            doc_info = docs.get(doc_id)
            fp = getattr(doc_info, "file_path", "") if doc_info is not None else ""
            if fp:
                ctx.file_paths.add(fp)


def _is_remote_source_path(path: str) -> bool:
    return path.startswith(("azure://", "s3://"))


def _source_root_for_stored_path(path: Path, input_root: Path) -> Path:
    if path.is_absolute():
        try:
            path.resolve().relative_to(input_root.resolve())
        except ValueError:
            return input_root
        return path.parent
    if len(path.parts) > 1:
        return input_root / Path(*path.parts[:-1])
    return input_root


__all__ = [
    "DeletionContext",
    "cascade_delete",
    "collect_deletion_context",
    "remove_deleted_files",
]
