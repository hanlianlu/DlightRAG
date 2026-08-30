# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""LightRAG deletion helpers."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dlightrag.engine.rag.corpus.contracts import DocStatusLookup
from dlightrag.engine.rag.corpus.metadata_index import MetadataIndexProtocol

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
    *,
    metadata_index: MetadataIndexProtocol | None,
    doc_status_lookup: DocStatusLookup | None,
) -> DeletionContext:
    """Resolve one deletion identifier through bounded indexed reads only.

    Identifiers are exact source locators, stored filenames, or status
    ``file_path`` values. Bare stems are intentionally not expanded because
    that is ambiguous and previously required corpus-wide status scans.
    """
    ctx = DeletionContext(identifier=identifier)
    normalized = str(identifier).strip()
    if not normalized:
        return ctx
    basename = Path(normalized).name

    metadata_lookup_failed = False
    status_lookup_failed = False

    # An exact durable locator owns identity. A filename fallback is permitted
    # only when the caller supplied a bare display name and exact resolution
    # completed without error; failures must never widen deletion identity.
    if metadata_index is not None:
        try:
            ctx.doc_ids.update(await metadata_index.find_by_download_locator(normalized))
            if ctx.doc_ids:
                ctx.sources_used.append("metadata_index")
        except Exception as exc:
            metadata_lookup_failed = True
            logger.warning("Metadata index lookup failed for %s: %s", identifier, exc)

    async def merge_status_matches(*, file_paths: tuple[str, ...]) -> None:
        nonlocal status_lookup_failed
        if doc_status_lookup is None:
            return
        try:
            matches = await doc_status_lookup.resolve_deletion_matches(
                file_paths=file_paths,
                doc_ids=tuple(sorted(ctx.doc_ids)),
            )
        except Exception as exc:
            status_lookup_failed = True
            logger.warning("Document status lookup failed for %s: %s", identifier, exc)
            return
        for match in matches:
            ctx.doc_ids.add(match.doc_id)
            if match.file_path:
                ctx.file_paths.add(match.file_path)
        if matches and "doc_status" not in ctx.sources_used:
            ctx.sources_used.append("doc_status")

    await merge_status_matches(file_paths=(normalized,))

    if not ctx.doc_ids and normalized == basename and not metadata_lookup_failed:
        if metadata_index is not None:
            try:
                ctx.doc_ids.update(await metadata_index.find_by_filename(basename))
                if ctx.doc_ids:
                    ctx.sources_used.append("metadata_index")
            except Exception as exc:
                metadata_lookup_failed = True
                logger.warning("Metadata filename lookup failed for %s: %s", identifier, exc)
        if not metadata_lookup_failed and not status_lookup_failed:
            await merge_status_matches(file_paths=(basename,))

    # A metadata locator can differ from the status file_path. Expand once on
    # the exact hydrated paths to include duplicate receipts sharing identity.
    duplicate_paths = tuple(sorted(ctx.file_paths.difference({normalized})))
    if duplicate_paths and not status_lookup_failed:
        await merge_status_matches(file_paths=duplicate_paths)

    logger.info(
        "Deletion context for %s: doc_ids=%d, file_paths=%d, sources=%s",
        identifier,
        len(ctx.doc_ids),
        len(ctx.file_paths),
        ctx.sources_used,
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


def _is_remote_source_path(path: str) -> bool:
    return path.startswith(("azure://", "s3://", "https://"))


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
