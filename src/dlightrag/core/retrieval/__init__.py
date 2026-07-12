# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared retrieval types and utilities."""

from typing import Any

from dlightrag.core.retrieval.models import MetadataFilter
from dlightrag.core.retrieval.protocols import (
    ChunkContext,
    EntityContext,
    RelationshipContext,
    RetrievalBackend,
    RetrievalContexts,
    RetrievalResult,
)
from dlightrag.core.retrieval.source_links import SourceDownloadLinkBuilder


def canonicalize_reference_ids(
    chunks: list[dict[str, Any]],
    *,
    references: list[dict[str, Any]] | None = None,
    federated: bool = False,
) -> list[dict[str, Any]]:
    """Assign stable document-level ``reference_id`` values to chunks.

    LightRAG ``aquery_data`` already returns a document reference list. When
    provided, that list is the seed for doc-level citations. DlightRAG then
    extends it for chunks introduced after LightRAG retrieval, such as BM25 or
    direct visual hits.

    Re-running on chunks that already have reference_ids is safe: existing
    mappings are preserved and only empty slots are filled.

    With ``federated=True``, chunks are keyed by ``(_workspace, file_path)``
    so cross-workspace duplicates (two workspaces both ingesting different
    docs that happen to share a filename) get distinct reference_ids.
    """
    if not chunks:
        return chunks

    if not federated:
        return _canonicalize_single_workspace(chunks, references=references)

    from lightrag.utils import generate_reference_list_from_chunks

    # \x00 is a sentinel separator never present in real file paths.
    proxy: list[dict[str, Any]] = []
    for c in chunks:
        c2 = dict(c)
        fp = c2.get("file_path", "")
        ws = c2.get("_workspace", "")
        if fp:
            c2["file_path"] = f"\x00{ws}\x00{fp}"
        proxy.append(c2)

    _, updated = generate_reference_list_from_chunks(proxy)
    # Restore original file_path; reference_id is the new federated value.
    for u, original in zip(updated, chunks, strict=True):
        u["file_path"] = original.get("file_path", "")
    return updated


def _canonicalize_single_workspace(
    chunks: list[dict[str, Any]],
    *,
    references: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    updated = [dict(chunk) for chunk in chunks]

    path_to_ref: dict[str, str] = {}
    used_refs: set[str] = set()

    for ref in references or []:
        file_path = str(ref.get("file_path") or "")
        ref_id = str(ref.get("reference_id") or "")
        if file_path and ref_id and file_path not in path_to_ref:
            path_to_ref[file_path] = ref_id
            used_refs.add(ref_id)

    for chunk in updated:
        file_path = str(chunk.get("file_path") or "")
        ref_id = str(chunk.get("reference_id") or "")
        if file_path and ref_id and file_path not in path_to_ref:
            path_to_ref[file_path] = ref_id
            used_refs.add(ref_id)

    missing_paths = _ordered_missing_paths(updated, known_paths=set(path_to_ref))
    next_ref = _next_reference_id(used_refs)
    for file_path in missing_paths:
        while str(next_ref) in used_refs:
            next_ref += 1
        ref_id = str(next_ref)
        path_to_ref[file_path] = ref_id
        used_refs.add(ref_id)
        next_ref += 1

    for chunk in updated:
        file_path = str(chunk.get("file_path") or "")
        if file_path and file_path in path_to_ref:
            chunk["reference_id"] = path_to_ref[file_path]

    return updated


def _ordered_missing_paths(
    chunks: list[dict[str, Any]],
    *,
    known_paths: set[str],
) -> list[str]:
    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    for idx, chunk in enumerate(chunks):
        file_path = str(chunk.get("file_path") or "")
        if not file_path or file_path == "unknown_source" or file_path in known_paths:
            continue
        counts[file_path] = counts.get(file_path, 0) + 1
        first_seen.setdefault(file_path, idx)

    return sorted(counts, key=lambda path: (-counts[path], first_seen[path]))


def _next_reference_id(used_refs: set[str]) -> int:
    numeric = [int(ref) for ref in used_refs if ref.isdigit()]
    return max(numeric, default=0) + 1


__all__ = [
    "ChunkContext",
    "EntityContext",
    "MetadataFilter",
    "RelationshipContext",
    "RetrievalBackend",
    "RetrievalContexts",
    "RetrievalResult",
    "SourceDownloadLinkBuilder",
    "canonicalize_reference_ids",
]
