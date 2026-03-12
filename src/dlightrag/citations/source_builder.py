"""Shared utility for building document-level SourceReferences from contexts.

Groups chunks by reference_id into SourceReference objects with ChunkSnippets.
Called by both web routes and API server after retrieval.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .schemas import ChunkSnippet, SourceReference
from .utils import filter_content_for_display

if TYPE_CHECKING:
    from dlightrag.core.retrieval.path_resolver import PathResolver


def build_sources(
    contexts: dict[str, Any],
    path_resolver: PathResolver | None = None,
) -> list[SourceReference]:
    """Group chunks by reference_id into document-level SourceReferences.

    Args:
        contexts: Retrieval contexts dict with "chunks" key.
        path_resolver: Optional PathResolver with .resolve(path) -> url method.

    Returns:
        List of SourceReference, ordered by first chunk appearance.
        Chunks within each source are sorted by page_idx.
    """
    chunks = contexts.get("chunks", [])
    if not chunks:
        return []

    # Group by reference_id, preserving first-appearance order
    groups: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        ref_id = str(chunk.get("reference_id", ""))
        if not ref_id:
            continue
        groups.setdefault(ref_id, []).append(chunk)

    sources: list[SourceReference] = []
    for ref_id, group in groups.items():
        first = group[0]
        file_path = first.get("file_path", "")
        metadata = first.get("metadata", {})

        # Sort by page_idx (None last)
        sorted_chunks = sorted(group, key=lambda c: (c.get("page_idx") is None, c.get("page_idx") or 0))

        snippets = [
            ChunkSnippet(
                chunk_id=c["chunk_id"],
                chunk_idx=idx,
                page_idx=c.get("page_idx"),
                content=filter_content_for_display(c.get("content", "")),
                image_data=c.get("image_data"),
            )
            for idx, c in enumerate(sorted_chunks)
        ]

        url = None
        if path_resolver and file_path:
            try:
                url = path_resolver.resolve(file_path)
            except Exception:
                pass

        sources.append(
            SourceReference(
                id=ref_id,
                title=metadata.get("file_name") or (Path(file_path).name if file_path else None),
                path=file_path,
                type=metadata.get("file_type"),
                url=url,
                chunks=snippets,
            )
        )

    return sources
