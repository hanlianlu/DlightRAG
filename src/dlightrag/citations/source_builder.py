"""Shared utility for building document-level SourceReferences from contexts.

Groups chunks by reference_id into SourceReference objects with ChunkSnippets.
Called by both web routes and API server after retrieval.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dlightrag.core.retrieval.path_resolver import PathResolver
from dlightrag.core.retrieval.protocols import RetrievalContexts

from .indexer import CitationIndexer
from .schemas import ChunkSnippet, SourceReference
from .utils import filter_content_for_display


def build_sources(
    contexts: RetrievalContexts,
    path_resolver: PathResolver | None = None,
    *,
    indexer: CitationIndexer | None = None,
) -> list[SourceReference]:
    """Group chunks by reference_id into document-level SourceReferences.

    Args:
        contexts: Retrieval contexts dict with "chunks" key.
        path_resolver: Optional PathResolver with .resolve(path) -> url method.

    Returns:
        List of SourceReference, ordered by first chunk appearance.
        Chunks within each source follow citation-index order.
    """
    return build_sources_from_chunks(
        contexts.get("chunks", []),
        path_resolver=path_resolver,
        indexer=indexer,
    )


def build_sources_from_chunks(
    chunks: list[dict[str, Any]],
    path_resolver: PathResolver | None = None,
    *,
    indexer: CitationIndexer | None = None,
    cited_chunks: dict[str, list[str]] | None = None,
    source_catalog: list[SourceReference] | None = None,
) -> list[SourceReference]:
    """Project chunk contexts into document sources using citation index order."""
    if not chunks:
        return []

    if indexer is None:
        indexer = CitationIndexer()
        indexer.build_index(chunks)
    chunks = indexer.inject_chunk_idx(chunks)

    catalog_by_id = {source.id: source for source in source_catalog or []}
    allowed_by_ref = {ref_id: set(chunk_ids) for ref_id, chunk_ids in (cited_chunks or {}).items()}

    # Group by reference_id, preserving first-appearance order
    groups: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        ref_id = str(chunk.get("reference_id", ""))
        if not ref_id:
            continue
        if cited_chunks is not None:
            allowed = allowed_by_ref.get(ref_id, set())
            if not allowed or chunk.get("chunk_id") not in allowed:
                continue
        groups.setdefault(ref_id, []).append(chunk)

    sources: list[SourceReference] = []
    ref_ids = list(cited_chunks) if cited_chunks is not None else list(groups)
    for ref_id in ref_ids:
        group = groups.get(ref_id, [])
        if not group:
            continue
        first = group[0]
        metadata = first.get("metadata", {})
        catalog = catalog_by_id.get(ref_id)
        file_path = catalog.path if catalog else first.get("file_path", "")

        def _citation_order(chunk: dict[str, Any], ref_id: str = ref_id) -> int:
            chunk_idx = chunk.get("chunk_idx")
            if isinstance(chunk_idx, int):
                return chunk_idx
            resolved = indexer.get_chunk_idx(ref_id, chunk.get("chunk_id", ""))
            return resolved if resolved is not None else 10**9

        ordered_chunks = sorted(group, key=_citation_order)

        snippets = [
            ChunkSnippet(
                chunk_id=c["chunk_id"],
                chunk_idx=c.get("chunk_idx") or indexer.get_chunk_idx(ref_id, c["chunk_id"]),
                page_idx=c.get("page_idx")
                if c.get("page_idx") is not None
                else (c.get("metadata", {}) or {}).get("page_idx"),
                content=filter_content_for_display(c.get("content", "")),
                image_data=c.get("image_data"),
            )
            for c in ordered_chunks
        ]

        url = catalog.url if catalog else None
        if url is None and path_resolver and file_path:
            try:
                url = path_resolver.resolve(file_path)
            except Exception:
                pass

        cited_chunk_ids = None
        if cited_chunks is not None:
            present = {c.get("chunk_id") for c in ordered_chunks}
            cited_chunk_ids = [cid for cid in cited_chunks.get(ref_id, []) if cid in present]

        sources.append(
            SourceReference(
                id=ref_id,
                title=(catalog.title if catalog else None)
                or metadata.get("file_name")
                or (Path(file_path).name if file_path else None),
                path=file_path,
                type=(catalog.type if catalog else None) or metadata.get("file_type"),
                url=url,
                cited_chunk_ids=cited_chunk_ids,
                chunks=snippets,
            )
        )

    return sources
