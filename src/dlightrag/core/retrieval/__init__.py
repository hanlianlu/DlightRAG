# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared retrieval types and utilities."""

from __future__ import annotations

from typing import Any

from dlightrag.core.retrieval.models import MetadataFilter
from dlightrag.core.retrieval.path_resolver import PathResolver
from dlightrag.core.retrieval.protocols import (
    ChunkContext,
    EntityContext,
    RelationshipContext,
    RetrievalBackend,
    RetrievalContexts,
    RetrievalResult,
)


def canonicalize_reference_ids(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-assign ``reference_id`` to all chunks via lightrag's canonical mapping.

    Single point that fixes the empty ``reference_id`` left on chunks that
    bypass ``aquery_data`` (metadata-path injections from
    ``fetch_missing_chunks``, visual-only hits from
    ``query_by_visual_embedding``). After this call every chunk with a
    non-empty ``file_path`` carries a stable doc-level reference_id, which
    the citation indexer needs to mint chunk-level ``[n-m]`` tags.

    Re-running on chunks that already have reference_ids is safe: the
    upstream helper rebuilds the mapping from file_path frequency, so the
    output is internally consistent within one query.
    """
    if not chunks:
        return chunks
    from lightrag.utils import generate_reference_list_from_chunks

    _, updated = generate_reference_list_from_chunks(chunks)
    return updated


__all__ = [
    "ChunkContext",
    "EntityContext",
    "MetadataFilter",
    "PathResolver",
    "RelationshipContext",
    "RetrievalBackend",
    "RetrievalContexts",
    "RetrievalResult",
    "canonicalize_reference_ids",
]
