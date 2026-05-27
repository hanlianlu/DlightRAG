# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Retrieval backend protocol and shared types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NotRequired, Protocol, TypedDict, runtime_checkable

from dlightrag.citations.schemas import SourceReference
from dlightrag.models.schemas import Reference

# ── Structured context types ──────────────────────────────────────


class ChunkContext(TypedDict):
    """A single text or image chunk from retrieval."""

    chunk_id: str
    reference_id: str
    file_path: str
    content: str
    page_idx: NotRequired[int | None]
    bbox: NotRequired[dict[str, Any] | None]
    image_data: NotRequired[str | None]
    image_mime_type: NotRequired[str | None]
    image_url: NotRequired[str | None]
    thumbnail_url: NotRequired[str | None]
    relevance_score: NotRequired[float | None]
    metadata: NotRequired[dict[str, Any]]
    _workspace: NotRequired[str]


class EntityContext(TypedDict):
    """A KG entity from retrieval."""

    entity_name: str
    entity_type: str
    description: str
    source_id: str
    reference_id: NotRequired[str]
    _workspace: NotRequired[str]


class RelationshipContext(TypedDict):
    """A KG relationship from retrieval."""

    src_id: str
    tgt_id: str
    description: str
    source_id: str
    reference_id: NotRequired[str]
    _workspace: NotRequired[str]


# RetrievalContexts is intentionally a plain type alias rather than a TypedDict.
# Contexts flow through JSON/DB boundaries as ``dict[str, Any]`` and are built
# incrementally in federation merges, so a strict TypedDict creates invariance
# errors throughout the codebase.  The per-field TypedDicts above (ChunkContext,
# EntityContext, RelationshipContext) still document the expected shape.
RetrievalContexts = dict[str, list[dict[str, Any]]]


# ── Result dataclass ──────────────────────────────────────────────


@dataclass
class RetrievalResult:
    """Wrapper for RAG query results."""

    answer: str | None = field(default=None)
    contexts: RetrievalContexts = field(
        default_factory=lambda: {"chunks": [], "entities": [], "relationships": []}
    )
    references: list[Reference] = field(default_factory=list)
    sources: list[SourceReference] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)
    image_descriptions: list[str] = field(default_factory=list)
    current_image_ids: list[str] = field(default_factory=list)


# ── Backend protocol ──────────────────────────────────────────────


@runtime_checkable
class RetrievalBackend(Protocol):
    """Retrieval interface for the LightRAG-main runtime path."""

    async def aretrieve(
        self,
        query: str,
        *,
        mode: str = "mix",
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        **kwargs: Any,
    ) -> RetrievalResult: ...


__all__ = [
    "ChunkContext",
    "EntityContext",
    "RelationshipContext",
    "RetrievalBackend",
    "RetrievalContexts",
    "RetrievalResult",
]
