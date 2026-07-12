# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Retrieval backend protocol and shared types."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NotRequired, Protocol, TypedDict

from dlightrag.models.schemas import Reference

if TYPE_CHECKING:
    from dlightrag.citations.schemas import SourceReference

# ── Structured context types ──────────────────────────────────────


class ChunkContext(TypedDict):
    """A single text or image chunk from retrieval."""

    chunk_id: str
    reference_id: str
    file_path: str
    content: str
    full_doc_id: NotRequired[str]
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


ContextRow = dict[str, Any]


# RetrievalContexts is intentionally a plain type alias rather than a TypedDict.
# Contexts flow through JSON/DB boundaries as ``dict[str, Any]`` and are built
# incrementally in federation merges, so a strict TypedDict creates invariance
# errors throughout the codebase.  The per-field TypedDicts above (ChunkContext,
# EntityContext, RelationshipContext) still document the expected shape.
RetrievalContexts = dict[str, list[ContextRow]]


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
    answer_images: list[dict[str, Any]] = field(default_factory=list)
    answer_blocks: list[dict[str, Any]] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)
    image_descriptions: list[str] = field(default_factory=list)


# ── Backend protocol ──────────────────────────────────────────────


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
    ) -> RetrievalResult:
        raise NotImplementedError


class BM25Retriever(Protocol):
    """Lexical chunk retriever used by UnifiedRetriever."""

    async def search(
        self,
        query: str,
        *,
        candidate_ids: set[str] | None,
        top_k: int | None = None,
    ) -> list[ContextRow]:
        raise NotImplementedError


class MetadataChunkStore(Protocol):
    """Store methods needed to resolve metadata hits into chunk ids."""

    async def chunk_ids_for_docs(self, doc_ids: list[str]) -> list[str]:
        raise NotImplementedError


__all__ = [
    "ChunkContext",
    "ContextRow",
    "BM25Retriever",
    "EntityContext",
    "MetadataChunkStore",
    "RelationshipContext",
    "RetrievalBackend",
    "RetrievalContexts",
    "RetrievalResult",
]
