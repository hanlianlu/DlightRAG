# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Retrieval backend protocol and shared types."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from dlightrag.models.schemas import Reference

if TYPE_CHECKING:
    from dlightrag.citations.schemas import SourceReference
    from dlightrag.core.retrieval.models import MetadataScope

ContextRow = dict[str, Any]


# RetrievalContexts is intentionally a plain type alias rather than a TypedDict.
# Contexts flow through JSON/DB boundaries as ``dict[str, Any]`` and are built
# incrementally in federation merges, so a strict TypedDict creates invariance
# errors throughout the codebase. docs/interfaces.md documents the row shapes.
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
    warnings: list[str] = field(default_factory=list)


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
        scope: MetadataScope | None,
        top_k: int | None = None,
    ) -> list[ContextRow]:
        raise NotImplementedError


class MetadataChunkStore(Protocol):
    """Store methods needed to size a metadata hit's chunk fan-out."""

    async def count_chunks_for_docs(self, doc_ids: list[str]) -> int:
        raise NotImplementedError


__all__ = [
    "ContextRow",
    "BM25Retriever",
    "MetadataChunkStore",
    "RetrievalBackend",
    "RetrievalContexts",
    "RetrievalResult",
]
