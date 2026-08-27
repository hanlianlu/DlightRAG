# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Storage-neutral retrieval interfaces."""

from typing import Any, Protocol

from dlightrag.engine.rag.retrieval.models import ContextRow, MetadataScope
from dlightrag.engine.rag.retrieval.results import RetrievalResult


class RetrievalBackend(Protocol):
    async def aretrieve(
        self,
        query: str,
        *,
        mode: str = "mix",
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        **kwargs: Any,
    ) -> RetrievalResult: ...


class BM25Search(Protocol):
    async def search(
        self,
        query: str,
        *,
        scope: MetadataScope | None,
        top_k: int | None = None,
    ) -> list[ContextRow]: ...


class BM25ProfileSearch(Protocol):
    async def search_profile(
        self,
        query: str,
        *,
        profile_name: str,
        language: str | None,
        doc_ids: list[str] | None,
        limit: int,
    ) -> list[ContextRow]: ...


class MetadataChunkStore(Protocol):
    async def count_chunks_for_docs(self, doc_ids: list[str]) -> int: ...


class CorpusChunkStore(MetadataChunkStore, Protocol):
    async def overwrite_chunk_vectors(
        self,
        vectors: dict[str, list[float]],
        *,
        embedding_dim: int,
    ) -> None: ...

    async def fetch_chunk_contents(self, chunk_ids: list[str]) -> list[dict[str, Any]]: ...

    async def update_chunk_bm25_languages(self, labels: dict[str, str]) -> None: ...


class FilteredVectorSearch(Protocol):
    async def search(
        self,
        embedding: list[float],
        *,
        scope: MetadataScope,
        top_k: int,
    ) -> list[dict[str, Any]]: ...

    async def ensure_document_scope_index(self) -> None: ...


__all__ = [
    "BM25Search",
    "BM25ProfileSearch",
    "CorpusChunkStore",
    "FilteredVectorSearch",
    "MetadataChunkStore",
    "RetrievalBackend",
]
