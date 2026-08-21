# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Storage-neutral ports: text embedding and recall candidate search.

These ports are the P4 substrate. ``TextEmbedder`` keeps dense recall optional
and backend-independent; ``NullEmbedder`` is the zero-configuration default
for standalone hosts (sparse + exact legs only). ``MemorySearch`` is the
optional search surface a storage adapter implements when it can generate
relevance candidates; adapters without it fall back to the recency window.
PostgreSQL-specific connection and migration shapes live in ``_storage.pg``,
not here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol

from dlightrag_memory.models import MemoryRecord

type Vector = Sequence[float]


class TextEmbedder(Protocol):
    """Produce one embedding space for memory bodies.

    ``fingerprint`` identifies the embedding model; an adapter stores it with
    every vector so a model change invalidates the dense index instead of
    silently comparing across spaces.
    """

    fingerprint: str
    dim: int

    async def embed_documents(self, texts: Sequence[str]) -> Sequence[Vector]: ...

    async def embed_query(self, text: str) -> Vector: ...


class NullEmbedder:
    """The zero-configuration embedder: dense recall stays off."""

    fingerprint = "none"
    dim = 0

    async def embed_documents(self, texts: Sequence[str]) -> Sequence[Vector]:
        raise RuntimeError("NullEmbedder produces no vectors; disable the dense leg")

    async def embed_query(self, text: str) -> Vector:
        raise RuntimeError("NullEmbedder produces no vectors; disable the dense leg")


SearchLeg = Literal["dense", "sparse", "exact"]


class SearchCandidate:
    """One recalled record with its source leg and a comparable score."""

    __slots__ = ("record", "leg", "score")

    def __init__(self, *, record: MemoryRecord, leg: SearchLeg, score: float) -> None:
        self.record = record
        self.leg = leg
        self.score = score


class MemorySearch(Protocol):
    """Relevance candidate generation over one owner's active records."""

    async def search_candidates(
        self, *, owner_id: str, query: str, limit: int
    ) -> tuple[SearchCandidate, ...]: ...


__all__ = [
    "MemorySearch",
    "NullEmbedder",
    "SearchCandidate",
    "SearchLeg",
    "TextEmbedder",
    "Vector",
]
