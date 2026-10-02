# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Storage-neutral ports: text embedding, BM25 languages and candidate shapes.

``TextEmbedder`` keeps dense recall optional and backend-independent;
``NullEmbedder`` is the zero-configuration default for standalone hosts
(sparse + exact legs only). ``BM25Languages`` decides which language each fact
is indexed under; ``ChineseAndEnglish`` is the standalone default.
``SearchCandidate`` is the leg-tagged candidate a storage adapter returns in
per-leg rank order. PostgreSQL-specific connection and migration shapes live in
``_storage.pg``, not here.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Literal, Protocol

from dlightrag_memory.models import MemoryRecord

type Vector = Sequence[float]


class TextEmbedder(Protocol):
    """Produce one embedding space for memory bodies.

    ``embedding_fingerprint`` identifies the embedding model; an adapter
    stores it with every vector so a model change invalidates the dense index
    instead of silently comparing across spaces. ``relevance_floor`` is the
    cosine similarity at or above which a query and a remembered fact are
    related in this space; ``None`` means the model is uncalibrated, so dense
    similarity alone never makes a fact relevant.
    """

    @property
    def embedding_fingerprint(self) -> str: ...

    @property
    def relevance_floor(self) -> float | None: ...

    dim: int

    async def embed_documents(self, texts: Sequence[str]) -> Sequence[Vector]: ...

    async def embed_query(self, text: str) -> Vector: ...


class NullEmbedder:
    """The zero-configuration embedder: dense recall stays off."""

    dim = 0

    @property
    def embedding_fingerprint(self) -> str:
        return "none"

    @property
    def relevance_floor(self) -> float | None:
        return None

    async def aclose(self) -> None:
        return None

    async def embed_documents(self, texts: Sequence[str]) -> Sequence[Vector]:
        raise RuntimeError("NullEmbedder produces no vectors; disable the dense leg")

    async def embed_query(self, text: str) -> Vector:
        raise RuntimeError("NullEmbedder produces no vectors; disable the dense leg")


class BM25Languages(Protocol):
    """Which language a text is indexed under, and each language's analyzer.

    ``text_configs`` maps every language ``language_of`` returns to a PostgreSQL
    text search configuration, so each language keeps its own stopwords and
    stemming. It always holds ``simple``, the language of what fits no other.
    """

    @property
    def text_configs(self) -> Mapping[str, str]: ...

    def language_of(self, text: str) -> str: ...


_CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
_LATIN = re.compile(r"[A-Za-z]")


class ChineseAndEnglish:
    """The standalone default: Chinese by its script, else English, else simple."""

    text_configs: Mapping[str, str] = MappingProxyType(
        {"zh": "public.jiebacfg", "en": "english", "simple": "simple"}
    )

    def language_of(self, text: str) -> str:
        if _CJK.search(text):
            return "zh"
        return "en" if _LATIN.search(text) else "simple"


SearchLeg = Literal["dense", "sparse", "exact"]


class SearchCandidate:
    """One matching record, the leg that matched it, and that leg's own score."""

    __slots__ = ("record", "leg", "score")

    def __init__(self, *, record: MemoryRecord, leg: SearchLeg, score: float) -> None:
        self.record = record
        self.leg = leg
        self.score = score


__all__ = [
    "BM25Languages",
    "ChineseAndEnglish",
    "NullEmbedder",
    "SearchCandidate",
    "SearchLeg",
    "TextEmbedder",
    "Vector",
]
