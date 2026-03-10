# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Retrieval backend protocol and shared types."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class RetrievalResult:
    """Wrapper for RAG query results."""

    answer: str | None = field(default=None)
    contexts: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class RetrievalBackend(Protocol):
    """Unified retrieval interface for caption and unified RAG modes."""

    async def aretrieve(
        self, query: str, *, mode: str = "mix",
        top_k: int | None = None, chunk_top_k: int | None = None,
        **kwargs: Any,
    ) -> RetrievalResult: ...

    async def aanswer(
        self, query: str, *, mode: str = "mix",
        top_k: int | None = None, chunk_top_k: int | None = None,
        **kwargs: Any,
    ) -> RetrievalResult: ...

    async def aanswer_stream(
        self, query: str, *, mode: str = "mix",
        top_k: int | None = None, chunk_top_k: int | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any], AsyncIterator[str]]: ...


__all__ = ["RetrievalBackend", "RetrievalResult"]
