# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for RetrievalBackend Protocol and RetrievalResult."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from dlightrag.core.retrieval.protocols import RetrievalBackend, RetrievalResult


class TestRetrievalResult:
    def test_defaults(self) -> None:
        r = RetrievalResult()
        assert r.answer is None
        assert r.contexts == {}
        assert r.raw == {}

    def test_with_values(self) -> None:
        r = RetrievalResult(answer="hello", contexts={"k": "v"}, raw={"x": 1})
        assert r.answer == "hello"
        assert r.contexts == {"k": "v"}
        assert r.raw == {"x": 1}


class TestRetrievalBackendProtocol:
    def test_structural_subtyping(self) -> None:
        class FakeBackend:
            async def aretrieve(self, query: str, *, mode: str = "mix",
                top_k: int | None = None, chunk_top_k: int | None = None,
                **kwargs: Any) -> RetrievalResult:
                return RetrievalResult()
            async def aanswer(self, query: str, *, mode: str = "mix",
                top_k: int | None = None, chunk_top_k: int | None = None,
                **kwargs: Any) -> RetrievalResult:
                return RetrievalResult()
            async def aanswer_stream(self, query: str, *, mode: str = "mix",
                top_k: int | None = None, chunk_top_k: int | None = None,
                **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any], AsyncIterator[str]]:
                raise NotImplementedError

        backend: RetrievalBackend = FakeBackend()
        assert isinstance(backend, FakeBackend)
