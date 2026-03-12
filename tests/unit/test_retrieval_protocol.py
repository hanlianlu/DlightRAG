# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for RetrievalBackend Protocol and RetrievalResult."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from dlightrag.core.retrieval.protocols import (
    RetrievalBackend,
    RetrievalContexts,
    RetrievalResult,
)


class TestRetrievalResult:
    def test_defaults(self) -> None:
        r = RetrievalResult()
        assert r.answer is None
        assert r.contexts == {"chunks": [], "entities": [], "relationships": []}

    def test_with_values(self) -> None:
        ctx = RetrievalContexts(
            chunks=[
                {
                    "chunk_id": "c1",
                    "reference_id": "r1",
                    "file_path": "/f",
                    "content": "hello",
                }
            ],
            entities=[],
            relationships=[],
        )
        r = RetrievalResult(answer="hello", contexts=ctx)
        assert r.answer == "hello"
        assert len(r.contexts["chunks"]) == 1
        assert r.contexts["chunks"][0]["chunk_id"] == "c1"

    def test_typeddict_structural_compat(self) -> None:
        """Plain dicts that match the TypedDict structure work fine."""
        plain_dict = {
            "chunks": [
                {
                    "chunk_id": "c1",
                    "reference_id": "r1",
                    "file_path": "/f",
                    "content": "text",
                }
            ],
            "entities": [],
            "relationships": [],
        }
        r = RetrievalResult(answer=None, contexts=plain_dict)
        assert r.contexts["chunks"][0]["content"] == "text"


class TestRetrievalBackendProtocol:
    def test_structural_subtyping(self) -> None:
        class FakeBackend:
            async def aretrieve(
                self,
                query: str,
                *,
                mode: str = "mix",
                top_k: int | None = None,
                chunk_top_k: int | None = None,
                **kwargs: Any,
            ) -> RetrievalResult:
                return RetrievalResult()

            async def aanswer(
                self,
                query: str,
                *,
                mode: str = "mix",
                top_k: int | None = None,
                chunk_top_k: int | None = None,
                **kwargs: Any,
            ) -> RetrievalResult:
                return RetrievalResult()

            async def aanswer_stream(
                self,
                query: str,
                *,
                mode: str = "mix",
                top_k: int | None = None,
                chunk_top_k: int | None = None,
                **kwargs: Any,
            ) -> tuple[dict[str, Any], AsyncIterator[str] | None]:
                raise NotImplementedError

        backend: RetrievalBackend = FakeBackend()
        assert isinstance(backend, FakeBackend)
