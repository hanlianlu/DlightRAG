# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for data source base contracts."""

from __future__ import annotations

from collections.abc import AsyncIterator

from dlightrag.sourcing.base import AsyncDataSource


class StreamingOnlySource(AsyncDataSource):
    async def aiter_documents(self, prefix: str | None = None) -> AsyncIterator[str]:
        base = prefix or ""
        yield f"{base}a.pdf"
        yield f"{base}b.pdf"

    async def aload_document(self, doc_id: str) -> bytes:
        return doc_id.encode()


async def test_async_data_source_list_collects_streaming_documents() -> None:
    source = StreamingOnlySource()

    assert await source.alist_documents(prefix="docs/") == ["docs/a.pdf", "docs/b.pdf"]
