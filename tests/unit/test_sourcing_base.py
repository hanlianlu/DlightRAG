# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for data source base contracts."""

from collections.abc import AsyncIterator
from pathlib import Path

from dlightrag.rag.sourcing.base import AsyncDataSource, SourceDocument


class StreamingOnlySource(AsyncDataSource):
    async def aiter_documents(self, prefix: str | None = None) -> AsyncIterator[SourceDocument]:
        base = prefix or ""
        yield SourceDocument(key=f"{base}a.pdf")
        yield SourceDocument(key=f"{base}b.pdf")

    async def amaterialize_document(self, document: SourceDocument, destination: Path) -> None:
        destination.write_bytes(document.key.encode())


async def test_async_data_source_list_collects_streaming_documents() -> None:
    source = StreamingOnlySource()

    assert [d async for d in source.aiter_documents(prefix="docs/")] == [
        SourceDocument(key="docs/a.pdf"),
        SourceDocument(key="docs/b.pdf"),
    ]
