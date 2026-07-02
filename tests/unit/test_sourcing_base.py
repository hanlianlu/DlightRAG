# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for data source base contracts."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

import dlightrag.sourcing as sourcing
from dlightrag.sourcing.base import AsyncDataSource


class StreamingOnlySource(AsyncDataSource):
    async def aiter_documents(self, prefix: str | None = None) -> AsyncIterator[str]:
        base = prefix or ""
        yield f"{base}a.pdf"
        yield f"{base}b.pdf"

    async def amaterialize_document(self, doc_id: str, destination: Path) -> None:
        destination.write_bytes(doc_id.encode())


async def test_async_data_source_list_collects_streaming_documents() -> None:
    source = StreamingOnlySource()

    assert await source.alist_documents(prefix="docs/") == ["docs/a.pdf", "docs/b.pdf"]


def test_sourcing_public_api_is_async_only() -> None:
    assert "DataSource" not in sourcing.__all__
    assert "LocalDataSource" not in sourcing.__all__

    data_source_name = "DataSource"
    local_source_name = "LocalDataSource"

    with pytest.raises(AttributeError):
        getattr(sourcing, data_source_name)

    with pytest.raises(AttributeError):
        getattr(sourcing, local_source_name)
