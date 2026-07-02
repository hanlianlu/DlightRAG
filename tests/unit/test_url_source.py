# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for URL-backed ingestion sources."""

from __future__ import annotations

import pytest

from dlightrag.sourcing.uri import parse_remote_uri
from dlightrag.sourcing.url import URLDataSource


class _Response:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


class _Client:
    def __init__(self) -> None:
        self.urls: list[str] = []
        self.closed = False

    async def get(self, url: str) -> _Response:
        self.urls.append(url)
        return _Response(b"document")

    async def aclose(self) -> None:
        self.closed = True


async def test_url_data_source_maps_extensionless_url_to_html_filename() -> None:
    client = _Client()
    source = URLDataSource(
        urls=["https://api.bynder.com/docs/getting-started"],
        client=client,
    )

    assert await source.alist_documents() == ["getting-started.html"]
    assert source.source_uri_for_key("getting-started.html") == (
        "https://api.bynder.com/docs/getting-started"
    )
    assert await source.aload_document("getting-started.html") == b"document"
    assert client.urls == ["https://api.bynder.com/docs/getting-started"]


async def test_url_data_source_uses_explicit_filename_for_opaque_single_url() -> None:
    source = URLDataSource(
        urls=["https://cdn.example.com/download?id=asset-1"],
        filename="asset.pdf",
        client=_Client(),
    )

    assert await source.alist_documents() == ["asset.pdf"]
    assert source.source_uri_for_key("asset.pdf") == "https://cdn.example.com/download?id=asset-1"


def test_url_data_source_rejects_insecure_or_private_urls() -> None:
    with pytest.raises(ValueError, match="https"):
        URLDataSource(urls=["http://example.com/report.pdf"], client=_Client())

    with pytest.raises(ValueError, match="public"):
        URLDataSource(urls=["https://127.0.0.1/report.pdf"], client=_Client())


def test_parse_remote_uri_treats_https_as_url_source() -> None:
    assert parse_remote_uri("https://api.bynder.com/docs/getting-started") == (
        "url",
        {"url": "https://api.bynder.com/docs/getting-started"},
    )
