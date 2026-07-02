# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for URL-backed ingestion sources."""

from __future__ import annotations

import pytest

from dlightrag.sourcing.uri import parse_remote_uri
from dlightrag.sourcing.url import URLDataSource


class _Response:
    def __init__(self, content: bytes, *, url: str = "https://cdn.example.com/report.pdf") -> None:
        self.content = content
        self.url = url

    def raise_for_status(self) -> None:
        return None


class _Client:
    def __init__(self, *, final_url: str = "https://cdn.example.com/report.pdf") -> None:
        self.final_url = final_url
        self.urls: list[str] = []
        self.closed = False

    async def get(self, url: str) -> _Response:
        self.urls.append(url)
        return _Response(b"document", url=self.final_url)

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
    assert source.source_uri_for_key("asset.pdf") == "https://cdn.example.com/download"


async def test_url_data_source_accepts_explicit_stable_source_uri() -> None:
    source = URLDataSource(
        urls=["https://cdn.example.com/download?id=asset-1&signature=secret"],
        filename="asset.pdf",
        source_uri="bynder://asset/asset-1",
        client=_Client(),
    )

    assert source.source_uri_for_key("asset.pdf") == "bynder://asset/asset-1"


async def test_url_data_source_revalidates_final_response_url() -> None:
    source = URLDataSource(
        urls=["https://cdn.example.com/report.pdf"],
        client=_Client(final_url="https://127.0.0.1/report.pdf"),
    )

    with pytest.raises(ValueError, match="public"):
        await source.aload_document("report.pdf")


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
