# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for URL-backed ingestion sources."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from dlightrag.sourcing.uri import parse_remote_uri
from dlightrag.sourcing.url import URLDataSource


class _Response:
    def __init__(
        self,
        content: bytes,
        *,
        url: str = "https://cdn.example.com/report.pdf",
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._content = content
        self.url = url
        self.status_code = status_code
        self.headers = headers or {}

    async def __aenter__(self) -> _Response:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    async def aiter_bytes(self):
        midpoint = len(self._content) // 2
        yield self._content[:midpoint]
        yield self._content[midpoint:]


class _Client:
    def __init__(
        self,
        *,
        content: bytes = b"document",
        final_url: str = "https://cdn.example.com/report.pdf",
    ) -> None:
        self.content = content
        self.final_url = final_url
        self.urls: list[str] = []
        self.closed = False

    def stream(self, method: str, url: str) -> _Response:
        assert method == "GET"
        self.urls.append(url)
        return _Response(self.content, url=self.final_url)

    async def aclose(self) -> None:
        self.closed = True


async def test_url_data_source_maps_extensionless_url_to_html_filename(tmp_path: Path) -> None:
    client = _Client()
    source = URLDataSource(
        urls=["https://api.bynder.com/docs/getting-started"],
        client=client,
    )

    documents = await source.alist_documents()
    assert [document.key for document in documents] == ["getting-started.html"]
    assert source.source_uri_for_key("getting-started.html") == (
        "https://api.bynder.com/docs/getting-started"
    )
    destination = tmp_path / "getting-started.html"
    await source.amaterialize_document(documents[0], destination)
    assert destination.read_bytes() == b"document"
    assert client.urls == ["https://api.bynder.com/docs/getting-started"]


async def test_url_data_source_uses_explicit_filename_for_opaque_single_url() -> None:
    source = URLDataSource(
        urls=["https://cdn.example.com/download?id=asset-1"],
        filename="asset.pdf",
        client=_Client(),
    )

    documents = await source.alist_documents()
    assert [document.key for document in documents] == ["asset.pdf"]
    assert source.source_uri_for_key("asset.pdf") == "https://cdn.example.com/download"


async def test_url_data_source_accepts_explicit_stable_source_uri() -> None:
    source = URLDataSource(
        urls=["https://cdn.example.com/download?id=asset-1&signature=secret"],
        filename="asset.pdf",
        source_uri="bynder://asset/asset-1",
        client=_Client(),
    )

    assert source.source_uri_for_key("asset.pdf") == "bynder://asset/asset-1"


async def test_url_data_source_revalidates_final_response_url(tmp_path: Path) -> None:
    source = URLDataSource(
        urls=["https://cdn.example.com/report.pdf"],
        client=_Client(final_url="https://127.0.0.1/report.pdf"),
    )

    with pytest.raises(ValueError, match="public"):
        await source.amaterialize_document(
            (await source.alist_documents())[0], tmp_path / "report.pdf"
        )


async def test_url_data_source_rejects_private_redirect_before_following(tmp_path: Path) -> None:
    class RedirectClient:
        def __init__(self) -> None:
            self.urls: list[str] = []

        def stream(self, method: str, url: str) -> _Response:
            assert method == "GET"
            self.urls.append(url)
            return _Response(
                b"",
                url=url,
                status_code=302,
                headers={"location": "https://127.0.0.1/admin.pdf"},
            )

    client = RedirectClient()
    source = URLDataSource(urls=["https://cdn.example.com/start.pdf"], client=client)

    with pytest.raises(ValueError, match="public"):
        await source.amaterialize_document(
            (await source.alist_documents())[0], tmp_path / "start.pdf"
        )

    assert client.urls == ["https://cdn.example.com/start.pdf"]
    assert not (tmp_path / "start.pdf").exists()


async def test_url_data_source_enforces_download_size_limit(tmp_path: Path) -> None:
    source = URLDataSource(
        urls=["https://cdn.example.com/report.pdf"],
        client=_Client(content=b"document"),
        max_download_bytes=3,
    )

    with pytest.raises(ValueError, match="maximum"):
        await source.amaterialize_document(
            (await source.alist_documents())[0], tmp_path / "report.pdf"
        )

    assert not (tmp_path / "report.pdf").exists()


def test_url_data_source_rejects_insecure_or_private_urls() -> None:
    with pytest.raises(ValueError, match="https"):
        URLDataSource(urls=["http://example.com/report.pdf"], client=_Client())

    with pytest.raises(ValueError, match="public"):
        URLDataSource(urls=["https://127.0.0.1/report.pdf"], client=_Client())


def test_url_data_source_rejects_hostname_that_resolves_private(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                ("10.0.0.1", 443),
            )
        ],
    )

    with pytest.raises(ValueError, match="public"):
        URLDataSource(urls=["https://private.example/report.pdf"])


def test_url_data_source_allows_allowlisted_private_hostname(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                ("10.0.0.1", 443),
            )
        ],
    )

    source = URLDataSource(
        urls=["https://docs.corp.example/report.pdf"],
        allow_private_hosts=["*.corp.example"],
    )

    assert source.source_uri_for_key("report.pdf") == "https://docs.corp.example/report.pdf"


def test_parse_remote_uri_treats_https_as_url_source() -> None:
    assert parse_remote_uri("https://api.bynder.com/docs/getting-started") == (
        "url",
        {"url": "https://api.bynder.com/docs/getting-started"},
    )
