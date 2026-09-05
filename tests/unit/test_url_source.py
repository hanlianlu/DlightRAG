# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for URL-backed ingestion sources."""

import logging
import socket
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx
import pytest

from dlightrag.engine.public_http import (
    PublicHttpPresentation,
    avalidate_public_http_url,
    fetch_public_http,
    normalize_public_http_url_identity,
)
from dlightrag.engine.rag.corpus.sources.base import SourceDocument
from dlightrag.engine.rag.corpus.sources.source_contract import safe_source_filename
from dlightrag.engine.rag.corpus.sources.uri import parse_remote_uri
from dlightrag.engine.rag.corpus.sources.url import URLDataSource


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

    def stream(self, method: str, url: str, **kwargs) -> _Response:
        assert method == "GET"
        self.urls.append(_logical_url(url, kwargs))
        return _Response(self.content, url=self.final_url)

    async def aclose(self) -> None:
        self.closed = True


class _RedirectClient:
    def __init__(self, start_url: str, target_url: str, *, content: bytes = b"final body") -> None:
        self.start_url = start_url
        self.target_url = target_url
        self.content = content
        self.urls: list[str] = []

    def stream(self, method: str, url: str, **kwargs) -> _Response:
        assert method == "GET"
        logical_url = _logical_url(url, kwargs)
        self.urls.append(logical_url)
        if logical_url == self.start_url:
            return _Response(
                b"",
                url=url,
                status_code=302,
                headers={"location": self.target_url},
            )
        return _Response(self.content, url=logical_url)


def _logical_url(url: str, kwargs: dict) -> str:
    parts = urlsplit(url)
    host = (kwargs.get("headers") or {}).get("host", parts.netloc)
    return urlunsplit((parts.scheme, host, parts.path, parts.query, ""))


@pytest.fixture(autouse=True)
def _public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port, *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))
        ],
    )


def test_safe_source_filename_preserves_extension_when_bounded() -> None:
    result = safe_source_filename(f"{'a' * 200}.pdf")

    assert result == f"{'a' * 124}.pdf"
    assert len(result) == 128


async def test_url_data_source_maps_extensionless_url_to_html_filename(tmp_path: Path) -> None:
    client = _Client()
    source = URLDataSource(
        urls=["https://api.bynder.com/docs/getting-started"],
        client=client,
    )

    documents = [d async for d in source.aiter_documents()]
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

    documents = [d async for d in source.aiter_documents()]
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


async def test_url_data_source_separates_fetch_identity_and_download_uri() -> None:
    source = URLDataSource(
        urls=["https://fetch.example.com/download?sig=secret"],
        filename="asset.pdf",
        source_uri="bynder://asset/1",
        download_uri="https://cdn.example.com/assets/1.pdf",
        client=_Client(),
    )

    document = ([d async for d in source.aiter_documents()])[0]

    assert document.source_uri == "bynder://asset/1"
    assert document.download_uri == "https://cdn.example.com/assets/1.pdf"
    assert source.download_uri_for_key("asset.pdf") == ("https://cdn.example.com/assets/1.pdf")


async def test_url_data_source_does_not_derive_download_uri_from_signed_fetch_url(
    caplog: pytest.LogCaptureFixture,
) -> None:
    hostile_filename = "https://files.example.com/report.pdf?display_token=secret"
    with caplog.at_level(logging.INFO, logger="dlightrag.engine.rag.corpus.sources.url"):
        source = URLDataSource(
            urls=["https://fetch.example.com/download?sig=secret"],
            filename=hostile_filename,
            client=_Client(),
        )

    document = ([d async for d in source.aiter_documents()])[0]

    assert document.download_uri is None
    outcome = next(
        record for record in caplog.records if record.message == "source_download_locator_outcome"
    )
    outcome_fields = vars(outcome)
    assert outcome_fields["outcome"] == "ephemeral"
    assert outcome_fields["locator_kind"] == "https"
    assert outcome_fields["source_filename"] == "report.pdf"
    assert "sig=secret" not in caplog.text
    assert "display_token=secret" not in caplog.text
    assert all("display_token=secret" not in str(value) for value in outcome_fields.values())


async def test_url_data_source_derives_download_uri_from_queryless_fetch_url() -> None:
    source = URLDataSource(
        urls=["https://fetch.example.com/assets/1.pdf"],
        client=_Client(),
    )

    document = ([d async for d in source.aiter_documents()])[0]

    assert document.download_uri == "https://fetch.example.com/assets/1.pdf"
    assert source.download_uri_for_key("1.pdf") == "https://fetch.example.com/assets/1.pdf"


async def test_url_data_source_uses_source_document_download_uri() -> None:
    source = URLDataSource(
        documents=[
            SourceDocument(
                key="https://fetch.example.com/download?sig=secret",
                source_uri="bynder://asset/1",
                download_uri="https://cdn.example.com/assets/1.pdf",
                display_filename="asset.pdf",
            )
        ],
        client=_Client(),
    )

    document = ([d async for d in source.aiter_documents()])[0]

    assert document.download_uri == "https://cdn.example.com/assets/1.pdf"


def test_url_data_source_download_uri_cardinality_is_strict() -> None:
    with pytest.raises(ValueError, match="download_uris"):
        URLDataSource(
            urls=["https://fetch.example.com/a.pdf", "https://fetch.example.com/b.pdf"],
            download_uris=["https://cdn.example.com/a.pdf"],
            client=_Client(),
        )


def test_url_data_source_rejects_non_durable_explicit_download_uri() -> None:
    with pytest.raises(ValueError, match="durable download_uri"):
        URLDataSource(
            urls=["https://fetch.example.com/download?sig=secret"],
            download_uri="https://cdn.example.com/download?sig=secret",
            client=_Client(),
        )


async def test_url_data_source_uses_validated_target_not_transport_reported_url(
    tmp_path: Path,
) -> None:
    source = URLDataSource(
        urls=["https://cdn.example.com/report.pdf"],
        client=_Client(final_url="https://127.0.0.1/report.pdf"),
    )
    destination = tmp_path / "report.pdf"

    await source.amaterialize_document(
        ([d async for d in source.aiter_documents()])[0], destination
    )

    assert destination.read_bytes() == b"document"


async def test_url_data_source_rejects_private_redirect_before_following(tmp_path: Path) -> None:
    client = _RedirectClient(
        "https://cdn.example.com/start.pdf",
        "https://127.0.0.1/admin.pdf",
    )
    source = URLDataSource(urls=["https://cdn.example.com/start.pdf"], client=client)

    with pytest.raises(ValueError, match="public"):
        await source.amaterialize_document(
            ([d async for d in source.aiter_documents()])[0], tmp_path / "start.pdf"
        )

    assert client.urls == ["https://cdn.example.com/start.pdf"]
    assert not (tmp_path / "start.pdf").exists()


async def test_url_data_source_rejects_redirect_hostname_that_resolves_private(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def resolver(host: str, port: int, *args: object, **kwargs: object):
        address = "10.0.0.1" if host == "private.example" else "93.184.216.34"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]

    monkeypatch.setattr(socket, "getaddrinfo", resolver)

    client = _RedirectClient(
        "https://public.example/start.pdf",
        "https://private.example/admin.pdf",
        content=b"private",
    )
    source = URLDataSource(urls=["https://public.example/start.pdf"], client=client)

    with pytest.raises(ValueError, match="public"):
        await source.amaterialize_document(
            ([document async for document in source.aiter_documents()])[0],
            tmp_path / "start.pdf",
        )

    assert client.urls == ["https://public.example/start.pdf"]


async def test_url_data_source_enforces_download_size_limit(tmp_path: Path) -> None:
    source = URLDataSource(
        urls=["https://cdn.example.com/report.pdf"],
        client=_Client(content=b"document"),
        max_download_bytes=3,
    )

    with pytest.raises(ValueError, match="maximum"):
        await source.amaterialize_document(
            ([d async for d in source.aiter_documents()])[0], tmp_path / "report.pdf"
        )

    assert not (tmp_path / "report.pdf").exists()


def test_url_data_source_accepts_public_http_urls() -> None:
    source = URLDataSource(urls=["http://example.com/report.pdf"], client=_Client())

    assert source.source_uri_for_key("report.pdf") == "http://example.com/report.pdf"


def test_url_data_source_rejects_non_http_or_private_urls() -> None:
    with pytest.raises(ValueError, match="http or https"):
        URLDataSource(urls=["ftp://example.com/report.pdf"], client=_Client())

    with pytest.raises(ValueError, match="public"):
        URLDataSource(urls=["https://127.0.0.1/report.pdf"], client=_Client())

    with pytest.raises(ValueError, match="public"):
        URLDataSource(urls=["http://127.0.0.1/report.pdf"], client=_Client())


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


async def test_url_data_source_keeps_allowlisted_private_fetch_url_out_of_download_uri() -> None:
    source = URLDataSource(
        urls=["https://10.0.0.1/report.pdf"],
        allow_private_hosts=["10.*"],
        client=_Client(),
    )

    document = ([d async for d in source.aiter_documents()])[0]

    assert document.download_uri is None


def test_public_url_identity_normalizes_authority_and_discards_fragment() -> None:
    assert (
        normalize_public_http_url_identity("HTTPS://EXAMPLE.COM.:443/report?id=7#section")
        == "https://example.com/report?id=7"
    )


def test_parse_remote_uri_treats_http_and_https_as_url_source() -> None:
    assert parse_remote_uri("https://api.bynder.com/docs/getting-started") == (
        "url",
        {"url": "https://api.bynder.com/docs/getting-started"},
    )
    assert parse_remote_uri("http://api.bynder.com/docs/getting-started") == (
        "url",
        {"url": "http://api.bynder.com/docs/getting-started"},
    )


async def test_fetch_public_http_returns_bounded_content_and_final_identity() -> None:
    client = _Client(content=b"hello world", final_url="https://cdn.example.com/report.txt")

    result = await fetch_public_http(
        "https://cdn.example.com/report.txt", max_bytes=1024, client=client
    )

    assert result.content == b"hello world"
    assert result.final_url == "https://cdn.example.com/report.txt"
    assert client.urls == ["https://cdn.example.com/report.txt"]


async def test_injected_httpx_client_cannot_add_credentials_or_arbitrary_headers() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=b"body")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        headers={"authorization": "Bearer secret", "x-extra": "forbidden"},
        cookies={"session": "secret"},
    ) as client:
        result = await fetch_public_http(
            "https://cdn.example.com/report.txt",
            max_bytes=1024,
            presentation=PublicHttpPresentation(user_agent="Allowed/1"),
            client=client,
        )

    assert result.content == b"body"
    assert requests[0].headers["user-agent"] == "Allowed/1"
    assert requests[0].extensions["sni_hostname"] == "cdn.example.com"
    assert "authorization" not in requests[0].headers
    assert "cookie" not in requests[0].headers
    assert "x-extra" not in requests[0].headers


async def test_fetch_public_http_enforces_max_bytes() -> None:
    client = _Client(content=b"x" * 100, final_url="https://cdn.example.com/big.txt")
    with pytest.raises(ValueError, match="maximum"):
        await fetch_public_http("https://cdn.example.com/big.txt", max_bytes=10, client=client)


async def test_fetch_public_http_follows_redirects_and_pins_validated_ip() -> None:
    client = _RedirectClient(
        "https://cdn.example.com/start.txt",
        "https://cdn.example.com/final.txt",
    )

    result = await fetch_public_http(
        "https://cdn.example.com/start.txt", max_bytes=1024, client=client
    )

    assert result.content == b"final body"
    assert result.final_url == "https://cdn.example.com/final.txt"
    assert client.urls == [
        "https://cdn.example.com/start.txt",
        "https://cdn.example.com/final.txt",
    ]


async def test_fetch_public_http_accepts_http_and_http_to_https() -> None:
    client = _RedirectClient(
        "http://cdn.example.com/start.txt",
        "https://cdn.example.com/final.txt",
    )
    result = await fetch_public_http(
        "http://cdn.example.com/start.txt", max_bytes=1024, client=client
    )
    assert result.content == b"final body"


async def test_fetch_public_http_rejects_scheme_downgrade_and_private_redirect() -> None:
    with pytest.raises(ValueError, match="downgrade"):
        await fetch_public_http(
            "https://cdn.example.com/start.txt",
            max_bytes=1024,
            client=_RedirectClient(
                "https://cdn.example.com/start.txt",
                "http://cdn.example.com/final.txt",
            ),
        )
    with pytest.raises(ValueError, match="public"):
        await fetch_public_http(
            "https://cdn.example.com/start.txt",
            max_bytes=1024,
            client=_RedirectClient(
                "https://cdn.example.com/start.txt",
                "https://127.0.0.1/admin.txt",
            ),
        )


async def test_fetch_rejects_redirect_hostname_that_resolves_private(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def resolver(host: str, port: int, *args: object, **kwargs: object):
        address = "10.0.0.1" if host == "private.example" else "93.184.216.34"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]

    monkeypatch.setattr(socket, "getaddrinfo", resolver)
    client = _RedirectClient(
        "https://public.example/start.txt",
        "https://private.example/admin.txt",
    )
    with pytest.raises(ValueError, match="public"):
        await fetch_public_http("https://public.example/start.txt", max_bytes=1024, client=client)
    assert client.urls == ["https://public.example/start.txt"]


async def test_public_url_dns_validation_runs_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    loop_thread = threading.get_ident()
    resolver_threads: list[int] = []

    def resolver(host: str, port: int, *args: object, **kwargs: object):
        resolver_threads.append(threading.get_ident())
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]

    monkeypatch.setattr(socket, "getaddrinfo", resolver)
    assert await avalidate_public_http_url("https://public.example/doc.pdf") == (
        "https://public.example/doc.pdf"
    )
    assert resolver_threads and loop_thread not in resolver_threads


async def test_async_public_url_validation_still_rejects_private_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port, *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", port))
        ],
    )
    with pytest.raises(ValueError, match="public"):
        await avalidate_public_http_url("https://private.example/doc.pdf")
