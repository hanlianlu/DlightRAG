# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""A parser service outage is named at LightRAG's parser transport boundary."""

import asyncio
import inspect
import socket
import ssl
import struct
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from lightrag.parser.external.docling import client as docling_client
from lightrag.parser.external.mineru import client as mineru_client
from lightrag.utils_pipeline import doc_status_parse_failure_fields

from dlightrag.engine.dependencies import ParserUnavailableError, classify_transient_dependency
from dlightrag.engine.rag.corpus.ingestion.parser_transport import (
    apply_parser_outage_reporting,
    parser_unavailable_recorded,
)

type Handler = Callable[[httpx.Request], httpx.Response]

_CLIENTS = {
    "mineru": (mineru_client, mineru_client.MinerURawClient),
    "docling": (docling_client, docling_client.DoclingRawClient),
}


@pytest.fixture
def unpatched_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start from LightRAG's own client methods; teardown restores the patched ones."""
    for module, client_class in _CLIENTS.values():
        monkeypatch.setattr(
            client_class, "download_into", inspect.unwrap(client_class.download_into)
        )
        monkeypatch.setattr(
            module,
            "raise_for_status_with_detail",
            inspect.unwrap(module.raise_for_status_with_detail),
        )


@pytest.fixture
def parser_service(
    unpatched_clients: None,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[list[Handler]]:
    """Serve both parser clients from one in-test handler."""
    del unpatched_clients
    monkeypatch.setenv("MINERU_API_MODE", "local")
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://mineru.test")
    monkeypatch.setenv("MINERU_POLL_INTERVAL_SECONDS", "0")
    monkeypatch.setenv("DOCLING_ENDPOINT", "http://docling.test")
    handlers: list[Handler] = []
    async_client = httpx.AsyncClient

    def client(**kwargs: Any) -> httpx.AsyncClient:
        transport = httpx.MockTransport(lambda request: handlers[0](request))
        return async_client(transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    yield handlers


async def _download(parser: str, tmp_path: Path, filename: str = "report.pdf") -> None:
    source = tmp_path / filename
    source.write_bytes(b"%PDF-1.4")
    raw_dir = tmp_path / "raw"
    if parser == "mineru":
        await mineru_client.MinerURawClient().download_into(raw_dir, source)
    else:
        await docling_client.DoclingRawClient().download_into(raw_dir, source)


def _raising(error: type[httpx.TransportError]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error("parser transport failed", request=request)

    return handler


def _status(status: int, detail: str = "parser says no") -> Handler:
    return lambda _request: httpx.Response(status, json={"detail": detail})


def _tls_error(reason: str) -> ssl.SSLError:
    # OpenSSL sets ``reason`` on the errors it raises; a constructed one has none.
    error = ssl.SSLError(1, f"[SSL: {reason}]")
    error.reason = reason
    return error


def _connect_failure(cause: Callable[[], BaseException]) -> Handler:
    """Fail to connect as httpx does, with the socket or TLS error as the cause."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(str(cause()), request=request) from cause()

    return handler


@pytest.mark.parametrize("parser", ["mineru", "docling"])
@pytest.mark.parametrize(
    "handler",
    [
        _raising(httpx.ConnectError),
        _raising(httpx.ReadError),
        _raising(httpx.ReadTimeout),
        _raising(httpx.ConnectTimeout),
        _raising(httpx.RemoteProtocolError),
        _raising(httpx.ProxyError),
        _status(429),
        _status(502),
        _status(503),
        _status(522),
    ],
    ids=[
        "refused",
        "reset",
        "read-timeout",
        "connect-timeout",
        "disconnect",
        "proxy",
        "429",
        "502",
        "503",
        "522",
    ],
)
async def test_transient_parser_failures_name_the_parser_unavailable(
    parser_service: list[Handler],
    tmp_path: Path,
    parser: str,
    handler: Handler,
) -> None:
    parser_service.append(handler)
    assert apply_parser_outage_reporting() is True

    with pytest.raises(ParserUnavailableError) as raised:
        await _download(parser, tmp_path)

    assert str(raised.value) == "Document parser is temporarily unavailable"
    # The verdict carries no cause, so classifying it never reads client text.
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert classify_transient_dependency(raised.value) == "parser"


@pytest.mark.parametrize("parser", ["mineru", "docling"])
@pytest.mark.parametrize(
    "filename",
    ["db_schema.pdf", "unsupported-formats.pdf", "credentials-policy.pdf"],
)
async def test_an_outage_is_named_whatever_the_file_or_response_text_says(
    parser_service: list[Handler],
    tmp_path: Path,
    parser: str,
    filename: str,
) -> None:
    # LightRAG's error text names the uploaded file and quotes the response
    # body, so words the provider classification reads as a rejection
    # ("schema", "unsupported", "credential") must not decide a parser outage.
    parser_service.append(_status(503, detail="worker pool unsupported state; retry later"))
    apply_parser_outage_reporting()

    with pytest.raises(ParserUnavailableError):
        await _download(parser, tmp_path, filename)


@pytest.mark.parametrize("parser", ["mineru", "docling"])
async def test_an_outage_is_named_whatever_the_endpoint_is_called(
    parser_service: list[Handler],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parser: str,
) -> None:
    # MinerU's transport error text quotes the configured endpoint.
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://credential-schema-parser.test")
    monkeypatch.setenv("DOCLING_ENDPOINT", "http://credential-schema-parser.test")
    parser_service.append(_raising(httpx.ConnectError))
    apply_parser_outage_reporting()

    with pytest.raises(ParserUnavailableError):
        await _download(parser, tmp_path, "unsupported_schema.pdf")


@pytest.mark.parametrize("parser", ["mineru", "docling"])
@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 413, 422, 501])
async def test_parser_rejections_stay_document_failures_with_their_status(
    parser_service: list[Handler],
    tmp_path: Path,
    parser: str,
    status: int,
) -> None:
    parser_service.append(_status(status))
    apply_parser_outage_reporting()

    with pytest.raises(RuntimeError, match=f"HTTP {status}") as raised:
        await _download(parser, tmp_path)

    assert not isinstance(raised.value, ParserUnavailableError)
    assert getattr(raised.value, "status_code", None) == status
    assert classify_transient_dependency(raised.value) is None


@pytest.mark.parametrize("parser", ["mineru", "docling"])
async def test_misconfigured_parser_url_is_not_an_outage(
    parser_service: list[Handler],
    tmp_path: Path,
    parser: str,
) -> None:
    parser_service.append(_raising(httpx.UnsupportedProtocol))
    apply_parser_outage_reporting()

    with pytest.raises((RuntimeError, httpx.UnsupportedProtocol)) as raised:
        await _download(parser, tmp_path)

    assert not isinstance(raised.value, ParserUnavailableError)


@pytest.mark.parametrize("parser", ["mineru", "docling"])
@pytest.mark.parametrize(
    "cause",
    [
        lambda: ssl.SSLCertVerificationError(1, "[SSL: CERTIFICATE_VERIFY_FAILED]"),
        lambda: _tls_error("WRONG_VERSION_NUMBER"),
    ],
    ids=["tls-certificate", "https-to-plain-http"],
)
async def test_misconfigured_parser_endpoint_is_not_an_outage(
    parser_service: list[Handler],
    tmp_path: Path,
    parser: str,
    cause: Callable[[], BaseException],
) -> None:
    parser_service.append(_connect_failure(cause))
    apply_parser_outage_reporting()

    with pytest.raises((RuntimeError, httpx.ConnectError)) as raised:
        await _download(parser, tmp_path)

    assert not isinstance(raised.value, ParserUnavailableError)


@pytest.mark.parametrize("parser", ["mineru", "docling"])
@pytest.mark.parametrize(
    "cause",
    [
        # Compose's DNS answers NXDOMAIN for a parser service that is stopped or
        # restarting, which is exactly how a parser killed by a document looks.
        lambda: socket.gaierror(socket.EAI_NONAME, "nodename nor servname provided"),
        lambda: socket.gaierror(socket.EAI_AGAIN, "Temporary failure in name resolution"),
        lambda: ssl.SSLEOFError(8, "EOF occurred in violation of protocol"),
        lambda: ConnectionRefusedError(61, "Connection refused"),
    ],
    ids=["dns-no-such-name", "dns-temporary-failure", "tls-dropped-mid-handshake", "refused"],
)
async def test_unreachable_parser_endpoint_is_an_outage(
    parser_service: list[Handler],
    tmp_path: Path,
    parser: str,
    cause: Callable[[], BaseException],
) -> None:
    parser_service.append(_connect_failure(cause))
    apply_parser_outage_reporting()

    with pytest.raises(ParserUnavailableError):
        await _download(parser, tmp_path)


@pytest.mark.usefixtures("unpatched_clients")
@pytest.mark.parametrize("parser", ["mineru", "docling"])
async def test_a_real_tls_handshake_reset_is_an_outage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parser: str,
) -> None:
    # The real httpcore chain for a peer resetting the handshake carries an
    # implicit SSLWantReadError context; it is a dropped connection.
    async def reset(_reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        connection = writer.get_extra_info("socket")
        connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        writer.transport.abort()

    server = await asyncio.start_server(reset, "127.0.0.1", 0)
    endpoint = f"https://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    monkeypatch.setenv("MINERU_API_MODE", "local")
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", endpoint)
    monkeypatch.setenv("DOCLING_ENDPOINT", endpoint)
    apply_parser_outage_reporting()
    try:
        with pytest.raises(ParserUnavailableError):
            await _download(parser, tmp_path)
    finally:
        server.close()
        await server.wait_closed()


async def test_exhausted_polling_budget_is_not_an_outage(
    parser_service: list[Handler],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A document the parser cannot finish within the budget would exhaust it on
    # every attempt, so waiting for the parser to recover cannot help.
    monkeypatch.setenv("MINERU_MAX_POLLS", "1")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"task_id": "task-1"})
        return httpx.Response(200, json={"status": "processing"})

    parser_service.append(handler)
    apply_parser_outage_reporting()

    with pytest.raises(TimeoutError, match="polling timeout"):
        await _download("mineru", tmp_path)


async def test_parser_reported_conversion_failure_is_not_an_outage(
    parser_service: list[Handler],
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"task_id": "task-1"})
        return httpx.Response(200, json={"status": "failed", "error": "encrypted PDF"})

    parser_service.append(handler)
    apply_parser_outage_reporting()

    with pytest.raises(RuntimeError, match="encrypted PDF") as raised:
        await _download("mineru", tmp_path)

    assert not isinstance(raised.value, ParserUnavailableError)


@pytest.mark.usefixtures("parser_service")
def test_both_parser_clients_are_patched_once() -> None:
    # A per-file parser directive can route a document to either engine.
    assert apply_parser_outage_reporting() is True
    assert apply_parser_outage_reporting() is False

    for _module, client_class in _CLIENTS.values():
        download = client_class.download_into
        assert inspect.unwrap(download) is not download
        assert inspect.unwrap(inspect.unwrap(download)) is inspect.unwrap(download)


@pytest.mark.usefixtures("parser_service")
@pytest.mark.parametrize("parser", ["mineru", "docling"])
def test_the_status_stamp_passes_every_argument_through(
    monkeypatch: pytest.MonkeyPatch,
    parser: str,
) -> None:
    # An upstream call site gaining an argument must reach LightRAG unchanged
    # rather than fail every parse inside the stamping wrapper.
    module, _client_class = _CLIENTS[parser]
    received: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def upstream(*args: Any, **kwargs: Any) -> None:
        received.append((args, kwargs))
        raise RuntimeError("parser upload failed: HTTP 503 busy")

    monkeypatch.setattr(module, "raise_for_status_with_detail", upstream)
    apply_parser_outage_reporting()
    response = httpx.Response(503)

    with pytest.raises(RuntimeError) as raised:
        module.raise_for_status_with_detail(
            response, "parser upload", body=b"busy", detail_limit=200
        )

    assert received == [((response, "parser upload"), {"body": b"busy", "detail_limit": 200})]
    assert getattr(raised.value, "status_code", None) == 503


def test_lightrag_records_the_verdict_that_is_recognized_afterwards() -> None:
    fields, metadata = doc_status_parse_failure_fields(
        ParserUnavailableError(),
        status_doc={"content_summary": "", "metadata": {}},
        engine_hint="mineru",
    )

    assert metadata["error_stage"] == "parse"
    assert parser_unavailable_recorded({"status": "failed", **fields}) is True
    rejected, _ = doc_status_parse_failure_fields(
        RuntimeError("MinerU local parse failed for task t: encrypted PDF"),
        status_doc={"content_summary": "", "metadata": {}},
        engine_hint="mineru",
    )
    assert parser_unavailable_recorded({"status": "failed", **rejected}) is False
    assert parser_unavailable_recorded(None) is False
