# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""A parser service outage is named at LightRAG's parser transport boundary."""

import inspect
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
def parser_service(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[Handler]]:
    """Serve both parser clients from one in-test handler on unpatched clients.

    Teardown restores whatever client methods were installed before the test.
    """
    for module, client_class in _CLIENTS.values():
        monkeypatch.setattr(
            client_class, "download_into", inspect.unwrap(client_class.download_into)
        )
        monkeypatch.setattr(
            module,
            "raise_for_status_with_detail",
            inspect.unwrap(module.raise_for_status_with_detail),
        )
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


async def _download(parser: str, tmp_path: Path) -> None:
    source = tmp_path / "report.pdf"
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


def _status(status: int) -> Handler:
    return lambda _request: httpx.Response(status, json={"detail": "parser says no"})


@pytest.mark.parametrize("parser", ["mineru", "docling"])
@pytest.mark.parametrize(
    "handler",
    [
        _raising(httpx.ConnectError),
        _raising(httpx.ReadError),
        _raising(httpx.ReadTimeout),
        _raising(httpx.ConnectTimeout),
        _raising(httpx.RemoteProtocolError),
        _status(429),
        _status(502),
        _status(503),
    ],
    ids=["refused", "reset", "read-timeout", "connect-timeout", "disconnect", "429", "502", "503"],
)
async def test_transient_parser_failures_name_the_parser_unavailable(
    parser_service: list[Handler],
    tmp_path: Path,
    parser: str,
    handler: Handler,
) -> None:
    parser_service.append(handler)
    assert apply_parser_outage_reporting(docling_active=parser == "docling") is True

    with pytest.raises(ParserUnavailableError) as raised:
        await _download(parser, tmp_path)

    assert str(raised.value) == "Document parser is temporarily unavailable"
    assert raised.value.__cause__ is not None
    assert classify_transient_dependency(raised.value) == "parser"


@pytest.mark.parametrize("parser", ["mineru", "docling"])
@pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 422, 501])
async def test_parser_rejections_stay_document_failures_with_their_status(
    parser_service: list[Handler],
    tmp_path: Path,
    parser: str,
    status: int,
) -> None:
    parser_service.append(_status(status))
    apply_parser_outage_reporting(docling_active=parser == "docling")

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
    apply_parser_outage_reporting(docling_active=parser == "docling")

    with pytest.raises((RuntimeError, httpx.UnsupportedProtocol)) as raised:
        await _download(parser, tmp_path)

    assert not isinstance(raised.value, ParserUnavailableError)


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
    apply_parser_outage_reporting(docling_active=False)

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
    apply_parser_outage_reporting(docling_active=False)

    with pytest.raises(RuntimeError, match="encrypted PDF") as raised:
        await _download("mineru", tmp_path)

    assert not isinstance(raised.value, ParserUnavailableError)


@pytest.mark.usefixtures("parser_service")
def test_patch_installs_only_on_the_active_parser_and_is_idempotent() -> None:
    assert apply_parser_outage_reporting(docling_active=False) is True
    assert apply_parser_outage_reporting(docling_active=False) is False

    mineru_download = mineru_client.MinerURawClient.download_into
    docling_download = docling_client.DoclingRawClient.download_into
    assert inspect.unwrap(mineru_download) is not mineru_download
    assert inspect.unwrap(docling_download) is docling_download


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
