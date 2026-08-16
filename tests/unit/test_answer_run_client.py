# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The one async REST helper that owns create-and-wait for durable Answer runs."""

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx
import pytest

import dlightrag.sdk.client as client_module
from dlightrag.sdk import (
    EVENT_READ_IDLE_SECONDS,
    MAX_RECONNECT_ATTEMPTS,
    AnswerAttachmentUpload,
    AnswerRunCancelledError,
    AnswerRunClient,
    AnswerRunFailedError,
    parse_sse_frames,
)

_DESCRIPTOR = {
    "run_id": "run-1",
    "status": "queued",
    "status_url": "/answer/run-1",
    "events_url": "/answer/run-1/events",
    "cancel_url": "/answer/run-1",
}
_RESULT = {"answer": "grounded", "contexts": {"chunks": []}}


@pytest.fixture(autouse=True)
def _no_delays(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backoff and poll cadence are timing, not behavior; every test runs at once."""
    monkeypatch.setattr(client_module, "RECONNECT_BACKOFF_SECONDS", 0.0)
    monkeypatch.setattr(client_module, "STATUS_POLL_SECONDS", 0.0)


def _frame(sequence: int, event: str, payload: dict[str, Any]) -> str:
    return f"id: {sequence}\nevent: {event}\ndata: {json.dumps(payload)}\n\n"


def _client(handler) -> tuple[httpx.AsyncClient, AnswerRunClient]:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://rag.test")
    return http, AnswerRunClient(http)


def _dropped_stream(chunks: Sequence[str], error: type[httpx.HTTPError]) -> httpx.Response:
    """An events response that delivers ``chunks`` and then loses its connection."""

    async def _body() -> AsyncIterator[bytes]:
        for chunk in chunks:
            yield chunk.encode()
        raise error("connection dropped")

    return httpx.Response(200, content=_body(), headers={"content-type": "text/event-stream"})


# ---------------------------------------------------------------------------
# SSE parsing
# ---------------------------------------------------------------------------


def test_parser_keeps_a_split_frame_until_it_completes() -> None:
    whole = _frame(1, "token", {"text": "hello"})
    head, tail = whole[:20], whole[20:]

    events, buffer = parse_sse_frames(head)
    assert events == []

    events, buffer = parse_sse_frames(tail, buffer=buffer)
    assert [event.sequence for event in events] == [1]
    assert events[0].payload == {"text": "hello"}
    assert buffer == ""


def test_parser_ignores_keepalive_comments() -> None:
    events, buffer = parse_sse_frames(": keepalive\n\n" + _frame(4, "progress", {"phase": "x"}))

    assert [event.sequence for event in events] == [4]
    assert buffer == ""


# ---------------------------------------------------------------------------
# Create and wait
# ---------------------------------------------------------------------------


async def test_answer_creates_then_follows_events_to_the_result() -> None:
    seen: list[str] = []
    tokens: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path.endswith("/events"):
            body = _frame(1, "token", {"text": "grou"}) + _frame(
                2, "done", {"status": "succeeded", "result": _RESULT}
            )
            return httpx.Response(200, text=body)
        return httpx.Response(202, json=_DESCRIPTOR)

    http, runs = _client(handler)
    async with http:
        result = await runs.answer({"query": "q"}, on_token=tokens.append)

    assert result == _RESULT
    assert tokens == ["grou"]
    assert seen == ["/answer", "/answer/run-1/events"]


async def test_multipart_create_sends_the_request_part_and_files() -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["content_type"] = request.headers["content-type"]
        captured["body"] = request.content
        return httpx.Response(202, json=_DESCRIPTOR)

    http, runs = _client(handler)
    async with http:
        descriptor = await runs.create(
            {"query": "q"},
            attachments=[AnswerAttachmentUpload(filename="a.txt", content=b"bytes")],
            idempotency_key="key-1",
        )

    assert descriptor.run_id == "run-1"
    assert captured["content_type"].startswith("multipart/form-data")
    assert b'name="request"' in captured["body"]
    assert b"bytes" in captured["body"]


async def test_reconnect_resumes_after_the_last_sequence_without_gaps() -> None:
    cursors: list[str | None] = []
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        if not request.url.path.endswith("/events"):
            return httpx.Response(202, json=_DESCRIPTOR)
        cursors.append(request.headers.get("Last-Event-ID"))
        attempts += 1
        if attempts == 1:
            return httpx.Response(200, text=_frame(1, "token", {"text": "a"}))
        return httpx.Response(
            200,
            text=_frame(2, "token", {"text": "b"})
            + _frame(3, "done", {"status": "succeeded", "result": _RESULT}),
        )

    tokens: list[str] = []
    http, runs = _client(handler)
    async with http:
        result = await runs.answer({"query": "q"}, on_token=tokens.append)

    assert result == _RESULT
    assert tokens == ["a", "b"]
    assert cursors == [None, "1"]


@pytest.mark.parametrize("error", [httpx.ReadError, httpx.ReadTimeout])
async def test_a_dropped_stream_resumes_after_the_last_sequence(
    error: type[httpx.HTTPError],
) -> None:
    creates = 0
    cursors: list[str | None] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal creates
        if not request.url.path.endswith("/events"):
            creates += 1
            return httpx.Response(202, json=_DESCRIPTOR)
        cursors.append(request.headers.get("Last-Event-ID"))
        if len(cursors) == 1:
            return _dropped_stream([_frame(1, "token", {"text": "a"})], error)
        return httpx.Response(
            200,
            text=_frame(2, "token", {"text": "b"})
            + _frame(3, "done", {"status": "succeeded", "result": _RESULT}),
        )

    tokens: list[str] = []
    http, runs = _client(handler)
    async with http:
        result = await runs.answer({"query": "q"}, on_token=tokens.append)

    assert result == _RESULT
    assert tokens == ["a", "b"]
    assert cursors == [None, "1"]
    assert creates == 1


async def test_reconnect_attempts_are_bounded() -> None:
    creates = 0
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal creates, attempts
        if not request.url.path.endswith("/events"):
            creates += 1
            return httpx.Response(202, json=_DESCRIPTOR)
        attempts += 1
        return _dropped_stream([], httpx.ReadError)

    http, runs = _client(handler)
    async with http:
        with pytest.raises(httpx.ReadError):
            await runs.answer({"query": "q"})

    assert attempts == MAX_RECONNECT_ATTEMPTS
    assert creates == 1


async def test_the_event_stream_reads_with_a_bounded_idle_timeout() -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events"):
            captured["timeout"] = dict(request.extensions["timeout"])
            return httpx.Response(
                200, text=_frame(1, "done", {"status": "succeeded", "result": _RESULT})
            )
        return httpx.Response(202, json=_DESCRIPTOR)

    http, runs = _client(handler)
    async with http:
        assert await runs.answer({"query": "q"}) == _RESULT

    assert captured["timeout"]["read"] == EVENT_READ_IDLE_SECONDS
    assert all(value is not None for value in captured["timeout"].values())


async def test_expired_event_log_falls_back_to_the_status_result() -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        if request.url.path.endswith("/events"):
            attempts += 1
            return httpx.Response(410, json={"detail": "expired"})
        if request.method == "GET":
            return httpx.Response(200, json={"status": "succeeded", "result": _RESULT})
        return httpx.Response(202, json=_DESCRIPTOR)

    http, runs = _client(handler)
    async with http:
        assert await runs.answer({"query": "q"}) == _RESULT

    assert attempts == 1


async def test_a_status_error_is_never_retried() -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        if request.url.path.endswith("/events"):
            attempts += 1
            return httpx.Response(503, json={"detail": "unavailable"})
        return httpx.Response(202, json=_DESCRIPTOR)

    http, runs = _client(handler)
    async with http:
        with pytest.raises(httpx.HTTPStatusError):
            await runs.answer({"query": "q"})

    assert attempts == 1


async def test_cancelling_the_wait_detaches_without_cancelling_the_run() -> None:
    calls: list[str] = []
    streaming = asyncio.Event()
    forever = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.method} {request.url.path}")
        if not request.url.path.endswith("/events"):
            return httpx.Response(202, json=_DESCRIPTOR)

        async def _body() -> AsyncIterator[bytes]:
            yield _frame(1, "token", {"text": "a"}).encode()
            streaming.set()
            await forever.wait()

        return httpx.Response(200, content=_body())

    http, runs = _client(handler)
    async with http:
        waiting = asyncio.create_task(runs.answer({"query": "q"}))
        await streaming.wait()
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting

    assert calls == ["POST /answer", "GET /answer/run-1/events"]


async def test_a_failed_run_raises_its_public_kind() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events"):
            return httpx.Response(
                200, text=_frame(1, "error", {"kind": "run_abandoned", "message": "gone"})
            )
        return httpx.Response(202, json=_DESCRIPTOR)

    http, runs = _client(handler)
    async with http:
        with pytest.raises(AnswerRunFailedError) as raised:
            await runs.answer({"query": "q"})

    assert raised.value.error_kind == "run_abandoned"


async def test_a_cancelled_run_raises_cancellation() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events"):
            return httpx.Response(200, text=_frame(1, "done", {"status": "cancelled"}))
        return httpx.Response(202, json=_DESCRIPTOR)

    http, runs = _client(handler)
    async with http:
        with pytest.raises(AnswerRunCancelledError):
            await runs.answer({"query": "q"})


async def test_a_stream_that_closes_early_polls_the_run_row() -> None:
    statuses = ["running", "succeeded"]

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events"):
            return httpx.Response(200, text="")
        if request.method == "GET":
            status = statuses.pop(0) if len(statuses) > 1 else statuses[0]
            return httpx.Response(
                200,
                json={"status": status, "result": _RESULT if status == "succeeded" else None},
            )
        return httpx.Response(202, json=_DESCRIPTOR)

    http, runs = _client(handler)
    async with http:
        assert await runs.answer({"query": "q"}) == _RESULT


async def test_cancel_is_a_plain_delete() -> None:
    seen: dict[str, str] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        return httpx.Response(200, json={"status": "cancelled"})

    http, runs = _client(handler)
    async with http:
        assert (await runs.cancel("run-1"))["status"] == "cancelled"

    assert seen == {"method": "DELETE", "path": "/answer/run-1"}
