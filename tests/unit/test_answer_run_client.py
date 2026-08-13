# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The one async REST helper that owns create-and-wait for durable Answer runs."""

import json
from typing import Any

import httpx
import pytest

from dlightrag.client import (
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


def _frame(sequence: int, event: str, payload: dict[str, Any]) -> str:
    return f"id: {sequence}\nevent: {event}\ndata: {json.dumps(payload)}\n\n"


def _client(handler) -> tuple[httpx.AsyncClient, AnswerRunClient]:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://rag.test")
    return http, AnswerRunClient(http)


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


async def test_expired_event_log_falls_back_to_the_status_result() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events"):
            return httpx.Response(410, json={"detail": "expired"})
        if request.method == "GET":
            return httpx.Response(200, json={"status": "succeeded", "result": _RESULT})
        return httpx.Response(202, json=_DESCRIPTOR)

    http, runs = _client(handler)
    async with http:
        assert await runs.answer({"query": "q"}) == _RESULT


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

    import dlightrag.client as client_module

    original = client_module.STATUS_POLL_SECONDS
    client_module.STATUS_POLL_SECONDS = 0.0
    try:
        http, runs = _client(handler)
        async with http:
            assert await runs.answer({"query": "q"}) == _RESULT
    finally:
        client_module.STATUS_POLL_SECONDS = original


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
