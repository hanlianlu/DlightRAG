# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The one durable-event streaming contract REST and the browser both follow.

Cursor resolution, keepalive cadence, and subscriber detach are transport
neutral, so they are proven once here and each transport contributes only its
event-to-frame projection. The two projections must stay distinct: REST serves
the canonical result, the browser serves rendered presentation.
"""

import asyncio
import json
from collections.abc import AsyncIterator
from functools import partial
from typing import Any

import pytest
from fastapi import HTTPException
from starlette.requests import Request

import dlightrag.api.routes.answer_runs as rest_routes
import dlightrag.web.answer_events as web_events
import dlightrag.web.routes.chat as web_routes
from dlightrag.api import answer_stream
from dlightrag.api.answer_stream import follow_run_frames, resume_cursor
from dlightrag.storage.answer_runs import AnswerRunEvent
from tests.unit.web.answer_run_fixtures import stored_result

_RENDERERS = {
    "rest": partial(
        rest_routes.answer_run_frame, downloadable_workspaces=None, visual_workspaces=None
    ),
    "web": partial(web_events.browser_frame, downloadable_workspaces=None, visual_workspaces=None),
}


def _request(*, header: str | None = None, query: str | None = None) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/answer/run/events",
            "query_string": b"" if query is None else f"after={query}".encode(),
            "headers": [] if header is None else [(b"last-event-id", header.encode())],
        }
    )


def _event(sequence: int, event_type: str, payload: dict[str, Any]) -> AnswerRunEvent:
    import datetime

    return AnswerRunEvent(
        sequence=sequence,
        event_type=event_type,  # type: ignore[arg-type]
        payload=payload,
        created_at=datetime.datetime.now(datetime.UTC),
    )


class _QuietSubscription:
    """A run that has committed no event yet and never terminates on its own."""

    def __init__(self, release: asyncio.Event | None = None) -> None:
        self.closed = False
        self._release = release

    async def events(self) -> AsyncIterator[AnswerRunEvent]:
        try:
            if self._release is None:
                await asyncio.Event().wait()
            else:
                await self._release.wait()
            yield _event(1, "token", {"text": "Rev"})
        finally:
            self.closed = True


# ---------------------------------------------------------------------------
# One cursor implementation
# ---------------------------------------------------------------------------


def test_both_transports_resolve_the_same_cursor_implementation() -> None:
    assert rest_routes.resume_cursor is resume_cursor
    assert web_routes.resume_cursor is resume_cursor


def test_no_transport_keeps_a_private_cursor_or_keepalive_copy() -> None:
    for module in (rest_routes, web_routes, web_events):
        assert not hasattr(module, "_parse_cursor")
        assert not hasattr(module, "_resume_cursor")
        assert not hasattr(module, "SSE_KEEPALIVE_SECONDS")
        assert not hasattr(module, "_KEEPALIVE_FRAME")


@pytest.mark.parametrize(
    ("header", "query", "expected"),
    [
        (None, None, 0),
        ("7", None, 7),
        (None, "3", 3),
        ("", None, 0),
        ("2", "2", 2),
        (None, "0", 0),
    ],
    ids=["no-cursor", "last-event-id", "after", "blank-last-event-id", "matching", "zero"],
)
def test_the_resume_cursor_comes_from_either_form(
    header: str | None, query: str | None, expected: int
) -> None:
    assert resume_cursor(_request(header=header, query=query)) == expected


@pytest.mark.parametrize(
    ("header", "query"),
    [("1", "2"), (None, "abc"), (None, "-1"), (None, "1.5"), (None, ""), ("abc", None)],
    ids=["conflicting", "non-numeric", "negative", "fractional", "blank-after", "bad-header"],
)
def test_an_unusable_cursor_is_a_400(header: str | None, query: str | None) -> None:
    with pytest.raises(HTTPException) as failure:
        resume_cursor(_request(header=header, query=query))

    assert failure.value.status_code == 400


# ---------------------------------------------------------------------------
# One follow loop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("render", list(_RENDERERS.values()), ids=list(_RENDERERS))
async def test_a_quiet_run_keeps_alive_and_still_delivers_the_next_event(
    render: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(answer_stream, "SSE_KEEPALIVE_SECONDS", 0.01)
    released = asyncio.Event()
    subscription = _QuietSubscription(released)

    frames = follow_run_frames(subscription.events(), render)
    first = await anext(frames)
    second = await anext(frames)
    released.set()
    event = await anext(frames)
    await frames.aclose()

    assert first == second == ": keepalive\n\n"
    assert first.startswith(":")  # a comment consumes no durable sequence
    assert event.startswith("id: 1\nevent: token\n")
    assert subscription.closed is True


@pytest.mark.parametrize("render", list(_RENDERERS.values()), ids=list(_RENDERERS))
async def test_closing_early_detaches_the_subscriber(
    render: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(answer_stream, "SSE_KEEPALIVE_SECONDS", 0.01)
    subscription = _QuietSubscription()

    frames = follow_run_frames(subscription.events(), render)
    assert await anext(frames) == ": keepalive\n\n"
    await frames.aclose()

    assert subscription.closed is True


@pytest.mark.parametrize("render", list(_RENDERERS.values()), ids=list(_RENDERERS))
async def test_cancelling_a_waiting_subscriber_propagates_and_still_detaches(
    render: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(answer_stream, "SSE_KEEPALIVE_SECONDS", 60.0)
    subscription = _QuietSubscription()
    frames = follow_run_frames(subscription.events(), render)
    waiting = asyncio.create_task(anext(frames))

    await asyncio.sleep(0)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting

    assert subscription.closed is True


@pytest.mark.parametrize("render", list(_RENDERERS.values()), ids=list(_RENDERERS))
async def test_the_stream_ends_when_the_run_committed_its_terminal_event(render: Any) -> None:
    async def _events() -> AsyncIterator[AnswerRunEvent]:
        yield _event(1, "progress", {"phase": "planning"})
        yield _event(2, "done", {"status": "cancelled"})

    frames = [frame async for frame in follow_run_frames(_events(), render)]

    assert [frame.split("\n")[0] for frame in frames] == ["id: 1", "id: 2"]


# ---------------------------------------------------------------------------
# Two distinct projections
# ---------------------------------------------------------------------------


def _done_data(render: Any) -> Any:
    frame = render(_event(9, "done", {"status": "succeeded", "result": stored_result()}))
    assert frame.startswith("id: 9\nevent: done\n")
    return json.loads(frame.split("data: ", 1)[1].strip())


def test_rest_serves_the_canonical_result_and_the_browser_serves_presentation() -> None:
    canonical = _done_data(_RENDERERS["rest"])
    browser = _done_data(_RENDERERS["web"])

    assert set(canonical) == {"status", "result"}
    assert canonical["result"]["answer"] == browser["answer"]
    assert set(browser) == {"status", "html", "answer", "answer_images"}
    assert "<" in browser["html"]
    assert "result" not in browser
