# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The one durable-event streaming contract REST and the browser both follow.

Cursor resolution, keepalive cadence, and subscriber detach are transport
neutral: both transports resolve cursors with one function, whose forms the REST
endpoint tests prove, and the follow loop is proven once here. Each transport
contributes only its event-to-frame projection. The two projections must stay
distinct: REST serves the canonical result, the browser serves rendered
presentation.
"""

import asyncio
import inspect
import json
from collections.abc import AsyncIterator
from functools import partial
from typing import Any

import pytest

import dlightrag.adapters.http.browser.answer_events as web_events
import dlightrag.adapters.http.browser.routes.chat as web_routes
import dlightrag.adapters.http.rest.routes.runs as rest_routes
from dlightrag.adapters.http.streaming import answer_stream
from dlightrag.adapters.http.streaming.answer_stream import follow_frames, resume_cursor
from dlightrag.application.runs import RunEvent
from tests.unit.web.answer_run_fixtures import stored_result

_RENDERERS = {
    "rest": partial(
        rest_routes.run_frame,
        downloadable_workspaces=None,
        visual_workspaces=None,
        run_id="019893f4-0000-7000-8000-000000000001",
        run_kind="answer",
        mask_reader_text=lambda text: text,
    ),
    "web": partial(
        web_events.browser_frame,
        downloadable_workspaces=None,
        visual_workspaces=None,
        mask_reader_text=lambda text: text,
    ),
}


def _event(sequence: int, event_type: str, payload: dict[str, Any]) -> RunEvent:
    import datetime

    return RunEvent(
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

    async def events(self) -> AsyncIterator[RunEvent]:
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

    frames = follow_frames(subscription.events(), render)
    first = await anext(frames)
    second = await anext(frames)
    released.set()
    # A keepalive may still land between the release and the event on a busy host; the
    # contract is that the event arrives, not that nothing precedes it.
    event = await anext(frames)
    while event == ": keepalive\n\n":
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

    frames = follow_frames(subscription.events(), render)
    assert await anext(frames) == ": keepalive\n\n"
    await frames.aclose()

    assert subscription.closed is True


@pytest.mark.parametrize("render", list(_RENDERERS.values()), ids=list(_RENDERERS))
async def test_cancelling_a_waiting_subscriber_propagates_and_still_detaches(
    render: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(answer_stream, "SSE_KEEPALIVE_SECONDS", 60.0)
    subscription = _QuietSubscription()
    frames = follow_frames(subscription.events(), render)
    waiting = asyncio.create_task(anext(frames))

    await asyncio.sleep(0)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting

    assert subscription.closed is True


@pytest.mark.parametrize("render", list(_RENDERERS.values()), ids=list(_RENDERERS))
async def test_the_stream_ends_when_the_run_committed_its_terminal_event(render: Any) -> None:
    async def _events() -> AsyncIterator[RunEvent]:
        yield _event(1, "progress", {"phase": "planning"})
        yield _event(2, "done", {"status": "cancelled"})

    frames = [frame async for frame in follow_frames(_events(), render)]

    assert [frame.split("\n")[0] for frame in frames] == ["id: 1", "id: 2"]


# ---------------------------------------------------------------------------
# Two distinct projections
# ---------------------------------------------------------------------------


async def _done_data(render: Any) -> Any:
    frame = render(_event(9, "done", {"status": "succeeded", "result": stored_result()}))
    if inspect.isawaitable(frame):
        frame = await frame
    assert frame.startswith("id: 9\nevent: done\n")
    return json.loads(frame.split("data: ", 1)[1].strip())


async def test_rest_serves_the_canonical_result_and_the_browser_serves_presentation() -> None:
    canonical = await _done_data(_RENDERERS["rest"])
    browser = await _done_data(_RENDERERS["web"])

    assert set(canonical) == {"status", "result"}
    presentation = browser["presentation"]
    assert canonical["result"]["answer"] == presentation["answer_text"]
    # Retrieval counts stay on the canonical result; the browser shows published sources.
    assert set(browser) == {"status", "presentation", "usage"}
    assert browser["usage"] == canonical["result"].get("usage", {})
    assert "<" in presentation["parts"][0]["html"]
    assert set(presentation) == {
        "answer_text",
        "parts",
        "sources",
        "evidence_images",
        "link_cards",
        "video_links",
        "artifacts",
        "artifact_outcome",
    }
    assert presentation["artifacts"] == []
    assert "result" not in browser


async def test_both_transports_carry_the_facts_a_settled_tool_reported() -> None:
    """A step card reads the same structure from a live frame as from the stored transcript."""
    facts = {
        "provider": "exa",
        "hits": 2,
        "degraded": False,
        "cost": 0.0123,
        "sources": [{"url": "https://example.org/a", "domain": "example.org"}],
    }
    payload = {
        "tool_name": "search_web",
        "call_id": "call-1",
        "source_index": 0,
        "outcome": "succeeded",
        "duration_ms": 812,
        "facts": facts,
    }

    frames: list[str] = []
    for render in _RENDERERS.values():
        frame = render(_event(7, "tool_end", payload))
        if inspect.isawaitable(frame):
            frame = await frame
        frames.append(frame)

    for frame in frames:
        assert frame.startswith("id: 7\nevent: tool_end\n")
        data = json.loads(frame.split("data: ", 1)[1].strip())
        assert data["facts"] == facts
        assert data["duration_ms"] == 812


async def test_a_tool_event_without_facts_carries_none() -> None:
    frames: list[str] = []
    for render in _RENDERERS.values():
        frame = render(_event(8, "tool_end", {"tool_name": "bash", "call_id": "call-2"}))
        if inspect.isawaitable(frame):
            frame = await frame
        frames.append(frame)

    for frame in frames:
        data = json.loads(frame.split("data: ", 1)[1].strip())
        assert "facts" not in data


async def test_a_frame_without_a_content_gate_omits_facts() -> None:
    """The canonical frame never ships a structured fact it could not mask (ADR 0061)."""
    payload = {
        "tool_name": "search_web",
        "call_id": "call-1",
        "source_index": 0,
        "outcome": "succeeded",
        "facts": {"query": "a query a reader's gate could not read"},
    }
    frame = rest_routes.run_frame(
        _event(9, "tool_end", payload),
        downloadable_workspaces=None,
        visual_workspaces=None,
        run_id="019893f4-0000-7000-8000-000000000001",
        run_kind="answer",
    )
    data = json.loads(frame.split("data: ", 1)[1].strip())
    assert "facts" not in data
