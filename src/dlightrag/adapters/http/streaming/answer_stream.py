# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Event streaming shared by the Server-Sent Events transports: cursors, keepalive, detachment.

Resuming from a cursor, keeping a quiet subscription's connection alive, and detaching a
subscriber are properties of the SSE event iterator, not of any one route that serves one, so
they live here once. A route contributes only how it renders an event: REST serves the
canonical result and the browser serves rendered presentation, and both then hand an
``EventRenderer`` to ``follow_frames``.

The live browser's frames socket is deliberately not one of them. It is a different protocol
— a WebSocket whose binary opcode is a JPEG and whose text opcode is the state beside it,
with one keepalive of its own — so what it shares with this module is the *shape* of the
race (a pending read against a keepalive clock, so a keepalive cannot lose the event it
raced), not this code. The two keepalive windows are different for that reason:
``SSE_KEEPALIVE_SECONDS`` here is 10 s because a comment frame on a proxied response is
cheap and frequent, while the socket's 30 s heartbeat (``HEARTBEAT_SECONDS`` in
``browser/routes/live_browser.py``) answers Cloudflare's ~100 s idle close with a margin
that survives one lost beat.

Nothing here writes durable state. Closing the returned generator detaches one
subscriber and never cancels the run it was watching.
"""

import asyncio
import contextlib
import inspect
import json
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from typing import Any

from fastapi import Request

from dlightrag.adapters.http.errors import invalid_request
from dlightrag.engine.answer.client_contracts import model_dump_json_safe

#: A queued or quiet SSE subscription keeps its connection alive with comments, not events.
SSE_KEEPALIVE_SECONDS = 10.0
KEEPALIVE_FRAME = ": keepalive\n\n"

#: How one SSE route turns an event into the frame its clients read. Most renderers are a pure
#: function of the event, so their frames are returned; the browser's ``done`` frame also derives a
#: Run's Artifact attestation from that Run's own durable records, so it returns an awaitable
#: instead. Both are the same contract: a route that needs to read something before it can render
#: says so by being ``async``, and every other route keeps the shape its work actually has.
type EventRenderer[T] = Callable[[T], str | Awaitable[str]]


def resume_cursor(request: Request) -> int:
    """Resolve the durable sequence this subscriber resumes after.

    An empty ``Last-Event-ID`` is how a browser reports "no cursor yet"; an
    explicitly supplied ``after`` is always parsed strictly.
    """
    header = request.headers.get("Last-Event-ID")
    query = request.query_params.get("after")
    from_header = _parse_cursor(header, "header", "Last-Event-ID") if header else None
    from_query = _parse_cursor(query, "query", "after") if query is not None else None
    if from_header is not None and from_query is not None and from_header != from_query:
        raise invalid_request(
            "Last-Event-ID and 'after' request different cursors", "query", "after"
        )
    if from_query is not None:
        return from_query
    return from_header or 0


def _parse_cursor(value: str, *location: str) -> int:
    if not value.isdigit():
        raise invalid_request("Event cursor must be a non-negative integer", *location)
    return int(value)


def sse_frame(*, sequence: int, event_type: str, payload: Any) -> str:
    """Serialize one durable event, keyed by the sequence a reconnect resumes from."""
    data = json.dumps(model_dump_json_safe(payload), ensure_ascii=False)
    return f"id: {sequence}\nevent: {event_type}\ndata: {data}\n\n"


async def follow_frames[T](
    events: AsyncIterator[T], render: EventRenderer[T]
) -> AsyncGenerator[str]:
    """Follow one SSE subscription as text frames, keeping it alive while it is quiet.

    Every frame is a string, because that is the whole of what this transport can send, and
    the pending read survives every keepalive: a comment written while nothing is happening
    neither restarts the read nor loses the event that arrives next.
    """
    iterator = events.__aiter__()
    pending: asyncio.Task[T] | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(anext(iterator))
            try:
                event = await asyncio.wait_for(asyncio.shield(pending), SSE_KEEPALIVE_SECONDS)
            except TimeoutError:
                yield KEEPALIVE_FRAME
                continue
            except StopAsyncIteration:
                pending = None
                return
            pending = None
            frame = render(event)
            yield (await frame) if inspect.isawaitable(frame) else frame
    finally:
        if pending is not None:
            pending.cancel()
            # A cancelled child is returned as a value; cancellation of this
            # enclosing subscriber still propagates instead of being swallowed.
            await asyncio.gather(pending, return_exceptions=True)
        with contextlib.suppress(Exception):
            await events.aclose()  # pyright: ignore[reportAttributeAccessIssue]


__all__ = [
    "KEEPALIVE_FRAME",
    "SSE_KEEPALIVE_SECONDS",
    "EventRenderer",
    "follow_frames",
    "resume_cursor",
    "sse_frame",
]
