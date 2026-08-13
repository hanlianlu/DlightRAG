# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Project one durable Answer run's events into browser SSE frames.

The browser subscribes to a run it already owns, so this module holds no
execution state: it renames nothing, commits nothing, and cancels nothing.
Closing the response detaches one subscriber. Each frame carries the run's
durable sequence as its SSE ``id``, so a reconnect resumes with
``Last-Event-ID`` and sees neither a gap nor a duplicate.
"""

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from dlightrag.core.answer_runs.results import project_answer_result
from dlightrag.core.client_contracts import model_dump_json_safe
from dlightrag.core.retrieval.source_links import SourceDownloadLinkBuilder
from dlightrag.storage.answer_runs import AnswerRunEvent
from dlightrag.web.conversations import WEB_SOURCE_DOWNLOAD_BASE
from dlightrag.web.events import AnswerDoneEvent, AnswerErrorEvent, AnswerProgressEvent
from dlightrag.web.safe_html import safe_answer_done

logger = logging.getLogger(__name__)

#: A queued or quiet run keeps its connection alive with comments, not events.
SSE_KEEPALIVE_SECONDS = 10.0
_KEEPALIVE_FRAME = ": keepalive\n\n"


def render_done_event(
    payload: dict[str, Any],
    *,
    downloadable_workspaces: set[str] | None,
    visual_workspaces: set[str] | None,
) -> AnswerDoneEvent:
    """Derive the finished presentation from the run's canonical result."""
    if str(payload.get("status")) == "cancelled":
        return AnswerDoneEvent(status="cancelled")
    projected = project_answer_result(
        payload.get("result") or {},
        source_link_builder=SourceDownloadLinkBuilder(base_url=WEB_SOURCE_DOWNLOAD_BASE),
        downloadable_workspaces=downloadable_workspaces,
        visual_workspaces=visual_workspaces,
    )
    answer = str(projected["answer"])
    return AnswerDoneEvent(
        status="succeeded",
        html=safe_answer_done(
            answer=answer,
            sources=projected["sources"],
            answer_images=projected["answer_images"],
        ),
        answer=answer,
        answer_images=projected["answer_images"],
    )


def _browser_payload(
    event: AnswerRunEvent,
    *,
    downloadable_workspaces: set[str] | None,
    visual_workspaces: set[str] | None,
) -> Any:
    payload = dict(event.payload)
    match event.event_type:
        case "progress":
            return AnswerProgressEvent(phase=payload["phase"])
        case "token":
            return str(payload.get("text") or "")
        case "reset":
            return {}
        case "done":
            return render_done_event(
                payload,
                downloadable_workspaces=downloadable_workspaces,
                visual_workspaces=visual_workspaces,
            )
        case _:
            return AnswerErrorEvent(
                message=str(payload.get("message") or "Service error. Please try again."),
                error_kind=str(payload.get("kind") or "answer_stream_failed"),
            )


def _frame(sequence: int, event_type: str, payload: Any) -> str:
    data = json.dumps(model_dump_json_safe(payload), ensure_ascii=False)
    return f"id: {sequence}\nevent: {event_type}\ndata: {data}\n\n"


async def stream_answer_events(
    events: AsyncIterator[AnswerRunEvent],
    *,
    downloadable_workspaces: set[str] | None = None,
    visual_workspaces: set[str] | None = None,
) -> AsyncGenerator[str]:
    """Replay and follow one run's durable events as browser SSE frames."""
    iterator = events.__aiter__()
    pending: asyncio.Task[AnswerRunEvent] | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(anext(iterator))
            try:
                event = await asyncio.wait_for(asyncio.shield(pending), SSE_KEEPALIVE_SECONDS)
            except TimeoutError:
                yield _KEEPALIVE_FRAME
                continue
            except StopAsyncIteration:
                pending = None
                return
            pending = None
            yield _frame(
                event.sequence,
                event.event_type,
                _browser_payload(
                    event,
                    downloadable_workspaces=downloadable_workspaces,
                    visual_workspaces=visual_workspaces,
                ),
            )
    finally:
        if pending is not None:
            pending.cancel()
            with contextlib.suppress(BaseException):
                await pending
        with contextlib.suppress(Exception):
            await events.aclose()  # pyright: ignore[reportAttributeAccessIssue]


__all__ = ["SSE_KEEPALIVE_SECONDS", "render_done_event", "stream_answer_events"]
