# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Project one durable Answer run's events into browser SSE frames.

The browser subscribes to a run it already owns, so this module holds no
execution state: it renames nothing, commits nothing, and cancels nothing. It
contributes only the projection; replay, keepalive, and detach belong to
``dlightrag.api.answer_stream``. Each frame carries the run's durable sequence as
its SSE ``id``, so a reconnect resumes with ``Last-Event-ID`` and sees neither a
gap nor a duplicate.

Unlike the REST projection, a browser ``done`` frame carries rendered
presentation -- sanitized ``html``, the answer text, and answer images -- instead
of the canonical stored result.
"""

from typing import Any

from dlightrag.api.answer_stream import sse_frame
from dlightrag.core.answer_runs.results import project_answer_result
from dlightrag.core.retrieval.source_links import SourceDownloadLinkBuilder
from dlightrag.storage.answer_runs import AnswerRunEvent
from dlightrag.web.conversations import WEB_SOURCE_DOWNLOAD_BASE
from dlightrag.web.events import AnswerDoneEvent, AnswerErrorEvent, AnswerProgressEvent
from dlightrag.web.safe_html import safe_answer_done


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


def browser_frame(
    event: AnswerRunEvent,
    *,
    downloadable_workspaces: set[str] | None = None,
    visual_workspaces: set[str] | None = None,
) -> str:
    """Render one durable event as the frame this browser session reads."""
    return sse_frame(
        sequence=event.sequence,
        event_type=event.event_type,
        payload=_browser_payload(
            event,
            downloadable_workspaces=downloadable_workspaces,
            visual_workspaces=visual_workspaces,
        ),
    )


__all__ = ["browser_frame", "render_done_event"]
