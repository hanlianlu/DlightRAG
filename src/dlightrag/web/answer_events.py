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

from dlightrag.answer.runs.results import project_answer_result
from dlightrag.answer.sources import SourceDownloadLinkBuilder
from dlightrag.api.answer_stream import sse_frame
from dlightrag.runtime import AnswerRunEvent
from dlightrag.web.conversations import WEB_IMAGE_URL_BASE, WEB_SOURCE_DOWNLOAD_BASE
from dlightrag.web.events import AnswerDoneEvent, AnswerErrorEvent, AnswerProgressEvent
from dlightrag.web.presentation import build_answer_presentation


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
        image_url_prefix=WEB_IMAGE_URL_BASE,
    )
    answer = str(projected["answer"])
    handle = projected.get("primary_report")
    return AnswerDoneEvent(
        status="succeeded",
        usage=dict(projected.get("usage") or {}),
        evidence=dict(projected.get("evidence") or {}),
        presentation=build_answer_presentation(
            answer=answer,
            sources=projected["sources"],
            answer_images=projected["answer_images"],
            primary_report=handle if isinstance(handle, str) and handle else None,
        ),
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
        case "tool_start" | "tool_progress" | "tool_end":
            allowed = {
                "tool_name",
                "call_id",
                "source_position",
                "update_sequence",
                "outcome",
                "duration_ms",
                "elapsed_ms",
                "output_bytes",
                "spill_state",
                "attachment_count",
            }
            return {key: value for key, value in payload.items() if key in allowed}
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
