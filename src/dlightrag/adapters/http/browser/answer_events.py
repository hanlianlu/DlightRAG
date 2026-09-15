# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Project one durable Answer run's events into browser SSE frames.

The browser subscribes to a run it already owns, so this module holds no
execution state: it renames nothing, commits nothing, and cancels nothing. It
contributes only the projection; replay, keepalive, and detach belong to
``dlightrag.adapters.http.streaming.answer_stream``. Each frame carries the run's durable sequence as
its SSE ``id``, so a reconnect resumes with ``Last-Event-ID`` and sees neither a
gap nor a duplicate.

Unlike the REST projection, a browser ``done`` frame carries rendered
presentation -- sanitized ``html``, the answer text, and answer images -- instead
of the canonical stored result.
"""

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from dlightrag.adapters.http.browser.conversations import (
    WEB_IMAGE_URL_BASE,
    WEB_SOURCE_DOWNLOAD_BASE,
)
from dlightrag.adapters.http.browser.events import (
    AnswerDoneEvent,
    AnswerErrorEvent,
    AnswerProgressEvent,
)
from dlightrag.adapters.http.browser.presentation import build_answer_presentation
from dlightrag.adapters.http.streaming.answer_stream import sse_frame
from dlightrag.application.runs import RunEvent
from dlightrag.engine.answer.citations.sources import SourceDownloadLinkBuilder
from dlightrag.engine.answer.results import project_answer_result


def render_done_event(
    payload: dict[str, Any],
    *,
    downloadable_workspaces: set[str] | None,
    visual_workspaces: set[str] | None,
    run_id: str | None = None,
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
        run_id=run_id,
        artifact_url_prefix="/web/api/answer",
    )
    answer = str(projected["answer"])
    return AnswerDoneEvent(
        status="succeeded",
        usage=dict(projected.get("usage") or {}),
        evidence=dict(projected.get("evidence") or {}),
        presentation=build_answer_presentation(
            answer=answer,
            sources=projected["sources"],
            evidence_images=projected["evidence_images"],
            artifacts=projected["artifacts"],
            artifact_outcome=projected["artifact_outcome"],
        ),
    )


def _is_live(event: RunEvent, live_after: int | None) -> bool:
    """Whether this event was committed after the subscriber arrived."""
    return live_after is None or event.sequence > live_after


def _elapsed_ms_since(committed_at: datetime) -> int:
    """Whole milliseconds from one committed event to the frame rendering it.

    This is the one place the browser edge reads a wall clock, and it measures a
    duration rather than an instant, so no reader has to synchronize clocks. A
    deployment whose workers disagree about the clock shows the difference only
    in a replayed counter.
    """
    return max(0, round((datetime.now(UTC) - committed_at).total_seconds() * 1000))


def _browser_payload(
    event: RunEvent,
    *,
    downloadable_workspaces: set[str] | None,
    visual_workspaces: set[str] | None,
    live_after: int | None,
    run_id: str | None,
    tool_labels: Mapping[str, str] | None = None,
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
                "object_label",
            }
            projected = {key: value for key, value in payload.items() if key in allowed}
            label = projected.get("object_label")
            if isinstance(label, str):
                projected["object_label"] = label[:64]
            name = projected.get("tool_name")
            display = tool_labels.get(name) if isinstance(name, str) and tool_labels else None
            if display:
                # Resolved here, never stored: the durable event keeps transport-neutral
                # identity, and only this edge knows how to name an owner's tool.
                projected["tool_label"] = display
            if event.event_type == "tool_start" and not _is_live(event, live_after):
                # A replayed start has already been running for a while. Saying how long
                # lets a reloaded trace keep counting instead of restarting from zero.
                projected["elapsed_ms"] = _elapsed_ms_since(event.created_at)
            return projected
        case "memory_operation_settled":
            allowed = {
                "body",
                "change_id",
                "intent_id",
                "kind",
                "memory_ids",
                "operation",
                "outcome",
                "session_id",
                "supersedes_id",
                "target_change_id",
            }
            safe = {key: value for key, value in payload.items() if key in allowed}
            safe["live"] = _is_live(event, live_after)
            return safe
        case "done":
            done = render_done_event(
                payload,
                downloadable_workspaces=downloadable_workspaces,
                visual_workspaces=visual_workspaces,
                run_id=run_id,
            )
            return {"status": "cancelled"} if done.status == "cancelled" else done
        case _:
            return AnswerErrorEvent(
                kind=str(payload.get("kind") or "answer_stream_failed"),
                message=str(payload.get("message") or "Service error. Please try again."),
            )


def browser_frame(
    event: RunEvent,
    *,
    downloadable_workspaces: set[str] | None = None,
    visual_workspaces: set[str] | None = None,
    live_after: int | None = None,
    run_id: str | None = None,
    tool_labels: Mapping[str, str] | None = None,
) -> str:
    """Render one durable event as the frame this browser session reads."""
    return sse_frame(
        sequence=event.sequence,
        event_type=event.event_type,
        payload=_browser_payload(
            event,
            downloadable_workspaces=downloadable_workspaces,
            visual_workspaces=visual_workspaces,
            live_after=live_after,
            run_id=run_id,
            tool_labels=tool_labels,
        ),
    )


__all__ = ["browser_frame", "render_done_event"]
