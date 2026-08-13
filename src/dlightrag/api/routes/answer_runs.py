# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The durable Answer run REST contract: create, status, events, cancel.

``POST /answer`` always accepts, persists, and returns a 202 descriptor; the run
outlives its creating request. Every read is owner-scoped, so an unknown run and
another owner's run are indistinguishable. Stored results carry transport-neutral
identities only, and each authenticated read projects fresh URLs from them.
"""

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import ValidationError
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from dlightrag.access_control import AccessAction
from dlightrag.api.auth import UserContext, get_current_user
from dlightrag.api.models import (
    ANSWER_REQUEST_PART_MAX_BYTES,
    AnswerRequest,
    AnswerRunDescriptor,
    AnswerRunStatusResponse,
)
from dlightrag.api.principal import owner_id_from_user
from dlightrag.app_state import request_config
from dlightrag.config import AnswerConfig
from dlightrag.core.answer_runs.execution import (
    AnswerRunInput,
    AttachmentReference,
    LinkReference,
)
from dlightrag.core.answer_runs.results import project_answer_result
from dlightrag.core.client_contracts import conversation_history_as_dicts, model_dump_json_safe
from dlightrag.core.retrieval.source_links import SourceDownloadLinkBuilder
from dlightrag.sourcing.source_contract import safe_source_filename
from dlightrag.storage.answer_runs import (
    AnswerRunEvent,
    AnswerRunRecord,
    IdempotencyKeyConflict,
    artifact_digest,
)

from .deps import authorized_workspaces, get_manager, resolve_authorized_query_workspaces

logger = logging.getLogger(__name__)
router = APIRouter()

#: A queued or quiet run keeps its connection alive with comments, not events.
SSE_KEEPALIVE_SECONDS = 10.0
_KEEPALIVE_FRAME = ": keepalive\n\n"

_ALLOWED_ANSWER_PARTS = {"request", "attachments"}
_MAX_ANSWER_FORM_FIELDS = 8
# Comfortably holds the JSON `request` part (query, history, filters, links) so a
# small per-attachment cap never truncates the request envelope.
_ANSWER_REQUEST_PART_CEILING = ANSWER_REQUEST_PART_MAX_BYTES


@dataclass(frozen=True, slots=True)
class _UploadedAttachment:
    """One multipart file admitted before the run-creation transaction."""

    filename: str
    mime_type: str
    content: bytes


# ---------------------------------------------------------------------------
# Request parsing
# ---------------------------------------------------------------------------


def _enforce_answer_attachment_count(count: int, max_attachments: int) -> None:
    """Reject over-limit attachment counts with a stable 413 and the safe limit."""
    if count > max_attachments:
        raise HTTPException(
            status_code=413,
            detail=f"Too many attachments; at most {max_attachments} are allowed",
        )


async def _parse_answer_body(
    request: Request, answer_cfg: AnswerConfig
) -> tuple[AnswerRequest, list[_UploadedAttachment]]:
    """Parse a JSON or multipart answer request bounded by attachment limits.

    JSON bodies carry the complete request with optional HTTPS link descriptors.
    Multipart bodies carry exactly one JSON ``request`` part plus repeated
    ``attachments`` file parts; uploaded files and JSON links may mix. Count,
    per-attachment, and total-byte admission are enforced here, before the run is
    accepted, without buffering an unbounded body.
    """
    content_type = request.headers.get("content-type", "").lower()
    if "multipart/form-data" not in content_type:
        try:
            body = AnswerRequest.model_validate_json(await request.body())
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        _enforce_answer_attachment_count(len(body.attachments or []), answer_cfg.max_attachments)
        return body, []

    max_attachments = answer_cfg.max_attachments
    max_item = max(1, answer_cfg.max_attachment_bytes)
    max_total = answer_cfg.max_total_attachment_bytes
    try:
        form = await request.form(
            max_files=max_attachments + 2,
            max_fields=_MAX_ANSWER_FORM_FIELDS,
            max_part_size=_ANSWER_REQUEST_PART_CEILING,
        )
    except StarletteHTTPException as exc:
        detail = str(exc.detail)
        if exc.status_code == 400 and detail.startswith(
            ("Too many files.", "Too many fields.", "Part exceeded maximum size")
        ):
            raise HTTPException(status_code=413, detail=detail) from exc
        raise
    try:
        unexpected = sorted({key for key, _ in form.multi_items()} - _ALLOWED_ANSWER_PARTS)
        if unexpected:
            raise HTTPException(
                status_code=400,
                detail=f"Unexpected multipart field(s): {', '.join(unexpected)}",
            )
        request_parts = form.getlist("request")
        if len(request_parts) != 1:
            raise HTTPException(
                status_code=400,
                detail="multipart answer requires exactly one 'request' part",
            )
        raw_request = request_parts[0]
        if isinstance(raw_request, StarletteUploadFile):
            if raw_request.size is not None and raw_request.size > _ANSWER_REQUEST_PART_CEILING:
                raise HTTPException(status_code=413, detail="Answer request part is too large")
            request_json = await raw_request.read(_ANSWER_REQUEST_PART_CEILING + 1)
            if len(request_json) > _ANSWER_REQUEST_PART_CEILING:
                raise HTTPException(status_code=413, detail="Answer request part is too large")
        else:
            request_json = raw_request
        try:
            body = AnswerRequest.model_validate_json(request_json)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc

        uploads: list[_UploadedAttachment] = []
        total = 0
        for part in form.getlist("attachments"):
            if not isinstance(part, StarletteUploadFile):
                raise HTTPException(
                    status_code=400, detail="'attachments' parts must be uploaded files"
                )
            if part.size is not None and part.size > max_item:
                raise HTTPException(
                    status_code=413, detail="An attachment exceeds the per-attachment size limit"
                )
            data = await part.read(max_item + 1)
            if len(data) > max_item:
                raise HTTPException(
                    status_code=413, detail="An attachment exceeds the per-attachment size limit"
                )
            total += len(data)
            if total > max_total:
                raise HTTPException(
                    status_code=413, detail="Attachments exceed the total size limit"
                )
            uploads.append(
                _UploadedAttachment(
                    filename=safe_source_filename(part.filename),
                    mime_type=part.content_type or "application/octet-stream",
                    content=data,
                )
            )
        _enforce_answer_attachment_count(
            len(body.attachments or []) + len(uploads), max_attachments
        )
        return body, uploads
    finally:
        await form.close()


def _idempotency_key(request: Request) -> str | None:
    """Read the run's optional replay key; a blank header is no key at all.

    An empty or whitespace-only header would otherwise become a real owner-unique
    key that unrelated requests collide on. A meaningful key is passed through
    verbatim, so a caller's byte-exact key stays its own.
    """
    value = request.headers.get("Idempotency-Key")
    return value if value and value.strip() else None


def _run_input(
    body: AnswerRequest,
    uploads: list[_UploadedAttachment],
    *,
    workspaces: list[str],
) -> AnswerRunInput:
    """Normalize one validated request into the run's immutable input."""
    filters = body.filters.model_dump(exclude_none=True, mode="json") if body.filters else None
    return AnswerRunInput(
        query=body.query,
        workspaces=tuple(workspaces),
        history=tuple(conversation_history_as_dicts(body.history) or ()),
        top_k=body.top_k,
        chunk_top_k=body.chunk_top_k,
        filters=filters,
        semantic_highlights=body.semantic_highlights,
        links=tuple(
            LinkReference(url=link.url, filename=link.filename, ordinal=ordinal)
            for ordinal, link in enumerate(body.attachments or [])
        ),
        attachments=tuple(
            AttachmentReference(
                digest=artifact_digest(upload.content),
                filename=upload.filename,
                mime_type=upload.mime_type,
                ordinal=ordinal,
            )
            for ordinal, upload in enumerate(uploads)
        ),
    )


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def _descriptor(record: AnswerRunRecord) -> dict[str, Any]:
    """Project the owner-scoped URLs of one run; they are never stored."""
    return {
        "run_id": record.run_id,
        "status": record.status,
        "status_url": f"/answer/{record.run_id}",
        "events_url": f"/answer/{record.run_id}/events",
        "cancel_url": f"/answer/{record.run_id}",
    }


async def _status_payload(
    request: Request, user: UserContext, record: AnswerRunRecord
) -> dict[str, Any]:
    """Project one run's authoritative state for this authenticated reader."""
    result: dict[str, Any] | None = None
    if record.result is not None:
        workspaces = [str(value) for value in record.request.get("workspaces") or ()]
        result = project_answer_result(
            record.result,
            source_link_builder=SourceDownloadLinkBuilder(),
            downloadable_workspaces=await authorized_workspaces(
                request, user, workspaces, AccessAction.WORKSPACE_DOWNLOAD_SOURCE
            ),
            visual_workspaces=await authorized_workspaces(
                request, user, workspaces, AccessAction.WORKSPACE_READ_VISUAL_ASSET
            ),
        )
    return {
        **_descriptor(record),
        "phase": record.phase,
        "completed_turns": record.completed_turns,
        "cancel_requested": record.cancel_requested,
        "result": result,
        "error_kind": record.error_kind,
        "error_message": record.error_message,
        "created_at": record.created_at,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/answer", response_model=AnswerRunDescriptor, status_code=202)
async def create_answer_run(
    request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any]:
    """Accept one durable answer run and return its owner-scoped descriptor.

    Accepts ``application/json`` (link descriptors only) or ``multipart/form-data``
    with one JSON ``request`` part plus repeated ``attachments`` files. Uploaded
    bytes and their references are committed with the run itself.
    """
    manager = get_manager(request)
    body, uploads = await _parse_answer_body(request, request_config(request).answer)
    workspaces = await resolve_authorized_query_workspaces(
        request,
        user,
        workspaces=body.workspaces,
        all_workspaces=body.all_workspaces,
    )
    try:
        creation = await manager.astart_answer_run(
            owner_id=owner_id_from_user(user),
            request=_run_input(body, uploads, workspaces=workspaces),
            idempotency_key=_idempotency_key(request),
            attachment_bytes=[upload.content for upload in uploads],
        )
    except IdempotencyKeyConflict:
        raise HTTPException(
            status_code=409,
            detail="Idempotency-Key was reused with a different answer request",
        ) from None
    return _descriptor(creation.run)


@router.get("/answer/{run_id}", response_model=AnswerRunStatusResponse)
async def get_answer_run(
    run_id: str, request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any]:
    """Return one owned run's status and, once it succeeded, its result."""
    manager = get_manager(request)
    record = await manager.aget_answer_run(owner_id=owner_id_from_user(user), run_id=run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Answer run not found")
    return await _status_payload(request, user, record)


@router.delete(
    "/answer/{run_id}",
    response_model=AnswerRunStatusResponse,
    responses={
        202: {
            "model": AnswerRunStatusResponse,
            "description": "Cancellation requested; a running worker must still observe it.",
        }
    },
)
async def cancel_answer_run(
    run_id: str,
    request: Request,
    response: Response,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    """Request cancellation; repeating it on a terminal run is a no-op."""
    manager = get_manager(request)
    outcome = await manager.acancel_answer_run(owner_id=owner_id_from_user(user), run_id=run_id)
    if outcome.outcome == "unknown" or outcome.run is None:
        raise HTTPException(status_code=404, detail="Answer run not found")
    # 202 only while a running worker still has to observe the request.
    response.status_code = 202 if outcome.outcome == "pending" else 200
    return await _status_payload(request, user, outcome.run)


@router.get("/answer/{run_id}/events")
async def stream_answer_run_events(
    run_id: str, request: Request, user: UserContext = Depends(get_current_user)
) -> StreamingResponse:
    """Replay this run's durable events from a cursor, then follow it live."""
    manager = get_manager(request)
    cursor = _resume_cursor(request)
    owner_id = owner_id_from_user(user)
    record = await manager.aget_answer_run(owner_id=owner_id, run_id=run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Answer run not found")
    if record.events_trimmed_at is not None:
        raise HTTPException(
            status_code=410,
            detail="Answer run events expired; read its result from the status endpoint",
        )
    workspaces = [str(value) for value in record.request.get("workspaces") or ()]
    downloadable = await authorized_workspaces(
        request, user, workspaces, AccessAction.WORKSPACE_DOWNLOAD_SOURCE
    )
    visual = await authorized_workspaces(
        request, user, workspaces, AccessAction.WORKSPACE_READ_VISUAL_ASSET
    )
    events = await manager.asubscribe_answer_run(
        owner_id=owner_id, run_id=run_id, after_sequence=cursor
    )
    return StreamingResponse(
        _sse_frames(events, downloadable_workspaces=downloadable, visual_workspaces=visual),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _resume_cursor(request: Request) -> int:
    """Resolve the durable sequence this subscriber resumes after.

    An empty ``Last-Event-ID`` is how a browser reports "no cursor yet"; an
    explicitly supplied ``after`` is always parsed strictly.
    """
    header = request.headers.get("Last-Event-ID")
    query = request.query_params.get("after")
    from_header = _parse_cursor(header) if header else None
    from_query = _parse_cursor(query) if query is not None else None
    if from_header is not None and from_query is not None and from_header != from_query:
        raise HTTPException(
            status_code=400, detail="Last-Event-ID and 'after' request different cursors"
        )
    if from_query is not None:
        return from_query
    return from_header or 0


def _parse_cursor(value: str | None) -> int:
    if value is None or not value.isdigit():
        raise HTTPException(status_code=400, detail="Event cursor must be a non-negative integer")
    return int(value)


async def _sse_frames(
    events: AsyncIterator[AnswerRunEvent],
    *,
    downloadable_workspaces: set[str] | None,
    visual_workspaces: set[str] | None,
) -> AsyncGenerator[str]:
    """Serialize durable events, keeping a quiet run's connection alive.

    Closing this generator detaches one subscriber. It never cancels the run and
    never writes durable state.
    """
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
            yield _sse_frame(
                event,
                downloadable_workspaces=downloadable_workspaces,
                visual_workspaces=visual_workspaces,
            )
    finally:
        if pending is not None:
            pending.cancel()
            with contextlib.suppress(BaseException):
                await pending
        with contextlib.suppress(Exception):
            await events.aclose()  # pyright: ignore[reportAttributeAccessIssue]


def _sse_frame(
    event: AnswerRunEvent,
    *,
    downloadable_workspaces: set[str] | None,
    visual_workspaces: set[str] | None,
) -> str:
    payload = dict(event.payload)
    stored = payload.get("result")
    if isinstance(stored, dict):
        payload["result"] = project_answer_result(
            stored,
            source_link_builder=SourceDownloadLinkBuilder(),
            downloadable_workspaces=downloadable_workspaces,
            visual_workspaces=visual_workspaces,
        )
    data = json.dumps(model_dump_json_safe(payload), ensure_ascii=False)
    return f"id: {event.sequence}\nevent: {event.event_type}\ndata: {data}\n\n"


__all__ = ["SSE_KEEPALIVE_SECONDS", "router"]
