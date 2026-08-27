# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The durable Answer run REST contract: create, status, events, cancel.

``POST /answer`` always accepts, persists, and returns a 202 descriptor; the run
outlives its creating request. Every read is owner-scoped, so an unknown run and
another owner's run are indistinguishable. Stored results carry transport-neutral
identities only, and each authenticated read projects fresh URLs from them.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from dlightrag.answer.resources.links import answer_link_resources
from dlightrag.answer.resources.models import ResourceInput
from dlightrag.api.answer_stream import follow_run_frames, resume_cursor, sse_frame
from dlightrag.api.auth import get_current_user
from dlightrag.api.models import (
    ANSWER_REQUEST_PART_MAX_BYTES,
    AnswerRequest,
    AnswerResponse,
    AnswerRunDescriptor,
    AnswerRunStatusResponse,
)
from dlightrag.application.access import AccessAction, UserContext, owner_id_from_user
from dlightrag.application.answer_runs import AnswerRequest as ServiceAnswerRequest
from dlightrag.application.answer_runs.client_contracts import conversation_history_as_dicts
from dlightrag.application.answer_runs.results import (
    answer_parts_from_markdown,
    project_answer_result,
    project_report_sources,
)
from dlightrag.application.answer_runs.sources import SourceDownloadLinkBuilder
from dlightrag.application.config import AnswerConfig
from dlightrag.engine.rag.corpus.sources.source_contract import safe_source_filename
from dlightrag.engine.runtime import (
    AnswerRunEvent,
    AnswerRunRecord,
    IdempotencyKeyConflict,
)

from .deps import authorized_workspaces, get_application, resolve_authorized_query_workspaces

logger = logging.getLogger(__name__)
router = APIRouter()

_ALLOWED_ANSWER_PARTS = {"request", "attachments"}
_MAX_ANSWER_FORM_FIELDS = 8
_INERT_SVG_CSP = "sandbox; default-src 'none'; img-src data:"


class _AgentControlBody(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    content: str = Field(min_length=1, max_length=20_000)


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
            max_part_size=ANSWER_REQUEST_PART_MAX_BYTES,
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
            if raw_request.size is not None and raw_request.size > ANSWER_REQUEST_PART_MAX_BYTES:
                raise HTTPException(status_code=413, detail="Answer request part is too large")
            request_json = await raw_request.read(ANSWER_REQUEST_PART_MAX_BYTES + 1)
            if len(request_json) > ANSWER_REQUEST_PART_MAX_BYTES:
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


def _service_request(
    body: AnswerRequest,
    uploads: list[_UploadedAttachment],
    *,
    workspaces: list[str],
) -> ServiceAnswerRequest:
    """Project one validated wire request into the Answer application contract."""
    from dlightrag.engine.rag.retrieval import MetadataFilter

    resources = answer_link_resources(body.attachments)
    resources.extend(
        ResourceInput(
            filename=upload.filename,
            content=upload.content,
            declared_mime=upload.mime_type,
        )
        for upload in uploads
    )
    return ServiceAnswerRequest(
        query=body.query,
        workspaces=tuple(workspaces),
        history=tuple(conversation_history_as_dicts(body.history) or ()),
        top_k=body.top_k,
        chunk_top_k=body.chunk_top_k,
        filters=(
            MetadataFilter.model_validate(body.filters.model_dump(exclude_none=True, mode="json"))
            if body.filters
            else None
        ),
        semantic_highlights=body.semantic_highlights,
        resources=tuple(resources),
        mode=body.mode,
    )


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def _descriptor(record: AnswerRunRecord) -> dict[str, Any]:
    """Project owner-scoped URLs and durable continuation lineage."""
    accepted = record.request_input()
    return {
        "run_id": record.run_id,
        "status": record.status,
        "status_url": f"/answer/{record.run_id}",
        "events_url": f"/answer/{record.run_id}/events",
        "cancel_url": f"/answer/{record.run_id}",
        "parent_run_id": accepted.get("parent_run_id"),
        "continuation_kind": accepted.get("continuation_kind"),
    }


def _published_artifact(
    result: Mapping[str, Any] | None, resource_id: str
) -> Mapping[str, Any] | None:
    for item in (result or {}).get("artifacts") or ():
        if isinstance(item, Mapping) and item.get("resource_id") == resource_id:
            return item
    return None


def _artifact_response_headers(descriptor: Mapping[str, Any], *, download: bool) -> tuple[str, str]:
    media_type = str(descriptor.get("media_type") or "application/octet-stream")
    safe_inline = media_type.startswith("image/") or media_type == "application/pdf"
    effective_type = media_type if safe_inline and not download else "application/octet-stream"
    filename = str(descriptor.get("filename") or "artifact").replace('"', "_")
    disposition = "attachment" if download or not safe_inline else "inline"
    return effective_type, f'{disposition}; filename="{filename}"'


async def _status_payload(
    request: Request, user: UserContext, record: AnswerRunRecord
) -> dict[str, Any]:
    """Project one run's authoritative state for this authenticated reader."""
    result: dict[str, Any] | None = None
    if record.result is not None:
        workspaces = [str(value) for value in record.request_input().get("workspaces") or ()]
        result = project_answer_result(
            record.result,
            source_link_builder=SourceDownloadLinkBuilder(),
            downloadable_workspaces=await authorized_workspaces(
                request, user, workspaces, AccessAction.WORKSPACE_DOWNLOAD_SOURCE
            ),
            visual_workspaces=await authorized_workspaces(
                request, user, workspaces, AccessAction.WORKSPACE_READ_VISUAL_ASSET
            ),
            run_id=record.run_id,
            artifact_url_prefix="/answer",
        )
    return {
        **_descriptor(record),
        "phase": record.phase,
        "durable_progress_version": record.durable_progress_version,
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


@router.get("/answer")
async def list_answer_runs(
    request: Request,
    user: UserContext = Depends(get_current_user),
    after: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """List this owner's durable runs, oldest first."""
    application = get_application(request)
    rows = await application.answers.list(
        owner_id=owner_id_from_user(user),
        after_run_id=after,
        limit=min(max(limit, 1), 100),
    )
    return {"runs": [_descriptor(record) for record in rows]}


@router.get("/answer/{run_id}/artifacts")
async def list_answer_artifacts(
    run_id: str, request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any]:
    application = get_application(request)
    record = await application.answers.get(owner_id=owner_id_from_user(user), run_id=run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Answer run not found")
    if record.result is None:
        raise HTTPException(
            status_code=409,
            detail="Answer artifacts are not available until the run has a stored result",
        )
    projected = project_answer_result(
        record.result,
        run_id=run_id,
        artifact_url_prefix="/answer",
    )
    return {"artifacts": projected["artifacts"], "artifact_outcome": projected["artifact_outcome"]}


@router.get(
    "/answer/{run_id}/artifacts/{resource_id}/presentation",
    response_model=AnswerResponse,
)
async def read_answer_artifact_presentation(
    run_id: str,
    resource_id: str,
    request: Request,
    response: Response,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    """Return one authenticated Markdown Artifact as a typed Answer presentation."""
    application = get_application(request)
    owner_id = owner_id_from_user(user)
    record = await application.answers.get(owner_id=owner_id, run_id=run_id)
    descriptor = _published_artifact(record.result if record else None, resource_id)
    if (
        record is None
        or record.status != "succeeded"
        or descriptor is None
        or descriptor.get("status") != "available"
        or descriptor.get("media_type") != "text/markdown"
    ):
        raise HTTPException(status_code=404, detail="artifact presentation not found")
    blob = await application.answers.read_artifact(
        owner_id=owner_id, run_id=run_id, resource_id=resource_id
    )
    if blob is None:
        raise HTTPException(status_code=404, detail="artifact not found")
    try:
        markdown = blob.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=422, detail="artifact is not UTF-8") from exc

    workspaces = [str(value) for value in record.request_input().get("workspaces") or ()]
    downloadable = await authorized_workspaces(
        request, user, workspaces, AccessAction.WORKSPACE_DOWNLOAD_SOURCE
    )
    visual = await authorized_workspaces(
        request, user, workspaces, AccessAction.WORKSPACE_READ_VISUAL_ASSET
    )
    projected = project_answer_result(
        record.result or {},
        source_link_builder=SourceDownloadLinkBuilder(),
        downloadable_workspaces=downloadable,
        visual_workspaces=visual,
        run_id=run_id,
        artifact_url_prefix="/answer",
    )
    report_sources = (
        project_report_sources(
            record.result or {},
            source_link_builder=SourceDownloadLinkBuilder(),
            downloadable_workspaces=downloadable,
            visual_workspaces=visual,
        )
        if descriptor.get("role") == "primary_report"
        else []
    )
    projected.update(
        answer=markdown,
        parts=answer_parts_from_markdown(
            markdown,
            artifacts=projected["artifacts"],
            evidence_images=[],
        ),
        contexts={},
        references=[
            {"id": source.id, "title": source.title or "Source"} for source in report_sources
        ],
        sources=[source.model_dump() for source in report_sources],
        evidence_images=[],
    )
    response.headers["Cache-Control"] = "private, no-store"
    return projected


@router.get("/answer/{run_id}/artifacts/{resource_id}")
async def read_answer_artifact(
    run_id: str,
    resource_id: str,
    request: Request,
    download: bool = False,
    user: UserContext = Depends(get_current_user),
) -> StreamingResponse:
    application = get_application(request)
    owner_id = owner_id_from_user(user)
    record = await application.answers.get(owner_id=owner_id, run_id=run_id)
    descriptor = _published_artifact(record.result if record else None, resource_id)
    if descriptor is None or descriptor.get("status") != "available":
        raise HTTPException(status_code=404, detail="artifact not found")
    header = request.headers.get("range", "").strip()
    total = await application.answers.artifact_size(
        owner_id=owner_id,
        run_id=run_id,
        resource_id=resource_id,
    )
    if total is None:
        raise HTTPException(status_code=404, detail="artifact not found")
    offset = 0
    length = None
    status_code = 200
    content_range = None
    if header:
        if not header.lower().startswith("bytes=") or "," in header:
            raise HTTPException(
                status_code=416,
                detail="range not satisfiable",
                headers={"Content-Range": f"bytes */{total}"},
            )
        spec = header.split("=", 1)[1]
        start_s, _, end_s = spec.partition("-")
        try:
            if start_s == "":
                suffix = int(end_s)
                if suffix <= 0 or total == 0:
                    raise ValueError
                suffix = min(suffix, total)
                offset = total - suffix
                length = suffix
            else:
                start = int(start_s)
                end = int(end_s) if end_s else total - 1
                if start >= total or end < start:
                    raise ValueError
                offset = start
                length = min(end, total - 1) - start + 1
        except ValueError as exc:
            raise HTTPException(
                status_code=416,
                detail="range not satisfiable",
                headers={"Content-Range": f"bytes */{total}"},
            ) from exc
        status_code = 206
        content_range = f"bytes {offset}-{offset + length - 1}/{total}"
    stream = await application.answers.open_artifact(
        owner_id=owner_id,
        run_id=run_id,
        resource_id=resource_id,
        offset=offset,
        length=length,
    )
    if stream is None:
        raise HTTPException(status_code=404, detail="artifact not found")
    media_type, disposition = _artifact_response_headers(descriptor, download=download)
    return StreamingResponse(
        stream,
        media_type=media_type,
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
            "Content-Disposition": disposition,
            **(
                {"Content-Security-Policy": _INERT_SVG_CSP}
                if descriptor.get("media_type") == "image/svg+xml"
                else {}
            ),
            **({"Content-Range": content_range} if content_range else {}),
        },
        status_code=status_code,
    )


@router.post("/answer", response_model=AnswerRunDescriptor, status_code=202)
async def create_answer_run(
    request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any]:
    """Accept one durable answer run and return its owner-scoped descriptor.

    Accepts ``application/json`` (link descriptors only) or ``multipart/form-data``
    with one JSON ``request`` part plus repeated ``attachments`` files. Uploaded
    bytes and their references are committed with the run itself.
    """
    application = get_application(request)
    body, uploads = await _parse_answer_body(request, application.config.answer.generation)
    workspaces = await resolve_authorized_query_workspaces(
        request,
        user,
        workspaces=body.workspaces,
        all_workspaces=body.all_workspaces,
    )
    try:
        creation = await application.answers.create(
            request=_service_request(body, uploads, workspaces=workspaces),
            owner_id=owner_id_from_user(user),
            idempotency_key=_idempotency_key(request),
            auth_mode=user.auth_mode,
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
    application = get_application(request)
    record = await application.answers.get(owner_id=owner_id_from_user(user), run_id=run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Answer run not found")
    return await _status_payload(request, user, record)


@router.post("/answer/{run_id}/steer", status_code=202)
async def steer_answer_run(
    run_id: str,
    body: _AgentControlBody,
    request: Request,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    receipt = await get_application(request).answers.steer(
        owner_id=owner_id_from_user(user),
        run_id=run_id,
        instruction=body.content,
    )
    if receipt is None:
        raise HTTPException(status_code=409, detail="Run is not a live Research session")
    return {
        "run_id": receipt.run_id,
        "control_sequence": receipt.control_sequence,
        "kind": receipt.kind,
    }


async def _continue_answer_run(
    *,
    operation: str,
    run_id: str,
    body: _AgentControlBody,
    request: Request,
    user: UserContext,
) -> dict[str, Any]:
    answers = get_application(request).answers
    owner_id = owner_id_from_user(user)
    parent = await answers.get(owner_id=owner_id, run_id=run_id)
    authorized_workspaces: Sequence[str] | None = None
    if parent is not None and parent.terminal:
        authorized_workspaces = await resolve_authorized_query_workspaces(
            request,
            user,
            workspaces=[str(item) for item in parent.request_input().get("workspaces") or ()],
            all_workspaces=False,
        )
    method = answers.follow_up if operation == "follow-up" else answers.fork
    try:
        creation = await method(
            owner_id=owner_id,
            run_id=run_id,
            query=body.content,
            idempotency_key=_idempotency_key(request),
            auth_mode=user.auth_mode,
            authorized_workspaces=authorized_workspaces,
        )
    except IdempotencyKeyConflict:
        raise HTTPException(
            status_code=409,
            detail="Idempotency-Key was reused with a different continuation",
        ) from None
    if creation is None:
        raise HTTPException(status_code=409, detail="Continuation requires a terminal owned run")
    return _descriptor(creation.run)


@router.post("/answer/{run_id}/follow-up", status_code=202)
async def follow_up_answer_run(
    run_id: str,
    body: _AgentControlBody,
    request: Request,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    return await _continue_answer_run(
        operation="follow-up", run_id=run_id, body=body, request=request, user=user
    )


@router.post("/answer/{run_id}/fork", status_code=202)
async def fork_answer_run(
    run_id: str,
    body: _AgentControlBody,
    request: Request,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    return await _continue_answer_run(
        operation="fork", run_id=run_id, body=body, request=request, user=user
    )


@router.get("/answer/{run_id}/transcript")
async def answer_run_transcript(
    run_id: str,
    request: Request,
    limit: int = 20,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    transcript = await get_application(request).answers.transcript_tail(
        owner_id=owner_id_from_user(user), run_id=run_id, limit=limit
    )
    if transcript is None:
        raise HTTPException(status_code=404, detail="Answer run not found")
    return {
        "run_id": transcript.run_id,
        "status": transcript.status,
        "messages": list(transcript.messages),
    }


@router.get("/answer/{run_id}/children")
async def answer_run_children(
    run_id: str,
    request: Request,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    children = await get_application(request).answers.children(
        owner_id=owner_id_from_user(user), run_id=run_id
    )
    if children is None:
        raise HTTPException(status_code=404, detail="Answer run not found")
    return {"run_id": run_id, "children": [dict(child) for child in children]}


@router.post("/answer/{run_id}/resume")
async def resume_answer_run(
    run_id: str,
    request: Request,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    record = await get_application(request).answers.resume(
        owner_id=owner_id_from_user(user), run_id=run_id
    )
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
    application = get_application(request)
    outcome = await application.answers.cancel(owner_id=owner_id_from_user(user), run_id=run_id)
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
    application = get_application(request)
    cursor = resume_cursor(request)
    owner_id = owner_id_from_user(user)
    record = await application.answers.get(owner_id=owner_id, run_id=run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Answer run not found")
    if record.events_trimmed_at is not None:
        raise HTTPException(
            status_code=410,
            detail="Answer run events expired; read its result from the status endpoint",
        )
    workspaces = [str(value) for value in record.request_input().get("workspaces") or ()]
    downloadable = await authorized_workspaces(
        request, user, workspaces, AccessAction.WORKSPACE_DOWNLOAD_SOURCE
    )
    visual = await authorized_workspaces(
        request, user, workspaces, AccessAction.WORKSPACE_READ_VISUAL_ASSET
    )
    events = application.answers.subscribe(owner_id=owner_id, run_id=run_id, after_sequence=cursor)
    return StreamingResponse(
        follow_run_frames(
            events,
            partial(
                answer_run_frame,
                downloadable_workspaces=downloadable,
                visual_workspaces=visual,
                run_id=run_id,
            ),
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def answer_run_frame(
    event: AnswerRunEvent,
    *,
    downloadable_workspaces: set[str] | None,
    visual_workspaces: set[str] | None,
    run_id: str | None = None,
) -> str:
    """Render one durable event for REST: stored identities, freshly projected URLs."""
    payload = dict(event.payload)
    stored = payload.get("result")
    if isinstance(stored, dict):
        payload["result"] = project_answer_result(
            stored,
            source_link_builder=SourceDownloadLinkBuilder(),
            downloadable_workspaces=downloadable_workspaces,
            visual_workspaces=visual_workspaces,
            run_id=run_id,
            artifact_url_prefix="/answer",
        )
    return sse_frame(sequence=event.sequence, event_type=event.event_type, payload=payload)


__all__ = ["answer_run_frame", "router"]
