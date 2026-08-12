# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""RAG operations API routes."""

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import ValidationError
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from dlightrag.access_control import AccessAction
from dlightrag.api.auth import UserContext, get_current_user
from dlightrag.api.events import (
    AnswerContextStreamEvent,
    AnswerDoneStreamEvent,
    AnswerErrorStreamEvent,
    AnswerImageMetaStreamEvent,
    AnswerSourcesStreamEvent,
    AnswerTokenStreamEvent,
    AnswerTraceStreamEvent,
    sse_data_event,
)
from dlightrag.api.models import (
    ANSWER_REQUEST_PART_MAX_BYTES,
    AnswerRequest,
    AnswerResponse,
    IngestJobStatusResponse,
    IngestRequest,
    ResetRequest,
    ResetResponse,
    RetrievalResponse,
    RetrieveRequest,
    UploadIngestJobResponse,
)
from dlightrag.app_state import request_config
from dlightrag.citations import finalize_answer
from dlightrag.citations.streaming import aclose_answer_stream, iter_answer_tokens
from dlightrag.config import AnswerConfig
from dlightrag.core.access import workspace_names
from dlightrag.core.answer.errors import (
    ANSWER_STREAM_FAILED,
    InvalidToolConfigurationError,
    classify_answer_error,
)
from dlightrag.core.answer.highlights import enrich_semantic_highlights
from dlightrag.core.answer.media import answer_blocks_from_markdown, answer_images_from_sources
from dlightrag.core.client_attachments import answer_link_resources
from dlightrag.core.client_contracts import IngestSpec, conversation_history_as_dicts
from dlightrag.core.client_execution import execute_answer, execute_retrieve
from dlightrag.core.client_payloads import (
    answer_payload,
    project_contexts_for_client,
    project_source_payloads,
    retrieval_payload,
)
from dlightrag.core.client_requests import (
    ingest_spec_from_payload,
    managed_local_ingest_documents,
    managed_local_ingest_path,
    query_kwargs_from_payload,
)
from dlightrag.core.ingestion.uploads import (
    UploadTooLargeError,
    safe_upload_basename,
    write_upload_stream,
)
from dlightrag.core.resources.models import ResourceInput
from dlightrag.core.retrieval.source_links import SourceDownloadLinkBuilder
from dlightrag.core.servicemanager import answer_trace_output
from dlightrag.observability import trace_observation

from .deps import (
    enforce_access,
    filter_workspace_records,
    get_manager,
    resolve_authorized_query_workspaces,
    resolve_workspace,
)

logger = logging.getLogger(__name__)
router = APIRouter()


async def _authorized_workspaces(
    request: Request,
    user: UserContext,
    workspaces: list[str],
    action: str,
) -> set[str]:
    records = await filter_workspace_records(
        request,
        user,
        action,
        [{"workspace": workspace} for workspace in workspaces],
    )
    return workspace_names(records)


def _job_response(job: dict[str, Any]) -> dict[str, Any]:
    job["status_url"] = f"/ingest/jobs/{job['job_id']}"
    return job


@router.post(
    "/ingest",
    response_model=IngestJobStatusResponse,
    status_code=202,
)
async def ingest(
    body: IngestRequest, request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any]:
    """Bulk document ingestion."""
    manager = get_manager(request)
    cfg = request_config(request)
    ws = resolve_workspace(body.workspace, request)
    await enforce_access(request, user, AccessAction.WORKSPACE_INGEST, workspace=ws)
    ingest_spec = ingest_spec_from_payload(body)
    if body.source_type == "local":
        try:
            path = managed_local_ingest_path(
                source_type=body.source_type,
                path=ingest_spec.path,
                input_dir=cfg.input_dir_path,
                workspace=ws,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        documents = managed_local_ingest_documents(
            source_type=body.source_type,
            documents=ingest_spec.documents,
            input_dir=cfg.input_dir_path,
            workspace=ws,
        )
        ingest_spec = ingest_spec.model_copy(update={"path": path, "documents": documents})

    job = await manager.astart_ingest_job(ws, ingest_spec)
    return _job_response(job)


@router.get("/ingest/jobs/{job_id}", response_model=IngestJobStatusResponse)
async def get_ingest_job(
    job_id: str,
    request: Request,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    """Return durable ingest job status."""
    manager = get_manager(request)
    job = await manager.aget_ingest_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Ingest job not found")
    workspace = job.get("workspace")
    await enforce_access(
        request,
        user,
        AccessAction.JOB_READ,
        workspace=str(workspace) if workspace else None,
    )
    return job


@router.post("/ingest/jobs/{job_id}/cancel", response_model=IngestJobStatusResponse)
async def cancel_ingest_job(
    job_id: str,
    request: Request,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    """Stop a running ingest job, keeping whatever it already ingested."""
    manager = get_manager(request)
    job = await manager.aget_ingest_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Ingest job not found")
    workspace = job.get("workspace")
    await enforce_access(
        request,
        user,
        AccessAction.JOB_CANCEL,
        workspace=str(workspace) if workspace else None,
    )
    cancelled = await manager.acancel_ingest_job(job_id)
    return cancelled if cancelled is not None else job


@router.post("/retrieve", response_model=RetrievalResponse)
async def retrieve(
    body: RetrieveRequest, request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any]:
    """Retrieve contexts and sources without LLM answer generation."""
    manager = get_manager(request)
    resolved_workspaces = await resolve_authorized_query_workspaces(
        request,
        user,
        workspaces=body.workspaces,
        all_workspaces=body.all_workspaces,
    )
    downloadable_workspaces = await _authorized_workspaces(
        request,
        user,
        resolved_workspaces,
        AccessAction.WORKSPACE_DOWNLOAD_SOURCE,
    )
    visual_workspaces = await _authorized_workspaces(
        request,
        user,
        resolved_workspaces,
        AccessAction.WORKSPACE_READ_VISUAL_ASSET,
    )
    result = await execute_retrieve(
        manager=manager,
        payload=body,
        resolved_workspaces=resolved_workspaces,
    )
    link_builder = SourceDownloadLinkBuilder()
    return retrieval_payload(
        result,
        source_link_builder=link_builder,
        downloadable_workspaces=downloadable_workspaces,
        visual_workspaces=visual_workspaces,
    )


_ALLOWED_ANSWER_PARTS = {"request", "attachments"}
_MAX_ANSWER_FORM_FIELDS = 8
_ALLOWED_INGEST_PARTS = {"file", "workspace", "title", "author", "metadata"}
_MAX_INGEST_FORM_FIELDS = 8
_INGEST_FORM_FIELD_MAX_BYTES = 1024 * 1024
# Comfortably holds the JSON `request` part (query, history, filters, links) so a
# small per-attachment cap never truncates the request envelope.
_ANSWER_REQUEST_PART_CEILING = ANSWER_REQUEST_PART_MAX_BYTES


def _enforce_answer_attachment_count(count: int, max_attachments: int) -> None:
    """Reject over-limit attachment counts with a stable 413 and the safe limit."""
    if count > max_attachments:
        raise HTTPException(
            status_code=413,
            detail=f"Too many attachments; at most {max_attachments} are allowed",
        )


async def _parse_answer_body(
    request: Request, answer_cfg: AnswerConfig
) -> tuple[AnswerRequest, list[ResourceInput]]:
    """Parse a JSON or multipart answer request bounded by attachment limits.

    JSON bodies carry the complete request with optional HTTPS link descriptors.
    Multipart bodies carry exactly one JSON ``request`` part plus repeated
    ``attachments`` file parts; uploaded files and JSON links may mix. Count,
    per-attachment, and total-byte admission are enforced here, before the
    orchestrator ever runs, without buffering an unbounded body.
    """
    content_type = request.headers.get("content-type", "").lower()
    if "multipart/form-data" not in content_type:
        try:
            body = AnswerRequest.model_validate_json(await request.body())
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        link_resources = answer_link_resources(body.attachments)
        _enforce_answer_attachment_count(len(link_resources), answer_cfg.max_attachments)
        return body, link_resources

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

        file_resources: list[ResourceInput] = []
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
            file_resources.append(
                ResourceInput(
                    filename=part.filename,
                    content=data,
                    declared_mime=part.content_type,
                )
            )
        link_resources = answer_link_resources(body.attachments)
        _enforce_answer_attachment_count(len(link_resources) + len(file_resources), max_attachments)
        return body, [*link_resources, *file_resources]
    finally:
        await form.close()


@router.post("/answer", response_model=AnswerResponse)
async def answer(request: Request, user: UserContext = Depends(get_current_user)):
    """RAG query with LLM-generated answer. Set stream=true for SSE.

    Accepts ``application/json`` (link descriptors only) or ``multipart/form-data``
    with one JSON ``request`` part plus repeated ``attachments`` files.
    """
    manager = get_manager(request)
    body, resources = await _parse_answer_body(request, request_config(request).answer)
    kwargs = query_kwargs_from_payload(body)
    resolved_workspaces = await resolve_authorized_query_workspaces(
        request,
        user,
        workspaces=body.workspaces,
        all_workspaces=body.all_workspaces,
    )
    downloadable_workspaces = await _authorized_workspaces(
        request,
        user,
        resolved_workspaces,
        AccessAction.WORKSPACE_DOWNLOAD_SOURCE,
    )
    visual_workspaces = await _authorized_workspaces(
        request,
        user,
        resolved_workspaces,
        AccessAction.WORKSPACE_READ_VISUAL_ASSET,
    )
    history = conversation_history_as_dicts(body.history)

    if not body.stream:
        result = await execute_answer(
            manager=manager,
            payload=body,
            resolved_workspaces=resolved_workspaces,
            resources=resources,
        )
        link_builder = SourceDownloadLinkBuilder()
        return answer_payload(
            result,
            source_link_builder=link_builder,
            downloadable_workspaces=downloadable_workspaces,
            visual_workspaces=visual_workspaces,
        )

    async def event_generator() -> AsyncIterator[str]:
        token_iter: AsyncIterator[str] | None = None
        async with trace_observation(
            "answer_pipeline",
            as_type="chain",
            input={"query": body.query},
            metadata={
                "stream": True,
                "workspaces": resolved_workspaces,
            },
        ) as observation:
            try:
                contexts, token_iter = await manager.aanswer_stream(
                    body.query,
                    workspaces=resolved_workspaces,
                    top_k=body.top_k,
                    chunk_top_k=body.chunk_top_k,
                    history=history,
                    resources=resources,
                    **kwargs,
                )
                public_contexts = project_contexts_for_client(
                    contexts,
                    visual_workspaces=visual_workspaces,
                )
                yield sse_data_event(AnswerContextStreamEvent(data=public_contexts))
                answer_parts: list[str] = []
                async for chunk in iter_answer_tokens(
                    token_iter, idle_timeout=manager.config.answer_stream_idle_timeout
                ):
                    answer_parts.append(chunk)
                    yield sse_data_event(AnswerTokenStreamEvent(content=chunk))

                full_answer = "".join(answer_parts)
                clean_answer = getattr(token_iter, "answer", None) or full_answer
                _link_builder = SourceDownloadLinkBuilder()
                finalized = finalize_answer(
                    clean_answer,
                    contexts,
                )
                if body.semantic_highlights:
                    finalized.sources = await enrich_semantic_highlights(
                        finalized.sources,
                        answer_text=finalized.answer,
                        config=manager.config,
                    )

                source_payloads = project_source_payloads(
                    finalized.sources,
                    resolver=_link_builder,
                    downloadable_workspaces=downloadable_workspaces,
                    visual_workspaces=visual_workspaces,
                )
                observation.update(
                    output=answer_trace_output(finalized.answer, finalized.sources, contexts)
                )
                yield sse_data_event(AnswerSourcesStreamEvent(data=source_payloads))
                trace = getattr(token_iter, "trace", None)
                if isinstance(trace, dict) and trace:
                    yield sse_data_event(AnswerTraceStreamEvent(data=trace))
                image_descriptions = getattr(token_iter, "image_descriptions", None)
                if image_descriptions:
                    yield sse_data_event(
                        AnswerImageMetaStreamEvent(
                            image_descriptions=image_descriptions or [],
                        )
                    )
                answer_images = answer_images_from_sources(
                    finalized.sources,
                    contexts=contexts,
                    visual_workspaces=visual_workspaces,
                )
                yield sse_data_event(
                    AnswerDoneStreamEvent(
                        answer=finalized.answer,
                        answer_images=answer_images,
                        answer_blocks=answer_blocks_from_markdown(finalized.answer, answer_images),
                    )
                )
            except asyncio.CancelledError:
                logger.debug("Client disconnected during SSE streaming")
                raise
            except Exception as exc:
                error_kind = classify_answer_error(exc)
                if isinstance(exc, InvalidToolConfigurationError):
                    logger.exception("Answer tool composition is invalid")
                    message = exc.public_message
                elif error_kind == ANSWER_STREAM_FAILED:
                    logger.exception("Error during SSE streaming")
                    message = "Internal server error during streaming"
                else:
                    # Actionable capability/transport error: surface the reason and kind.
                    logger.info("Answer image request rejected during streaming (%s)", error_kind)
                    message = str(exc)
                yield sse_data_event(AnswerErrorStreamEvent(message=message, error_kind=error_kind))
            finally:
                await aclose_answer_stream(token_iter)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post(
    "/ingest/blob",
    response_model=UploadIngestJobResponse,
    status_code=202,
)
async def ingest_blob(
    request: Request,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    """Direct file upload ingestion via multipart/form-data.

    File is persisted to input_dir/<workspace>/<filename> for citation
    download links, then ingested via the local file pipeline.
    """
    import json as _json

    manager = get_manager(request)
    cfg = request_config(request)
    try:
        form = await request.form(
            max_files=2,
            max_fields=_MAX_INGEST_FORM_FIELDS,
            max_part_size=_INGEST_FORM_FIELD_MAX_BYTES,
        )
    except StarletteHTTPException as exc:
        detail = str(exc.detail)
        if exc.status_code == 400 and detail.startswith(
            ("Too many files.", "Too many fields.", "Part exceeded maximum size")
        ):
            raise HTTPException(status_code=413, detail=detail) from exc
        raise
    try:
        unexpected = sorted({key for key, _ in form.multi_items()} - _ALLOWED_INGEST_PARTS)
        if unexpected:
            raise HTTPException(
                status_code=400,
                detail=f"Unexpected multipart field(s): {', '.join(unexpected)}",
            )
        files = form.getlist("file")
        if len(files) != 1 or not isinstance(files[0], StarletteUploadFile):
            raise HTTPException(status_code=400, detail="Exactly one file is required")
        file = files[0]
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")

        def optional_text(name: str) -> str | None:
            value = form.get(name)
            if value in (None, ""):
                return None
            if not isinstance(value, str):
                raise HTTPException(status_code=400, detail=f"{name} must be a text field")
            return value

        workspace = optional_text("workspace")
        title = optional_text("title")
        author = optional_text("author")
        metadata = optional_text("metadata")
        ws = resolve_workspace(workspace, request)
        await enforce_access(request, user, AccessAction.WORKSPACE_INGEST, workspace=ws)

        try:
            safe_name = safe_upload_basename(file.filename)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid filename") from None

        target_dir = cfg.input_dir_path / ws
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / safe_name

        try:
            await write_upload_stream(file, target_path, max_bytes=cfg.max_upload_bytes)
        except UploadTooLargeError:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum size of {cfg.max_upload_bytes} bytes",
            ) from None
    finally:
        await form.close()

    # Parse optional metadata JSON
    meta_dict: dict[str, Any] | None = None
    if metadata:
        try:
            meta_dict = _json.loads(metadata)
        except _json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid metadata JSON") from None

    kwargs: dict[str, Any] = {"source_type": "local", "path": str(target_path)}
    if title is not None:
        kwargs["title"] = title
    if author is not None:
        kwargs["author"] = author
    if meta_dict is not None:
        kwargs["metadata"] = meta_dict

    job = await manager.astart_ingest_job(ws, IngestSpec(**kwargs))
    job["uploaded_file"] = str(target_path)
    job["filename"] = safe_name
    return _job_response(job)


@router.post("/reset", response_model=ResetResponse)
async def reset_workspace(
    body: ResetRequest, request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any]:
    """Reset all RAG data for a workspace."""
    manager = get_manager(request)
    ws = resolve_workspace(body.workspace, request)
    await enforce_access(request, user, AccessAction.WORKSPACE_RESET, workspace=ws)
    return await manager.areset(
        workspace=ws,
        keep_files=body.keep_files,
        dry_run=body.dry_run,
    )
