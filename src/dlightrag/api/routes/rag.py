# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""RAG operations API routes."""

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse

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
    AnswerRequest,
    AnswerResponse,
    IngestJobStatusResponse,
    IngestRequest,
    ResetRequest,
    ResetResponse,
    RetrievalResponse,
    RetrieveRequest,
)
from dlightrag.app_state import request_config
from dlightrag.citations import finalize_answer
from dlightrag.citations.streaming import aclose_answer_stream, iter_answer_tokens
from dlightrag.core.answer_highlights import enrich_semantic_highlights
from dlightrag.core.answer_media import answer_blocks_from_markdown, answer_images_from_sources
from dlightrag.core.client_contracts import IngestSpec, dump_optional_list
from dlightrag.core.client_payloads import (
    answer_payload,
    project_contexts_for_client,
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
from dlightrag.core.retrieval.source_url_resolver import SourceUrlResolver
from dlightrag.observability import trace_observation

from .deps import (
    enforce_access,
    enforce_workspaces_access,
    get_manager,
    request_scope,
    resolve_workspace,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _job_response(job: dict[str, Any]) -> JSONResponse:
    job["status_url"] = f"/ingest/jobs/{job['job_id']}"
    return JSONResponse(status_code=202, content=jsonable_encoder(job))


@router.post("/ingest", response_model=None)
async def ingest(
    body: IngestRequest, request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any] | JSONResponse:
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


@router.post("/retrieve", response_model=RetrievalResponse)
async def retrieve(
    body: RetrieveRequest, request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any]:
    """Retrieve contexts and sources without LLM answer generation."""
    manager = get_manager(request)
    kwargs = query_kwargs_from_payload(body)
    await enforce_workspaces_access(request, user, AccessAction.WORKSPACE_QUERY, body.workspaces)
    scope = request_scope(user, body.workspaces)

    result = await manager.aretrieve(
        body.query,
        workspaces=body.workspaces,
        top_k=body.top_k,
        chunk_top_k=body.chunk_top_k,
        scope=scope,
        **kwargs,
    )
    resolver = SourceUrlResolver(input_dir=str(manager.config.input_dir_path))
    return retrieval_payload(result, source_url_resolver=resolver)


@router.post("/answer", response_model=AnswerResponse)
async def answer(
    body: AnswerRequest, request: Request, user: UserContext = Depends(get_current_user)
):
    """RAG query with LLM-generated answer. Set stream=true for SSE."""
    manager = get_manager(request)
    kwargs = query_kwargs_from_payload(body)
    await enforce_workspaces_access(request, user, AccessAction.WORKSPACE_QUERY, body.workspaces)
    scope = request_scope(user, body.workspaces)

    if not body.stream:
        result = await manager.aanswer(
            body.query,
            conversation_history=dump_optional_list(body.conversation_history),
            workspaces=body.workspaces,
            top_k=body.top_k,
            chunk_top_k=body.chunk_top_k,
            answer_context_top_k=body.answer_context_top_k,
            semantic_highlights=body.semantic_highlights,
            scope=scope,
            **kwargs,
        )
        return answer_payload(result)

    async def event_generator() -> AsyncIterator[str]:
        token_iter: AsyncIterator[str] | None = None
        async with trace_observation(
            "answer_stream_pipeline",
            as_type="chain",
            input={"query": body.query},
            metadata={
                "workspaces": body.workspaces,
                "history_turns": len(body.conversation_history or []),
            },
        ):
            try:
                contexts, token_iter = await manager.aanswer_stream(
                    body.query,
                    conversation_history=dump_optional_list(body.conversation_history),
                    workspaces=body.workspaces,
                    top_k=body.top_k,
                    chunk_top_k=body.chunk_top_k,
                    answer_context_top_k=body.answer_context_top_k,
                    scope=scope,
                    **kwargs,
                )
                public_contexts = project_contexts_for_client(contexts)
                yield sse_data_event(AnswerContextStreamEvent(data=public_contexts))
                answer_parts: list[str] = []
                async for chunk in iter_answer_tokens(
                    token_iter, idle_timeout=manager.config.answer_stream_idle_timeout
                ):
                    answer_parts.append(chunk)
                    yield sse_data_event(AnswerTokenStreamEvent(content=chunk))

                full_answer = "".join(answer_parts)
                clean_answer = getattr(token_iter, "answer", None) or full_answer
                _resolver = SourceUrlResolver(input_dir=str(manager.config.input_dir_path))
                finalized = finalize_answer(
                    clean_answer,
                    contexts,
                    source_contexts=public_contexts,
                    source_url_resolver=_resolver,
                )
                if body.semantic_highlights:
                    finalized.sources = await enrich_semantic_highlights(
                        finalized.sources,
                        answer_text=finalized.answer,
                        config=manager.config,
                    )

                yield sse_data_event(AnswerSourcesStreamEvent(data=finalized.sources))
                trace = getattr(token_iter, "trace", None)
                if isinstance(trace, dict) and trace:
                    yield sse_data_event(AnswerTraceStreamEvent(data=trace))
                image_ids = getattr(token_iter, "current_image_ids", None)
                image_descriptions = getattr(token_iter, "image_descriptions", None)
                if image_ids or image_descriptions:
                    yield sse_data_event(
                        AnswerImageMetaStreamEvent(
                            current_image_ids=image_ids or [],
                            image_descriptions=image_descriptions or [],
                        )
                    )
                answer_images = answer_images_from_sources(finalized.sources, contexts=contexts)
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
            except Exception:
                logger.exception("Error during SSE streaming")
                yield sse_data_event(
                    AnswerErrorStreamEvent(message="Internal server error during streaming")
                )
            finally:
                await aclose_answer_stream(token_iter)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/ingest/blob", response_model=None)
async def ingest_blob(
    request: Request,
    file: UploadFile = File(...),
    workspace: str | None = Form(None),
    title: str | None = Form(None),
    author: str | None = Form(None),
    metadata: str | None = Form(None),
    metadata_policy: str | None = Form(None),
    user: UserContext = Depends(get_current_user),
) -> JSONResponse:
    """Direct file upload ingestion via multipart/form-data.

    File is persisted to input_dir/<workspace>/<filename> for citation
    download links, then ingested via the local file pipeline.
    """
    import json as _json

    manager = get_manager(request)
    ws = resolve_workspace(workspace, request)
    cfg = request_config(request)
    await enforce_access(request, user, AccessAction.WORKSPACE_INGEST, workspace=ws)

    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

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

    # Parse optional metadata JSON
    meta_dict: dict[str, Any] | None = None
    if metadata:
        try:
            meta_dict = _json.loads(metadata)
        except _json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid metadata JSON") from None

    policy = None
    if metadata_policy:
        _valid_policies = frozenset({"validate", "reject_unknown", "store_only"})
        if metadata_policy not in _valid_policies:
            raise HTTPException(
                status_code=400, detail=f"Invalid metadata_policy: {metadata_policy}"
            )
        policy = metadata_policy

    kwargs: dict[str, Any] = {"source_type": "local", "path": str(target_path)}
    if title is not None:
        kwargs["title"] = title
    if author is not None:
        kwargs["author"] = author
    if meta_dict is not None:
        kwargs["metadata"] = meta_dict
    if policy is not None:
        kwargs["metadata_policy"] = policy

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
