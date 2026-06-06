# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""RAG operations API routes."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from dlightrag.api.auth import UserContext, get_current_user
from dlightrag.api.models import (
    AnswerRequest,
    IngestRequest,
    ResetRequest,
    RetrieveRequest,
)
from dlightrag.citations import finalize_answer
from dlightrag.core.client_payloads import (
    answer_payload,
    metadata_filter_from_payload,
    project_contexts_for_client,
    retrieval_payload,
)
from dlightrag.core.retrieval.source_url_resolver import SourceUrlResolver

from .deps import get_manager, request_scope, resolve_workspace

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ingest")
async def ingest(
    body: IngestRequest, request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any]:
    """Bulk document ingestion."""
    manager = get_manager(request)
    ws = resolve_workspace(body.workspace)

    kwargs: dict[str, Any] = {}
    if body.replace is not None:
        kwargs["replace"] = body.replace

    if body.source_type == "local":
        kwargs["path"] = body.path
    elif body.source_type == "azure_blob":
        if body.blob_path and body.prefix is not None:
            raise HTTPException(
                status_code=400, detail="'blob_path' and 'prefix' are mutually exclusive"
            )
        kwargs["container_name"] = body.container_name
        if body.blob_path:
            kwargs["blob_path"] = body.blob_path
        if body.prefix is not None:
            kwargs["prefix"] = body.prefix
    elif body.source_type == "s3":
        kwargs["bucket"] = body.bucket
        kwargs["key"] = body.key
    if body.title is not None:
        kwargs["title"] = body.title
    if body.author is not None:
        kwargs["author"] = body.author
    if body.metadata is not None:
        kwargs["metadata"] = body.metadata
    if body.metadata_policy is not None:
        kwargs["metadata_policy"] = body.metadata_policy

    return await manager.aingest(ws, source_type=body.source_type, **kwargs)


@router.post("/retrieve")
async def retrieve(
    body: RetrieveRequest, request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any]:
    """Retrieve contexts and sources without LLM answer generation."""
    manager = get_manager(request)
    kwargs: dict[str, Any] = {}
    filters = metadata_filter_from_payload(body.filters)
    if filters is not None:
        kwargs["filters"] = filters
    if body.multimodal_content:
        kwargs["multimodal_content"] = body.multimodal_content
    if body.query_images:
        kwargs["query_images"] = body.query_images
    if body.session_id:
        kwargs["session_id"] = body.session_id
    if body.referenced_image_ids:
        kwargs["referenced_image_ids"] = body.referenced_image_ids
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


@router.post("/answer", response_model=None)
async def answer(
    body: AnswerRequest, request: Request, user: UserContext = Depends(get_current_user)
):
    """RAG query with LLM-generated answer. Set stream=true for SSE."""
    manager = get_manager(request)
    kwargs: dict[str, Any] = {}
    filters = metadata_filter_from_payload(body.filters)
    if filters is not None:
        kwargs["filters"] = filters
    if body.multimodal_content:
        kwargs["multimodal_content"] = body.multimodal_content
    if body.query_images:
        kwargs["query_images"] = body.query_images
    if body.session_id:
        kwargs["session_id"] = body.session_id
    if body.referenced_image_ids:
        kwargs["referenced_image_ids"] = body.referenced_image_ids
    scope = request_scope(user, body.workspaces)

    if not body.stream:
        result = await manager.aanswer(
            body.query,
            conversation_history=body.conversation_history,
            workspaces=body.workspaces,
            top_k=body.top_k,
            chunk_top_k=body.chunk_top_k,
            answer_candidate_top_k=body.answer_candidate_top_k,
            answer_context_top_k=body.answer_context_top_k,
            scope=scope,
            **kwargs,
        )
        return answer_payload(result)

    contexts, token_iter = await manager.aanswer_stream(
        body.query,
        conversation_history=body.conversation_history,
        workspaces=body.workspaces,
        top_k=body.top_k,
        chunk_top_k=body.chunk_top_k,
        answer_candidate_top_k=body.answer_candidate_top_k,
        answer_context_top_k=body.answer_context_top_k,
        scope=scope,
        **kwargs,
    )

    async def event_generator() -> AsyncIterator[str]:
        public_contexts = project_contexts_for_client(contexts)
        yield f"data: {json.dumps({'type': 'context', 'data': public_contexts}, ensure_ascii=False)}\n\n"
        answer_parts: list[str] = []
        try:
            if token_iter is None:
                pass
            elif isinstance(token_iter, str):
                answer_parts.append(token_iter)
                yield f"data: {json.dumps({'type': 'token', 'content': token_iter}, ensure_ascii=False)}\n\n"
            else:
                async for chunk in token_iter:
                    answer_parts.append(chunk)
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk}, ensure_ascii=False)}\n\n"

            full_answer = "".join(answer_parts)
            clean_answer = getattr(token_iter, "answer", None) or full_answer
            _resolver = SourceUrlResolver(input_dir=str(manager.config.input_dir_path))
            finalized = finalize_answer(
                clean_answer,
                contexts,
                source_contexts=public_contexts,
                source_url_resolver=_resolver,
            )

            yield f"data: {json.dumps({'type': 'sources', 'data': [s.model_dump() for s in finalized.sources]}, ensure_ascii=False)}\n\n"
            trace = getattr(token_iter, "trace", None)
            if isinstance(trace, dict) and trace:
                yield f"data: {json.dumps({'type': 'trace', 'data': trace}, ensure_ascii=False)}\n\n"
            image_ids = getattr(token_iter, "current_image_ids", None)
            image_descriptions = getattr(token_iter, "image_descriptions", None)
            if image_ids or image_descriptions:
                yield f"data: {json.dumps({'type': 'image_meta', 'current_image_ids': image_ids or [], 'image_descriptions': image_descriptions or []}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'answer': finalized.answer}, ensure_ascii=False)}\n\n"
        except asyncio.CancelledError:
            logger.debug("Client disconnected during SSE streaming")
            raise
        except Exception:
            logger.exception("Error during SSE streaming")
            yield f"data: {json.dumps({'type': 'error', 'message': 'Internal server error during streaming'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/ingest/blob")
async def ingest_blob(
    request: Request,
    file: UploadFile = File(...),
    workspace: str | None = Form(None),
    title: str | None = Form(None),
    author: str | None = Form(None),
    metadata: str | None = Form(None),
    metadata_policy: str | None = Form(None),
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    """Direct file upload ingestion via multipart/form-data.

    File is persisted to input_dir/<workspace>/<filename> for citation
    download links, then ingested via the local file pipeline.
    """
    import json as _json

    manager = get_manager(request)
    ws = resolve_workspace(workspace)
    cfg = manager.config

    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    # Sanitize filename -- reject path traversal
    safe_name = Path(file.filename).name
    if safe_name != file.filename or ".." in safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Check file size against config limit
    contents = await file.read()
    if len(contents) > cfg.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum size of {cfg.max_upload_bytes} bytes",
        )

    # Persist to input_dir/<workspace>/<safe_name>
    target_dir = cfg.input_dir_path / ws
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / safe_name
    target_path.write_bytes(contents)

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

    kwargs: dict[str, Any] = {"path": str(target_path)}
    if title is not None:
        kwargs["title"] = title
    if author is not None:
        kwargs["author"] = author
    if meta_dict is not None:
        kwargs["metadata"] = meta_dict
    if policy is not None:
        kwargs["metadata_policy"] = policy

    result = await manager.aingest(ws, source_type="local", **kwargs)
    result["uploaded_file"] = str(target_path)
    result["filename"] = safe_name
    return result


@router.post("/reset")
async def reset_workspace(
    body: ResetRequest, request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any]:
    """Reset all RAG data for a workspace."""
    manager = get_manager(request)
    ws = resolve_workspace(body.workspace)
    return await manager.areset(
        workspace=ws,
        keep_files=body.keep_files,
        dry_run=body.dry_run,
    )
