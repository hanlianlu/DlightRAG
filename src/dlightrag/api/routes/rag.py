# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""RAG operations API routes."""

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from dlightrag.api.auth import UserContext, get_current_user
from dlightrag.api.models import AnswerRequest, IngestRequest, ResetRequest, RetrieveRequest
from dlightrag.citations.processor import CitationProcessor
from dlightrag.citations.source_builder import build_sources

from .deps import get_manager, resolve_workspace

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
    if body.metadata:
        kwargs["metadata"] = body.metadata

    return await manager.aingest(ws, source_type=body.source_type, **kwargs)


@router.post("/retrieve")
async def retrieve(
    body: RetrieveRequest, request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any]:
    """Retrieve contexts and sources without LLM answer generation."""
    manager = get_manager(request)
    kwargs: dict[str, Any] = {}
    if body.filters:
        from dlightrag.core.retrieval.models import MetadataFilter

        kwargs["filters"] = MetadataFilter(**body.filters.model_dump(exclude_none=True))

    result = await manager.aretrieve(
        body.query,
        workspaces=body.workspaces,
        mode=body.mode,
        top_k=body.top_k,
        chunk_top_k=body.chunk_top_k,
        **kwargs,
    )
    sources = build_sources(result.contexts)
    return {
        "answer": result.answer,
        "contexts": result.contexts,
        "sources": [s.model_dump() for s in sources],
    }


@router.post("/answer", response_model=None)
async def answer(
    body: AnswerRequest, request: Request, user: UserContext = Depends(get_current_user)
):
    """RAG query with LLM-generated answer. Set stream=true for SSE."""
    manager = get_manager(request)
    kwargs: dict[str, Any] = {}
    if body.multimodal_content:
        kwargs["multimodal_content"] = body.multimodal_content

    if not body.stream:
        result = await manager.aanswer(
            body.query,
            conversation_history=body.conversation_history,
            workspaces=body.workspaces,
            mode=body.mode,
            top_k=body.top_k,
            chunk_top_k=body.chunk_top_k,
            **kwargs,
        )
        flat_contexts: list[dict[str, Any]] = []
        for items in result.contexts.values():
            if isinstance(items, list):
                flat_contexts.extend(items)
        all_sources = build_sources(result.contexts)
        if result.answer and flat_contexts:
            processor = CitationProcessor(contexts=flat_contexts, available_sources=all_sources)
            cited = processor.process(result.answer)
            sources = cited.sources
        else:
            sources = []
        cited_refs = [{"id": s.id, "title": s.title} for s in sources]
        return {
            "answer": result.answer,
            "contexts": result.contexts,
            "references": cited_refs,
            "sources": [s.model_dump() for s in sources],
        }

    contexts, token_iter = await manager.aanswer_stream(
        body.query,
        conversation_history=body.conversation_history,
        workspaces=body.workspaces,
        mode=body.mode,
        top_k=body.top_k,
        chunk_top_k=body.chunk_top_k,
        **kwargs,
    )

    async def event_generator() -> AsyncIterator[str]:
        yield f"data: {json.dumps({'type': 'context', 'data': contexts}, ensure_ascii=False)}\n\n"
        full_answer = ""
        try:
            if token_iter is None:
                pass
            elif isinstance(token_iter, str):
                full_answer = token_iter
                yield f"data: {json.dumps({'type': 'token', 'content': token_iter}, ensure_ascii=False)}\n\n"
            else:
                async for chunk in token_iter:
                    full_answer += chunk
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk}, ensure_ascii=False)}\n\n"

            clean_answer = getattr(token_iter, "answer", None) or full_answer
            flat_contexts: list[dict[str, Any]] = []
            for items in contexts.values():
                if isinstance(items, list):
                    flat_contexts.extend(items)
            all_sources = build_sources(contexts)
            if clean_answer and flat_contexts:
                processor = CitationProcessor(contexts=flat_contexts, available_sources=all_sources)
                cited = processor.process(clean_answer)
                sources = cited.sources
            else:
                sources = []

            yield f"data: {json.dumps({'type': 'sources', 'data': [s.model_dump() for s in sources]}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception:
            logger.exception("Error during SSE streaming")
            yield f"data: {json.dumps({'type': 'error', 'message': 'Internal server error during streaming'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
