# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""API route definitions for DlightRAG REST server.

All endpoint logic lives here; server.py is a minimal app factory.
"""

from __future__ import annotations

import json
import logging
import mimetypes
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from starlette.responses import RedirectResponse

from dlightrag.api.auth import UserContext, get_current_user
from dlightrag.api.models import (
    AnswerRequest,
    DeleteRequest,
    IngestRequest,
    MetadataFilterRequest,
    MetadataUpdateRequest,
    ResetRequest,
    RetrieveRequest,
)
from dlightrag.citations.processor import CitationProcessor
from dlightrag.citations.source_builder import build_sources
from dlightrag.config import get_config
from dlightrag.core.servicemanager import RAGServiceManager
from dlightrag.sourcing.azure_blob import generate_azure_sas_url

logger = logging.getLogger(__name__)

router = APIRouter()

# ═══════════════════════════════════════════════════════════════════
# DRY helpers
# ═══════════════════════════════════════════════════════════════════


def _get_manager(request: Request) -> RAGServiceManager:
    return request.app.state.manager


def _resolve_workspace(ws: str | None) -> str:
    from dlightrag.utils import normalize_workspace

    return normalize_workspace(ws or get_config().workspace)


# ═══════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════


@router.post("/ingest")
async def ingest(
    body: IngestRequest, request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any]:
    """Bulk document ingestion."""
    manager = _get_manager(request)
    ws = _resolve_workspace(body.workspace)

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

    result = await manager.aingest(ws, source_type=body.source_type, **kwargs)
    return result


@router.post("/retrieve")
async def retrieve(
    body: RetrieveRequest, request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any]:
    """Retrieve contexts and sources without LLM answer generation."""
    manager = _get_manager(request)

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
    manager = _get_manager(request)
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
        # Build cited-only sources via CitationProcessor
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
        # Derive references from cited-only sources (not LLM raw output)
        cited_refs = [{"id": s.id, "title": s.title} for s in sources]
        return {
            "answer": result.answer,
            "contexts": result.contexts,
            "references": cited_refs,
            "sources": [s.model_dump() for s in sources],
        }

    # Streaming mode
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
        # Send raw contexts first (sources computed after answer completes)
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

            # AnswerStream cleans invalid citations; use its answer if available
            clean_answer = getattr(token_iter, "answer", None) or full_answer

            # Build cited-only sources
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


@router.get("/files")
async def list_files(
    request: Request,
    workspace: str | None = Query(default=None),
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    """List all ingested documents."""
    manager = _get_manager(request)
    ws = _resolve_workspace(workspace)
    try:
        files = await manager.list_ingested_files(ws)
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=400,
            detail="File listing is not supported in unified RAG mode",
        ) from exc
    return {"files": files, "count": len(files), "workspace": ws}


@router.delete("/files")
async def delete_files(
    body: DeleteRequest, request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any]:
    """Delete documents from knowledge base."""
    manager = _get_manager(request)
    ws = _resolve_workspace(body.workspace)
    try:
        results = await manager.delete_files(
            ws,
            file_paths=body.file_paths,
            filenames=body.filenames,
            delete_source=body.delete_source,
        )
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=400,
            detail="File deletion is not supported in unified RAG mode",
        ) from exc
    return {"results": results, "workspace": ws}


@router.post("/reset")
async def reset_workspace(
    body: ResetRequest, request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any]:
    """Reset all RAG data for a workspace.

    Always targets a single workspace (defaults to config workspace).
    Use ``scripts/reset.py --all`` for multi-workspace reset.
    """
    manager = _get_manager(request)
    ws = _resolve_workspace(body.workspace)
    result = await manager.areset(
        workspace=ws,
        keep_files=body.keep_files,
        dry_run=body.dry_run,
    )
    return result


# ── Metadata CRUD ─────────────────────────────────────────────────


@router.get("/metadata/{doc_id}")
async def get_metadata(
    doc_id: str,
    request: Request,
    workspace: str | None = Query(default=None),
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    """Get document metadata by doc_id."""
    manager = _get_manager(request)
    ws = _resolve_workspace(workspace)
    try:
        meta = await manager.aget_metadata(ws, doc_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not meta:
        raise HTTPException(status_code=404, detail=f"No metadata for doc_id={doc_id}")
    # Convert non-serializable types
    result = {}
    for k, v in meta.items():
        if hasattr(v, "isoformat"):
            result[k] = v.isoformat()
        else:
            result[k] = v
    return {"doc_id": doc_id, "metadata": result}


@router.put("/metadata/{doc_id}")
async def update_metadata(
    doc_id: str,
    body: MetadataUpdateRequest,
    request: Request,
    workspace: str | None = Query(default=None),
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    """Update document metadata (merge with existing)."""
    manager = _get_manager(request)
    ws = _resolve_workspace(workspace)
    try:
        await manager.aupdate_metadata(ws, doc_id, body.metadata)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"doc_id": doc_id, "status": "updated"}


@router.post("/metadata/search")
async def search_metadata(
    body: MetadataFilterRequest,
    request: Request,
    workspace: str | None = Query(default=None),
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    """Search documents by metadata filters."""
    manager = _get_manager(request)
    ws = _resolve_workspace(workspace)
    try:
        doc_ids = await manager.asearch_metadata(ws, body.model_dump(exclude_none=True))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"doc_ids": doc_ids, "count": len(doc_ids)}


# ── Utility endpoints ─────────────────────────────────────────────


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    """Health check including RAG service status."""
    config = get_config()
    manager = _get_manager(request)

    is_degraded = manager.is_degraded()
    warnings = manager.get_warnings()

    status: dict[str, Any] = {
        "status": "degraded" if is_degraded else "healthy",
        "rag_initialized": manager.is_ready(),
        "rag_mode": config.rag_mode,
        "crafted_by": "hllyu",
        "maintained_by": "HanlianLyu",
        "storage": {
            "vector": config.vector_storage,
            "graph": config.graph_storage,
            "kv": config.kv_storage,
        },
    }
    if warnings:
        status["warnings"] = warnings

    # Check PostgreSQL connectivity if using PG backends
    if config.kv_storage.startswith("PG"):
        try:
            import asyncpg

            conn = await asyncpg.connect(
                host=config.postgres_host,
                port=config.postgres_port,
                user=config.postgres_user,
                password=config.postgres_password,
                database=config.postgres_database,
            )
            await conn.fetchval("SELECT 1")
            await conn.close()
            status["postgres"] = "connected"
        except Exception as e:
            status["postgres"] = f"error: {e}"
            status["status"] = "degraded"

    return status


@router.get("/workspaces")
async def workspaces(
    request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any]:
    """List all available workspaces."""
    manager = _get_manager(request)
    ws_list = await manager.list_workspaces()
    return {"workspaces": ws_list}


@router.get("/api/files/{file_path:path}", response_model=None)
async def serve_file(
    file_path: str, user: UserContext = Depends(get_current_user)
) -> StreamingResponse | RedirectResponse:
    """Serve a file by relative path.

    - Local paths: 200 + StreamingResponse
    - azure://: 302 redirect to SAS signed URL
    - s3://: 501 Not Implemented (S3 presigned URL support in Task 5)
    """
    config = get_config()

    # --- Azure blob: 302 redirect ---
    if file_path.startswith("azure://"):
        if not config.blob_connection_string:
            raise HTTPException(503, "Azure blob storage not configured")
        sas_url = generate_azure_sas_url(
            connection_string=config.blob_connection_string,
            raw_path=file_path,
            expiry_seconds=config.azure_sas_expiry,
        )
        return RedirectResponse(url=sas_url, status_code=302)

    # --- S3: not yet implemented ---
    if file_path.startswith("s3://"):
        raise HTTPException(501, "S3 presigned URL support not yet implemented")

    # --- Unknown remote scheme ---
    if "://" in file_path:
        raise HTTPException(400, f"Unsupported scheme: {file_path.split('://', 1)[0]}")

    # --- Local file: stream from working_dir ---
    working_dir = config.working_dir_path.resolve()
    full_path = (working_dir / file_path).resolve()

    # Security: path traversal + symlink escape check
    if not full_path.is_relative_to(working_dir):
        raise HTTPException(403, "Access denied")
    if not full_path.is_file():
        raise HTTPException(404, "File not found")

    content_type, _ = mimetypes.guess_type(str(full_path))
    return StreamingResponse(
        _stream_file(full_path),
        media_type=content_type or "application/octet-stream",
    )


async def _stream_file(path: Path, chunk_size: int = 64 * 1024) -> AsyncIterator[bytes]:
    """Stream a file in chunks to avoid loading it entirely into memory."""
    import aiofiles

    async with aiofiles.open(path, "rb") as f:
        while chunk := await f.read(chunk_size):
            yield chunk
