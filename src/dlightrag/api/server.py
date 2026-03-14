# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""FastAPI REST server for bulk ingestion and queries.

Entry point: dlightrag-api
Primary interface for offline/batch data ingestion operations.
"""

from __future__ import annotations

import logging
import mimetypes
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, field_validator
from starlette.responses import RedirectResponse

from dlightrag.citations.parser import parse_freetext_references
from dlightrag.citations.processor import CitationProcessor
from dlightrag.citations.source_builder import build_sources
from dlightrag.config import DlightragConfig, get_config
from dlightrag.core.servicemanager import RAGServiceManager, RAGServiceUnavailableError
from dlightrag.sourcing.azure_blob import generate_azure_sas_url

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    try:
        _app.state.manager = await RAGServiceManager.create()
    except Exception:
        logger.exception("Failed to initialize RAG service manager")
        raise
    yield
    await _app.state.manager.close()


app = FastAPI(
    title="dlightrag",
    description="DlightRAG - Dual-mode (Caption based & Unified representation based) multi-modal RAG service",
    version=__import__("dlightrag").__version__,
    lifespan=lifespan,
)


# ═══════════════════════════════════════════════════════════════════
# Auth middleware
# ═══════════════════════════════════════════════════════════════════


def _get_config() -> DlightragConfig:
    return get_config()


async def _verify_auth(request: Request, config: DlightragConfig = Depends(_get_config)) -> None:
    """Verify bearer token if DLIGHTRAG_API_AUTH_TOKEN is set."""
    token = config.api_auth_token
    if not token:
        return  # No auth required

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    provided_token = auth_header[7:]
    if provided_token != token:
        raise HTTPException(status_code=403, detail="Invalid token")


# ═══════════════════════════════════════════════════════════════════
# Request/Response Models
# ═══════════════════════════════════════════════════════════════════


class MetadataFilterRequest(BaseModel):
    """Structured metadata filter for retrieval queries."""

    filename: str | None = None
    filename_pattern: str | None = None
    file_extension: str | None = None
    doc_title: str | None = None
    doc_author: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    rag_mode: str | None = None
    custom: dict[str, Any] | None = None


class IngestRequest(BaseModel):
    source_type: Literal["local", "azure_blob", "snowflake"]
    path: str | None = None
    container_name: str | None = None
    blob_path: str | None = None
    prefix: str | None = None
    query: str | None = None
    table: str | None = None
    replace: bool | None = None
    workspace: str | None = None
    metadata: dict[str, Any] | None = None


class RetrieveRequest(BaseModel):
    query: str
    mode: Literal["local", "global", "hybrid", "naive", "mix"] = "mix"
    top_k: int | None = None
    chunk_top_k: int | None = None
    workspaces: list[str] | None = None
    filters: MetadataFilterRequest | None = None


class AnswerRequest(BaseModel):
    query: str
    mode: Literal["local", "global", "hybrid", "naive", "mix"] = "mix"
    stream: bool = False
    top_k: int | None = None
    chunk_top_k: int | None = None
    workspaces: list[str] | None = None
    multimodal_content: list[dict[str, Any]] | None = None

    @field_validator("multimodal_content")
    @classmethod
    def validate_image_count(cls, v: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if v and len(v) > 3:
            raise ValueError("Maximum 3 multimodal items per query")
        return v


class DeleteRequest(BaseModel):
    file_paths: list[str] | None = None
    filenames: list[str] | None = None
    delete_source: bool = True
    workspace: str | None = None


# ═══════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════


def _get_manager(request: Request) -> RAGServiceManager:
    return request.app.state.manager


@app.post("/ingest", dependencies=[Depends(_verify_auth)])
async def ingest(body: IngestRequest, request: Request) -> dict[str, Any]:
    """Bulk document ingestion."""
    manager = _get_manager(request)
    ws = body.workspace or get_config().workspace

    kwargs: dict[str, Any] = {}
    if body.replace is not None:
        kwargs["replace"] = body.replace

    if body.source_type == "local":
        if not body.path:
            raise HTTPException(status_code=400, detail="'path' is required for local ingestion")
        kwargs["path"] = body.path

    elif body.source_type == "azure_blob":
        if not body.container_name:
            raise HTTPException(
                status_code=400, detail="'container_name' is required for azure_blob"
            )
        if body.blob_path and body.prefix is not None:
            raise HTTPException(
                status_code=400, detail="'blob_path' and 'prefix' are mutually exclusive"
            )
        kwargs["container_name"] = body.container_name
        if body.blob_path:
            kwargs["blob_path"] = body.blob_path
        if body.prefix is not None:
            kwargs["prefix"] = body.prefix

    elif body.source_type == "snowflake":
        if not body.query:
            raise HTTPException(status_code=400, detail="'query' is required for snowflake")
        kwargs["query"] = body.query
        if body.table:
            kwargs["table"] = body.table

    result = await manager.aingest(ws, source_type=body.source_type, **kwargs)
    return result


@app.post("/retrieve", dependencies=[Depends(_verify_auth)])
async def retrieve(body: RetrieveRequest, request: Request) -> dict[str, Any]:
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


@app.post("/answer", dependencies=[Depends(_verify_auth)], response_model=None)
async def answer(body: AnswerRequest, request: Request):
    """RAG query with LLM-generated answer. Set stream=true for SSE."""
    import json

    manager = _get_manager(request)
    kwargs: dict[str, Any] = {}
    if body.multimodal_content:
        kwargs["multimodal_content"] = body.multimodal_content

    if not body.stream:
        result = await manager.aanswer(
            body.query,
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
        return {
            "answer": result.answer,
            "contexts": result.contexts,
            "references": [r.model_dump() for r in result.references] if result.references else [],
            "sources": [s.model_dump() for s in sources],
        }

    # Streaming mode
    contexts, token_iter = await manager.aanswer_stream(
        body.query,
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

            # Extract references: structured (AnswerStream) or freetext (parse)
            refs = getattr(token_iter, "references", None)
            if not refs and full_answer:
                full_answer, parsed_refs = parse_freetext_references(full_answer)
                refs = parsed_refs or None

            logger.info(
                "[API] SSE: token_iter type=%s refs_count=%s",
                type(token_iter).__name__,
                len(refs) if refs else 0,
            )

            refs_data = [r.model_dump() for r in refs] if refs else []
            yield f"data: {json.dumps({'type': 'references', 'data': refs_data}, ensure_ascii=False)}\n\n"

            # Build cited-only sources
            flat_contexts: list[dict[str, Any]] = []
            for items in contexts.values():
                if isinstance(items, list):
                    flat_contexts.extend(items)
            all_sources = build_sources(contexts)
            if full_answer and flat_contexts:
                processor = CitationProcessor(contexts=flat_contexts, available_sources=all_sources)
                cited = processor.process(full_answer)
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


@app.get("/files", dependencies=[Depends(_verify_auth)])
async def list_files(
    request: Request, workspace: str | None = Query(default=None)
) -> dict[str, Any]:
    """List all ingested documents."""
    manager = _get_manager(request)
    ws = workspace or get_config().workspace
    try:
        files = await manager.list_ingested_files(ws)
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=400,
            detail="File listing is not supported in unified RAG mode",
        ) from exc
    return {"files": files, "count": len(files), "workspace": ws}


@app.delete("/files", dependencies=[Depends(_verify_auth)])
async def delete_files(body: DeleteRequest, request: Request) -> dict[str, Any]:
    """Delete documents from knowledge base."""
    manager = _get_manager(request)
    ws = body.workspace or get_config().workspace
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


# ── Metadata CRUD ─────────────────────────────────────────────────


@app.get("/metadata/{doc_id}", dependencies=[Depends(_verify_auth)])
async def get_metadata(
    doc_id: str, request: Request, workspace: str | None = Query(default=None)
) -> dict[str, Any]:
    """Get document metadata by doc_id."""
    manager = _get_manager(request)
    ws = workspace or get_config().workspace
    svc = await manager._get_service(ws)
    if svc._metadata_index is None:
        raise HTTPException(status_code=400, detail="Metadata index not available (non-PG backend)")
    meta = await svc._metadata_index.get(doc_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"No metadata for doc_id={doc_id}")
    # Convert non-serializable types
    result = {}
    for k, v in meta.items():
        if hasattr(v, "isoformat"):
            result[k] = v.isoformat()
        else:
            result[k] = v
    return {"doc_id": doc_id, "metadata": result}


@app.put("/metadata/{doc_id}", dependencies=[Depends(_verify_auth)])
async def update_metadata(
    doc_id: str, request: Request, workspace: str | None = Query(default=None)
) -> dict[str, Any]:
    """Update document metadata (merge with existing)."""
    manager = _get_manager(request)
    ws = workspace or get_config().workspace
    svc = await manager._get_service(ws)
    if svc._metadata_index is None:
        raise HTTPException(status_code=400, detail="Metadata index not available (non-PG backend)")
    body = await request.json()
    await svc._metadata_index.upsert(doc_id, body)
    return {"doc_id": doc_id, "status": "updated"}


@app.post("/metadata/search", dependencies=[Depends(_verify_auth)])
async def search_metadata(
    body: MetadataFilterRequest, request: Request, workspace: str | None = Query(default=None)
) -> dict[str, Any]:
    """Search documents by metadata filters."""
    manager = _get_manager(request)
    ws = workspace or get_config().workspace
    svc = await manager._get_service(ws)
    if svc._metadata_index is None:
        raise HTTPException(status_code=400, detail="Metadata index not available (non-PG backend)")
    from dlightrag.core.retrieval.models import MetadataFilter

    filters = MetadataFilter(**body.model_dump(exclude_none=True))
    doc_ids = await svc._metadata_index.query(filters)
    return {"doc_ids": doc_ids, "count": len(doc_ids)}


@app.get("/health")
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


@app.get("/workspaces", dependencies=[Depends(_verify_auth)])
async def workspaces(request: Request) -> dict[str, Any]:
    """List all available workspaces."""
    manager = _get_manager(request)
    ws_list = await manager.list_workspaces()
    return {"workspaces": ws_list}


@app.get("/api/files/{file_path:path}", response_model=None, dependencies=[Depends(_verify_auth)])
async def serve_file(file_path: str) -> StreamingResponse | RedirectResponse:
    """Serve a file by relative path.

    - Local paths: 200 + StreamingResponse
    - azure://: 302 redirect to SAS signed URL
    - snowflake://: 400 (no file to serve)
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

    # --- Snowflake: no file to serve ---
    if file_path.startswith("snowflake://"):
        raise HTTPException(400, "Snowflake sources have no downloadable file")

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


@app.exception_handler(RAGServiceUnavailableError)
async def rag_unavailable_handler(
    request: Request,  # noqa: ARG001
    exc: RAGServiceUnavailableError,
) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": exc.detail},
    )


# Mount web frontend (optional — only if jinja2 is installed)
try:
    import importlib.util

    if importlib.util.find_spec("jinja2"):
        from fastapi.staticfiles import StaticFiles

        from dlightrag.web.deps import _TEMPLATE_DIR
        from dlightrag.web.routes import router as web_router

        app.include_router(web_router)
        _static_dir = _TEMPLATE_DIR.parent / "static"
        if _static_dir.exists():
            app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
except ImportError:
    pass  # Web frontend not installed


def main() -> None:
    """Entry point for dlightrag-api."""
    import argparse

    import uvicorn
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(description="dlightrag REST API server")
    parser.add_argument("--env-file", help="Path to .env configuration file")
    args = parser.parse_args()

    if args.env_file:
        load_dotenv(args.env_file, override=True)

    config = get_config()
    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO))

    uvicorn.run(
        "dlightrag.api.server:app",
        host=config.api_host,
        port=config.api_port,
        log_level=config.log_level,
    )


__all__ = ["app", "main"]
