"""Web routes — page rendering + htmx fragment endpoints."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from markupsafe import Markup

from dlightrag.citations import (
    CitationProcessor,
    SourceReference,
    extract_highlights_for_sources,
)

from .deps import get_manager, get_workspace, templates

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/web", tags=["web"])


# ---------------------------------------------------------------------------
# Helpers — data preparation (no HTML)
# ---------------------------------------------------------------------------


def _contexts_to_flat_list(contexts: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten DlightRAG contexts dict into list for CitationIndexer."""
    flat: list[dict[str, Any]] = []
    for items in contexts.values():
        if isinstance(items, list):
            flat.extend(items)
    return flat


def _extract_available_sources(contexts: dict[str, Any]) -> list[SourceReference]:
    """Build available SourceReference list from context metadata."""
    seen: dict[str, SourceReference] = {}
    for items in contexts.values():
        if not isinstance(items, list):
            continue
        for ctx in items:
            ref_id = str(ctx.get("reference_id", ""))
            if not ref_id or ref_id in seen:
                continue
            metadata = ctx.get("metadata", {})
            seen[ref_id] = SourceReference(
                id=ref_id,
                path=metadata.get("file_name", metadata.get("file_path", "")),
                title=metadata.get("file_name"),
                type=metadata.get("file_type"),
            )
    return list(seen.values())


def _render_partial(name: str, **ctx: Any) -> str:
    """Render a Jinja2 partial template to string."""
    return templates.env.get_template(name).render(**ctx)


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------


@router.get("/", response_class=HTMLResponse)
async def index(request: Request, workspace: str = Depends(get_workspace)):
    """Main page."""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "workspace": workspace},
    )


# ---------------------------------------------------------------------------
# Answer streaming (SSE)
# ---------------------------------------------------------------------------


@router.post("/answer")
async def answer_stream(
    request: Request,
    workspace: str = Depends(get_workspace),
):
    """Stream answer via SSE, then swap in enriched citations."""
    body = await request.json()
    query = str(body.get("query", ""))
    if not query:
        return HTMLResponse("<span>Please enter a question.</span>")

    conversation_history: list[dict[str, str]] | None = body.get("conversation_history")

    manager = get_manager(request)

    async def event_generator() -> AsyncIterator[str]:
        full_answer = ""
        kwargs: dict[str, Any] = {}
        if conversation_history:
            kwargs["conversation_history"] = conversation_history
        try:
            contexts, raw, token_iter = await manager.aanswer_stream(
                query=query,
                workspace=workspace,
                **kwargs,
            )

            # token_iter may be an AsyncIterator, a plain str, or None
            # depending on the RAG mode and provider capabilities.
            if token_iter is None:
                pass
            elif isinstance(token_iter, str):
                full_answer = token_iter
                escaped = Markup.escape(token_iter)
                yield f"event: token\ndata: {escaped}\n\n"
            else:
                async for chunk in token_iter:
                    full_answer += chunk
                    escaped = Markup.escape(chunk)
                    yield f"event: token\ndata: {escaped}\n\n"

            # Citation processing
            flat_contexts = _contexts_to_flat_list(contexts)
            available_sources = _extract_available_sources(contexts)

            processor = CitationProcessor(
                contexts=flat_contexts,
                available_sources=available_sources,
            )
            result = processor.process(full_answer)

            # Highlight extraction (best-effort)
            try:
                from dlightrag.config import get_config
                from dlightrag.models.llm import get_llm_model_func

                llm_func = get_llm_model_func(get_config())
                result_sources = await extract_highlights_for_sources(
                    sources=result.sources,
                    answer_text=result.answer,
                    llm_func=llm_func,
                )
            except Exception:
                logger.warning("Highlight extraction failed", exc_info=True)
                result_sources = result.sources

            # Render done payload via Jinja2 partials
            done_html = _render_partial(
                "partials/answer_done.html",
                answer=result.answer,
                sources=result_sources,
            )
            # SSE: flatten to single line
            done_line = done_html.replace("\n", " ")
            yield f"event: done\ndata: {done_line}\n\n"

            # Server-driven budget: tell the frontend the char budget for next request.
            # Conservative multiplier: 2 chars/token handles CJK (~1.5) with headroom.
            try:
                from dlightrag.config import get_config

                cfg = get_config()
                max_tokens = cfg.max_conversation_tokens
                max_msgs = cfg.max_conversation_turns * 2
                char_budget = max_tokens * 2
                meta = {"history_char_budget": char_budget, "history_max_messages": max_msgs}
                yield f"event: meta\ndata: {json.dumps(meta)}\n\n"
            except Exception:
                pass  # Non-critical; frontend falls back to defaults

        except Exception:
            logger.exception("Answer streaming failed")
            yield "event: error\ndata: Service error. Please try again.\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# File management
# ---------------------------------------------------------------------------


@router.get("/files", response_class=HTMLResponse)
async def file_list(request: Request, workspace: str = Depends(get_workspace)):
    """Return file list HTML fragment for panel."""
    manager = get_manager(request)
    try:
        files = await manager.list_ingested_files(workspace)
    except Exception:
        files = []

    return templates.TemplateResponse(
        "partials/file_list.html",
        {"request": request, "files": files, "workspace": workspace},
    )


@router.post("/files/upload", response_class=HTMLResponse)
async def upload_files(
    request: Request,
    files: list[UploadFile] = File(...),
    workspace: str = Depends(get_workspace),
):
    """Upload and ingest files."""
    manager = get_manager(request)

    tmp_dir = Path(tempfile.mkdtemp(prefix="dlightrag_upload_"))
    try:
        for f in files:
            if not f.filename:
                continue
            dest = tmp_dir / f.filename
            with open(dest, "wb") as out:
                shutil.copyfileobj(f.file, out)

        await manager.aingest(
            workspace=workspace,
            source_type="local",
            input_dir=str(tmp_dir),
        )
    except Exception as e:
        logger.exception("Upload/ingestion failed")
        return HTMLResponse(_render_partial("partials/error.html", message=f"Upload failed: {e}"))

    return await file_list(request, workspace)


@router.delete("/files", response_class=HTMLResponse)
async def delete_files(
    request: Request,
    workspace: str = Depends(get_workspace),
):
    """Delete files from workspace."""
    body = await request.json()
    file_paths = body.get("file_paths", [])
    manager = get_manager(request)

    try:
        await manager.delete_files(workspace, file_paths=file_paths)
    except Exception as e:
        logger.exception("Delete failed")
        return HTMLResponse(_render_partial("partials/error.html", message=f"Delete failed: {e}"))

    return await file_list(request, workspace)


# ---------------------------------------------------------------------------
# Workspaces
# ---------------------------------------------------------------------------


@router.get("/workspaces", response_class=HTMLResponse)
async def workspace_list(request: Request, workspace: str = Depends(get_workspace)):
    """Return workspace list fragment."""
    manager = get_manager(request)
    try:
        workspaces = await manager.list_workspaces()
    except Exception:
        workspaces = ["default"]

    return templates.TemplateResponse(
        "partials/workspace_list.html",
        {"request": request, "workspaces": workspaces, "current_workspace": workspace},
    )


@router.post("/workspaces/switch")
async def switch_workspace(request: Request):
    """Switch workspace via cookie."""
    form = await request.form()
    workspace = str(form.get("workspace", "default"))
    response = RedirectResponse(url="/web/", status_code=303)
    response.set_cookie("dlightrag_workspace", workspace, httponly=True)
    return response
