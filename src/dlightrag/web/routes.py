"""Web routes — page rendering + htmx fragment endpoints."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import shutil
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse

from dlightrag.citations import (
    CitationProcessor,
    extract_highlights_for_sources,
)
from dlightrag.citations.source_builder import build_sources

from .deps import get_manager, get_workspace, templates

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/web", tags=["web"])


# ---------------------------------------------------------------------------
# Helpers — data preparation (no HTML)
# ---------------------------------------------------------------------------


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

    # Extract images (base64 from frontend)
    images_b64: list[str] = body.get("images", [])
    multimodal_content: list[dict[str, Any]] | None = None
    tmp_paths: list[str] = []
    if images_b64:
        multimodal_content = []
        for b64 in images_b64[:3]:  # enforce 3-image limit
            try:
                img_bytes = base64.b64decode(b64)
                if len(img_bytes) > 10 * 1024 * 1024:  # 10MB server-side limit
                    logger.warning("Skipping oversized image (%d bytes)", len(img_bytes))
                    continue
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tmp.write(img_bytes)
                tmp.close()
                tmp_paths.append(tmp.name)
                multimodal_content.append({"type": "image", "img_path": tmp.name})
            except Exception:
                logger.warning("Failed to decode uploaded image", exc_info=True)
        if not multimodal_content:
            multimodal_content = None

    # Extract workspaces (multi-select from frontend)
    workspaces: list[str] | None = body.get("workspaces")

    manager = get_manager(request)

    async def event_generator() -> AsyncIterator[str]:
        full_answer = ""
        kwargs: dict[str, Any] = {}
        if conversation_history:
            kwargs["conversation_history"] = conversation_history
        try:
            contexts, token_iter = await manager.aanswer_stream(
                query=query,
                workspace=workspace,
                workspaces=workspaces,
                multimodal_content=multimodal_content,
                **kwargs,
            )

            # token_iter may be an AsyncIterator, a plain str, or None
            # depending on the RAG mode and provider capabilities.
            #
            # Tokens are JSON-encoded so newlines and special chars are
            # properly escaped within the SSE data line.
            if token_iter is None:
                pass
            elif isinstance(token_iter, str):
                full_answer = token_iter
                yield f"event: token\ndata: {json.dumps(token_iter)}\n\n"
            else:
                async for chunk in token_iter:
                    full_answer += chunk
                    yield f"event: token\ndata: {json.dumps(chunk)}\n\n"

            # Emit structured references if available (from AnswerStream)
            if hasattr(token_iter, "references") and token_iter.references:
                refs_data = [r.model_dump() for r in token_iter.references]
                yield f"event: references\ndata: {json.dumps(refs_data)}\n\n"

            # Build sources from contexts
            flat_contexts = []
            for items in contexts.values():
                if isinstance(items, list):
                    flat_contexts.extend(items)

            sources = build_sources(contexts)

            processor = CitationProcessor(
                contexts=flat_contexts,
                available_sources=sources,
            )
            result = processor.process(full_answer)

            # Send done event immediately (sources without highlights)
            done_html = _render_partial(
                "partials/answer_done.html",
                answer=result.answer,
                sources=result.sources,
            )
            yield f"event: done\ndata: {json.dumps(done_html)}\n\n"

            # Highlight extraction (best-effort, async follow-up)
            # Sends a separate SSE event to patch source panel with highlights.
            try:
                from dlightrag.config import get_config
                from dlightrag.models.llm import get_llm_model_func

                llm_func = get_llm_model_func(get_config())
                highlighted_sources = await asyncio.wait_for(
                    extract_highlights_for_sources(
                        sources=result.sources,
                        answer_text=result.answer,
                        llm_func=llm_func,
                    ),
                    timeout=30.0,
                )
                highlights_html = _render_partial(
                    "partials/source_panel.html",
                    sources=highlighted_sources,
                )
                yield f"event: highlights\ndata: {json.dumps(highlights_html)}\n\n"
            except TimeoutError:
                logger.warning("Highlight extraction timed out, skipping")
            except Exception:
                logger.warning("Highlight extraction failed", exc_info=True)

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
        finally:
            # Clean up temp image files
            for p in tmp_paths:
                Path(p).unlink(missing_ok=True)

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
