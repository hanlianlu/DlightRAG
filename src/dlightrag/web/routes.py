"""Web routes — page rendering + htmx fragment endpoints."""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse

from dlightrag.citations import (
    CitationProcessor,
    SourceReference,
    extract_highlights_for_sources,
)
from dlightrag.citations.parser import CITATION_PATTERN

from .deps import get_manager, get_workspace, templates

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/web", tags=["web"])


# === Page Routes ===


@router.get("/", response_class=HTMLResponse)
async def index(request: Request, workspace: str = Depends(get_workspace)):
    """Main page."""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "workspace": workspace},
    )


# === Answer Streaming ===


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


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _render_enriched_answer(answer_text: str) -> str:
    """Replace [ref_id-chunk_idx] with clickable citation badges.

    Strategy: split on citation pattern, escape non-citation segments,
    wrap citation matches in badge HTML. Avoids double-escaping.
    """
    parts = CITATION_PATTERN.split(answer_text)
    # split gives: [text, ref_id, chunk_idx, text, ref_id, chunk_idx, ...]
    result = []
    i = 0
    while i < len(parts):
        if i % 3 == 0:
            # Text segment — escape HTML
            result.append(_escape_html(parts[i]))
        else:
            # ref_id (i%3==1) and chunk_idx (i%3==2) pair
            ref_id = parts[i]
            chunk_idx = parts[i + 1]
            result.append(
                f'<span class="citation-badge" data-ref="{_escape_html(ref_id)}" '
                f'data-chunk="{_escape_html(chunk_idx)}" onclick="filterSource(this)">'
                f"[{_escape_html(ref_id)}-{_escape_html(chunk_idx)}]</span>"
            )
            i += 1  # Skip chunk_idx, already consumed
        i += 1
    return "".join(result)


def _render_sources(sources: list[SourceReference]) -> str:
    """Render source chunks as HTML for the source panel."""
    parts: list[str] = []
    for src in sources:
        if not src.chunks:
            continue
        for chunk in src.chunks:
            safe_content = _escape_html(chunk.content)
            if chunk.highlight_phrases:
                for phrase in chunk.highlight_phrases:
                    safe_phrase = _escape_html(phrase)
                    pattern = re.escape(safe_phrase)
                    safe_content = re.sub(
                        f"({pattern})",
                        r'<span class="highlight">\1</span>',
                        safe_content,
                        flags=re.IGNORECASE,
                        count=1,
                    )

            title = _escape_html(src.title or src.path)
            page = f"p.{chunk.page_idx}" if chunk.page_idx else ""
            parts.append(
                f'<div class="source-chunk" data-ref="{_escape_html(src.id)}" '
                f'data-chunk="{chunk.chunk_idx}">'
                f'<div class="source-chunk-header">'
                f'<span class="source-chunk-title">[{_escape_html(src.id)}-{chunk.chunk_idx}] {title}</span>'
                f'<span class="source-chunk-page">{page}</span>'
                f"</div>"
                f'<div class="source-chunk-content">{safe_content}</div>'
                f"</div>"
            )

    if parts:
        parts.append(
            '<button class="show-all-btn" onclick="showAllSources()" '
            'style="display:none">Show all sources</button>'
        )

    return "\n".join(parts)


@router.post("/answer")
async def answer_stream(
    request: Request,
    workspace: str = Depends(get_workspace),
):
    """Stream answer via SSE, then swap in enriched citations."""
    form = await request.form()
    query = str(form.get("query", ""))
    if not query:
        return HTMLResponse("<span>Please enter a question.</span>")

    manager = get_manager(request)

    async def event_generator() -> AsyncIterator[str]:
        full_answer = ""
        try:
            contexts, raw, token_iter = await manager.aanswer_stream(
                query=query,
                workspace=workspace,
            )

            async for chunk in token_iter:
                full_answer += chunk
                escaped = _escape_html(chunk)
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

            enriched_answer = _render_enriched_answer(result.answer)
            source_html = _render_sources(result_sources)

            done_html = (
                f'<div id="answer-content">{enriched_answer}</div>'
                f'<div id="source-data" style="display:none">{source_html}</div>'
            )
            # SSE requires each line to be a separate data: field
            done_lines = done_html.replace("\n", " ")  # Flatten to single line
            yield f"event: done\ndata: {done_lines}\n\n"

        except Exception:
            logger.exception("Answer streaming failed")
            yield "event: error\ndata: <span>Service error. Please try again.</span>\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# === File Management ===


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
        return HTMLResponse(
            f'<div class="file-item" style="color:#c44">'
            f"Upload failed: {_escape_html(str(e))}</div>"
        )

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
        return HTMLResponse(
            f'<div style="color:#c44">Delete failed: {_escape_html(str(e))}</div>'
        )

    return await file_list(request, workspace)


# === Workspaces ===


@router.get("/workspaces", response_class=HTMLResponse)
async def workspace_list(request: Request, workspace: str = Depends(get_workspace)):
    """Return workspace list fragment."""
    manager = get_manager(request)
    try:
        workspaces = await manager.list_workspaces()
    except Exception:
        workspaces = ["default"]

    parts = []
    for ws in workspaces:
        active = " style='border-color:var(--gold-500)'" if ws == workspace else ""
        parts.append(
            f'<div class="file-item"{active}>'
            f'<span class="file-name">{_escape_html(ws)}</span>'
            f'<form method="post" action="/web/workspaces/switch">'
            f'<input type="hidden" name="workspace" value="{_escape_html(ws)}">'
            f'<button type="submit" class="topbar-btn" '
            f'style="padding:4px 8px;font-size:11px">Switch</button>'
            f"</form></div>"
        )
    return HTMLResponse("\n".join(parts))


@router.post("/workspaces/switch")
async def switch_workspace(request: Request):
    """Switch workspace via cookie."""
    form = await request.form()
    workspace = str(form.get("workspace", "default"))
    response = RedirectResponse(url="/web/", status_code=303)
    response.set_cookie("dlightrag_workspace", workspace, httponly=True)
    return response
