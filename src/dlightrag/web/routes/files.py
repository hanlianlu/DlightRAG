# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web routes for file management."""

import json
import logging
import shutil
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse

from dlightrag.web.deps import get_manager, get_workspace, templates

logger = logging.getLogger(__name__)

router = APIRouter()


def _render_partial(name: str, **ctx: Any) -> str:
    """Render a Jinja2 partial template to string."""
    return templates.env.get_template(name).render(**ctx)


@router.get("/files", response_class=HTMLResponse)
async def file_list(request: Request, workspace: str = Depends(get_workspace)):
    """Return file list HTML fragment for panel."""
    manager = get_manager(request)
    try:
        files = await manager.list_ingested_files(workspace)
    except Exception:
        files = []

    return templates.TemplateResponse(
        request,
        "partials/file_list.html",
        {"files": files, "workspace": workspace},
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
            path=str(tmp_dir),
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
    # htmx sends hx-vals as query params for DELETE requests
    file_path = request.query_params.get("file_path", "")
    file_paths = [file_path] if file_path else []
    manager = get_manager(request)

    try:
        await manager.delete_files(workspace, file_paths=file_paths)
    except Exception as e:
        logger.exception("Delete failed")
        return HTMLResponse(_render_partial("partials/error.html", message=f"Delete failed: {e}"))

    return await file_list(request, workspace)


@router.get("/ingest/progress")
async def ingest_progress(request: Request):
    """SSE endpoint for ingestion progress -- pushes task event dicts."""
    ingest_mgr = getattr(request.app.state, "ingest_task_manager", None)
    if ingest_mgr is None:

        async def _no_mgr() -> AsyncIterator[str]:
            yield f"event: error\ndata: {json.dumps({'detail': 'Ingest task manager not available'})}\n\n"

        return StreamingResponse(
            _no_mgr(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Map internal event types to client SSE event names
    _EVENT_MAP = {
        "snapshot": "snapshot",
        "task_created": "task_created",
        "task_updated": "progress",
        "task_progress": "progress",
        "task_done": "done",
        "task_failed": "failed",
    }

    async def stream() -> AsyncIterator[str]:
        async for event in ingest_mgr.subscribe():
            if await request.is_disconnected():
                break

            etype = event.get("type", "")
            sse_name = _EVENT_MAP.get(etype)
            if not sse_name:
                continue

            # Encode the event payload as JSON
            if sse_name == "snapshot":
                tasks = event.get("tasks", [])
                yield f"event: snapshot\ndata: {json.dumps(tasks)}\n\n"
            else:
                task = event.get("task", {})
                yield f"event: {sse_name}\ndata: {json.dumps(task)}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
