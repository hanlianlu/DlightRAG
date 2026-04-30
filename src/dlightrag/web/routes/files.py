# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web routes for file management."""

import json
import logging
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
    from dlightrag.config import get_config

    cfg = get_config()
    max_bytes = cfg.max_upload_size_mb * 1024 * 1024

    # Reject oversized requests up front (Content-Length header). This is a
    # cheap pre-check; the per-file budget below catches malformed CL too.
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > max_bytes:
        return HTMLResponse(
            _render_partial(
                "partials/error.html",
                message=f"Upload exceeds limit ({cfg.max_upload_size_mb} MB)",
            ),
            status_code=413,
        )

    manager = get_manager(request)

    tmp_dir = Path(tempfile.mkdtemp(prefix="dlightrag_upload_"))
    bytes_written = 0
    try:
        for f in files:
            if not f.filename:
                continue
            # Strip any path components from the user-supplied filename —
            # ``tmp_dir / "../etc/passwd"`` or ``tmp_dir / "/etc/passwd"``
            # would otherwise escape tmp_dir on open().
            safe_name = Path(f.filename).name
            if not safe_name or safe_name in (".", ".."):
                logger.warning("Rejected upload with unsafe filename: %r", f.filename)
                continue
            dest = tmp_dir / safe_name
            # Stream-copy with running budget so a missing/lying Content-Length
            # can't blow past max_bytes.
            with open(dest, "wb") as out:
                while chunk := await f.read(1024 * 1024):
                    bytes_written += len(chunk)
                    if bytes_written > max_bytes:
                        out.close()
                        dest.unlink(missing_ok=True)
                        return HTMLResponse(
                            _render_partial(
                                "partials/error.html",
                                message=f"Upload exceeds limit ({cfg.max_upload_size_mb} MB)",
                            ),
                            status_code=413,
                        )
                    out.write(chunk)

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
