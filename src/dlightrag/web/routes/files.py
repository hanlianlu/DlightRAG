# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web routes for file management."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from fastapi.responses import HTMLResponse

from dlightrag.web.deps import (
    error_response,
    get_manager,
    get_workspace,
    render_partial,
    templates,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Track per-workspace background ingest tasks so the Upload button doesn't
# launch a duplicate while an ingest is already running.
_ingest_tasks: dict[str, asyncio.Task[Any] | None] = {}


def _resolve_workspace(requested: str | None, fallback: str) -> str:
    from dlightrag.utils import normalize_workspace

    if not requested:
        return fallback
    normalized = normalize_workspace(requested)
    return normalized or fallback


def _file_view_models(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in files:
        row = dict(item)
        file_path = str(row.get("file_path") or "")
        file_name = str(row.get("file_name") or row.get("filename") or "")
        if not file_name and file_path:
            file_name = Path(file_path).name
        if not file_name:
            file_name = str(row.get("doc_id") or "Untitled file")
        row["file_name"] = file_name
        row["file_path"] = file_path
        rows.append(row)
    return rows


def _staging_dir(workspace: str) -> Path:
    """Persistent staging directory for this workspace's uploaded files."""
    from dlightrag.config import get_config

    cfg = get_config()
    p = cfg.input_dir_path / workspace
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_relative_path(filename: str) -> Path:
    """Sanitize a webkitRelativePath — reject path traversal attempts."""
    if not filename:
        raise ValueError("Empty filename")
    if ".." in filename or "\0" in filename:
        raise ValueError(f"Unsafe filename: {filename!r}")
    parts = Path(filename).parts
    if not parts:
        raise ValueError(f"Empty path parts: {filename!r}")
    return Path(*parts)


# ---------------------------------------------------------------------------
# GET /web/files — file list panel content
# ---------------------------------------------------------------------------


@router.get("/files", response_class=HTMLResponse)
async def file_list(
    request: Request,
    workspace: str = Depends(get_workspace),
    workspace_name: str | None = Query(default=None, alias="workspace"),
):
    """Return file list HTML fragment for panel."""
    selected_workspace = _resolve_workspace(workspace_name, workspace)
    return await _file_list_response(request, selected_workspace)


async def _file_list_response(request: Request, workspace: str):
    manager = get_manager(request)
    try:
        files = _file_view_models(await manager.list_ingested_files(workspace))
    except Exception:
        files = []

    # Check whether an ingest is currently active so reopening the panel
    # shows the progress bar again instead of stale state.
    ingest_busy = _ingest_tasks.get(workspace) is not None
    if ingest_busy:
        try:
            ps = await manager.get_pipeline_status(workspace)
            ingest_busy = ps.get("busy", False) or ps.get("pending_enqueues", 0) > 1
        except Exception:
            pass

    return templates.TemplateResponse(
        request,
        "partials/file_list.html",
        {
            "files": files,
            "workspace": workspace,
            "ingest_busy": ingest_busy,
        },
    )


# ---------------------------------------------------------------------------
# POST /web/files/upload — non-blocking upload + background ingest
# ---------------------------------------------------------------------------


@router.post("/files/upload", response_class=HTMLResponse)
async def upload_files(
    request: Request,
    files: list[UploadFile] = File(...),
    workspace_name: str | None = Form(default=None, alias="workspace"),
    workspace: str = Depends(get_workspace),
):
    """Upload files and start background ingest.  Returns immediately."""
    from dlightrag.config import get_config

    cfg = get_config()
    max_bytes = cfg.max_upload_size_mb * 1024 * 1024

    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > max_bytes:
        return error_response(
            f"Upload exceeds limit ({cfg.max_upload_size_mb} MB)",
            status_code=413,
        )

    manager = get_manager(request)
    selected_workspace = _resolve_workspace(workspace_name, workspace)

    # Reject if an ingest is already running in this workspace.
    existing = _ingest_tasks.get(selected_workspace)
    if existing is not None and not existing.done():
        return error_response(
            "An ingest is already in progress for this workspace. "
            "Wait for it to finish before uploading more files.",
            status_code=409,
        )

    bytes_written = 0
    try:
        dest_dir = _staging_dir(selected_workspace)
        saved_files = 0
        for f in files:
            if not f.filename:
                continue
            try:
                rel = _safe_relative_path(f.filename)
            except ValueError:
                logger.warning("Rejected upload with unsafe filename: %r", f.filename)
                continue
            dest = dest_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as out:
                while chunk := await f.read(1024 * 1024):
                    bytes_written += len(chunk)
                    if bytes_written > max_bytes:
                        out.close()
                        dest.unlink(missing_ok=True)
                        return error_response(
                            f"Upload exceeds limit ({cfg.max_upload_size_mb} MB)",
                            status_code=413,
                        )
                    out.write(chunk)
            saved_files += 1

        if saved_files == 0:
            return error_response("No valid files selected")

    except Exception as e:
        logger.exception("Upload staging failed")
        return error_response(f"Upload failed: {e}", status_code=500)

    # Fire-and-forget: schedule ingest, return immediately with progress UI.
    async def _background_ingest() -> None:
        try:
            await manager.aingest(
                workspace=selected_workspace,
                source_type="local",
                path=str(dest_dir),
            )
        except Exception:
            logger.exception("Background ingest failed for workspace %s", selected_workspace)
        finally:
            _ingest_tasks.pop(selected_workspace, None)

    _ingest_tasks[selected_workspace] = asyncio.create_task(_background_ingest())

    return templates.TemplateResponse(
        request,
        "partials/file_list.html",
        {
            "files": [],
            "workspace": selected_workspace,
            "ingest_busy": True,
            "status": {"latest_message": "Starting ingest..."},
        },
    )


# ---------------------------------------------------------------------------
# GET /web/ingest-status — htmx polling endpoint
# ---------------------------------------------------------------------------


@router.get("/ingest-status", response_class=HTMLResponse)
async def ingest_status(
    request: Request,
    workspace: str = Depends(get_workspace),
    workspace_name: str | None = Query(default=None, alias="workspace"),
):
    """Return either a progress partial (while busy) or the file list (done)."""
    selected_workspace = _resolve_workspace(workspace_name, workspace)
    manager = get_manager(request)

    try:
        ps = await manager.get_pipeline_status(selected_workspace)
    except Exception:
        ps = {"busy": False, "latest_message": "Status unavailable"}

    still_busy = ps.get("busy", False) or ps.get("pending_enqueues", 0) > 1

    if still_busy:
        return HTMLResponse(
            render_partial(
                "partials/ingest_progress.html",
                workspace=selected_workspace,
                status=ps,
            )
        )

    # Ingest finished — clean up task tracker and return refreshed file list.
    _ingest_tasks.pop(selected_workspace, None)
    return await _file_list_response(request, selected_workspace)


# ---------------------------------------------------------------------------
# DELETE /web/files
# ---------------------------------------------------------------------------


@router.delete("/files", response_class=HTMLResponse)
async def delete_files(
    request: Request,
    workspace: str = Depends(get_workspace),
):
    """Delete files from workspace."""
    file_path = request.query_params.get("file_path", "")
    file_paths = [file_path] if file_path else []
    manager = get_manager(request)
    selected_workspace = _resolve_workspace(request.query_params.get("workspace"), workspace)

    try:
        await manager.delete_files(selected_workspace, file_paths=file_paths)
    except Exception as e:
        logger.exception("Delete failed")
        return error_response(f"Delete failed: {e}", status_code=500)

    return await _file_list_response(request, selected_workspace)
