# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web routes for file management."""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path, PureWindowsPath
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


def _resolve_workspace(requested: str | None, cookie_workspace: str) -> str:
    from dlightrag.utils import normalize_workspace

    if not requested:
        return cookie_workspace
    normalized = normalize_workspace(requested)
    return normalized or cookie_workspace


async def _resolve_registered_workspace(
    request: Request,
    workspace: str,
) -> str | None:
    """Return the requested workspace when it is registered."""
    from dlightrag.utils import normalize_workspace

    manager = get_manager(request)
    try:
        known = {
            normalized
            for item in await manager.list_workspaces()
            if (normalized := normalize_workspace(item))
        }
    except Exception:
        return workspace
    if workspace in known:
        return workspace
    return None


async def _workspace_is_registered(request: Request, workspace: str) -> bool:
    """Return whether a workspace is registered; fail open on registry outages."""
    from dlightrag.utils import normalize_workspace

    manager = get_manager(request)
    try:
        known = {
            normalized
            for item in await manager.list_workspaces()
            if (normalized := normalize_workspace(item))
        }
    except Exception:
        return True
    return workspace in known


def _stale_workspace_response() -> HTMLResponse:
    return error_response(
        "Workspace no longer exists. Refresh and choose an existing workspace.",
        status_code=409,
    )


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


def _upload_batch_dir(workspace: str) -> Path:
    """Per-request staging directory so batch ingest never scans old uploads."""
    root = _staging_dir(workspace) / "__uploads__"
    root.mkdir(parents=True, exist_ok=True)
    p = root / uuid.uuid4().hex
    p.mkdir(parents=True, exist_ok=False)
    return p


def _safe_relative_path(filename: str) -> Path:
    """Sanitize a webkitRelativePath — reject path traversal attempts."""
    if not filename:
        raise ValueError("Empty filename")
    if "\0" in filename:
        raise ValueError(f"Unsafe filename: {filename!r}")
    candidate = Path(filename)
    windows_candidate = PureWindowsPath(filename)
    if candidate.is_absolute() or windows_candidate.is_absolute() or windows_candidate.drive:
        raise ValueError(f"Unsafe filename: {filename!r}")
    parts = candidate.parts
    if not parts or ".." in parts:
        raise ValueError(f"Unsafe filename: {filename!r}")
    return Path(*parts)


def _safe_upload_destination(dest_dir: Path, filename: str) -> Path:
    """Return a destination under dest_dir, rejecting escapes after resolution."""
    rel = _safe_relative_path(filename)
    root = dest_dir.resolve()
    dest = (root / rel).resolve()
    if not dest.is_relative_to(root):
        raise ValueError(f"Unsafe filename: {filename!r}")
    return dest


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
    selected_workspace = await _resolve_registered_workspace(request, selected_workspace)
    if selected_workspace is None:
        return _stale_workspace_response()
    return await _file_list_response(request, selected_workspace)


async def _file_list_response(request: Request, workspace: str):
    manager = get_manager(request)
    try:
        files = _file_view_models(await manager.list_ingested_files(workspace))
    except Exception:
        files = []

    # LightRAG's shared-storage pipeline_status is the authoritative ingest
    # activity signal across API, MCP, Web, restart, and scale-out.
    ingest_busy = False
    status: dict[str, Any] = {}
    try:
        ps = await manager.get_pipeline_status(workspace)
        ingest_busy = ps.get("busy", False) or ps.get("pending_enqueues", 0) > 0
        if ingest_busy:
            status = ps
    except Exception:
        pass

    return templates.TemplateResponse(
        request,
        "partials/file_list.html",
        {
            "files": files,
            "workspace": workspace,
            "ingest_busy": ingest_busy,
            "status": status,
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
    if not await _workspace_is_registered(request, selected_workspace):
        return _stale_workspace_response()

    # Detect whether the pipeline is already busy so the UI can show a
    # "queued" state instead of "starting" — LightRAG's request_pending
    # mechanism picks up new enqueues automatically after the current batch.
    already_busy = False
    try:
        ps = await manager.get_pipeline_status(selected_workspace)
        already_busy = bool(ps.get("busy"))
    except Exception:
        pass

    bytes_written = 0
    upload_dir: Path | None = None
    try:
        upload_dir = _upload_batch_dir(selected_workspace)
        saved_paths: list[Path] = []
        _SYSTEM_FILES = {
            ".DS_Store",  # macOS folder metadata
            "Thumbs.db",  # Windows thumbnail cache
            "desktop.ini",  # Windows folder customization
        }

        for f in files:
            if not f.filename:
                continue
            basename = f.filename.rsplit("/", 1)[-1]
            if basename in _SYSTEM_FILES or basename.startswith("._"):
                continue
            try:
                dest = _safe_upload_destination(upload_dir, f.filename)
            except ValueError:
                logger.warning("Rejected upload with unsafe filename: %r", f.filename)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as out:
                while chunk := await f.read(1024 * 1024):
                    bytes_written += len(chunk)
                    if bytes_written > max_bytes:
                        out.close()
                        dest.unlink(missing_ok=True)
                        if upload_dir is not None:
                            shutil.rmtree(upload_dir, ignore_errors=True)
                        return error_response(
                            f"Upload exceeds limit ({cfg.max_upload_size_mb} MB)",
                            status_code=413,
                        )
                    out.write(chunk)
            saved_paths.append(dest)

        if not saved_paths:
            if upload_dir is not None:
                shutil.rmtree(upload_dir, ignore_errors=True)
            return error_response("No valid files selected")

    except Exception as e:
        logger.exception("Upload staging failed")
        if upload_dir is not None:
            shutil.rmtree(upload_dir, ignore_errors=True)
        return error_response(f"Upload failed: {e}", status_code=500)

    try:
        await manager._astart_staged_local_ingest_job(
            selected_workspace,
            path=str(upload_dir),
        )
    except Exception as e:
        logger.exception("Failed to start ingest job for workspace %s", selected_workspace)
        if upload_dir is not None:
            shutil.rmtree(upload_dir, ignore_errors=True)
        return error_response(f"Upload staged but ingest did not start: {e}", status_code=500)

    return templates.TemplateResponse(
        request,
        "partials/file_list.html",
        {
            "files": [],
            "workspace": selected_workspace,
            "ingest_busy": True,
            "is_queued": already_busy,
            "file_count": len(saved_paths),
            "status": {
                "latest_message": (
                    "Queued — processing after current batch"
                    if already_busy
                    else "Starting ingest..."
                ),
            },
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
    selected_workspace = await _resolve_registered_workspace(request, selected_workspace)
    if selected_workspace is None:
        return _stale_workspace_response()
    manager = get_manager(request)

    try:
        ps = await manager.get_pipeline_status(selected_workspace)
    except Exception:
        ps = {"busy": False, "latest_message": "Status unavailable"}

    still_busy = ps.get("busy", False) or ps.get("pending_enqueues", 0) > 0

    if still_busy:
        return HTMLResponse(
            render_partial(
                "partials/ingest_progress.html",
                workspace=selected_workspace,
                status=ps,
            )
        )

    # Ingest finished -- reload the full file list into the panel.
    response = await _file_list_response(request, selected_workspace)
    response.headers["HX-Retarget"] = "#panel-content"
    return response


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
    if not await _workspace_is_registered(request, selected_workspace):
        return _stale_workspace_response()

    try:
        await manager.delete_files(selected_workspace, file_paths=file_paths)
    except Exception as e:
        logger.exception("Delete failed")
        return error_response(f"Delete failed: {e}", status_code=500)

    return await _file_list_response(request, selected_workspace)
