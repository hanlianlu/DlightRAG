# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web routes for file management."""

import logging
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from fastapi.responses import HTMLResponse

from dlightrag.web.deps import get_manager, get_workspace, templates

logger = logging.getLogger(__name__)

router = APIRouter()


def _render_partial(name: str, **ctx: Any) -> str:
    """Render a Jinja2 partial template to string."""
    return templates.env.get_template(name).render(**ctx)


def _error_response(message: str, status_code: int = 400) -> HTMLResponse:
    return HTMLResponse(
        _render_partial("partials/error.html", message=message),
        status_code=status_code,
    )


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

    return templates.TemplateResponse(
        request,
        "partials/file_list.html",
        {"files": files, "workspace": workspace},
    )


@router.post("/files/upload", response_class=HTMLResponse)
async def upload_files(
    request: Request,
    files: list[UploadFile] = File(...),
    workspace_name: str | None = Form(default=None, alias="workspace"),
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
        return _error_response(
            f"Upload exceeds limit ({cfg.max_upload_size_mb} MB)",
            status_code=413,
        )

    manager = get_manager(request)
    selected_workspace = _resolve_workspace(workspace_name, workspace)

    bytes_written = 0
    try:
        with tempfile.TemporaryDirectory(prefix="dlightrag_upload_") as tmp:
            tmp_dir = Path(tmp)
            saved_files = 0
            for f in files:
                if not f.filename:
                    continue
                # Strip any path components from the user-supplied filename:
                # tmp_dir / "../etc/passwd" would otherwise escape tmp_dir.
                safe_name = Path(f.filename).name
                if not safe_name or safe_name in (".", ".."):
                    logger.warning("Rejected upload with unsafe filename: %r", f.filename)
                    continue
                dest = tmp_dir / safe_name
                # Stream-copy with running budget so a missing or lying
                # Content-Length cannot blow past max_bytes.
                with open(dest, "wb") as out:
                    while chunk := await f.read(1024 * 1024):
                        bytes_written += len(chunk)
                        if bytes_written > max_bytes:
                            out.close()
                            dest.unlink(missing_ok=True)
                            return _error_response(
                                f"Upload exceeds limit ({cfg.max_upload_size_mb} MB)",
                                status_code=413,
                            )
                        out.write(chunk)
                saved_files += 1

            if saved_files == 0:
                return _error_response("No valid files selected")

            await manager.aingest(
                workspace=selected_workspace,
                source_type="local",
                path=str(tmp_dir),
            )
    except Exception as e:
        logger.exception("Upload/ingestion failed")
        return _error_response(f"Upload failed: {e}", status_code=500)

    return await _file_list_response(request, selected_workspace)


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
    selected_workspace = _resolve_workspace(request.query_params.get("workspace"), workspace)

    try:
        await manager.delete_files(selected_workspace, file_paths=file_paths)
    except Exception as e:
        logger.exception("Delete failed")
        return _error_response(f"Delete failed: {e}", status_code=500)

    return await _file_list_response(request, selected_workspace)
