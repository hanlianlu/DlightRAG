# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web routes for file management."""

import logging
import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from dlightrag.access_control import AccessAction
from dlightrag.app_state import request_config
from dlightrag.core.client_contracts import IngestSpec
from dlightrag.core.ingestion.uploads import (
    UploadTooLargeError,
    ignored_upload,
    safe_upload_destination,
    upload_batch_dir,
    write_upload_stream,
)
from dlightrag.core.source_download import (
    LocalDownloadTarget,
    RedirectDownloadTarget,
    SourceDownloadInvalidError,
    SourceDownloadNotFoundError,
    SourceDownloadTarget,
    SourceDownloadUnavailableError,
)
from dlightrag.utils import log_safe
from dlightrag.web.deps import (
    enforce_web_access,
    error_response,
    get_manager,
    get_workspace,
    render_partial,
    templates,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/files/raw/{document_id:path}", response_model=None)
async def download_source(
    document_id: str,
    request: Request,
    workspace: str | None = Query(default=None),
) -> FileResponse | RedirectResponse:
    """Download one source document through the Web session boundary."""
    from dlightrag.utils import normalize_workspace

    safe_workspace = normalize_workspace(workspace or request_config(request).workspace)
    try:
        await enforce_web_access(
            request,
            AccessAction.WORKSPACE_DOWNLOAD_SOURCE,
            safe_workspace,
        )
    except HTTPException as exc:
        if exc.status_code == 403:
            logger.info(
                "source_download_projection_outcome",
                extra={"outcome": "unauthorized", "workspace": safe_workspace},
            )
        raise

    try:
        target = await get_manager(request).aprepare_source_download(
            safe_workspace,
            document_id,
        )
    except SourceDownloadInvalidError as exc:
        raise HTTPException(400, str(exc)) from exc
    except SourceDownloadNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except SourceDownloadUnavailableError as exc:
        raise HTTPException(503, str(exc)) from exc
    return _source_download_response(target)


def _source_download_response(
    target: SourceDownloadTarget,
) -> FileResponse | RedirectResponse:
    if isinstance(target, LocalDownloadTarget):
        return FileResponse(
            target.path,
            media_type=target.media_type,
            filename=target.filename,
        )
    if isinstance(target, RedirectDownloadTarget):
        return RedirectResponse(url=target.url, status_code=302)
    raise TypeError("Unsupported source download target")


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
            for item in await manager.alist_workspaces()
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
            for item in await manager.alist_workspaces()
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
    await enforce_web_access(request, AccessAction.WORKSPACE_LIST_FILES, selected_workspace)
    return await _file_list_response(request, selected_workspace)


async def _file_list_response(request: Request, workspace: str):
    manager = get_manager(request)
    status: dict[str, Any] = {}
    try:
        snapshot = await manager.aget_file_panel_snapshot(workspace)
        files = _file_view_models(list(snapshot.get("files") or []))
        status = dict(snapshot.get("pipeline_status") or {})
    except Exception:
        logger.debug(
            "Could not read files panel snapshot for workspace %s",
            log_safe(workspace),
            exc_info=True,
        )
        files = []

    ingest_busy = status.get("busy", False) or status.get("pending_enqueues", 0) > 0
    if not ingest_busy:
        status = {}

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
    manager = get_manager(request)
    cfg = request_config(request)
    max_bytes = cfg.max_upload_size_mb * 1024 * 1024

    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > max_bytes:
        return error_response(
            f"Upload exceeds limit ({cfg.max_upload_size_mb} MB)",
            status_code=413,
        )

    selected_workspace = _resolve_workspace(workspace_name, workspace)
    if not await _workspace_is_registered(request, selected_workspace):
        return _stale_workspace_response()
    await enforce_web_access(request, AccessAction.WORKSPACE_INGEST, selected_workspace)

    # Detect whether the pipeline is already busy so the UI can show a
    # "queued" state instead of "starting" — LightRAG's request_pending
    # mechanism picks up new enqueues automatically after the current batch.
    already_busy = False
    try:
        ps = await manager.aget_pipeline_status(selected_workspace)
        already_busy = bool(ps.get("busy"))
    except Exception:
        logger.debug(
            "Could not read pipeline status before upload for workspace %s",
            log_safe(selected_workspace),
            exc_info=True,
        )

    bytes_written = 0
    upload_dir: Path | None = None
    try:
        upload_dir = upload_batch_dir(cfg.input_dir_path / selected_workspace)
        saved_paths: list[Path] = []

        for f in files:
            if not f.filename:
                continue
            if ignored_upload(f.filename):
                continue
            try:
                dest = safe_upload_destination(upload_dir, f.filename)
            except ValueError:
                logger.warning("Rejected upload with unsafe filename: %r", f.filename)
                continue
            try:
                bytes_written = await write_upload_stream(
                    f,
                    dest,
                    max_bytes=max_bytes,
                    bytes_written=bytes_written,
                )
            except UploadTooLargeError:
                if upload_dir is not None:
                    shutil.rmtree(upload_dir, ignore_errors=True)
                return error_response(
                    f"Upload exceeds limit ({cfg.max_upload_size_mb} MB)",
                    status_code=413,
                )
            saved_paths.append(dest)

        if not saved_paths:
            if upload_dir is not None:
                shutil.rmtree(upload_dir, ignore_errors=True)
            return error_response("No valid files selected")

    except Exception:
        logger.exception("Upload staging failed")
        if upload_dir is not None:
            shutil.rmtree(upload_dir, ignore_errors=True)
        return error_response("Upload failed. Please try again.", status_code=500)

    try:
        await manager.astart_ingest_job(
            selected_workspace,
            IngestSpec(source_type="local", path=str(upload_dir)),
        )
    except Exception as e:
        logger.exception(
            "Failed to start ingest job for workspace %s",
            log_safe(selected_workspace),
        )
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
    await enforce_web_access(request, AccessAction.WORKSPACE_LIST_FILES, selected_workspace)
    manager = get_manager(request)

    try:
        ps = await manager.aget_pipeline_status(selected_workspace)
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
    response.headers["HX-Reswap"] = "innerHTML"
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
    await enforce_web_access(request, AccessAction.WORKSPACE_DELETE_FILES, selected_workspace)

    try:
        await manager.adelete_files(selected_workspace, file_paths=file_paths)
    except Exception:
        logger.exception("Delete failed")
        return error_response("Delete failed. Please try again.", status_code=500)

    return await _file_list_response(request, selected_workspace)
