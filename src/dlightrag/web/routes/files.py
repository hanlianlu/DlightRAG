# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web routes for file management."""

import logging
import shutil
from pathlib import Path
from typing import Any, NoReturn

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse

from dlightrag.access import AccessAction
from dlightrag.ai.telemetry import safe_log_text
from dlightrag.services.corpora import IngestSpec
from dlightrag.services.errors import (
    LocalDownloadTarget,
    RedirectDownloadTarget,
    SourceDownloadInvalidError,
    SourceDownloadNotFoundError,
    SourceDownloadTarget,
    SourceDownloadUnavailableError,
    UnsafeUploadNameError,
    UploadTooLargeError,
)
from dlightrag.web.deps import enforce_web_access, get_application, get_workspace
from dlightrag.web.file_models import (
    WebFileItem,
    WebFilePanelSnapshot,
    WebIngestStatus,
    WebUploadReceipt,
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
    from dlightrag.rag.workspaces import normalize_workspace

    safe_workspace = normalize_workspace(
        workspace or get_application(request).config.deployment.workspace
    )
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
                extra={"outcome": "unauthorized", "workspace": safe_log_text(safe_workspace)},
            )
        raise

    try:
        target = await get_application(request).corpora.prepare_source_download(
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
    from dlightrag.rag.workspaces import normalize_workspace

    if not requested:
        return cookie_workspace
    normalized = normalize_workspace(requested)
    return normalized or cookie_workspace


async def _resolve_registered_workspace(
    request: Request,
    workspace: str,
) -> str | None:
    """Return the requested workspace when it is registered."""
    from dlightrag.rag.workspaces import normalize_workspace

    application = get_application(request)
    try:
        known = {
            normalized
            for item in await application.corpora.list_workspaces()
            if (normalized := normalize_workspace(item))
        }
    except Exception:
        return workspace
    if workspace in known:
        return workspace
    return None


async def _workspace_is_registered(request: Request, workspace: str) -> bool:
    """Return whether a workspace is registered; fail open on registry outages."""
    from dlightrag.rag.workspaces import normalize_workspace

    application = get_application(request)
    try:
        known = {
            normalized
            for item in await application.corpora.list_workspaces()
            if (normalized := normalize_workspace(item))
        }
    except Exception:
        return True
    return workspace in known


def _stale_workspace() -> NoReturn:
    raise HTTPException(
        status_code=409,
        detail="Workspace no longer exists. Refresh and choose an existing workspace.",
    )


def _file_view_models(files: list[dict[str, Any]]) -> list[WebFileItem]:
    rows: list[WebFileItem] = []
    for item in files:
        file_path = str(item.get("file_path") or "")
        file_name = str(item.get("file_name") or item.get("filename") or "")
        if not file_name and file_path:
            file_name = Path(file_path).name
        if not file_name:
            file_name = str(item.get("doc_id") or "Untitled file")
        rows.append(WebFileItem(file_name=file_name, file_path=file_path))
    return rows


def _ingest_status(status: dict[str, Any], *, message: str = "") -> WebIngestStatus:
    pending = max(0, int(status.get("pending_enqueues") or 0))
    busy = bool(status.get("busy")) or pending > 0
    batches = max(0, int(status.get("batchs") or 0))
    current = max(0, int(status.get("cur_batch") or 0))
    documents = max(0, int(status.get("docs") or 0))
    progress = min(100, int(current / batches * 100)) if documents and batches else None
    return WebIngestStatus(
        busy=busy,
        message=str(status.get("latest_message") or message or ("Ingesting..." if busy else "")),
        progress_percent=progress,
        current_batch=current if documents and batches else None,
        total_batches=batches if documents and batches else None,
        documents=documents if documents and batches else None,
        pending_enqueues=pending,
    )


# ---------------------------------------------------------------------------
# GET /web/api/files — file list panel content
# ---------------------------------------------------------------------------


@router.get("/files", response_model=WebFilePanelSnapshot)
async def file_list(
    request: Request,
    workspace: str = Depends(get_workspace),
    workspace_name: str | None = Query(default=None, alias="workspace"),
) -> WebFilePanelSnapshot:
    """Return one typed Files panel snapshot."""
    selected_workspace = _resolve_workspace(workspace_name, workspace)
    selected_workspace = await _resolve_registered_workspace(request, selected_workspace)
    if selected_workspace is None:
        _stale_workspace()
    await enforce_web_access(request, AccessAction.WORKSPACE_LIST_FILES, selected_workspace)
    return await _file_panel_snapshot(request, selected_workspace)


async def _file_panel_snapshot(request: Request, workspace: str) -> WebFilePanelSnapshot:
    try:
        snapshot = await get_application(request).corpora.file_panel_snapshot(workspace)
    except Exception:
        logger.exception(
            "Could not read Files panel snapshot for workspace %s",
            safe_log_text(workspace),
        )
        raise HTTPException(status_code=503, detail="Files are temporarily unavailable") from None
    return WebFilePanelSnapshot(
        workspace=workspace,
        files=_file_view_models(list(snapshot.get("files") or [])),
        ingest=_ingest_status(dict(snapshot.get("pipeline_status") or {})),
    )


# ---------------------------------------------------------------------------
# POST /web/api/files/upload — non-blocking upload + background ingest
# ---------------------------------------------------------------------------


@router.post("/files/upload", response_model=WebUploadReceipt)
async def upload_files(
    request: Request,
    files: list[UploadFile] = File(...),
    workspace_name: str | None = Form(default=None, alias="workspace"),
    workspace: str = Depends(get_workspace),
):
    """Upload files and start background ingest.  Returns immediately."""
    application = get_application(request)
    cfg = application.config
    # Per-file document cap is the single shared limit used by every ingest
    # path (REST /ingest/blob, URL, web upload): one document may not exceed it.
    # The larger per-request cap is a temp-directory guard for multi-file
    # (folder) uploads.
    per_file_max_bytes = cfg.corpus.ingestion.max_upload_bytes
    batch_max_bytes = cfg.max_upload_batch_bytes
    per_file_max_mb = per_file_max_bytes // (1024 * 1024)

    selected_workspace = _resolve_workspace(workspace_name, workspace)
    if not await _workspace_is_registered(request, selected_workspace):
        _stale_workspace()
    await enforce_web_access(request, AccessAction.WORKSPACE_INGEST, selected_workspace)

    # Detect whether the pipeline is already busy so the UI can show a
    # "queued" state instead of "starting" — LightRAG's request_pending
    # mechanism picks up new enqueues automatically after the current batch.
    already_busy = False
    try:
        ps = await application.corpora.get_pipeline_status(selected_workspace)
        already_busy = bool(ps.get("busy"))
    except Exception:
        logger.debug(
            "Could not read pipeline status before upload for workspace %s",
            safe_log_text(selected_workspace),
            exc_info=True,
        )

    upload_dir: Path | None = None
    try:
        upload_dir, saved_paths = await application.corpora.stage_upload_batch(
            selected_workspace,
            [(f.filename or "", f) for f in files],
            per_file_max_bytes=per_file_max_bytes,
            batch_max_bytes=batch_max_bytes,
        )
        if not saved_paths:
            if upload_dir is not None:
                shutil.rmtree(upload_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail="No valid files selected")
    except UnsafeUploadNameError as exc:
        logger.warning("Rejected upload with unsafe filename: %s", exc)
        raise HTTPException(status_code=400, detail="Upload contains an unsafe filename") from None
    except UploadTooLargeError:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Upload exceeds limit ({per_file_max_mb} MB per file, "
                f"{cfg.interfaces.max_upload_size_mb} MB per request)"
            ),
        ) from None
    except Exception:
        logger.exception("Upload staging failed")
        if upload_dir is not None:
            shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail="Upload failed. Please try again.") from None

    try:
        await application.corpora.start_ingest_job(
            selected_workspace,
            IngestSpec(source_type="local", path=str(upload_dir)),
        )
    except Exception:
        logger.exception(
            "Failed to start ingest job for workspace %s",
            safe_log_text(selected_workspace),
        )
        if upload_dir is not None:
            shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(
            status_code=500,
            detail="Upload staged but ingest did not start. Please retry.",
        ) from None

    return WebUploadReceipt(
        workspace=selected_workspace,
        file_count=len(saved_paths),
        queued=already_busy,
        ingest=WebIngestStatus(
            busy=True,
            message=(
                "Queued — processing after current batch" if already_busy else "Starting ingest..."
            ),
        ),
    )


# ---------------------------------------------------------------------------
# GET /web/api/ingest-status — browser polling endpoint
# ---------------------------------------------------------------------------


@router.get("/ingest-status", response_model=WebIngestStatus)
async def ingest_status(
    request: Request,
    workspace: str = Depends(get_workspace),
    workspace_name: str | None = Query(default=None, alias="workspace"),
) -> WebIngestStatus:
    """Return one typed ingest status for browser polling."""
    selected_workspace = _resolve_workspace(workspace_name, workspace)
    selected_workspace = await _resolve_registered_workspace(request, selected_workspace)
    if selected_workspace is None:
        _stale_workspace()
    await enforce_web_access(request, AccessAction.WORKSPACE_LIST_FILES, selected_workspace)
    try:
        status = await get_application(request).corpora.get_pipeline_status(selected_workspace)
    except Exception:
        logger.exception(
            "Could not read ingest status for workspace %s",
            safe_log_text(selected_workspace),
        )
        raise HTTPException(status_code=503, detail="Ingest status is unavailable") from None
    return _ingest_status(dict(status or {}))


# ---------------------------------------------------------------------------
# DELETE /web/api/files
# ---------------------------------------------------------------------------


@router.delete("/files", response_model=WebFilePanelSnapshot)
async def delete_files(
    request: Request,
    workspace: str = Depends(get_workspace),
):
    """Delete files from workspace."""
    file_path = request.query_params.get("file_path", "")
    file_paths = [file_path] if file_path else []
    application = get_application(request)
    selected_workspace = _resolve_workspace(request.query_params.get("workspace"), workspace)
    if not await _workspace_is_registered(request, selected_workspace):
        _stale_workspace()
    await enforce_web_access(request, AccessAction.WORKSPACE_DELETE_FILES, selected_workspace)

    try:
        await application.corpora.delete_files(selected_workspace, file_paths=file_paths)
    except Exception:
        logger.exception("Delete failed")
        raise HTTPException(status_code=500, detail="Delete failed. Please try again.") from None

    return await _file_panel_snapshot(request, selected_workspace)
