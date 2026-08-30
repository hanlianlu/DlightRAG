# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""File operations API routes."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from starlette.responses import FileResponse, RedirectResponse

from dlightrag.adapters.http.rest.auth import get_current_user
from dlightrag.adapters.http.rest.models import (
    DeleteFilesResponse,
    DeleteRequest,
    FailedFilesResponse,
    FileListResponse,
)
from dlightrag.application.access import AccessAction, UserContext
from dlightrag.application.corpus_admin import (
    LocalDownloadTarget,
    RedirectDownloadTarget,
    SourceDownloadInvalidError,
    SourceDownloadNotFoundError,
    SourceDownloadTarget,
    SourceDownloadUnavailableError,
    safe_log_text,
)
from dlightrag.application.errors import WorkspaceWriteFencedError

from .deps import enforce_access, get_application, raise_fenced_http, resolve_workspace

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/files", response_model=FileListResponse)
async def list_files(
    request: Request,
    workspace: str | None = Query(default=None),
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    """List all ingested documents."""
    application = get_application(request)
    ws = resolve_workspace(workspace, request)
    await enforce_access(request, user, AccessAction.WORKSPACE_LIST_FILES, workspace=ws)
    files = await application.corpora.list_ingested_files(ws)
    return {"files": files, "count": len(files), "workspace": ws}


@router.delete("/files", response_model=DeleteFilesResponse)
async def delete_files(
    body: DeleteRequest, request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any]:
    """Delete documents from knowledge base."""
    application = get_application(request)
    ws = resolve_workspace(body.workspace, request)
    await enforce_access(request, user, AccessAction.WORKSPACE_DELETE_FILES, workspace=ws)
    try:
        results = await application.corpora.delete_files(
            ws,
            file_paths=body.file_paths,
            filenames=body.filenames,
            dry_run=body.dry_run,
        )
    except WorkspaceWriteFencedError as exc:
        raise raise_fenced_http(exc) from exc
    return {"results": results, "workspace": ws}


@router.get("/files/failed", response_model=FailedFilesResponse)
async def list_failed_files(
    request: Request,
    workspace: str | None = Query(default=None),
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    """List documents currently in DocStatus.FAILED."""
    application = get_application(request)
    ws = resolve_workspace(workspace, request)
    await enforce_access(request, user, AccessAction.WORKSPACE_LIST_FILES, workspace=ws)
    failed = await application.corpora.list_failed_docs(ws)
    return {"failed": failed, "count": len(failed), "workspace": ws}


@router.post("/files/retry")
async def retry_failed_files(
    request: Request,
    workspace: str | None = Query(default=None),
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    """Re-ingest FAILED documents from stored source/download metadata."""
    application = get_application(request)
    ws = resolve_workspace(workspace, request)
    await enforce_access(request, user, AccessAction.WORKSPACE_INGEST, workspace=ws)
    try:
        return await application.corpora.retry_failed_docs(ws)
    except WorkspaceWriteFencedError as exc:
        raise raise_fenced_http(exc) from exc


@router.get("/files/raw/{document_id:path}", response_model=None)
async def serve_file(
    document_id: str,
    request: Request,
    workspace: str | None = Query(default=None),
    user: UserContext = Depends(get_current_user),
) -> FileResponse | RedirectResponse:
    """Download one source document through the REST Bearer boundary."""
    safe_workspace = resolve_workspace(workspace, request)
    await _enforce_source_download_access(
        request,
        user,
        workspace=safe_workspace,
    )
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
    return _download_response(target)


async def _enforce_source_download_access(
    request: Request,
    user: UserContext,
    *,
    workspace: str,
) -> None:
    try:
        await enforce_access(
            request,
            user,
            AccessAction.WORKSPACE_DOWNLOAD_SOURCE,
            workspace=workspace,
        )
    except HTTPException as exc:
        if exc.status_code == 403:
            logger.info(
                "source_download_projection_outcome",
                extra={"outcome": "unauthorized", "workspace": safe_log_text(workspace)},
            )
        raise


def _download_response(target: SourceDownloadTarget) -> FileResponse | RedirectResponse:
    if isinstance(target, LocalDownloadTarget):
        return FileResponse(
            target.path,
            media_type=target.media_type,
            filename=target.filename,
        )
    if isinstance(target, RedirectDownloadTarget):
        return RedirectResponse(url=target.url, status_code=302)
    raise TypeError("Unsupported source download target")
