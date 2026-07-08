# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""File operations API routes."""

import mimetypes
import os
from pathlib import Path, PureWindowsPath
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from starlette.responses import FileResponse, RedirectResponse

from dlightrag.access_control import AccessAction
from dlightrag.api.auth import UserContext, get_current_user
from dlightrag.api.models import (
    DeleteFilesResponse,
    DeleteRequest,
    FailedFilesResponse,
    FileListResponse,
)
from dlightrag.app_state import request_config
from dlightrag.sourcing.aws_s3 import (
    S3CredentialsUnavailable,
    S3PresignError,
    generate_s3_presigned_url,
)
from dlightrag.sourcing.azure_blob import generate_azure_sas_url
from dlightrag.sourcing.url import validate_public_https_url
from dlightrag.utils import normalize_workspace

from .deps import enforce_access, get_manager, resolve_workspace

router = APIRouter()

# Upper bound on directory entries scanned when locating a source file by its
# canonical basename. Keeps a single download request cheap on large workspaces.
_MAX_WORKSPACE_SCAN_ENTRIES = 50_000


@router.get("/files", response_model=FileListResponse)
async def list_files(
    request: Request,
    workspace: str | None = Query(default=None),
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    """List all ingested documents."""
    manager = get_manager(request)
    ws = resolve_workspace(workspace, request)
    await enforce_access(request, user, AccessAction.WORKSPACE_LIST_FILES, workspace=ws)
    files = await manager.list_ingested_files(ws)
    return {"files": files, "count": len(files), "workspace": ws}


@router.delete("/files", response_model=DeleteFilesResponse)
async def delete_files(
    body: DeleteRequest, request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any]:
    """Delete documents from knowledge base."""
    manager = get_manager(request)
    ws = resolve_workspace(body.workspace, request)
    await enforce_access(request, user, AccessAction.WORKSPACE_DELETE_FILES, workspace=ws)
    results = await manager.delete_files(
        ws,
        file_paths=body.file_paths,
        filenames=body.filenames,
        dry_run=body.dry_run,
    )
    return {"results": results, "workspace": ws}


@router.get("/files/failed", response_model=FailedFilesResponse)
async def list_failed_files(
    request: Request,
    workspace: str | None = Query(default=None),
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    """List documents currently in DocStatus.FAILED."""
    manager = get_manager(request)
    ws = resolve_workspace(workspace, request)
    await enforce_access(request, user, AccessAction.WORKSPACE_LIST_FILES, workspace=ws)
    failed = await manager.list_failed_docs(ws)
    return {"failed": failed, "count": len(failed), "workspace": ws}


@router.post("/files/retry")
async def retry_failed_files(
    request: Request,
    workspace: str | None = Query(default=None),
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    """Re-ingest all FAILED documents (replace=True). Source type is derived
    from each doc's stored file_path scheme (azure://, s3://, https://, otherwise local)."""
    manager = get_manager(request)
    ws = resolve_workspace(workspace, request)
    await enforce_access(request, user, AccessAction.WORKSPACE_INGEST, workspace=ws)
    return await manager.retry_failed_docs(ws)


@router.get("/api/files/{file_path:path}", response_model=None)
async def serve_file(
    file_path: str,
    request: Request,
    workspace: str | None = Query(default=None),
    user: UserContext = Depends(get_current_user),
) -> FileResponse | RedirectResponse:
    """Serve a file by relative path.

    - Local paths under ``input_dir/<workspace>/``: 200 + FileResponse
    - azure://: 302 redirect to SAS signed URL
    - s3://: 302 redirect to S3 presigned URL
    - https://: 302 redirect to original source URL
    """
    config = request_config(request)
    if file_path.startswith(("azure://", "s3://", "https://")):
        await enforce_access(
            request,
            user,
            AccessAction.WORKSPACE_DOWNLOAD_SOURCE,
            workspace=resolve_workspace(workspace, request),
        )

    # --- Azure blob: 302 redirect ---
    if file_path.startswith("azure://"):
        if not config.blob_connection_string:
            raise HTTPException(503, "Azure blob storage not configured")
        sas_url = generate_azure_sas_url(
            connection_string=config.blob_connection_string,
            raw_path=file_path,
            expiry_seconds=config.azure_sas_expiry,
        )
        return RedirectResponse(url=sas_url, status_code=302)

    # --- S3 object: 302 redirect ---
    if file_path.startswith("s3://"):
        try:
            signed_url = await generate_s3_presigned_url(
                raw_path=file_path,
                expiry_seconds=config.s3_presign_expiry,
                region=config.s3_region,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except S3CredentialsUnavailable as exc:
            raise HTTPException(503, "S3 credentials not configured") from exc
        except S3PresignError as exc:
            raise HTTPException(503, "S3 presigned URL generation failed") from exc
        return RedirectResponse(url=signed_url, status_code=302)

    if file_path.startswith("https://"):
        # Only redirect to a safe public source URL — the same invariant URL
        # ingestion enforces. Blocks credentialed / localhost / private-IP-literal
        # targets so the endpoint cannot be abused as an open/SSRF-adjacent redirect.
        try:
            validate_public_https_url(file_path)
        except ValueError as exc:
            raise HTTPException(400, "Invalid source URL") from exc
        return RedirectResponse(url=file_path, status_code=302)

    # --- Unknown remote scheme ---
    if "://" in file_path:
        raise HTTPException(400, f"Unsupported scheme: {file_path.split('://', 1)[0]}")

    # --- Local file ---
    _ensure_safe_local_file_path(file_path)

    input_dir = config.input_dir_path.resolve()
    safe_workspace, relative_path = _resolve_local_file_scope(
        file_path,
        input_dir,
        workspace,
        default_workspace=config.workspace,
    )
    await enforce_access(
        request,
        user,
        AccessAction.WORKSPACE_DOWNLOAD_SOURCE,
        workspace=safe_workspace,
    )

    ws_dir = _contained_resolved_path(input_dir, input_dir / safe_workspace, must_exist=False)
    if ws_dir is None:
        raise HTTPException(403, "Access denied") from None

    resolved = _resolve_workspace_file(ws_dir, relative_path)
    if resolved is None:
        raise HTTPException(404, "File not found")
    return _file_response(resolved)


def _resolve_local_file_scope(
    file_path: str,
    input_dir: Path,
    workspace: str | None,
    *,
    default_workspace: str,
) -> tuple[str, Path]:
    relative_path = Path(file_path.lstrip("/"))
    explicit_workspace = normalize_workspace(workspace) if workspace is not None else None
    explicit_workspace = explicit_workspace or None

    parts = relative_path.parts
    embedded_workspace: str | None = None
    scoped_path = relative_path
    if len(parts) >= 2:
        candidate = normalize_workspace(parts[0])
        if candidate and (input_dir / candidate).is_dir():
            embedded_workspace = candidate
            scoped_path = Path(*parts[1:])

    if explicit_workspace and embedded_workspace and explicit_workspace != embedded_workspace:
        raise HTTPException(403, "Access denied")

    return explicit_workspace or embedded_workspace or normalize_workspace(
        default_workspace
    ), scoped_path


def _ensure_safe_local_file_path(file_path: str) -> None:
    candidate = Path(file_path)
    windows_candidate = PureWindowsPath(file_path)
    if (
        candidate.is_absolute()
        or windows_candidate.is_absolute()
        or windows_candidate.drive
        or ".." in candidate.parts
        or ".." in windows_candidate.parts
    ):
        raise HTTPException(403, "Access denied")


def _resolve_workspace_file(ws_dir: Path, relative_path: Path) -> Path | None:
    """Locate a source file under a workspace input dir, robust to nesting.

    Retrieval canonicalizes ``file_path`` to a bare basename, but the original
    source may be staged at the workspace root, inside a folder-upload subtree,
    or under staging dirs (``__uploads__/<batch>/``, ``__remote_sources__/<type>/``,
    ``__parsed__/``). Resolve in order of increasing cost, always confirming the
    resolved path stays inside the workspace (blocks symlink / traversal escapes).
    """
    # 1. Exact relative path -- handles URLs that already carry the full staged
    #    path (e.g. folder uploads projected from an absolute stored file_path).
    if relative_path.parts:
        exact = _contained_resolved_path(ws_dir, ws_dir / relative_path)
        if exact is not None and exact.is_file():
            return exact

    bare_name = os.path.basename(relative_path.as_posix())
    if not bare_name:
        return None

    # 2. Directly under the workspace root (canonical staged single file).
    root_candidate = _contained_resolved_path(ws_dir, ws_dir / bare_name)
    if root_candidate is not None and root_candidate.is_file():
        return root_candidate

    if not ws_dir.is_dir():
        return None

    # 3. Bounded, symlink-safe recursive search for the basename anywhere under
    #    the workspace (nested folder uploads, __uploads__/<batch>/,
    #    __remote_sources__/<type>/, __parsed__/ copies).
    return _find_file_by_basename(ws_dir, bare_name)


def _find_file_by_basename(ws_dir: Path, bare_name: str) -> Path | None:
    """Depth-first search for ``bare_name`` under ``ws_dir``.

    ``followlinks=False`` prevents descending through directory symlinks, and
    every candidate is re-validated with :func:`_contained_resolved_path` so a
    symlinked file that resolves outside the workspace is rejected. The scan is
    bounded to keep a single download request cheap on large workspaces.
    """
    scanned = 0
    for dirpath, dirnames, filenames in os.walk(ws_dir, followlinks=False):
        # Skip hidden staging/temp directories in place.
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        scanned += len(dirnames) + len(filenames)
        if scanned > _MAX_WORKSPACE_SCAN_ENTRIES:
            break
        if bare_name in filenames:
            candidate = _contained_resolved_path(ws_dir, Path(dirpath) / bare_name)
            if candidate is not None and candidate.is_file():
                return candidate
    return None


def _contained_resolved_path(
    root: Path,
    candidate: Path,
    *,
    must_exist: bool = True,
) -> Path | None:
    try:
        resolved = candidate.resolve(strict=must_exist)
        resolved.relative_to(root.resolve(strict=False))
    except OSError, ValueError:
        return None
    return resolved


def _file_response(path: Path) -> FileResponse:
    content_type, _ = mimetypes.guess_type(str(path))
    # Set an explicit download filename so the browser saves the original name
    # (Content-Disposition: attachment) instead of guessing from the URL or, on
    # error, saving an API JSON body.
    return FileResponse(
        path,
        media_type=content_type or "application/octet-stream",
        filename=path.name,
    )
