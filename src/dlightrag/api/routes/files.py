# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""File operations API routes."""

import mimetypes
from collections.abc import AsyncIterator
from pathlib import Path, PureWindowsPath
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from starlette.responses import RedirectResponse

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
from dlightrag.utils import normalize_workspace

from .deps import enforce_access, get_manager, resolve_workspace

router = APIRouter()


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
) -> StreamingResponse | RedirectResponse:
    """Serve a file by relative path.

    - Local paths under ``input_dir/<workspace>/``: 200 + StreamingResponse
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

    ws_dir = (input_dir / safe_workspace).resolve()
    try:
        ws_dir.relative_to(input_dir)
    except ValueError:
        raise HTTPException(403, "Access denied") from None

    full_path = (ws_dir / relative_path).resolve()
    try:
        full_path.relative_to(ws_dir)
    except ValueError:
        pass  # symlink escape, skip direct lookup
    else:
        try:
            if full_path.is_file():
                content_type, _ = mimetypes.guess_type(str(full_path))
                return StreamingResponse(
                    _stream_file(full_path),
                    media_type=content_type or "application/octet-stream",
                )
        except OSError:
            pass

    # --- Canonical basename lookup ---
    # LightRAG canonicalizes file_path to just the basename (no directory
    # components).  The original file may live in __parsed__/ or another
    # workspace subdirectory.
    bare_name = relative_path.name
    if ws_dir.is_dir():
        for sub in _iter_workspace_lookup_dirs(ws_dir):
            candidate = (sub / bare_name).resolve()
            try:
                candidate.relative_to(ws_dir)
            except ValueError:
                continue  # symlink escape, skip
            if candidate.is_file():
                content_type, _ = mimetypes.guess_type(str(candidate))
                return StreamingResponse(
                    _stream_file(candidate),
                    media_type=content_type or "application/octet-stream",
                )

    raise HTTPException(404, "File not found")


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


def _iter_workspace_lookup_dirs(ws_dir: Path) -> list[Path]:
    """Yield workspace subdirectories to search for bare filenames.

    ``__parsed__/`` comes first because every parsed document gets a copy
    there during ingestion.  Other immediate subdirectories follow.
    """
    dirs: list[Path] = []
    parsed: Path | None = None
    try:
        for entry in ws_dir.iterdir():
            if entry.is_dir():
                if entry.name == "__parsed__":
                    parsed = entry
                else:
                    dirs.append(entry)
    except OSError:
        return []
    if parsed is not None:
        dirs.insert(0, parsed)
    return dirs


async def _stream_file(path: Path, chunk_size: int = 64 * 1024) -> AsyncIterator[bytes]:
    """Stream a file in chunks to avoid loading it entirely into memory."""
    import asyncio

    with path.open("rb") as f:
        while chunk := await asyncio.to_thread(f.read, chunk_size):
            yield chunk
