# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""File operations API routes."""

import mimetypes
from collections.abc import AsyncIterator
from pathlib import Path, PureWindowsPath
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from starlette.responses import RedirectResponse

from dlightrag.api.auth import UserContext, get_current_user
from dlightrag.api.models import (
    DeleteFilesResponse,
    DeleteRequest,
    FailedFilesResponse,
    FileListResponse,
)
from dlightrag.config import get_config
from dlightrag.sourcing.aws_s3 import (
    S3CredentialsUnavailable,
    S3PresignError,
    generate_s3_presigned_url,
)
from dlightrag.sourcing.azure_blob import generate_azure_sas_url
from dlightrag.utils import normalize_workspace

from .deps import get_manager, resolve_workspace

router = APIRouter()


@router.get("/files", response_model=FileListResponse)
async def list_files(
    request: Request,
    workspace: str | None = Query(default=None),
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    """List all ingested documents."""
    manager = get_manager(request)
    ws = resolve_workspace(workspace)
    files = await manager.list_ingested_files(ws)
    return {"files": files, "count": len(files), "workspace": ws}


@router.delete("/files", response_model=DeleteFilesResponse)
async def delete_files(
    body: DeleteRequest, request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any]:
    """Delete documents from knowledge base."""
    manager = get_manager(request)
    ws = resolve_workspace(body.workspace)
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
    ws = resolve_workspace(workspace)
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
    ws = resolve_workspace(workspace)
    return await manager.retry_failed_docs(ws)


@router.get("/api/files/{file_path:path}", response_model=None)
async def serve_file(
    file_path: str,
    workspace: str = Query(default="default"),
    user: UserContext = Depends(get_current_user),
) -> StreamingResponse | RedirectResponse:
    """Serve a file by relative path.

    - Local paths under ``input_dir/<workspace>/``: 200 + StreamingResponse
    - azure://: 302 redirect to SAS signed URL
    - s3://: 302 redirect to S3 presigned URL
    - https://: 302 redirect to original source URL
    """
    config = get_config()

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

    # SourceUrlResolver embeds workspace in the URL path, so try
    # input_dir/<file_path> directly first. Also support explicit
    # ?workspace=... callers with input_dir/<workspace>/<file_path>.
    input_dir = config.input_dir_path.resolve()
    safe_workspace = normalize_workspace(workspace) or "default"
    attempts: list[Path] = []
    # Direct path (workspace already in file_path from SourceUrlResolver)
    direct = (input_dir / file_path.lstrip("/")).resolve()
    # Explicit ?workspace=... path
    ws_prefixed = (input_dir / safe_workspace / file_path.lstrip("/")).resolve()
    if direct != ws_prefixed:
        attempts = [direct, ws_prefixed]
    else:
        attempts = [direct]

    for full_path in attempts:
        try:
            full_path.relative_to(input_dir)
        except ValueError:
            continue  # path traversal attempt, skip
        if full_path.is_file():
            content_type, _ = mimetypes.guess_type(str(full_path))
            return StreamingResponse(
                _stream_file(full_path),
                media_type=content_type or "application/octet-stream",
            )

    # --- Canonical basename lookup ---
    # LightRAG canonicalizes file_path to just the basename (no directory
    # components).  The original file may live in __parsed__/ or another
    # workspace subdirectory.
    ws_dir = input_dir / safe_workspace
    try:
        ws_dir.relative_to(input_dir)
    except ValueError:
        pass  # traversal attempt, skip lookup
    else:
        bare_name = Path(file_path.lstrip("/")).name
        if ws_dir.is_dir():
            for sub in _iter_workspace_lookup_dirs(ws_dir):
                candidate = (sub / bare_name).resolve()
                try:
                    candidate.relative_to(input_dir)
                except ValueError:
                    continue  # symlink escape, skip
                if candidate.is_file():
                    content_type, _ = mimetypes.guess_type(str(candidate))
                    return StreamingResponse(
                        _stream_file(candidate),
                        media_type=content_type or "application/octet-stream",
                    )

    raise HTTPException(404, "File not found")


def _ensure_safe_local_file_path(file_path: str) -> None:
    candidate = Path(file_path)
    windows_candidate = PureWindowsPath(file_path)
    if (
        candidate.is_absolute()
        or windows_candidate.is_absolute()
        or windows_candidate.drive
        or ".." in candidate.parts
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
