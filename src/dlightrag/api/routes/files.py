# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""File operations API routes."""

import mimetypes
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from starlette.responses import RedirectResponse

from dlightrag.api.auth import UserContext, get_current_user
from dlightrag.api.models import DeleteRequest
from dlightrag.config import get_config
from dlightrag.sourcing.azure_blob import generate_azure_sas_url

from .deps import get_manager, resolve_workspace

router = APIRouter()


@router.get("/files")
async def list_files(
    request: Request,
    workspace: str | None = Query(default=None),
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    """List all ingested documents."""
    manager = get_manager(request)
    ws = resolve_workspace(workspace)
    try:
        files = await manager.list_ingested_files(ws)
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=400,
            detail="File listing is not supported in unified RAG mode",
        ) from exc
    return {"files": files, "count": len(files), "workspace": ws}


@router.delete("/files")
async def delete_files(
    body: DeleteRequest, request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any]:
    """Delete documents from knowledge base."""
    manager = get_manager(request)
    ws = resolve_workspace(body.workspace)
    try:
        results = await manager.delete_files(
            ws,
            file_paths=body.file_paths,
            filenames=body.filenames,
            delete_source=body.delete_source,
        )
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=400,
            detail="File deletion is not supported in unified RAG mode",
        ) from exc
    return {"results": results, "workspace": ws}


@router.get("/files/failed")
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
    from each doc's stored file_path scheme (azure://, s3://, otherwise local)."""
    manager = get_manager(request)
    ws = resolve_workspace(workspace)
    return await manager.retry_failed_docs(ws)


@router.get("/api/files/{file_path:path}", response_model=None)
async def serve_file(
    file_path: str, user: UserContext = Depends(get_current_user)
) -> StreamingResponse | RedirectResponse:
    """Serve a file by relative path.

    - Local paths: 200 + StreamingResponse
    - azure://: 302 redirect to SAS signed URL
    - s3://: 501 Not Implemented (S3 presigned URL support in Task 5)
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

    # --- S3: not yet implemented ---
    if file_path.startswith("s3://"):
        raise HTTPException(501, "S3 presigned URL support not yet implemented")

    # --- Unknown remote scheme ---
    if "://" in file_path:
        raise HTTPException(400, f"Unsupported scheme: {file_path.split('://', 1)[0]}")

    # --- Local file: stream from working_dir ---
    working_dir = config.working_dir_path.resolve()
    full_path = (working_dir / file_path).resolve()

    # Security: path traversal + symlink escape check
    if not full_path.is_relative_to(working_dir):
        raise HTTPException(403, "Access denied")
    if not full_path.is_file():
        raise HTTPException(404, "File not found")

    content_type, _ = mimetypes.guess_type(str(full_path))
    return StreamingResponse(
        _stream_file(full_path),
        media_type=content_type or "application/octet-stream",
    )


async def _stream_file(path: Path, chunk_size: int = 64 * 1024) -> AsyncIterator[bytes]:
    """Stream a file in chunks to avoid loading it entirely into memory."""
    import aiofiles

    async with aiofiles.open(path, "rb") as f:
        while chunk := await f.read(chunk_size):
            yield chunk
