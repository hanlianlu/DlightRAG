# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""File-serving endpoint — unified file access for all storage backends.

Local files are streamed directly. Azure blobs return 302 redirect to
a time-limited SAS signed URL (best practice per AWS/GCS/ActiveStorage).
"""
from __future__ import annotations

import mimetypes
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, HTTPException
from starlette.responses import RedirectResponse, StreamingResponse

from dlightrag.config import DlightragConfig
from dlightrag.sourcing.azure_blob import generate_azure_sas_url


def create_file_router(config: DlightragConfig) -> APIRouter:
    """Create the file-serving router bound to a config instance."""
    router = APIRouter()

    @router.get("/api/files/{file_path:path}", response_model=None)
    async def serve_file(file_path: str) -> StreamingResponse | RedirectResponse:
        """Serve a file by relative path.

        - Local paths: 200 + StreamingResponse
        - azure://: 302 redirect to SAS signed URL
        - snowflake://: 400 (no file to serve)
        """
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

        # --- Snowflake: no file to serve ---
        if file_path.startswith("snowflake://"):
            raise HTTPException(400, "Snowflake sources have no downloadable file")

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

    return router


async def _stream_file(path: Path, chunk_size: int = 64 * 1024) -> AsyncIterator[bytes]:
    """Stream a file in chunks to avoid loading it entirely into memory."""
    import aiofiles

    async with aiofiles.open(path, "rb") as f:
        while chunk := await f.read(chunk_size):
            yield chunk
