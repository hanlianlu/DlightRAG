# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Serve Vite-owned HTML entries through the authenticated Web adapter."""

from fastapi import HTTPException
from fastapi.responses import FileResponse

from dlightrag.web.static_files import APP_DIR

_INDEX_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
}


def app_html_response(filename: str) -> FileResponse:
    path = APP_DIR / filename
    if not path.is_file():
        raise HTTPException(
            status_code=503,
            detail="Web application assets are unavailable; run the frontend build",
        )
    return FileResponse(
        path,
        media_type="text/html; charset=utf-8",
        headers=_INDEX_HEADERS,
    )


__all__ = ["app_html_response"]
