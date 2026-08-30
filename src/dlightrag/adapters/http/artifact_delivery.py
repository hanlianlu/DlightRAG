# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared HTTP mechanics for authenticated Answer Artifact delivery."""

from collections.abc import Mapping
from typing import Any, TypeGuard

from fastapi import HTTPException

_INERT_SVG_CSP = "sandbox; default-src 'none'; img-src data:"


def artifact_descriptor(
    result: Mapping[str, Any] | None,
    resource_id: str,
) -> Mapping[str, Any] | None:
    """Return one published Artifact descriptor from an owner-scoped result."""
    for item in (result or {}).get("artifacts") or ():
        if isinstance(item, Mapping) and item.get("resource_id") == resource_id:
            return item
    return None


def artifact_range(header: str, total: int) -> tuple[int, int | None, int, str | None]:
    """Resolve one optional HTTP byte range against an Artifact size."""
    if not header:
        return 0, None, 200, None
    if not header.lower().startswith("bytes=") or "," in header:
        raise _range_not_satisfiable(total)
    start_s, _, end_s = header.split("=", 1)[1].partition("-")
    try:
        if start_s == "":
            suffix = int(end_s)
            if suffix <= 0 or total == 0:
                raise ValueError
            length = min(suffix, total)
            offset = total - length
        else:
            offset = int(start_s)
            end = int(end_s) if end_s else total - 1
            if offset >= total or end < offset:
                raise ValueError
            length = min(end, total - 1) - offset + 1
    except ValueError as exc:
        raise _range_not_satisfiable(total) from exc
    return offset, length, 206, f"bytes {offset}-{offset + length - 1}/{total}"


def artifact_response(
    descriptor: Mapping[str, Any],
    *,
    download: bool,
    content_range: str | None,
) -> tuple[str, dict[str, str]]:
    """Return the safe media type and headers for inert Artifact bytes."""
    media_type = str(descriptor.get("media_type") or "application/octet-stream")
    safe_inline = media_type.startswith("image/") or media_type == "application/pdf"
    effective_type = media_type if safe_inline and not download else "application/octet-stream"
    filename = str(descriptor.get("filename") or "artifact").replace('"', "_")
    disposition = "attachment" if download or not safe_inline else "inline"
    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "private, no-store",
        "Content-Disposition": f'{disposition}; filename="{filename}"',
        "X-Content-Type-Options": "nosniff",
    }
    if media_type == "image/svg+xml":
        headers["Content-Security-Policy"] = _INERT_SVG_CSP
    if content_range is not None:
        headers["Content-Range"] = content_range
    return effective_type, headers


def artifact_presentation_available(
    descriptor: Mapping[str, Any] | None,
) -> TypeGuard[Mapping[str, Any]]:
    """Whether one descriptor may be rendered as a Markdown presentation."""
    return bool(
        descriptor is not None
        and descriptor.get("status") == "available"
        and descriptor.get("media_type") == "text/markdown"
    )


def _range_not_satisfiable(total: int) -> HTTPException:
    return HTTPException(
        status_code=416,
        detail="range not satisfiable",
        headers={"Content-Range": f"bytes */{total}"},
    )


__all__ = [
    "artifact_descriptor",
    "artifact_presentation_available",
    "artifact_range",
    "artifact_response",
]
