# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unified path resolution for retrieval source/media URLs.

Normalizes raw storage paths (local, azure://, snowflake://) into
/api/files/{path} URLs. The file-serving endpoint handles dispatch:
local → stream, azure → 302 redirect to SAS URL.
"""
from __future__ import annotations


class PathResolver:
    """Normalize raw storage paths into unified /api/files/ URLs.

    All paths — local, azure://, snowflake:// — are mapped to the
    file-serving endpoint. The endpoint handles source-type dispatch
    internally (local → stream, azure → 302 redirect to SAS URL).

    This keeps the front-end completely decoupled from storage backends.
    The front-end only ever sees one URL format: /api/files/{path}.

    Args:
        working_dir: RAG storage root directory. Local absolute paths
            under this directory are converted to relative paths.
        base_url: URL prefix for the file-serving endpoint
            (default: "/api/files").

    Usage:
        resolver = PathResolver(working_dir="/data/rag_storage")

        # Local file:
        resolver.resolve("/data/rag_storage/sources/local/f.pdf")
        # → "/api/files/sources/local/f.pdf"

        # Azure blob:
        resolver.resolve("azure://mycontainer/doc.pdf")
        # → "/api/files/azure://mycontainer/doc.pdf"

        # Both go through the same endpoint. The endpoint decides
        # whether to stream locally or 302 redirect to a signed URL.
    """

    _FALLBACK_MARKERS = ("sources/", "artifacts/")

    def __init__(
        self,
        working_dir: str | None = None,
        base_url: str = "/api/files",
    ) -> None:
        self._working_dir = working_dir.rstrip("/") if working_dir else None
        self._base_url = base_url.rstrip("/")

    def resolve(self, raw_path: str) -> str:
        """Convert a raw storage path to a /api/files/ URL."""
        # Strip file:// scheme — treat as local
        if raw_path.startswith("file://"):
            raw_path = raw_path[7:]

        # Remote schemes (azure://, snowflake://) — wrap as-is
        if "://" in raw_path:
            return f"{self._base_url}/{raw_path}"

        # Local path — extract relative, wrap
        relative = self.resolve_relative(raw_path)
        if relative:
            return f"{self._base_url}/{relative}"
        return f"{self._base_url}/{raw_path}"

    def resolve_relative(self, raw_path: str) -> str | None:
        """Extract working_dir-relative path. None if not under working_dir.

        Tries working_dir prefix first, then falls back to known directory
        markers (sources/, artifacts/).

        E.g. "/data/rag_storage/sources/local/f.pdf" → "sources/local/f.pdf"
        """
        if self._working_dir:
            idx = raw_path.find(self._working_dir)
            if idx != -1:
                return raw_path[idx + len(self._working_dir) :].lstrip("/")

        for marker in self._FALLBACK_MARKERS:
            idx = raw_path.find(marker)
            if idx != -1:
                return raw_path[idx:]

        return None


__all__ = ["PathResolver"]
