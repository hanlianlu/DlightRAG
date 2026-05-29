# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unified path resolution for retrieval source/media URLs.

Normalizes raw storage paths (local, azure://, s3://) into
/api/files/{path} URLs. The file-serving endpoint handles dispatch:
local → stream, azure → 302 redirect to SAS URL.
"""

from __future__ import annotations

from pathlib import Path


class PathResolver:
    """Normalize raw storage paths into unified /api/files/ URLs.

    All paths — local, azure://, s3:// — are mapped to the
    file-serving endpoint. The endpoint handles source-type dispatch
    internally (local → stream, azure → 302 redirect to SAS URL).

    This keeps the front-end completely decoupled from storage backends.
    The front-end only ever sees one URL format: /api/files/{path}.

    Args:
        working_dir: Legacy RAG storage root directory. Local absolute paths
            under this directory are converted to relative paths.
            Kept for backward compatibility.
        input_dir: Input file storage directory. Paths under this
            directory are resolved relative to it. Takes priority over
            working_dir.
        workspace: Optional workspace name scoping. When set, resolution
            strips ``input_dir/<workspace>/`` prefix from matching paths.
        base_url: URL prefix for the file-serving endpoint
            (default: "/api/files").

    Usage:
        resolver = PathResolver(input_dir="/data/rag_storage/inputs")

        # Local file with workspace in path:
        resolver.resolve("/data/rag_storage/inputs/ws-a/files/f.pdf")
        # → "/api/files/ws-a/files/f.pdf"

        # Azure blob:
        resolver.resolve("azure://mycontainer/doc.pdf")
        # → "/api/files/azure://mycontainer/doc.pdf"

        # Both go through the same endpoint. The endpoint decides
        # whether to stream locally or 302 redirect to a signed URL.
    """

    _FALLBACK_MARKERS = ("artifacts/",)

    def __init__(
        self,
        working_dir: str | None = None,
        input_dir: str | None = None,
        workspace: str | None = None,
        base_url: str = "/api/files",
    ) -> None:
        self._working_dir = working_dir.rstrip("/") if working_dir else None
        self._input_dir = str(Path(input_dir).resolve()) if input_dir else None
        self._workspace = workspace
        self._base_url = base_url.rstrip("/")

    def resolve(self, raw_path: str) -> str:
        """Convert a raw storage path to a /api/files/ URL."""
        # Remote schemes (azure://, s3://) — wrap as-is
        if "://" in raw_path:
            return f"{self._base_url}/{raw_path}"

        # Local path — extract relative, wrap
        relative = self.resolve_relative(raw_path)
        if relative:
            return f"{self._base_url}/{relative}"
        return f"{self._base_url}/{raw_path}"

    def resolve_relative(self, raw_path: str) -> str | None:
        """Extract input_dir- or working_dir-relative path.

        Tries ``input_dir`` first (new), then ``working_dir`` (legacy),
        then falls back to known artifact directory markers.

        E.g. "/data/rag_storage/inputs/ws-a/doc.pdf" with
        input_dir="/data/rag_storage/inputs" → "ws-a/doc.pdf"
        """
        # Try input_dir first (new behaviour)
        if self._input_dir:
            idx = raw_path.find(self._input_dir)
            if idx != -1:
                return raw_path[idx + len(self._input_dir) :].lstrip("/")

        # Try working_dir (legacy backward compat)
        if self._working_dir:
            idx = raw_path.find(self._working_dir)
            if idx != -1:
                return raw_path[idx + len(self._working_dir) :].lstrip("/")

        for marker in self._FALLBACK_MARKERS:
            idx = raw_path.find(marker)
            if idx != -1:
                return raw_path[idx:]

        # Bare filename (no directory separators) — LightRAG canonicalizes
        # file_path to just the basename.  Prepend workspace so the
        # serve-file endpoint can locate it under input_dir/<workspace>/.
        if self._workspace and "/" not in raw_path and "\\" not in raw_path:
            return f"{self._workspace}/{raw_path}"

        return None


__all__ = ["PathResolver"]
