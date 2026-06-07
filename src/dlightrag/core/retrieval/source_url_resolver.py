# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""URL projection for retrieval source/media paths.

Normalizes raw storage paths (local, azure://, s3://) into
/api/files/{path} URLs. The file-serving endpoint handles dispatch:
local → stream, azure → 302 redirect to SAS URL,
s3 → 302 redirect to presigned URL.
"""

from __future__ import annotations

from pathlib import Path, PureWindowsPath


class SourceUrlResolver:
    """Normalize raw storage paths into unified /api/files/ URLs.

    All paths — local, azure://, s3:// — are mapped to the
    file-serving endpoint. The endpoint handles source-type dispatch
    internally (local → stream, azure/S3 → 302 redirect to signed URL).

    This keeps the front-end completely decoupled from storage backends.
    The front-end only ever sees one URL format: /api/files/{path}.

    Args:
        input_dir: Input file storage directory. Paths under this
            directory are resolved relative to it.
        workspace: Optional workspace name scoping. When set, resolution
            strips ``input_dir/<workspace>/`` prefix from matching paths.
        base_url: URL prefix for the file-serving endpoint
            (default: "/api/files").

    Usage:
        resolver = SourceUrlResolver(input_dir="/data/rag_storage/inputs")

        # Local file with workspace in path:
        resolver.resolve("/data/rag_storage/inputs/ws-a/files/f.pdf")
        # → "/api/files/ws-a/files/f.pdf"

        # Azure blob:
        resolver.resolve("azure://mycontainer/doc.pdf")
        # → "/api/files/azure://mycontainer/doc.pdf"

        # Both go through the same endpoint. The endpoint decides
        # whether to stream locally or 302 redirect to a signed URL.
    """

    def __init__(
        self,
        input_dir: str | None = None,
        workspace: str | None = None,
        base_url: str = "/api/files",
    ) -> None:
        self._input_dir = Path(input_dir).resolve() if input_dir else None
        self._workspace = workspace
        self._base_url = base_url.rstrip("/")

    def resolve(self, raw_path: str) -> str | None:
        """Convert a servable stored path to a /api/files/ URL."""
        if raw_path.startswith(("azure://", "s3://")):
            return f"{self._base_url}/{raw_path}"
        if "://" in raw_path:
            return None

        relative = self.resolve_relative(raw_path)
        return f"{self._base_url}/{relative}" if relative else None

    def resolve_relative(self, raw_path: str) -> str | None:
        """Extract an endpoint-relative source path.

        Example: "/data/rag_storage/inputs/ws-a/doc.pdf" with
        input_dir="/data/rag_storage/inputs" → "ws-a/doc.pdf"
        """
        if self._input_dir and raw_path.startswith("/"):
            try:
                relative = Path(raw_path).resolve().relative_to(self._input_dir)
            except ValueError:
                pass
            else:
                return relative.as_posix()

        if raw_path.startswith("/"):
            return None

        windows_path = PureWindowsPath(raw_path)
        if windows_path.is_absolute() or windows_path.drive:
            return None

        if "\\" in raw_path:
            parts = windows_path.parts
            if ".." in parts:
                return None
            if self._workspace and len(parts) == 1:
                return f"{self._workspace}/{parts[0]}"
            return Path(*parts).as_posix()

        # Bare filename (no directory separators) — LightRAG canonicalizes
        # file_path to just the basename.  Prepend workspace so the
        # serve-file endpoint can locate it under input_dir/<workspace>/.
        if self._workspace and "/" not in raw_path and "\\" not in raw_path:
            return f"{self._workspace}/{raw_path}"

        relative = Path(raw_path)
        if ".." in relative.parts or relative.is_absolute():
            return None
        return relative.as_posix()


__all__ = ["SourceUrlResolver"]
