# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""HTTP projection for internal source download locators.

Projects local, azure://, s3://, and https:// download locators into authenticated
/files/raw/{path} URLs. The file-serving endpoint handles dispatch:
local → stream, azure → 302 redirect to SAS URL,
s3 → 302 redirect to presigned URL, https → 302 redirect to source URL.
"""

from pathlib import Path, PureWindowsPath
from urllib.parse import quote


class SourceUrlResolver:
    """Project internal download locators into unified /files/raw/ URLs.

    All paths — local, azure://, s3://, https:// — are mapped to the
    file-serving endpoint. The endpoint handles source-type dispatch
    internally (local → stream, remote sources → 302 redirect).

    This keeps the front-end completely decoupled from storage backends.
    Raw locators stay inside the adapter boundary. The front-end only sees the
    projected /files/raw/{path} URL.

    Args:
        input_dir: Input file storage directory. Paths under this
            directory are resolved relative to it.
        workspace: Optional workspace name scoping. When set, resolution
            strips ``input_dir/<workspace>/`` prefix from matching paths.
        base_url: URL prefix for the file-serving endpoint
            (default: "/files/raw").

    Usage:
        resolver = SourceUrlResolver(input_dir="/data/rag_storage/inputs")

        # Local file with workspace in path:
        resolver.resolve("/data/rag_storage/inputs/ws-a/files/f.pdf")
        # → "/files/raw/ws-a/files/f.pdf"

        # Azure blob:
        resolver.resolve("azure://mycontainer/doc.pdf")
        # → "/files/raw/azure://mycontainer/doc.pdf"

        # Public URL source:
        resolver.resolve("https://example.com/doc.pdf")
        # → "/files/raw/https://example.com/doc.pdf"

        # Both go through the same endpoint. The endpoint decides
        # whether to stream locally or 302 redirect to a signed URL.
    """

    def __init__(
        self,
        input_dir: str | None = None,
        workspace: str | None = None,
        base_url: str = "/files/raw",
    ) -> None:
        self._input_dir = Path(input_dir).resolve() if input_dir else None
        self._workspace = workspace
        self._base_url = base_url.rstrip("/")

    def resolve(self, raw_path: str, *, workspace: str | None = None) -> str | None:
        """Project a validated internal download locator to a client URL."""
        effective_workspace = workspace if workspace is not None else self._workspace
        if raw_path.startswith(("azure://", "s3://")):
            resolved = f"{self._base_url}/{quote(raw_path, safe='/:')}"
            return self._with_workspace(resolved, effective_workspace)
        if raw_path.startswith("https://"):
            resolved = f"{self._base_url}/{quote(raw_path, safe='/:')}"
            return self._with_workspace(resolved, effective_workspace)
        if "://" in raw_path:
            return None

        relative = self.resolve_relative(raw_path, workspace=effective_workspace)
        if not relative:
            return None
        # Percent-encode path segments (spaces, unicode, reserved chars) while
        # keeping "/" as the separator so the resulting href is a valid URL.
        resolved = f"{self._base_url}/{quote(relative, safe='/')}"
        return self._with_workspace(resolved, effective_workspace)

    def resolve_relative(
        self,
        raw_path: str,
        *,
        workspace: str | None = None,
    ) -> str | None:
        """Extract an endpoint-relative path from a local download locator.

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
            if workspace and len(parts) == 1:
                return f"{workspace}/{parts[0]}"
            return Path(*parts).as_posix()

        # Bare filename (no directory separators) — LightRAG canonicalizes
        # file_path to just the basename.  Prepend workspace so the
        # serve-file endpoint can locate it under input_dir/<workspace>/.
        if workspace and "/" not in raw_path and "\\" not in raw_path:
            return f"{workspace}/{raw_path}"

        relative = Path(raw_path)
        if ".." in relative.parts or relative.is_absolute():
            return None
        return relative.as_posix()

    @staticmethod
    def _with_workspace(url: str, workspace: str | None) -> str:
        if not workspace:
            return url
        return f"{url}?workspace={quote(workspace, safe='')}"


__all__ = ["SourceUrlResolver"]
