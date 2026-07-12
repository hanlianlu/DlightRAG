# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Build adapter-owned source download links from internal document IDs."""

from urllib.parse import quote


class SourceDownloadLinkBuilder:
    """Project a workspace-scoped document ID into one HTTP adapter URL."""

    def __init__(self, base_url: str = "/files/raw") -> None:
        self._base_url = base_url.rstrip("/")

    def resolve(self, document_id: str, *, workspace: str) -> str | None:
        if not document_id or not workspace:
            return None
        return (
            f"{self._base_url}/{quote(document_id, safe='')}?workspace={quote(workspace, safe='')}"
        )


__all__ = ["SourceDownloadLinkBuilder"]
