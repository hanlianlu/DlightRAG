# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Lightweight PostgreSQL reads for the Web files panel."""

from typing import Any

from dlightrag.utils import normalize_workspace


class PGFilePanelStore:
    """Read file-panel data without constructing a LightRAG service."""

    def __init__(self, *, pool: Any = None) -> None:
        self._pool = pool

    async def _run(self, operation):
        if self._pool is not None:
            async with self._pool.acquire() as conn:
                return await operation(conn)

        from dlightrag.storage.pool import pg_pool

        return await pg_pool.run(operation)

    async def list_processed_files(self, workspace: str) -> list[dict[str, Any]]:
        """Return processed document rows for *workspace* from doc_status."""
        workspace_id = normalize_workspace(workspace)
        if not workspace_id:
            return []

        async def _operation(conn: Any) -> list[dict[str, Any]]:
            rows = await conn.fetch(
                "SELECT id, file_path, updated_at "
                "FROM LIGHTRAG_DOC_STATUS "
                "WHERE workspace = $1 AND status = 'processed' "
                "ORDER BY updated_at DESC, id ASC",
                workspace_id,
            )
            return [_file_row(row) for row in rows]

        return await self._run(_operation)


def _file_row(row: Any) -> dict[str, Any]:
    updated_at = row.get("updated_at")
    isoformat = getattr(updated_at, "isoformat", None)
    return {
        "doc_id": str(row.get("id") or ""),
        "file_path": str(row.get("file_path") or ""),
        "status": "processed",
        "updated_at": isoformat() if callable(isoformat) else str(updated_at or ""),
    }


__all__ = ["PGFilePanelStore"]
