# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Lightweight PostgreSQL reads for the Web files panel."""

from typing import Any

from dlightrag.adapters.postgres.core._operations import PostgresOperationRunner


class PGFilePanelStore(PostgresOperationRunner):
    """Read file-panel data without constructing a LightRAG service."""

    async def list_processed_files(self, workspace: str) -> list[dict[str, Any]]:
        """Return processed document rows for *workspace* from doc_status."""
        workspace_id = str(workspace).strip()
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
