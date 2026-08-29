# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Bounded PostgreSQL reads for the Web files panel."""

from typing import Any

from dlightrag.adapters.postgres.core._operations import PostgresOperationRunner
from dlightrag.application.corpus_admin import (
    FilePanelPageRequest,
    FilePanelRowPage,
    ProcessedFileRow,
)

_CREATE_PAGE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_dlightrag_file_panel_processed_updated_id
ON LIGHTRAG_DOC_STATUS (workspace, updated_at DESC NULLS FIRST, id ASC)
WHERE status = 'processed'
"""

_LIST_FIRST_PAGE = """
SELECT id, file_path, updated_at
FROM LIGHTRAG_DOC_STATUS
WHERE workspace = $1 AND status = 'processed'
ORDER BY updated_at DESC NULLS FIRST, id ASC
LIMIT $2
"""

_LIST_AFTER_NULL = """
SELECT id, file_path, updated_at
FROM LIGHTRAG_DOC_STATUS
WHERE workspace = $1 AND status = 'processed'
  AND (
    (updated_at IS NULL AND id > $2)
    OR updated_at IS NOT NULL
  )
ORDER BY updated_at DESC NULLS FIRST, id ASC
LIMIT $3
"""

_LIST_AFTER_TIMESTAMP = """
SELECT id, file_path, updated_at
FROM LIGHTRAG_DOC_STATUS
WHERE workspace = $1 AND status = 'processed'
  AND updated_at IS NOT NULL
  AND (
    updated_at < $2::timestamp
    OR (updated_at = $2::timestamp AND id > $3)
  )
ORDER BY updated_at DESC NULLS FIRST, id ASC
LIMIT $4
"""


class PGFilePanelStore(PostgresOperationRunner):
    """Read file-panel data without constructing a LightRAG runtime."""

    async def ensure_page_index(self) -> None:
        """Create the exact writer-owned index after LightRAG creates its table."""

        async def _operation(conn: Any) -> None:
            await conn.execute(_CREATE_PAGE_INDEX)

        await self._run(_operation)

    async def list_processed_files(
        self,
        workspace: str,
        *,
        page: FilePanelPageRequest,
    ) -> FilePanelRowPage:
        """Return one physical limit+1 keyset page for a canonical workspace."""
        workspace_id = str(workspace).strip()
        if not workspace_id:
            raise ValueError("workspace cannot be empty")
        validated = FilePanelPageRequest(limit=page.limit, cursor=page.cursor)
        if validated.cursor is not None and validated.cursor.workspace != workspace_id:
            raise ValueError("file-panel cursor belongs to another workspace")
        fetch_limit = validated.limit + 1

        async def _operation(conn: Any) -> FilePanelRowPage:
            cursor = validated.cursor
            if cursor is None:
                rows = await conn.fetch(_LIST_FIRST_PAGE, workspace_id, fetch_limit)
            elif cursor.updated_at is None:
                rows = await conn.fetch(
                    _LIST_AFTER_NULL,
                    workspace_id,
                    cursor.doc_id,
                    fetch_limit,
                )
            else:
                rows = await conn.fetch(
                    _LIST_AFTER_TIMESTAMP,
                    workspace_id,
                    cursor.updated_at,
                    cursor.doc_id,
                    fetch_limit,
                )
            fetched_rows = len(rows)
            return FilePanelRowPage(
                items=tuple(_file_row(row) for row in rows[: validated.limit]),
                has_more=fetched_rows > validated.limit,
                fetched_rows=fetched_rows,
            )

        return await self._run(_operation)


def _file_row(row: Any) -> ProcessedFileRow:
    return ProcessedFileRow(
        doc_id=str(row.get("id") or ""),
        file_path=str(row.get("file_path") or ""),
        updated_at=row.get("updated_at"),
    )


__all__ = ["PGFilePanelStore"]
