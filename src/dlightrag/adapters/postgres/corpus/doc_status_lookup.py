# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Indexed PostgreSQL lookups for deletion-time LightRAG status identity."""

from collections.abc import Sequence
from typing import Any

from dlightrag.adapters.postgres.core._operations import (
    ConnectionPool,
    PostgresOperationRunner,
)
from dlightrag.engine.rag.corpus.contracts import DocStatusMatch
from dlightrag.engine.rag.workspace.workspaces import require_canonical_workspace_id

_BY_IDS = """
SELECT id, file_path
FROM LIGHTRAG_DOC_STATUS
WHERE workspace = $1 AND id = ANY($2::varchar[])
ORDER BY id
"""

_BY_FILE_PATHS = """
SELECT id, file_path
FROM LIGHTRAG_DOC_STATUS
WHERE workspace = $1 AND file_path = ANY($2::varchar[])
ORDER BY id
"""


class PGDocStatusLookup(PostgresOperationRunner):
    """Resolve exact source identities without materializing status buckets."""

    def __init__(
        self,
        *,
        workspace: str,
        pool: ConnectionPool | None = None,
    ) -> None:
        super().__init__(pool=pool)
        self._workspace = require_canonical_workspace_id(workspace)

    async def resolve_deletion_matches(
        self,
        *,
        file_paths: Sequence[str],
        doc_ids: Sequence[str],
    ) -> tuple[DocStatusMatch, ...]:
        paths = tuple(dict.fromkeys(str(path).strip() for path in file_paths if str(path).strip()))
        ids = tuple(dict.fromkeys(str(doc_id).strip() for doc_id in doc_ids if str(doc_id).strip()))

        async def _operation(conn: Any) -> tuple[DocStatusMatch, ...]:
            rows: list[Any] = []
            if ids:
                rows.extend(await conn.fetch(_BY_IDS, self._workspace, list(ids)))
            if paths:
                rows.extend(await conn.fetch(_BY_FILE_PATHS, self._workspace, list(paths)))
            by_id: dict[str, DocStatusMatch] = {}
            for row in rows:
                doc_id = str(row["id"])
                by_id[doc_id] = DocStatusMatch(
                    doc_id=doc_id,
                    file_path=str(row.get("file_path") or ""),
                )
            return tuple(by_id[doc_id] for doc_id in sorted(by_id))

        return await self._run(_operation)


__all__ = ["PGDocStatusLookup"]
