# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Document-level metadata index for structured queries."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from dlightrag.core.retrieval.models import MetadataFilter

logger = logging.getLogger(__name__)

_SYSTEM_FIELDS = frozenset({
    "filename",
    "filename_stem",
    "file_path",
    "file_extension",
    "doc_title",
    "doc_author",
    "creation_date",
    "original_format",
    "page_count",
    "rag_mode",
})

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS dlightrag_doc_metadata (
    workspace       VARCHAR(255) NOT NULL,
    doc_id          VARCHAR(255) NOT NULL,
    filename        VARCHAR(512),
    filename_stem   VARCHAR(512),
    file_path       TEXT,
    file_extension  VARCHAR(32),
    doc_title       TEXT,
    doc_author      VARCHAR(255),
    creation_date   TIMESTAMPTZ,
    original_format VARCHAR(32),
    page_count      INTEGER,
    rag_mode        VARCHAR(32),
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),
    custom_metadata JSONB DEFAULT '{}',
    PRIMARY KEY (workspace, doc_id)
)"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_dm_filename ON dlightrag_doc_metadata USING gin (filename gin_trgm_ops)",
    "CREATE INDEX IF NOT EXISTS idx_dm_stem ON dlightrag_doc_metadata USING gin (filename_stem gin_trgm_ops)",
    "CREATE INDEX IF NOT EXISTS idx_dm_title ON dlightrag_doc_metadata USING gin (doc_title gin_trgm_ops)",
    "CREATE INDEX IF NOT EXISTS idx_dm_author ON dlightrag_doc_metadata (doc_author)",
    "CREATE INDEX IF NOT EXISTS idx_dm_date ON dlightrag_doc_metadata (creation_date)",
    "CREATE INDEX IF NOT EXISTS idx_dm_custom ON dlightrag_doc_metadata USING gin (custom_metadata)",
]

_UPSERT = """\
INSERT INTO dlightrag_doc_metadata
    (workspace, doc_id, filename, filename_stem, file_path, file_extension,
     doc_title, doc_author, creation_date, original_format,
     page_count, rag_mode, custom_metadata)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
ON CONFLICT (workspace, doc_id) DO UPDATE SET
    filename=EXCLUDED.filename, filename_stem=EXCLUDED.filename_stem,
    file_path=EXCLUDED.file_path, file_extension=EXCLUDED.file_extension,
    doc_title=EXCLUDED.doc_title, doc_author=EXCLUDED.doc_author,
    creation_date=EXCLUDED.creation_date, original_format=EXCLUDED.original_format,
    page_count=EXCLUDED.page_count, rag_mode=EXCLUDED.rag_mode,
    custom_metadata=EXCLUDED.custom_metadata"""


class PGMetadataIndex:
    """PostgreSQL-backed document metadata index.

    Stores system-extracted and user-defined metadata per document.
    Supports exact match, fuzzy match (pg_trgm), range, and JSONB queries.
    """

    def __init__(self, workspace: str = "default") -> None:
        self._workspace = workspace
        self._pool: Any = None

    async def initialize(self) -> None:
        """Create table and indexes. Call once during service startup."""
        from lightrag.kg.postgres_impl import ClientManager

        db = await ClientManager.get_client()
        self._pool = db.pool
        async with self._pool.acquire() as conn:
            await conn.execute(_CREATE_TABLE)
            for idx_sql in _CREATE_INDEXES:
                try:
                    await conn.execute(idx_sql)
                except Exception:
                    logger.debug("Index creation skipped (pg_trgm may not be available): %s", idx_sql[:60])

    async def upsert(self, doc_id: str, metadata: dict[str, Any]) -> None:
        """Insert or update document metadata."""
        system = {k: metadata.get(k) for k in _SYSTEM_FIELDS}
        custom = {k: v for k, v in metadata.items() if k not in _SYSTEM_FIELDS and k != "ingested_at"}

        async with self._pool.acquire() as conn:
            await conn.execute(
                _UPSERT,
                self._workspace,
                doc_id,
                system.get("filename"),
                system.get("filename_stem"),
                system.get("file_path"),
                system.get("file_extension"),
                system.get("doc_title"),
                system.get("doc_author"),
                system.get("creation_date"),
                system.get("original_format"),
                system.get("page_count"),
                system.get("rag_mode"),
                json.dumps(custom),
            )

    async def query(self, filters: MetadataFilter) -> list[str]:
        """Query for doc_ids matching the given filters."""
        conditions = ["workspace = $1"]
        params: list[Any] = [self._workspace]
        idx = 2

        if filters.filename:
            conditions.append(f"filename = ${idx}")
            params.append(filters.filename)
            idx += 1
        if filters.filename_pattern:
            conditions.append(f"filename ILIKE ${idx}")
            params.append(filters.filename_pattern)
            idx += 1
        if filters.file_extension:
            conditions.append(f"file_extension = ${idx}")
            params.append(filters.file_extension)
            idx += 1
        if filters.doc_title:
            conditions.append(f"doc_title ILIKE ${idx}")
            params.append(f"%{filters.doc_title}%")
            idx += 1
        if filters.doc_author:
            conditions.append(f"doc_author = ${idx}")
            params.append(filters.doc_author)
            idx += 1
        if filters.date_from:
            conditions.append(f"creation_date >= ${idx}")
            params.append(filters.date_from)
            idx += 1
        if filters.date_to:
            conditions.append(f"creation_date <= ${idx}")
            params.append(filters.date_to)
            idx += 1
        if filters.rag_mode:
            conditions.append(f"rag_mode = ${idx}")
            params.append(filters.rag_mode)
            idx += 1
        if filters.custom:
            conditions.append(f"custom_metadata @> ${idx}::jsonb")
            params.append(json.dumps(filters.custom))
            idx += 1

        where = " AND ".join(conditions)
        sql = f"SELECT doc_id FROM dlightrag_doc_metadata WHERE {where}"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        return [r["doc_id"] for r in rows]

    async def get(self, doc_id: str) -> dict[str, Any] | None:
        """Get metadata for a single document."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM dlightrag_doc_metadata WHERE workspace=$1 AND doc_id=$2",
                self._workspace,
                doc_id,
            )
        if not row:
            return None
        return dict(row)

    async def delete(self, doc_id: str) -> None:
        """Delete metadata for a document."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM dlightrag_doc_metadata WHERE workspace=$1 AND doc_id=$2",
                self._workspace,
                doc_id,
            )

    async def clear(self) -> None:
        """Delete all metadata for this workspace."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM dlightrag_doc_metadata WHERE workspace=$1",
                self._workspace,
            )
            logger.info("PGMetadataIndex cleared for workspace %s: %s", self._workspace, result)

    async def find_by_filename(self, name: str) -> list[str]:
        """Find doc_ids by exact filename match."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT doc_id FROM dlightrag_doc_metadata WHERE workspace=$1 AND filename=$2",
                self._workspace,
                name,
            )
        return [r["doc_id"] for r in rows]


def extract_system_metadata(
    file_path: str | Path,
    rag_mode: str = "caption",
    page_count: int | None = None,
) -> dict[str, Any]:
    """Extract system metadata from a file path.

    Returns a dict suitable for PGMetadataIndex.upsert().
    """
    p = Path(file_path)
    meta: dict[str, Any] = {
        "filename": p.name,
        "filename_stem": p.stem,
        "file_path": str(p),
        "file_extension": p.suffix.lower().lstrip(".") if p.suffix else None,
        "original_format": p.suffix.lower().lstrip(".") if p.suffix else None,
        "rag_mode": rag_mode,
    }
    if page_count is not None:
        meta["page_count"] = page_count
    return meta
