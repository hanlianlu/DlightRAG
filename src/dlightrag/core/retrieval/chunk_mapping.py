# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Bidirectional entity-chunk mapping for citation provenance.

Schema only — integration into ingestion/deletion pipelines is deferred.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

TABLE = "dlightrag_chunk_mapping"

_CREATE_TABLE = f"""\
CREATE TABLE IF NOT EXISTS {TABLE} (
    workspace   VARCHAR(255) NOT NULL,
    entity_id   VARCHAR(255) NOT NULL,
    chunk_id    VARCHAR(255) NOT NULL,
    source_type VARCHAR(32)  NOT NULL DEFAULT 'entity',
    PRIMARY KEY (workspace, entity_id, chunk_id)
)"""

_CREATE_INDEXES = [
    f"CREATE INDEX IF NOT EXISTS idx_cm_chunk ON {TABLE} (chunk_id)",
    f"CREATE INDEX IF NOT EXISTS idx_cm_workspace ON {TABLE} (workspace)",
]

_INSERT = f"""\
INSERT INTO {TABLE} (workspace, entity_id, chunk_id, source_type)
VALUES ($1, $2, $3, $4)
ON CONFLICT (workspace, entity_id, chunk_id) DO NOTHING"""

_BY_ENTITY = f"SELECT chunk_id FROM {TABLE} WHERE workspace = $1 AND entity_id = $2"
_BY_CHUNK = f"SELECT entity_id FROM {TABLE} WHERE chunk_id = $1"
_DEL_ENTITY = f"DELETE FROM {TABLE} WHERE workspace = $1 AND entity_id = $2"
_DEL_CHUNKS = f"DELETE FROM {TABLE} WHERE workspace = $1 AND chunk_id = ANY($2)"
_CLEAR = f"DELETE FROM {TABLE} WHERE workspace = $1"


class PGChunkMapping:
    """Bidirectional entity-chunk mapping stored in PostgreSQL."""

    def __init__(self, workspace: str) -> None:
        self._workspace = workspace
        self._pool: Any = None

    def _get_pool(self) -> Any:
        if self._pool is None:
            raise RuntimeError("PGChunkMapping not initialized — call initialize() first")
        return self._pool

    async def initialize(self, pool: Any = None) -> None:
        if pool is not None:
            self._pool = pool
        else:
            from dlightrag.storage.pool import pg_pool

            self._pool = await pg_pool.get()
        async with self._pool.acquire() as conn:
            await conn.execute(_CREATE_TABLE)
            for idx_sql in _CREATE_INDEXES:
                await conn.execute(idx_sql)

    async def add_mappings(
        self,
        entity_id: str,
        chunk_ids: list[str],
        source_type: str = "entity",
    ) -> None:
        if not chunk_ids:
            return
        rows = [(self._workspace, entity_id, cid, source_type) for cid in chunk_ids]
        async with self._get_pool().acquire() as conn:
            async with conn.transaction():
                await conn.executemany(_INSERT, rows)

    async def get_chunks_for_entity(self, entity_id: str) -> list[str]:
        async with self._get_pool().acquire() as conn:
            rows = await conn.fetch(_BY_ENTITY, self._workspace, entity_id)
        return [r["chunk_id"] for r in rows]

    async def get_entities_for_chunk(self, chunk_id: str) -> list[str]:
        async with self._get_pool().acquire() as conn:
            rows = await conn.fetch(_BY_CHUNK, chunk_id)
        return [r["entity_id"] for r in rows]

    async def delete_by_entity(self, entity_id: str) -> int:
        async with self._get_pool().acquire() as conn:
            result = await conn.execute(_DEL_ENTITY, self._workspace, entity_id)
        return int(result.split()[-1]) if result else 0

    async def delete_by_chunks(self, chunk_ids: list[str]) -> int:
        if not chunk_ids:
            return 0
        async with self._get_pool().acquire() as conn:
            result = await conn.execute(_DEL_CHUNKS, self._workspace, chunk_ids)
        return int(result.split()[-1]) if result else 0

    async def clear(self) -> None:
        async with self._get_pool().acquire() as conn:
            await conn.execute(_CLEAR, self._workspace)
