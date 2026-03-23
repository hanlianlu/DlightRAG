# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Generic JSONB-based KV storage for PostgreSQL.

LightRAG's ``PGKVStorage`` maps namespaces to 7 hardcoded tables via
``NAMESPACE_TABLE_MAP``. Custom namespaces (e.g. ``visual_chunks``) hit
three failure modes: ``KeyError`` in ``get_by_id`` (missing SQL template),
silent no-op in ``upsert`` (no matching elif branch), and SQL syntax errors
in ``filter_keys``/``delete`` (``namespace_to_table_name`` returns None).

This module sidesteps that with a single generic table keyed by
``(workspace, namespace, id)`` and a JSONB data column. Large binary fields
(e.g. ``image_data``) can be separated into a dedicated BYTEA column via
the ``blob_field`` parameter to keep JSONB lean and browsable.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from lightrag.base import BaseKVStorage

logger = logging.getLogger(__name__)

TABLE = "dlightrag_kv_store"

_CREATE_TABLE = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    workspace VARCHAR(255) NOT NULL,
    namespace VARCHAR(255) NOT NULL,
    id        VARCHAR(255) NOT NULL,
    data      JSONB NOT NULL,
    blob_data BYTEA,
    create_time TIMESTAMPTZ DEFAULT NOW(),
    update_time TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (workspace, namespace, id)
)
"""

_ADD_BLOB_COLUMN = f"""
ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS blob_data BYTEA
"""

_UPSERT = f"""
INSERT INTO {TABLE} (workspace, namespace, id, data, blob_data)
VALUES ($1, $2, $3, $4::jsonb, $5)
ON CONFLICT (workspace, namespace, id)
DO UPDATE SET data = EXCLUDED.data, blob_data = EXCLUDED.blob_data, update_time = NOW()
"""

_GET_BY_ID = (
    f"SELECT data, blob_data FROM {TABLE} WHERE workspace = $1 AND namespace = $2 AND id = $3"
)

_GET_BY_IDS = f"SELECT id, data, blob_data FROM {TABLE} WHERE workspace = $1 AND namespace = $2 AND id = ANY($3)"

_DELETE = f"DELETE FROM {TABLE} WHERE workspace = $1 AND namespace = $2 AND id = ANY($3)"

_FILTER_EXISTING = (
    f"SELECT id FROM {TABLE} WHERE workspace = $1 AND namespace = $2 AND id = ANY($3)"
)

_IS_EMPTY = f"SELECT 1 FROM {TABLE} WHERE workspace = $1 AND namespace = $2 LIMIT 1"

_DROP = f"DELETE FROM {TABLE} WHERE workspace = $1 AND namespace = $2"

_MIGRATE_BLOB = f"""
UPDATE {TABLE}
SET blob_data = decode(data->>'{{field}}', 'base64'),
    data = (data - '{{field}}') || '{{"_blob_field": "{{field}}"}}'::jsonb
WHERE workspace = $1 AND namespace = $2
  AND blob_data IS NULL
  AND data ? '{{field}}'
"""


@dataclass
class PGJsonbKVStorage(BaseKVStorage):
    """Generic JSONB KV storage for PostgreSQL.

    Uses a single ``dlightrag_kv_store`` table with (workspace, namespace, id)
    composite primary key. Uses DlightRAG's own dedicated asyncpg pool via
    ``pg_pool`` (independent of LightRAG's ClientManager pool).

    Parameters
    ----------
    blob_field : str or None
        If set, this key is extracted from the JSONB ``data`` and stored as
        raw bytes in the ``blob_data`` BYTEA column. On read, the field is
        merged back into the returned dict (base64-encoded, matching the
        original format). This keeps JSONB small and browsable in tools
        like DBeaver.
    """

    _pool: Any = field(default=None, repr=False)
    blob_field: str | None = field(default=None)

    def _get_pool(self) -> Any:
        if self._pool is None:
            raise RuntimeError("PGJsonbKVStorage not initialized — call initialize() first")
        return self._pool

    async def initialize(self) -> None:
        if self._pool is None:
            from dlightrag.storage.pool import pg_pool

            self._pool = await pg_pool.get()
        await self._ensure_table()

    async def _ensure_table(self) -> None:
        async with self._get_pool().acquire() as conn:
            await conn.execute(_CREATE_TABLE)
            # Migrate existing tables that lack the blob_data column
            await conn.execute(_ADD_BLOB_COLUMN)
            # Migrate existing rows: move blob_field from JSONB to BYTEA
            if self.blob_field:
                sql = _MIGRATE_BLOB.replace("{field}", self.blob_field)
                result = await conn.execute(sql, self.workspace, self.namespace)
                if result and "UPDATE" in result and result != "UPDATE 0":
                    logger.info(
                        "Migrated %s blob_field='%s': %s", self.namespace, self.blob_field, result
                    )

    async def finalize(self) -> None:
        # Pool is owned by pg_pool singleton — don't close it here.
        # RAGService.close() calls pg_pool.close() after all stores finalize.
        self._pool = None

    async def index_done_callback(self) -> None:
        # PostgreSQL commits immediately — no deferred persistence needed.
        pass

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return
        rows = []
        for k, v in data.items():
            blob_bytes = None
            if self.blob_field and self.blob_field in v:
                v = dict(v)  # Don't mutate caller's dict
                raw = v.pop(self.blob_field)
                v["_blob_field"] = self.blob_field  # Mark which field is in blob_data
                if isinstance(raw, str):
                    blob_bytes = base64.b64decode(raw)
                elif isinstance(raw, bytes):
                    blob_bytes = raw
            rows.append((self.workspace, self.namespace, k, json.dumps(v), blob_bytes))
        async with self._get_pool().acquire() as conn:
            await conn.executemany(_UPSERT, rows)

    @staticmethod
    def _parse_data(raw: Any) -> dict[str, Any]:
        """Parse JSONB data — asyncpg usually returns a dict, but some pool
        configurations may return a raw JSON string instead."""
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            return json.loads(raw)
        return raw

    def _merge_blob(self, data: dict[str, Any], blob_data: bytes | None) -> dict[str, Any]:
        """Merge blob_data back into the dict as base64 string."""
        data.pop("_blob_field", None)  # Remove internal marker
        if self.blob_field and blob_data is not None:
            data[self.blob_field] = base64.b64encode(blob_data).decode("ascii")
        return data

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        async with self._get_pool().acquire() as conn:
            row = await conn.fetchrow(_GET_BY_ID, self.workspace, self.namespace, id)
            if row is None:
                return None
            data = self._parse_data(row["data"])
            return self._merge_blob(data, row["blob_data"])

    async def get_by_ids(  # type: ignore[override]
        self, ids: list[str]
    ) -> list[dict[str, Any] | None]:
        if not ids:
            return []
        async with self._get_pool().acquire() as conn:
            rows = await conn.fetch(_GET_BY_IDS, self.workspace, self.namespace, ids)
        lookup: dict[str, dict[str, Any]] = {}
        for row in rows:
            data = self._parse_data(row["data"])
            lookup[row["id"]] = self._merge_blob(data, row["blob_data"])
        return [lookup.get(id_) for id_ in ids]

    async def filter_keys(self, keys: set[str]) -> set[str]:
        if not keys:
            return set()
        async with self._get_pool().acquire() as conn:
            rows = await conn.fetch(_FILTER_EXISTING, self.workspace, self.namespace, list(keys))
        existing = {row["id"] for row in rows}
        return keys - existing

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        async with self._get_pool().acquire() as conn:
            await conn.execute(_DELETE, self.workspace, self.namespace, ids)

    async def is_empty(self) -> bool:
        async with self._get_pool().acquire() as conn:
            row = await conn.fetchrow(_IS_EMPTY, self.workspace, self.namespace)
            return row is None

    async def drop(self) -> dict[str, str]:
        async with self._get_pool().acquire() as conn:
            result = await conn.execute(_DROP, self.workspace, self.namespace)
        logger.info("Dropped %s/%s: %s", self.workspace, self.namespace, result)
        return {"status": "success", "message": "data dropped"}
