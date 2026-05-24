# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL store for chunk provenance and embedding input kind."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

TABLE_NAME = "dlightrag_chunk_provenance"
EMBEDDING_INPUT_KINDS = frozenset({"text", "image", "multimodal"})
SIDECAR_TYPES = frozenset({"block", "drawing", "table", "equation", "native_image"})


def _json_dumps(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False)


def _json_loads(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def validate_chunk_provenance_record(record: Mapping[str, Any]) -> None:
    """Validate a chunk provenance record before writing it."""
    kind = record.get("embedding_input_kind")
    if kind not in EMBEDDING_INPUT_KINDS:
        raise ValueError(f"embedding_input_kind must be one of {sorted(EMBEDDING_INPUT_KINDS)}")
    sidecar_type = record.get("sidecar_type")
    if sidecar_type is not None and sidecar_type not in SIDECAR_TYPES:
        raise ValueError(f"sidecar_type must be one of {sorted(SIDECAR_TYPES)}")
    if not record.get("chunk_id"):
        raise ValueError("chunk_id is required")
    if not record.get("full_doc_id"):
        raise ValueError("full_doc_id is required")


class PGChunkProvenance:
    """PG-backed chunk provenance store."""

    def __init__(self, workspace: str = "default") -> None:
        self._workspace = workspace
        self._pool = None

    async def initialize(self) -> None:
        from dlightrag.storage.pool import pg_pool

        self._pool = await pg_pool.get()
        await self._pool.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                workspace TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                full_doc_id TEXT NOT NULL,
                embedding_input_kind TEXT NOT NULL,
                sidecar_type TEXT,
                sidecar_id TEXT,
                sidecar_path TEXT,
                page_index INTEGER,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (workspace, chunk_id)
            )
            """
        )
        await self._pool.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_doc ON {TABLE_NAME}(workspace, full_doc_id)"
        )

    async def upsert_many(self, records: list[Mapping[str, Any]]) -> None:
        for record in records:
            validate_chunk_provenance_record(record)
        if not records:
            return
        pool = self._require_pool()

        values = [
            (
                self._workspace,
                str(record["chunk_id"]),
                str(record["full_doc_id"]),
                str(record["embedding_input_kind"]),
                record.get("sidecar_type"),
                record.get("sidecar_id"),
                record.get("sidecar_path"),
                record.get("page_index"),
                _json_dumps(record.get("metadata")),
            )
            for record in records
        ]
        await pool.executemany(
            f"""
            INSERT INTO {TABLE_NAME} (
                workspace, chunk_id, full_doc_id, embedding_input_kind,
                sidecar_type, sidecar_id, sidecar_path, page_index, metadata, updated_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, CURRENT_TIMESTAMP)
            ON CONFLICT (workspace, chunk_id) DO UPDATE
            SET full_doc_id=EXCLUDED.full_doc_id,
                embedding_input_kind=EXCLUDED.embedding_input_kind,
                sidecar_type=EXCLUDED.sidecar_type,
                sidecar_id=EXCLUDED.sidecar_id,
                sidecar_path=EXCLUDED.sidecar_path,
                page_index=EXCLUDED.page_index,
                metadata=EXCLUDED.metadata,
                updated_at=CURRENT_TIMESTAMP
            """,
            values,
        )

    async def get_batch(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        pool = self._require_pool()
        if not chunk_ids:
            return {}
        rows = await pool.fetch(
            f"SELECT * FROM {TABLE_NAME} WHERE workspace = $1 AND chunk_id = ANY($2::text[])",
            self._workspace,
            chunk_ids,
        )
        return {row["chunk_id"]: self._row_to_dict(row) for row in rows}

    async def chunk_ids_for_docs(self, doc_ids: list[str]) -> list[str]:
        pool = self._require_pool()
        if not doc_ids:
            return []
        rows = await pool.fetch(
            f"""
            SELECT chunk_id FROM {TABLE_NAME}
            WHERE workspace = $1 AND full_doc_id = ANY($2::text[])
            ORDER BY full_doc_id, page_index, chunk_id
            """,
            self._workspace,
            doc_ids,
        )
        return [row["chunk_id"] for row in rows]

    async def delete_doc(self, full_doc_id: str) -> None:
        if not full_doc_id:
            raise ValueError("full_doc_id is required")
        pool = self._require_pool()
        await pool.execute(
            f"DELETE FROM {TABLE_NAME} WHERE workspace = $1 AND full_doc_id = $2",
            self._workspace,
            full_doc_id,
        )

    async def clear(self) -> None:
        pool = self._require_pool()
        await pool.execute(
            f"DELETE FROM {TABLE_NAME} WHERE workspace = $1",
            self._workspace,
        )

    def _require_pool(self) -> Any:
        if self._pool is None:
            raise RuntimeError("PGChunkProvenance is not initialized")
        return self._pool

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        data = dict(row)
        data["metadata"] = _json_loads(data.get("metadata"))
        return data
