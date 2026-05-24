# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL registry for LightRAG parser sidecar artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

TABLE_NAME = "dlightrag_document_artifacts"


def _json_dumps(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False)


def _json_loads(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


class PGDocumentArtifacts:
    """Registry of LightRAG sidecar locations and parser provenance."""

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
                full_doc_id TEXT NOT NULL,
                source_uri TEXT,
                parser TEXT,
                parse_engine TEXT,
                process_options TEXT,
                chunk_options JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                sidecar_location TEXT,
                content_hash TEXT,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                artifacts JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (workspace, full_doc_id)
            )
            """
        )

    async def upsert(self, record: Mapping[str, Any]) -> None:
        full_doc_id = str(record.get("full_doc_id") or "")
        if not full_doc_id:
            raise ValueError("full_doc_id is required")
        pool = self._require_pool()

        await pool.execute(
            f"""
            INSERT INTO {TABLE_NAME} (
                workspace, full_doc_id, source_uri, parser, parse_engine,
                process_options, chunk_options, sidecar_location, content_hash,
                metadata, artifacts, updated_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10::jsonb, $11::jsonb,
                    CURRENT_TIMESTAMP)
            ON CONFLICT (workspace, full_doc_id) DO UPDATE
            SET source_uri=EXCLUDED.source_uri,
                parser=EXCLUDED.parser,
                parse_engine=EXCLUDED.parse_engine,
                process_options=EXCLUDED.process_options,
                chunk_options=EXCLUDED.chunk_options,
                sidecar_location=EXCLUDED.sidecar_location,
                content_hash=EXCLUDED.content_hash,
                metadata=EXCLUDED.metadata,
                artifacts=EXCLUDED.artifacts,
                updated_at=CURRENT_TIMESTAMP
            """,
            self._workspace,
            full_doc_id,
            record.get("source_uri"),
            record.get("parser"),
            record.get("parse_engine"),
            record.get("process_options"),
            _json_dumps(record.get("chunk_options")),
            record.get("sidecar_location"),
            record.get("content_hash"),
            _json_dumps(record.get("metadata")),
            _json_dumps(record.get("artifacts")),
        )

    async def get(self, full_doc_id: str) -> dict[str, Any] | None:
        if not full_doc_id:
            raise ValueError("full_doc_id is required")
        pool = self._require_pool()
        row = await pool.fetchrow(
            f"SELECT * FROM {TABLE_NAME} WHERE workspace = $1 AND full_doc_id = $2",
            self._workspace,
            full_doc_id,
        )
        return self._row_to_dict(row)

    async def delete_doc(self, full_doc_id: str) -> dict[str, Any] | None:
        if not full_doc_id:
            raise ValueError("full_doc_id is required")
        pool = self._require_pool()
        row = await pool.fetchrow(
            f"DELETE FROM {TABLE_NAME} WHERE workspace = $1 AND full_doc_id = $2 RETURNING *",
            self._workspace,
            full_doc_id,
        )
        return self._row_to_dict(row)

    async def clear(self) -> None:
        pool = self._require_pool()
        await pool.execute(
            f"DELETE FROM {TABLE_NAME} WHERE workspace = $1",
            self._workspace,
        )

    def _require_pool(self) -> Any:
        if self._pool is None:
            raise RuntimeError("PGDocumentArtifacts is not initialized")
        return self._pool

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any] | None:
        if row is None:
            return None
        data = dict(row)
        for key in ("chunk_options", "metadata", "artifacts"):
            data[key] = _json_loads(data.get(key))
        return data
