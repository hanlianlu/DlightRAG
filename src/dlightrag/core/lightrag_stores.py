# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Single boundary for LightRAG private storage access."""

from __future__ import annotations

import asyncio
import datetime
from typing import Any, ClassVar


class LightRAGStores:
    """Typed accessor for LightRAG storage attributes DlightRAG writes directly."""

    _REQUIRED: ClassVar[frozenset[str]] = frozenset(
        {
            "chunks_vdb",
            "text_chunks",
            "full_docs",
            "doc_status",
            "entities_vdb",
            "relationships_vdb",
            "chunk_entity_relation_graph",
            "full_entities",
            "full_relations",
            "entity_chunks",
            "relation_chunks",
            "llm_response_cache",
            "_build_global_config",
        }
    )

    def __init__(self, lightrag: Any) -> None:
        missing = sorted(name for name in self._REQUIRED if not hasattr(lightrag, name))
        if missing:
            raise RuntimeError(f"LightRAGStores missing required surface(s): {missing}")
        self.raw = lightrag
        for name in self._REQUIRED - {"_build_global_config"}:
            setattr(self, name, getattr(lightrag, name))
        self._vector_write_lock = asyncio.Lock()

    def __getattr__(self, name: str) -> Any:
        if name in self._REQUIRED:
            return getattr(self.raw, name)
        raise AttributeError(name)

    def build_global_config(self) -> dict[str, Any]:
        return self.raw._build_global_config()

    async def get_doc_status(self, doc_id: str) -> dict[str, Any] | None:
        return await self.doc_status.get_by_id(doc_id)

    async def get_full_doc(self, doc_id: str) -> dict[str, Any] | None:
        return await self.full_docs.get_by_id(doc_id)

    async def upsert_chunks_with_vectors(
        self,
        rows: dict[str, dict[str, Any]],
        vectors: dict[str, list[float]],
        *,
        embedding_dim: int,
        max_token_size: int,  # noqa: ARG002
    ) -> None:
        """Write direct-image chunks while preserving precomputed image vectors."""
        if not rows:
            return

        values = self._build_chunk_values(rows, vectors, embedding_dim=embedding_dim)

        await self.text_chunks.upsert(rows)

        chunks_vdb = self.chunks_vdb
        if not hasattr(chunks_vdb, "table_name") or not hasattr(chunks_vdb, "db"):
            raise RuntimeError("Direct image vector writes require PGVectorStorage")

        sql = f"""
            INSERT INTO {chunks_vdb.table_name} (
                workspace, id, tokens, chunk_order_index, full_doc_id,
                content, content_vector, file_path, create_time, update_time
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (workspace,id) DO UPDATE
            SET tokens=EXCLUDED.tokens,
                chunk_order_index=EXCLUDED.chunk_order_index,
                full_doc_id=EXCLUDED.full_doc_id,
                content=EXCLUDED.content,
                content_vector=EXCLUDED.content_vector,
                file_path=EXCLUDED.file_path,
                update_time=EXCLUDED.update_time
        """

        async def _execute(connection: Any) -> None:
            await connection.executemany(sql, values)

        async with self._vector_write_lock:
            if hasattr(chunks_vdb.db, "_run_with_retry"):
                await chunks_vdb.db._run_with_retry(
                    _execute,
                    timing_label=f"{chunks_vdb.workspace} direct_image_chunk_upsert",
                )
                return
            async with chunks_vdb.db.pool.acquire() as connection:
                await _execute(connection)

    def _build_chunk_values(
        self,
        rows: dict[str, dict[str, Any]],
        vectors: dict[str, list[float]],
        *,
        embedding_dim: int,
    ) -> list[tuple[Any, ...]]:
        current_time = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
        workspace = getattr(self.chunks_vdb, "workspace", "default")
        values = []
        for chunk_id, row in rows.items():
            vector = vectors[chunk_id]
            if len(vector) != embedding_dim:
                raise ValueError(f"{chunk_id} vector dimension {len(vector)} != {embedding_dim}")
            values.append(
                (
                    workspace,
                    chunk_id,
                    int(row.get("tokens") or 0),
                    int(row.get("chunk_order_index") or 0),
                    str(row["full_doc_id"]),
                    str(row["content"]),
                    vector,
                    row.get("file_path"),
                    current_time,
                    current_time,
                )
            )
        return values
