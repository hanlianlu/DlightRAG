# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Single boundary for LightRAG private storage access.

**LightRAG coupling surface (lightrag-hku>=1.5.0):**

This module depends on the following LightRAG internals that are NOT part
of the public API.  A LightRAG major-version bump may break these:

Storage attributes (accessed via __getattr__):
    ``chunks_vdb``       — PGVectorStorage instance
    ``text_chunks``      — PGKVStorage instance
    ``full_docs``        — PGKVStorage instance
    ``doc_status``       — PGDocStatusStorage instance
    ``entities_vdb``     — PGVectorStorage instance
    ``relationships_vdb`` — PGVectorStorage instance
    ``chunk_entity_relation_graph`` — PGGraphStorage instance
    ``full_entities``    — PGKVStorage instance
    ``full_relations``   — PGKVStorage instance
    ``entity_chunks``    — PGKVStorage instance
    ``relation_chunks``  — PGKVStorage instance
    ``llm_response_cache`` — PGKVStorage instance

Storage internals (reached through storage attributes):
    ``chunks_vdb.table_name``  — LIGHTRAG_DOC_CHUNKS table name
    ``chunks_vdb.db``          — PostgreSQL ClientManager
    ``chunks_vdb.workspace``   — workspace name string
    ``text_chunks.db``         — PostgreSQL ClientManager
    ``text_chunks.workspace``  — workspace name string
    ``chunks_vdb.db._run_with_retry()`` — optional retry wrapper
    ``chunks_vdb.db.pool``     — asyncpg connection pool

LightRAG API methods (public, but shape-dependent):
    ``LightRAG._build_global_config()`` — returns config dict
    ``LightRAG.aquery_data()``          — returns {data: {...}, status: ...}
    ``apipeline_enqueue_documents()``   — enqueues for processing
    ``apipeline_process_enqueue_documents()`` — processes queue

Constants:
    ``lightrag.base.DocStatus``
    ``lightrag.constants.FULL_DOCS_FORMAT_*``
    ``lightrag.constants.PARSER_ENGINE_*``

When upgrading lightrag-hku, verify these surfaces still exist and behave
as expected.  The compat_guard module provides runtime version checks.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
from typing import Any, ClassVar

from lightrag.base import DocStatus

logger = logging.getLogger(__name__)


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
        chunks_by_doc = self._chunks_by_doc(rows)

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
            current_status = await self._load_parent_doc_status(chunks_by_doc)

            if hasattr(chunks_vdb.db, "_run_with_retry"):
                await chunks_vdb.db._run_with_retry(
                    _execute,
                    timing_label=f"{chunks_vdb.workspace} direct_image_chunk_upsert",
                )
            else:
                async with chunks_vdb.db.pool.acquire() as connection:
                    await _execute(connection)

            await self._append_doc_status_chunks(chunks_by_doc, current_status)

    async def upsert_document_record(
        self,
        *,
        doc_id: str,
        content: str,
        file_path: str,
        chunks: list[str],
        parse_engine: str,
        parse_format: str,
        content_hash: str | None,
        metadata: dict[str, Any] | None = None,
        sidecar_location: str | None = None,
        process_options: str | None = None,
        chunk_options: dict[str, Any] | None = None,
    ) -> None:
        """Create/update the LightRAG full_doc and doc_status parent record."""
        now = datetime.datetime.now(datetime.UTC).isoformat()
        await self.full_docs.upsert(
            {
                doc_id: {
                    "content": content,
                    "file_path": file_path,
                    "sidecar_location": sidecar_location,
                    "parse_format": parse_format,
                    "content_hash": content_hash,
                    "process_options": process_options,
                    "chunk_options": chunk_options or {},
                    "parse_engine": parse_engine,
                }
            }
        )
        await self.doc_status.upsert(
            {
                doc_id: {
                    "content_summary": content[:240],
                    "content_length": len(content),
                    "chunks_count": len(chunks),
                    "status": DocStatus.PROCESSED.value,
                    "file_path": file_path,
                    "chunks_list": list(chunks),
                    "metadata": metadata or {},
                    "error_msg": None,
                    "content_hash": content_hash,
                    "created_at": now,
                    "updated_at": now,
                }
            }
        )

    async def chunk_ids_for_docs(self, doc_ids: list[str]) -> list[str]:
        """Resolve full_doc ids to LightRAG text chunk ids."""
        if not doc_ids:
            return []
        text_chunks = self.text_chunks
        db = getattr(text_chunks, "db", None)
        if db is None:
            raise RuntimeError("LightRAG text_chunks storage does not expose a PostgreSQL db")
        workspace = getattr(text_chunks, "workspace", "default")
        sql = """
            SELECT id
            FROM LIGHTRAG_DOC_CHUNKS
            WHERE workspace = $1
              AND full_doc_id = ANY($2::text[])
            ORDER BY full_doc_id, chunk_order_index, id
        """

        async def _execute(connection: Any) -> list[Any]:
            return await connection.fetch(sql, workspace, list(doc_ids))

        if hasattr(db, "_run_with_retry"):
            rows = await db._run_with_retry(
                _execute,
                timing_label=f"{workspace} text_chunk_ids_for_docs",
            )
        else:
            async with db.pool.acquire() as connection:
                rows = await _execute(connection)
        return [str(row["id"]) for row in rows]

    async def cleanup_doc(self, doc_id: str) -> int:
        """Delete a document's LightRAG records (doc_status, doc_full, chunks).

        Returns the number of rows targeted for deletion (an upper bound,
        not an exact count of rows that existed).
        Does NOT attempt to clean entities, relations, vectors, or graphs --
        those are keyed by entity/relation name or chunk_id and do not block
        re-ingestion.  Orphaned rows in those tables are harmless.
        """
        deleted = 0

        # 1. Find chunk ids belonging to this doc so we can clean the text_chunks table.
        chunk_ids = await self.chunk_ids_for_docs([doc_id])

        # 2. doc_status -- the primary duplicate-detection gate.
        if hasattr(self.doc_status, "delete"):
            await self.doc_status.delete([doc_id])
            deleted += 1  # count the doc_status row

        # 3. doc_full -- parse config / sidecar location.
        if hasattr(self.full_docs, "delete"):
            await self.full_docs.delete([doc_id])
            deleted += 1

        # 4. doc_chunks (text_chunks) -- old chunks would conflict with new ones.
        if chunk_ids and hasattr(self.text_chunks, "delete"):
            await self.text_chunks.delete(chunk_ids)
            deleted += len(chunk_ids)

        return deleted

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

    @staticmethod
    def _chunks_by_doc(rows: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for chunk_id, row in rows.items():
            grouped.setdefault(str(row["full_doc_id"]), []).append(chunk_id)
        return grouped

    async def _load_parent_doc_status(
        self, chunks_by_doc: dict[str, list[str]]
    ) -> dict[str, dict[str, Any]]:
        current: dict[str, dict[str, Any]] = {}
        for doc_id in chunks_by_doc:
            record = await self.doc_status.get_by_id(doc_id)
            if record is None:
                raise RuntimeError(
                    f"Direct image chunk write requires LightRAG doc_status for {doc_id}"
                )
            current[doc_id] = record
        return current

    async def _append_doc_status_chunks(
        self,
        chunks_by_doc: dict[str, list[str]],
        current_status: dict[str, dict[str, Any]],
    ) -> None:
        if not chunks_by_doc:
            return
        now = datetime.datetime.now(datetime.UTC).isoformat()
        updates: dict[str, dict[str, Any]] = {}
        for doc_id, chunk_ids in chunks_by_doc.items():
            current = current_status[doc_id]
            merged = list(dict.fromkeys([*(current.get("chunks_list") or []), *chunk_ids]))
            if merged == (current.get("chunks_list") or []):
                continue
            status = current.get("status") or DocStatus.PROCESSED.value
            if isinstance(status, DocStatus):
                status = status.value
            metadata = current.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            updates[doc_id] = {
                "content_summary": current.get("content_summary") or "",
                "content_length": int(current.get("content_length") or 0),
                "chunks_count": len(merged),
                "status": status,
                "file_path": current.get("file_path") or "",
                "chunks_list": merged,
                "track_id": current.get("track_id"),
                "metadata": metadata,
                "error_msg": current.get("error_msg"),
                "content_hash": current.get("content_hash"),
                "created_at": current.get("created_at") or now,
                "updated_at": now,
            }
        if updates:
            await self.doc_status.upsert(updates)
            logger.debug(
                "Appended direct image chunks to %d LightRAG doc_status rows", len(updates)
            )
