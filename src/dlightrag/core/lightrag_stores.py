# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Single boundary for LightRAG private storage access.

**LightRAG coupling surface (lightrag-hku>=1.5.3):**

This module depends on the following LightRAG internals that are NOT part
of the public API.  A LightRAG major-version bump may break these:

Storage attributes (copied from the LightRAG instance):
    ``chunks_vdb``       — PGVectorStorage instance
    ``text_chunks``      — PGKVStorage instance
    ``full_docs``        — PGKVStorage instance
    ``doc_status``       — PGDocStatusStorage instance

Storage internals (reached through storage attributes):
    ``chunks_vdb.table_name``  — LIGHTRAG_DOC_CHUNKS table name
    ``chunks_vdb.db``          — PostgreSQL ClientManager
    ``chunks_vdb.workspace``   — workspace name string
    ``text_chunks.db``         — PostgreSQL ClientManager
    ``text_chunks.workspace``  — workspace name string
    ``chunks_vdb.db._run_with_retry()`` — PostgreSQL retry wrapper

LightRAG API methods (public, but shape-dependent):
    ``LightRAG._build_global_config()`` — returns config dict
    ``LightRAG.aquery_data()``          — returns {data: {...}, status: ...}
    ``apipeline_enqueue_documents()``   — enqueues for processing
    ``apipeline_process_enqueue_documents()`` — processes queue

When upgrading lightrag-hku, verify these surfaces still exist and behave
as expected.  The contract_guard module provides runtime contract checks.
"""

import asyncio
import datetime
import json
import logging
from typing import Any, ClassVar

from dlightrag.core.retrieval.bm25_language import BM25_LANGUAGE_COLUMN
from dlightrag.storage.sql_identifiers import pg_qualified_identifier

logger = logging.getLogger(__name__)


class LightRAGStores:
    """Typed accessor for the LightRAG storage surface DlightRAG touches directly."""

    chunks_vdb: Any
    text_chunks: Any
    full_docs: Any
    doc_status: Any

    _VECTOR_WRITE_MAX_BYTES: ClassVar[int] = 16 * 1024 * 1024
    _VECTOR_WRITE_MAX_RECORDS: ClassVar[int] = 200

    _REQUIRED: ClassVar[frozenset[str]] = frozenset(
        {
            "chunks_vdb",
            "text_chunks",
            "full_docs",
            "doc_status",
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

    def build_global_config(self) -> dict[str, Any]:
        return self.raw._build_global_config()

    async def get_doc_status(self, doc_id: str) -> dict[str, Any] | None:
        return await self.doc_status.get_by_id(doc_id)

    async def docs_by_status(self, status: Any) -> dict[str, Any]:
        return await self.doc_status.get_docs_by_status(status)

    async def get_full_doc(self, doc_id: str) -> dict[str, Any] | None:
        return await self.full_docs.get_by_id(doc_id)

    async def get_text_chunks(self, chunk_ids: list[str]) -> list[Any]:
        if not chunk_ids:
            return []
        return await self.text_chunks.get_by_ids(chunk_ids)

    async def context_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        """Fetch LightRAG text chunks and format them as retrieval context rows."""
        if not chunk_ids:
            return []
        inject_ids = list(dict.fromkeys(chunk_ids))
        raw_contents = await self.get_text_chunks(inject_ids)

        chunks: list[dict[str, Any]] = []
        for cid, content_raw in zip(inject_ids, raw_contents, strict=False):
            if content_raw is None:
                continue
            if isinstance(content_raw, str):
                content = content_raw
                file_path = ""
                full_doc_id = None
            else:
                content = content_raw.get("content", "")
                file_path = content_raw.get("file_path", "") or ""
                full_doc_id = content_raw.get("full_doc_id")
            chunk = {
                "chunk_id": cid,
                "content": content,
                "reference_id": "",
                "file_path": file_path,
            }
            if full_doc_id:
                chunk["full_doc_id"] = str(full_doc_id)
            if not isinstance(content_raw, str):
                for key in ("sidecar", "sidecar_location", "page_idx"):
                    if content_raw.get(key) is not None:
                        chunk[key] = content_raw[key]
            chunks.append(chunk)
        return chunks

    async def overwrite_chunk_vectors(
        self,
        vectors: dict[str, list[float]],
        *,
        embedding_dim: int,
    ) -> None:
        """Replace vectors for existing LightRAG chunks without changing text rows."""
        if not vectors:
            return
        values = self._build_vector_update_values(vectors, embedding_dim=embedding_dim)

        chunks_vdb = self.chunks_vdb
        if not hasattr(chunks_vdb, "table_name") or not hasattr(chunks_vdb, "db"):
            raise RuntimeError("Vector overwrite requires PGVectorStorage")

        chunks_table = pg_qualified_identifier(chunks_vdb.table_name)
        sql = (
            f"UPDATE {chunks_table} "  # noqa: S608 - validated by pg_qualified_identifier.
            "SET content_vector=$3, update_time=$4 "
            "WHERE workspace=$1 AND id=$2"
        )

        async def _execute(connection: Any) -> None:
            for batch in self._chunk_vector_values(values):
                await connection.executemany(sql, batch)

        async with self._vector_write_lock:
            await chunks_vdb.db._run_with_retry(
                _execute,
                timing_label=f"{chunks_vdb.workspace} chunk_vector_overwrite",
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

        rows = await db._run_with_retry(
            _execute,
            timing_label=f"{workspace} text_chunk_ids_for_docs",
        )
        return [str(row["id"]) for row in rows]

    async def fetch_chunk_contents(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        """Fetch LightRAG chunk content for BM25 language labeling."""
        if not chunk_ids:
            return []
        text_chunks = self.text_chunks
        db = getattr(text_chunks, "db", None)
        if db is None:
            raise RuntimeError("LightRAG text_chunks storage does not expose a PostgreSQL db")
        workspace = getattr(text_chunks, "workspace", "default")
        sql = """
            SELECT id, content
            FROM LIGHTRAG_DOC_CHUNKS
            WHERE workspace = $1
              AND id = ANY($2::text[])
            ORDER BY chunk_order_index, id
        """

        async def _execute(connection: Any) -> list[Any]:
            return await connection.fetch(sql, workspace, list(chunk_ids))

        rows = await db._run_with_retry(
            _execute,
            timing_label=f"{workspace} chunk_content_for_bm25_language",
        )
        return [{"id": str(row["id"]), "content": str(row["content"] or "")} for row in rows]

    async def update_chunk_bm25_languages(self, labels: dict[str, str]) -> None:
        """Batch update DlightRAG BM25 language labels on LightRAG chunks."""
        if not labels:
            return
        text_chunks = self.text_chunks
        db = getattr(text_chunks, "db", None)
        if db is None:
            raise RuntimeError("LightRAG text_chunks storage does not expose a PostgreSQL db")
        workspace = getattr(text_chunks, "workspace", "default")
        chunk_ids = list(labels)
        languages = [labels[chunk_id] for chunk_id in chunk_ids]
        sql = (
            "UPDATE LIGHTRAG_DOC_CHUNKS AS chunks "  # noqa: S608 - internal table constant.
            f"SET {BM25_LANGUAGE_COLUMN}=labels.language, update_time=CURRENT_TIMESTAMP "
            "FROM UNNEST($2::text[], $3::text[]) AS labels(id, language) "
            "WHERE chunks.workspace=$1 AND chunks.id=labels.id"
        )

        async def _execute(connection: Any) -> None:
            await connection.execute(sql, workspace, chunk_ids, languages)

        await db._run_with_retry(
            _execute,
            timing_label=f"{workspace} chunk_bm25_language_update",
        )

    def _build_vector_update_values(
        self,
        vectors: dict[str, list[float]],
        *,
        embedding_dim: int,
    ) -> list[tuple[Any, ...]]:
        current_time = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
        workspace = getattr(self.chunks_vdb, "workspace", "default")
        values = []
        for chunk_id, vector in vectors.items():
            if len(vector) != embedding_dim:
                raise ValueError(f"{chunk_id} vector dimension {len(vector)} != {embedding_dim}")
            values.append((workspace, chunk_id, vector, current_time))
        return values

    @classmethod
    def _chunk_vector_values(cls, values: list[tuple[Any, ...]]) -> list[list[tuple[Any, ...]]]:
        if not values:
            return []

        payload_limit = (
            cls._VECTOR_WRITE_MAX_BYTES if cls._VECTOR_WRITE_MAX_BYTES > 0 else float("inf")
        )
        records_limit = (
            cls._VECTOR_WRITE_MAX_RECORDS if cls._VECTOR_WRITE_MAX_RECORDS > 0 else float("inf")
        )
        batches: list[list[tuple[Any, ...]]] = []
        current: list[tuple[Any, ...]] = []
        current_bytes = 2

        for value in values:
            value_bytes = cls._estimate_vector_record_bytes(value)
            separator = 1 if current else 0
            next_bytes = current_bytes + separator + value_bytes
            if current and (len(current) >= records_limit or next_bytes > payload_limit):
                batches.append(current)
                current = []
                current_bytes = 2
                next_bytes = current_bytes + value_bytes

            current.append(value)
            current_bytes = next_bytes

        if current:
            batches.append(current)
        return batches

    @staticmethod
    def _estimate_vector_record_bytes(record: tuple[Any, ...]) -> int:
        total = 0
        for value in record:
            if isinstance(value, str):
                total += len(value.encode("utf-8"))
            elif isinstance(value, (bytes, bytearray)):
                total += len(value)
            elif value is None:
                continue
            elif isinstance(value, list) and all(isinstance(x, int | float) for x in value):
                total += len(value) * 8
            elif isinstance(value, (dict, list)):
                total += len(
                    json.dumps(
                        value,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        default=str,
                    ).encode("utf-8")
                )
            else:
                total += 16
        return total
