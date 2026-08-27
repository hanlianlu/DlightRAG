# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL implementation of LightRAG corpus chunk mutations and reads."""

import asyncio
import datetime
import json
from typing import Any, ClassVar

from dlightrag.adapters.postgres.core.identifiers import pg_qualified_identifier
from dlightrag.adapters.postgres.corpus._corpus_schema import (
    LIGHTRAG_CHUNKS_TABLE,
)
from dlightrag.adapters.postgres.corpus.corpus_languages import update_chunk_bm25_languages


class PGCorpusChunkStore:
    """Own raw PostgreSQL operations over LightRAG chunk stores."""

    _VECTOR_WRITE_MAX_BYTES: ClassVar[int] = 16 * 1024 * 1024
    _VECTOR_WRITE_MAX_RECORDS: ClassVar[int] = 200

    def __init__(self, lightrag: Any) -> None:
        self._chunks_vdb = lightrag.chunks_vdb
        self._text_chunks = lightrag.text_chunks
        self._vector_write_lock = asyncio.Lock()

    async def overwrite_chunk_vectors(
        self,
        vectors: dict[str, list[float]],
        *,
        embedding_dim: int,
    ) -> None:
        if not vectors:
            return
        chunks_vdb = self._chunks_vdb
        values = self._build_vector_update_values(vectors, embedding_dim=embedding_dim)
        if not hasattr(chunks_vdb, "table_name") or not hasattr(chunks_vdb, "db"):
            raise RuntimeError("Vector overwrite requires PGVectorStorage")
        chunks_table = pg_qualified_identifier(chunks_vdb.table_name)
        sql = (
            f"UPDATE {chunks_table} "  # noqa: S608 - validated identifier.
            "SET content_vector=$3, update_time=$4 "
            "WHERE workspace=$1 AND id=$2"
        )

        async def execute(connection: Any) -> None:
            for batch in self._chunk_vector_values(values):
                await connection.executemany(sql, batch)

        async with self._vector_write_lock:
            await chunks_vdb.db._run_with_retry(
                execute,
                timing_label=f"{chunks_vdb.workspace} chunk_vector_overwrite",
            )

    async def count_chunks_for_docs(self, doc_ids: list[str]) -> int:
        if not doc_ids:
            return 0
        db, workspace = self._text_db_and_workspace()
        sql = f"""
            SELECT count(*) AS chunk_count
            FROM {LIGHTRAG_CHUNKS_TABLE}
            WHERE workspace = $1
              AND full_doc_id = ANY($2::text[])
        """  # noqa: S608 - private constant.

        async def execute(connection: Any) -> Any:
            return await connection.fetchval(sql, workspace, list(doc_ids))

        count = await db._run_with_retry(
            execute,
            timing_label=f"{workspace} text_chunk_count_for_docs",
        )
        return int(count or 0)

    async def fetch_chunk_contents(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        if not chunk_ids:
            return []
        db, workspace = self._text_db_and_workspace()
        sql = f"""
            SELECT id, content
            FROM {LIGHTRAG_CHUNKS_TABLE}
            WHERE workspace = $1
              AND id = ANY($2::text[])
            ORDER BY chunk_order_index, id
        """  # noqa: S608 - private constant.

        async def execute(connection: Any) -> list[Any]:
            return await connection.fetch(sql, workspace, list(chunk_ids))

        rows = await db._run_with_retry(
            execute,
            timing_label=f"{workspace} chunk_content_for_bm25_language",
        )
        return [{"id": str(row["id"]), "content": str(row["content"] or "")} for row in rows]

    async def update_chunk_bm25_languages(self, labels: dict[str, str]) -> None:
        if not labels:
            return
        db, workspace = self._text_db_and_workspace()

        async def execute(connection: Any) -> None:
            await update_chunk_bm25_languages(
                connection,
                workspace=workspace,
                labels=labels,
            )

        await db._run_with_retry(
            execute,
            timing_label=f"{workspace} chunk_bm25_language_update",
        )

    def _text_db_and_workspace(self) -> tuple[Any, str]:
        db = getattr(self._text_chunks, "db", None)
        if db is None:
            raise RuntimeError("LightRAG text_chunks storage does not expose a PostgreSQL db")
        return db, str(getattr(self._text_chunks, "workspace", "default"))

    def _build_vector_update_values(
        self,
        vectors: dict[str, list[float]],
        *,
        embedding_dim: int,
    ) -> list[tuple[Any, ...]]:
        current_time = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
        workspace = getattr(self._chunks_vdb, "workspace", "default")
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
        payload_limit = cls._VECTOR_WRITE_MAX_BYTES or float("inf")
        records_limit = cls._VECTOR_WRITE_MAX_RECORDS or float("inf")
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
            elif isinstance(value, bytes | bytearray):
                total += len(value)
            elif value is None:
                continue
            elif isinstance(value, list) and all(isinstance(item, int | float) for item in value):
                total += len(value) * 8
            elif isinstance(value, dict | list):
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


__all__ = ["PGCorpusChunkStore"]
