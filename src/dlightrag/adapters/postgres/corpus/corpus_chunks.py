# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL implementation of LightRAG corpus chunk mutations and reads."""

import json
from typing import Any

from dlightrag.adapters.postgres.core.identifiers import pg_identifier, pg_qualified_identifier
from dlightrag.adapters.postgres.corpus._corpus_schema import (
    CHUNK_DOCUMENT_SCOPE_INDEX,
    LIGHTRAG_CHUNKS_TABLE,
)
from dlightrag.adapters.postgres.corpus.corpus_languages import update_chunk_bm25_languages
from dlightrag.adapters.postgres.corpus.corpus_vectors import EXACT_FILTER_THRESHOLD
from dlightrag.adapters.postgres.corpus.pg_metadata_index import (
    METADATA_TABLE,
    metadata_match_conditions,
    metadata_visibility_condition,
)
from dlightrag.adapters.postgres.corpus.pg_metadata_scope import build_bounded_scope_probe
from dlightrag.engine.rag.retrieval import MetadataFilter, MetadataScope


class PGCorpusChunkStore:
    """Own raw PostgreSQL operations over LightRAG chunk stores."""

    def __init__(self, lightrag: Any, *, exact_threshold: int = EXACT_FILTER_THRESHOLD) -> None:
        self._text_chunks = lightrag.text_chunks
        self._exact_threshold = exact_threshold

    async def ensure_document_scope_index(self) -> None:
        """Provision the chunk-side semi-join index independent of BM25.

        Metadata scope preflights join selective document matches back to the
        chunk table even when lexical retrieval is disabled. Owning this index
        here prevents that path from degenerating into a hot-partition scan.
        """
        db, workspace = self._text_db_and_workspace()
        index_name = pg_identifier(CHUNK_DOCUMENT_SCOPE_INDEX)
        table_name = pg_qualified_identifier(LIGHTRAG_CHUNKS_TABLE)

        async def execute(connection: Any) -> None:
            await connection.execute(
                f"CREATE INDEX IF NOT EXISTS {index_name} "  # noqa: S608
                f"ON {table_name}(workspace, full_doc_id)"
            )

        await db._run_with_retry(
            execute,
            timing_label=f"{workspace} chunk_document_scope_index",
        )

    async def resolve_scope(self, filters: MetadataFilter) -> MetadataScope:
        """Resolve filter facts plus a bounded matching-chunk probe.

        One preflight statement per attempted filename mode: EXACT filename
        first, then a rebuild with the escaped-literal contains clause only
        when the exact mode matched no document. No complete document-id set
        and no exact corpus-scale COUNT ever crosses into Python.
        """
        db, workspace = self._text_db_and_workspace()
        filename_mode = "exact"
        row = await self._probe_scope(db, workspace, filters, filename_mode=filename_mode)
        if not row["doc_exists"] and filters.filename:
            filename_mode = "contains"
            row = await self._probe_scope(db, workspace, filters, filename_mode=filename_mode)
        chunk_count = int(row["chunk_count"] or 0)
        return MetadataScope(
            filters=filters,
            filename_mode=filename_mode,
            doc_exists=bool(row["doc_exists"]),
            candidate_count=chunk_count,
            candidate_count_exact=chunk_count <= self._exact_threshold,
        )

    async def read_scoped(
        self,
        scope: MetadataScope | None,
        chunk_ids: list[str],
    ) -> list[dict[str, Any] | None]:
        """Read graph-referenced chunks under one active metadata scope.

        One query fuses the chunk fetch with the metadata guard: rows must
        belong to the authenticated workspace, carry a requested id, and own a
        matching metadata document. Returns the same LightRAG text-chunk
        columns with the same JSON decoding, preserving the requested
        positional order (duplicates included) with ``None`` for missing or
        out-of-scope ids.
        """
        if not chunk_ids:
            return []
        if scope is not None and not scope:
            return [None] * len(chunk_ids)
        db, workspace = self._text_db_and_workspace()
        if scope is None:
            conditions = [metadata_visibility_condition("m")]
            params: list[Any] = []
        else:
            conditions, params = metadata_match_conditions(
                workspace,
                scope.filters,
                filename_mode=scope.filename_mode,
                start_index=3,
                alias="m",
            )
        sql = (
            f"SELECT c.id, c.tokens, COALESCE(c.content, '') AS content, "  # noqa: S608
            f"c.chunk_order_index, c.full_doc_id, c.file_path, "
            f"COALESCE(c.llm_cache_list, '[]'::jsonb) AS llm_cache_list, "
            f"COALESCE(c.heading, '{{}}'::jsonb) AS heading, "
            f"COALESCE(c.sidecar, '{{}}'::jsonb) AS sidecar, "
            f"EXTRACT(EPOCH FROM c.create_time)::BIGINT AS create_time, "
            f"EXTRACT(EPOCH FROM c.update_time)::BIGINT AS update_time "
            f"FROM {LIGHTRAG_CHUNKS_TABLE} c "
            "WHERE c.workspace = $1 AND c.id = ANY($2::text[]) "
            "AND EXISTS ("
            f"SELECT 1 FROM {METADATA_TABLE} m "
            "WHERE m.workspace = c.workspace AND m.doc_id = c.full_doc_id "
            f"AND {' AND '.join(conditions)}"
            ")"
        )

        async def execute(connection: Any) -> list[Any]:
            return await connection.fetch(sql, workspace, list(chunk_ids), *params)

        rows = await db._run_with_retry(
            execute,
            timing_label=f"{workspace} scoped_chunk_read",
        )
        row_map = {str(row["id"]): self._decode_text_chunk_row(row) for row in rows}
        return [row_map.get(str(chunk_id)) for chunk_id in chunk_ids]

    async def _probe_scope(
        self,
        db: Any,
        workspace: str,
        filters: MetadataFilter,
        *,
        filename_mode: str,
    ) -> Any:
        sql, params = build_bounded_scope_probe(
            workspace,
            filters,
            filename_mode=filename_mode,
            threshold=self._exact_threshold,
        )

        async def execute(connection: Any) -> Any:
            return await connection.fetchrow(sql, *params)

        row = await db._run_with_retry(
            execute,
            timing_label=f"{workspace} metadata_scope_probe",
        )
        if row is None:
            raise RuntimeError("metadata scope probe returned no row")
        return row

    @staticmethod
    def _decode_text_chunk_row(row: Any) -> dict[str, Any]:
        """Decode one text-chunk row exactly like LightRAG's get_by_ids."""
        result = dict(row)
        for field, fallback in (("llm_cache_list", []), ("heading", {}), ("sidecar", {})):
            value = result.get(field)
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    parsed = fallback
                # LightRAG forces the heading/sidecar shapes back to dicts; a
                # parsed non-dict falls back just like a decode error.
                if field != "llm_cache_list" and not isinstance(parsed, dict):
                    parsed = fallback
                result[field] = parsed
            elif value is None:
                result[field] = fallback
        create_time = result.get("create_time", 0)
        update_time = result.get("update_time", 0)
        result["create_time"] = create_time
        result["update_time"] = create_time if update_time == 0 else update_time
        return result

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


__all__ = ["PGCorpusChunkStore"]
