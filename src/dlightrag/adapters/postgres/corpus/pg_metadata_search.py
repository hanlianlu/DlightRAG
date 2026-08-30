# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Bounded PostgreSQL reads for REST metadata search."""

from typing import Any

from dlightrag.adapters.postgres.core._operations import PostgresOperationRunner
from dlightrag.adapters.postgres.corpus.pg_metadata_index import metadata_match_conditions
from dlightrag.application.corpus_admin import (
    MetadataMatchRowPage,
    MetadataSearchFilenameMode,
    MetadataSearchPageRequest,
)
from dlightrag.engine.rag.retrieval import MetadataFilter


def _paged_sql(
    conditions: list[str],
    params: list[Any],
    *,
    after_doc_id: str | None,
) -> str:
    """Build one ordered doc_id-keyset page query without OFFSET.

    The cursor predicate is appended last, so the filter placeholders keep the
    indices ``metadata_match_conditions`` produced, and the LIMIT placeholder
    follows it. The caller binds parameters in the same order.
    """
    parts = list(conditions)
    if after_doc_id is not None:
        parts.append(f"doc_id > ${len(params) + 1}")
    limit_slot = len(params) + (2 if after_doc_id is not None else 1)
    where = " AND ".join(parts)
    return (
        f"SELECT doc_id FROM dlightrag_doc_metadata WHERE {where} "  # noqa: S608
        f"ORDER BY doc_id ASC LIMIT ${limit_slot}"
    )


class PGMetadataSearchStore(PostgresOperationRunner):
    """Serve bounded REST metadata-search pages without a LightRAG runtime."""

    async def search_metadata_page(
        self,
        workspace: str,
        filters: MetadataFilter,
        *,
        page: MetadataSearchPageRequest,
    ) -> MetadataMatchRowPage:
        """Return one physical limit+1 keyset page for a canonical workspace."""
        workspace_id = str(workspace).strip()
        if not workspace_id:
            raise ValueError("workspace cannot be empty")
        validated = MetadataSearchPageRequest(limit=page.limit, cursor=page.cursor)
        cursor = validated.cursor
        if cursor is not None and cursor.workspace != workspace_id:
            raise ValueError("metadata-search cursor belongs to another workspace")
        fetch_limit = validated.limit + 1

        async def _operation(conn: Any) -> MetadataMatchRowPage:
            mode: MetadataSearchFilenameMode = cursor.mode if cursor is not None else "exact"
            conditions, params = metadata_match_conditions(
                workspace_id,
                filters,
                filename_mode=mode,
            )
            sql = _paged_sql(
                conditions,
                params,
                after_doc_id=cursor.after_doc_id if cursor is not None else None,
            )
            args: list[Any] = [*params]
            if cursor is not None:
                args.append(cursor.after_doc_id)
            args.append(fetch_limit)
            rows = await conn.fetch(sql, *args)
            if cursor is None and not rows and filters.filename:
                # First page, exact name matched nothing verbatim: widen to the
                # literal-substring clause, exactly as the internal query path
                # does, and bind that decision into this page's cursor mode.
                mode = "contains"
                conditions, params = metadata_match_conditions(
                    workspace_id,
                    filters,
                    filename_mode=mode,
                )
                sql = _paged_sql(conditions, params, after_doc_id=None)
                rows = await conn.fetch(sql, *params, fetch_limit)
            fetched_rows = len(rows)
            document_ids = tuple(str(row["doc_id"]) for row in rows[: validated.limit])
            return MetadataMatchRowPage(
                document_ids=document_ids,
                has_more=fetched_rows > validated.limit,
                fetched_rows=fetched_rows,
                mode=mode,
            )

        return await self._run(_operation)


__all__ = ["PGMetadataSearchStore"]
