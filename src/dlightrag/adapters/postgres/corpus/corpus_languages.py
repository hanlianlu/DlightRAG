# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL chunk-language mutations shared by ingestion and BM25 rebuilds."""

from collections.abc import Mapping
from typing import Any

from dlightrag.adapters.postgres.corpus._corpus_schema import (
    BM25_LANGUAGE_COLUMN,
    LIGHTRAG_CHUNKS_TABLE,
)

_UPDATE_CHUNK_LANGUAGES_SQL = (
    f"UPDATE {LIGHTRAG_CHUNKS_TABLE} AS chunks "  # noqa: S608 - private constants.
    f"SET {BM25_LANGUAGE_COLUMN}=labels.language, update_time=CURRENT_TIMESTAMP "
    "FROM UNNEST($2::text[], $3::text[]) AS labels(id, language) "
    "WHERE chunks.workspace=$1 AND chunks.id=labels.id"
)


async def update_chunk_bm25_languages(
    connection: Any,
    *,
    workspace: str,
    labels: Mapping[str, str],
) -> int:
    """Persist one language bucket per chunk and return the updated row count."""
    if not labels:
        return 0
    chunk_ids = list(labels)
    languages = [labels[chunk_id] for chunk_id in chunk_ids]
    await connection.execute(
        _UPDATE_CHUNK_LANGUAGES_SQL,
        workspace,
        chunk_ids,
        languages,
    )
    return len(chunk_ids)


__all__ = ["update_chunk_bm25_languages"]
