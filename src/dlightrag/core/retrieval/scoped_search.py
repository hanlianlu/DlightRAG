# src/dlightrag/core/retrieval/scoped_search.py
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Scoped vector search — rank chunk_ids by semantic relevance using native DB filtered search.

Detects the VDB backend type and uses native filtered vector search:
- PostgreSQL (PGVectorStorage): WHERE id = ANY($1) ORDER BY embedding <=> query_vec
- Milvus (MilvusVectorDBStorage): search(filter="id in [...]")
- Qdrant (QdrantVectorDBStorage): search(query_filter=HasIdCondition(...))

Falls back to get_vectors_by_ids + manual cosine similarity for unknown backends.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


async def scoped_vector_search(
    query: str,
    chunk_ids: list[str],
    chunks_vdb: Any,
    top_k: int = 20,
) -> list[str]:
    """Rank chunk_ids by semantic relevance within a scoped set.

    If len(chunk_ids) <= top_k, returns all without vector search.
    Otherwise dispatches to backend-specific native filtered search.
    """
    if not chunk_ids:
        return []

    if len(chunk_ids) <= top_k:
        return list(chunk_ids)

    class_name = type(chunks_vdb).__name__

    if class_name == "PGVectorStorage":
        return await _pg_scoped_search(query, chunk_ids, chunks_vdb, top_k)
    elif class_name == "MilvusVectorDBStorage":
        return await _milvus_scoped_search(query, chunk_ids, chunks_vdb, top_k)
    elif class_name == "QdrantVectorDBStorage":
        return await _qdrant_scoped_search(query, chunk_ids, chunks_vdb, top_k)
    else:
        logger.info(
            "[ScopedSearch] Unknown backend %s, using fallback cosine similarity",
            class_name,
        )
        return await _fallback_cosine_search(query, chunk_ids, chunks_vdb, top_k)


async def _embed_query(chunks_vdb: Any, query: str) -> list[float]:
    """Embed query text using the VDB's embedding function."""
    result = await chunks_vdb.embedding_func.func([query])
    vec = result[0]
    if hasattr(vec, "tolist"):
        return vec.tolist()
    return list(vec)


async def _pg_scoped_search(
    query: str,
    chunk_ids: list[str],
    chunks_vdb: Any,
    top_k: int,
) -> list[str]:
    """PostgreSQL native filtered vector search using asyncpg."""
    try:
        query_vec = await _embed_query(chunks_vdb, query)
        pool = chunks_vdb._pool
        table = chunks_vdb._table_name
        vector_col = getattr(chunks_vdb, "_vector_col_name", "embedding")

        sql = f"""
            SELECT id
            FROM {table}
            WHERE workspace = $1 AND id = ANY($2)
            ORDER BY {vector_col} <=> $3::vector
            LIMIT $4
        """
        workspace = chunks_vdb._namespace
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, workspace, chunk_ids, str(query_vec), top_k)
        return [row["id"] for row in rows]
    except Exception as exc:
        logger.warning("[ScopedSearch] PG native search failed: %s, using fallback", exc)
        return await _fallback_cosine_search(query, chunk_ids, chunks_vdb, top_k)


async def _milvus_scoped_search(
    query: str,
    chunk_ids: list[str],
    chunks_vdb: Any,
    top_k: int,
) -> list[str]:
    """Milvus native filtered vector search."""
    try:
        query_vec = await _embed_query(chunks_vdb, query)
        client = chunks_vdb._client
        collection = chunks_vdb._collection_name

        id_list_str = ", ".join(f'"{cid}"' for cid in chunk_ids)
        filter_expr = f"id in [{id_list_str}]"

        results = client.search(
            collection_name=collection,
            data=[query_vec],
            filter=filter_expr,
            limit=top_k,
            output_fields=["id"],
        )
        if results and len(results) > 0:
            return [hit["entity"]["id"] for hit in results[0]]
        return []
    except Exception as exc:
        logger.warning("[ScopedSearch] Milvus native search failed: %s, using fallback", exc)
        return await _fallback_cosine_search(query, chunk_ids, chunks_vdb, top_k)


async def _qdrant_scoped_search(
    query: str,
    chunk_ids: list[str],
    chunks_vdb: Any,
    top_k: int,
) -> list[str]:
    """Qdrant native filtered vector search."""
    try:
        from qdrant_client import models

        query_vec = await _embed_query(chunks_vdb, query)
        client = chunks_vdb._client
        collection = chunks_vdb._collection_name

        results = client.query_points(
            collection_name=collection,
            query=query_vec,
            query_filter=models.Filter(
                must=[models.HasIdCondition(has_id=chunk_ids)]
            ),
            limit=top_k,
        )
        if hasattr(results, "points"):
            return [str(p.id) for p in results.points]
        return []
    except Exception as exc:
        logger.warning("[ScopedSearch] Qdrant native search failed: %s, using fallback", exc)
        return await _fallback_cosine_search(query, chunk_ids, chunks_vdb, top_k)


async def _fallback_cosine_search(
    query: str,
    chunk_ids: list[str],
    chunks_vdb: Any,
    top_k: int,
) -> list[str]:
    """Fallback: get_vectors_by_ids + manual cosine similarity."""
    try:
        vectors = await chunks_vdb.get_vectors_by_ids(chunk_ids)
        query_vecs = await chunks_vdb.embedding_func.func([query])
        query_vec = np.array(query_vecs[0])
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return list(chunk_ids)[:top_k]

        scores: dict[str, float] = {}
        for cid, vec in vectors.items():
            if vec is None:
                continue
            v = np.array(vec)
            norm = np.linalg.norm(v)
            if norm == 0:
                scores[cid] = 0.0
            else:
                scores[cid] = float(np.dot(query_vec, v) / (query_norm * norm))

        ranked = sorted(scores, key=lambda c: scores[c], reverse=True)
        return ranked[:top_k]
    except Exception as exc:
        logger.warning("[ScopedSearch] Fallback cosine search failed: %s", exc)
        return list(chunk_ids)[:top_k]


__all__ = ["scoped_vector_search"]
