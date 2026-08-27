# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""LightRAG mix retrieval backend."""

import logging
from typing import Any

from lightrag import QueryParam
from lightrag.constants import DEFAULT_CHUNK_TOP_K, DEFAULT_TOP_K

from dlightrag.engine.rag.retrieval import ContextRow, RetrievalResult
from dlightrag.engine.rag.retrieval.references import canonicalize_reference_ids

logger = logging.getLogger(__name__)


class LightRAGMixBackend:
    """RetrievalBackend over a LightRAG instance, always using mix mode."""

    def __init__(
        self,
        *,
        lightrag: Any,
        stores: Any,
        max_entity_tokens: int = 6000,
        max_relation_tokens: int = 8000,
        max_total_tokens: int = 40000,
    ) -> None:
        self._lightrag = lightrag
        self._stores = stores
        self._max_entity_tokens = max_entity_tokens
        self._max_relation_tokens = max_relation_tokens
        self._max_total_tokens = max_total_tokens

    async def aretrieve(
        self,
        query: str,
        *,
        mode: str = "mix",
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        del mode, kwargs
        limit = chunk_top_k or top_k or DEFAULT_CHUNK_TOP_K
        param = QueryParam(
            mode="mix",
            top_k=top_k or DEFAULT_TOP_K,
            chunk_top_k=limit,
            max_entity_tokens=self._max_entity_tokens,
            max_relation_tokens=self._max_relation_tokens,
            max_total_tokens=self._max_total_tokens,
            enable_rerank=False,
        )
        raw = await self._lightrag.aquery_data(query, param=param)
        data = raw.get("data", {})

        chunks = self._chunks_from_lightrag(data.get("chunks", []))
        trace: dict[str, Any] = {
            "lightrag_status": raw.get("status"),
            "lightrag_chunk_count": len(chunks),
            "lightrag_entity_count": len(data.get("entities", [])),
            "lightrag_relationship_count": len(data.get("relationships", [])),
        }

        chunks = canonicalize_reference_ids(chunks, references=data.get("references", []))

        context_chunks: list[ContextRow] = []
        for c in chunks[:limit]:
            context_chunk = {
                "chunk_id": c["chunk_id"],
                "reference_id": c.get("reference_id", ""),
                "file_path": c.get("file_path", ""),
                "content": c.get("content", ""),
                "image_data": c.get("image_data"),
                "image_mime_type": c.get("image_mime_type"),
                "relevance_score": c.get("relevance_score"),
            }
            if c.get("full_doc_id"):
                context_chunk["full_doc_id"] = c["full_doc_id"]
            if c.get("page_number") is not None:
                context_chunk["page_number"] = c["page_number"]
            if c.get("sidecar") is not None:
                context_chunk["sidecar"] = c["sidecar"]
            if c.get("sidecar_location") is not None:
                context_chunk["sidecar_location"] = c["sidecar_location"]
            context_chunks.append(context_chunk)

        return RetrievalResult(
            contexts={
                "entities": data.get("entities", []),
                "relationships": data.get("relationships", []),
                "chunks": context_chunks,
            },
            trace=trace,
        )

    @staticmethod
    def _chunks_from_lightrag(rows: list[ContextRow]) -> list[ContextRow]:
        chunks: list[ContextRow] = []
        seen: set[str] = set()
        for raw in rows:
            cid = raw.get("chunk_id") or raw.get("id")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            chunk = {
                "chunk_id": cid,
                "content": raw.get("content", ""),
                "reference_id": str(raw.get("reference_id", "")),
                "file_path": raw.get("file_path", ""),
                "relevance_score": (
                    raw.get("score") if raw.get("score") is not None else raw.get("distance")
                ),
            }
            for key in ("full_doc_id", "sidecar", "sidecar_location", "page_number"):
                if raw.get(key) is not None:
                    chunk[key] = raw[key]
            chunks.append(chunk)
        return chunks
