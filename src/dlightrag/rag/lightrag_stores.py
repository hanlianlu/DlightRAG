# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Single boundary for LightRAG private storage access.

**LightRAG coupling surface (lightrag-hku>=1.5.3):**

This module depends on the following LightRAG internals that are NOT part
of the public API.  A LightRAG major-version bump may break these:

Storage attributes (copied from the LightRAG instance):
    ``chunks_vdb``
    ``text_chunks``
    ``full_docs``
    ``doc_status``

LightRAG API methods (public, but shape-dependent):
    ``LightRAG.aquery_data()``          — returns {data: {...}, status: ...}
    ``apipeline_enqueue_documents()``   — enqueues for processing
    ``apipeline_process_enqueue_documents()`` — processes queue

Backend-specific chunk operations are delegated through ``CorpusChunkStore``.
When upgrading lightrag-hku, verify these surfaces still exist and behave as
expected. The host contract guard provides runtime contract checks.
"""

from typing import Any

from dlightrag.rag.ports import CorpusChunkStore


class LightRAGStores:
    """Typed accessor for the LightRAG storage surface DlightRAG touches directly."""

    chunks_vdb: Any
    text_chunks: Any
    full_docs: Any
    doc_status: Any

    def __init__(self, lightrag: Any, *, chunk_store: CorpusChunkStore) -> None:
        self.raw = lightrag
        self.chunks_vdb = lightrag.chunks_vdb
        self.text_chunks = lightrag.text_chunks
        self.full_docs = lightrag.full_docs
        self.doc_status = lightrag.doc_status
        self._chunk_store = chunk_store

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
                for key in ("sidecar", "sidecar_location", "page_number"):
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
        await self._chunk_store.overwrite_chunk_vectors(vectors, embedding_dim=embedding_dim)

    async def count_chunks_for_docs(self, doc_ids: list[str]) -> int:
        return await self._chunk_store.count_chunks_for_docs(doc_ids)

    async def fetch_chunk_contents(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        return await self._chunk_store.fetch_chunk_contents(chunk_ids)

    async def update_chunk_bm25_languages(self, labels: dict[str, str]) -> None:
        await self._chunk_store.update_chunk_bm25_languages(labels)
