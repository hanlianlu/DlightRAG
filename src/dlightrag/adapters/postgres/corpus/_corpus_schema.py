# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Private PostgreSQL corpus table and column identities."""

BM25_LANGUAGE_COLUMN = "dlightrag_bm25_language"
CHUNK_DOCUMENT_SCOPE_INDEX = "idx_lightrag_doc_chunks_dlightrag_full_doc_id"
LIGHTRAG_CHUNKS_TABLE = "LIGHTRAG_DOC_CHUNKS"

__all__ = [
    "BM25_LANGUAGE_COLUMN",
    "CHUNK_DOCUMENT_SCOPE_INDEX",
    "LIGHTRAG_CHUNKS_TABLE",
]
