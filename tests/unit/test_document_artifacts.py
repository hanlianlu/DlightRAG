# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for document artifact PG store guards."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from dlightrag.storage.document_artifacts import PGDocumentArtifacts


async def test_document_artifacts_clear_requires_initialize() -> None:
    store = PGDocumentArtifacts(workspace="default")

    with pytest.raises(RuntimeError, match="not initialized"):
        await store.clear()


async def test_document_artifacts_rejects_empty_doc_id() -> None:
    store = PGDocumentArtifacts(workspace="default")

    with pytest.raises(ValueError, match="full_doc_id"):
        await store.get("")

    with pytest.raises(ValueError, match="full_doc_id"):
        await store.delete_doc("")

    with pytest.raises(ValueError, match="full_doc_id"):
        await store.upsert({"full_doc_id": ""})


async def test_document_artifacts_upsert_uses_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = AsyncMock()

    async def fake_get():
        return pool

    from dlightrag.storage import pool as pool_module

    monkeypatch.setattr(pool_module.pg_pool, "get", fake_get)
    store = PGDocumentArtifacts(workspace="default")

    await store.initialize()
    await store.upsert(
        {
            "full_doc_id": "doc-1",
            "source_uri": "file:///tmp/doc.pdf",
            "parser": "lightrag",
            "parse_engine": "mineru",
            "process_options": "iteP",
            "chunk_options": {"chunk_token_size": 1200},
            "sidecar_location": "file:///tmp/doc.parsed/",
            "metadata": {"title": "Doc"},
        }
    )

    pool.execute.assert_awaited()
