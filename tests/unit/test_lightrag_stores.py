# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for LightRAG storage boundary adapter."""

from __future__ import annotations

import pytest

from dlightrag.core.lightrag_stores import LightRAGStores


class FakeLightRAG:
    chunks_vdb = object()
    text_chunks = object()
    full_docs = object()
    doc_status = object()
    entities_vdb = object()
    relationships_vdb = object()
    chunk_entity_relation_graph = object()
    full_entities = object()
    full_relations = object()
    entity_chunks = object()
    relation_chunks = object()
    llm_response_cache = object()

    def _build_global_config(self):
        return {"ok": True}


def test_lightrag_stores_validates_required_surfaces() -> None:
    stores = LightRAGStores(FakeLightRAG())

    assert stores.text_chunks is FakeLightRAG.text_chunks
    assert stores.build_global_config() == {"ok": True}


def test_lightrag_stores_reports_missing_surfaces() -> None:
    class Broken:
        chunks_vdb = object()

    with pytest.raises(RuntimeError, match="missing"):
        LightRAGStores(Broken())


async def test_upsert_chunks_with_vectors_requires_matching_dimension() -> None:
    stores = LightRAGStores(FakeLightRAG())

    with pytest.raises(ValueError, match="vector dimension"):
        await stores.upsert_chunks_with_vectors(
            {"chunk-1": {"content": "x", "full_doc_id": "doc-1"}},
            {"chunk-1": [0.1, 0.2]},
            embedding_dim=3,
            max_token_size=8192,
        )
