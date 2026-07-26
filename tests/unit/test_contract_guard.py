# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for LightRAG contract guard startup checks."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from dlightrag.core.contract_guard import LightRAGContractGuard


def _fake_lightrag(*, graph_storage: object | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        chunks_vdb=SimpleNamespace(
            db=SimpleNamespace(pool=object()),
            table_name="LIGHTRAG_DOC_CHUNKS",
            embedding_func=None,
        ),
        chunk_entity_relation_graph=graph_storage,
        full_docs=None,
        text_chunks=None,
        full_entities=None,
        full_relations=None,
        entity_chunks=None,
        relation_chunks=None,
        entities_vdb=None,
        relationships_vdb=None,
        llm_response_cache=None,
        doc_status=None,
    )


def _stub_runtime_checks(monkeypatch: pytest.MonkeyPatch, guard: LightRAGContractGuard) -> None:
    monkeypatch.setattr(guard, "_check_chunks_table_schema", AsyncMock())
    monkeypatch.setattr(guard, "_check_bm25_table", AsyncMock())
    monkeypatch.setattr(guard, "_check_embedding_func_attr", lambda errors: None)
    monkeypatch.setattr(guard, "_check_pool_access", lambda errors: None)
    monkeypatch.setattr(guard, "_check_patch_signatures", lambda errors: None)


async def test_verify_all_reports_missing_client_manager_attach_surfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lightrag.kg.postgres_impl as postgres_impl

    guard = LightRAGContractGuard(_fake_lightrag())
    _stub_runtime_checks(monkeypatch, guard)

    fake_manager = SimpleNamespace(
        get_config=lambda *, vector_storage=None: {"database": "db"},
        _build_vector_signature=lambda config, vector_storage: {"database": "db"},
        _assert_compatible_vector_signature=lambda signature: None,
        _lock=object(),
    )

    with patch.object(postgres_impl, "ClientManager", fake_manager):
        with pytest.raises(RuntimeError, match=r"ClientManager\._instances"):
            await guard.verify_all()


async def test_verify_all_reports_missing_workspace_graph_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lightrag.kg.postgres_impl as postgres_impl

    guard = LightRAGContractGuard(_fake_lightrag(graph_storage=SimpleNamespace(graph_name="graph")))
    _stub_runtime_checks(monkeypatch, guard)

    fake_manager = SimpleNamespace(
        get_config=lambda *, vector_storage=None: {"database": "db"},
        _build_vector_signature=lambda config, vector_storage: {"database": "db"},
        _assert_compatible_vector_signature=lambda signature: None,
        _lock=object(),
        _instances={"db": None, "ref_count": 0, "vector_signature": None},
    )

    with patch.object(postgres_impl, "ClientManager", fake_manager):
        with pytest.raises(RuntimeError, match="_get_workspace_graph_name"):
            await guard.verify_all()
