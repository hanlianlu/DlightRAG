# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for metadata retrieval path."""

from unittest.mock import AsyncMock

from dlightrag.engine.rag.retrieval import MetadataFilter, MetadataScope
from dlightrag.engine.rag.retrieval.metadata_path import metadata_retrieve


def _scope(**overrides: object) -> MetadataScope:
    fields: dict[str, object] = {
        "filters": MetadataFilter(filename="x.pdf"),
        "filename_mode": "exact",
        "doc_exists": True,
        "candidate_count": 1470,
        "candidate_count_exact": True,
    }
    fields.update(overrides)
    return MetadataScope(**fields)  # type: ignore[arg-type]


async def test_metadata_retrieve_returns_scope_facts_without_document_ids() -> None:
    stores = AsyncMock()
    stores.resolve_scope.return_value = _scope()

    scope = await metadata_retrieve(
        stores=stores,
        filters=MetadataFilter(filename="x.pdf"),
    )

    # The filter facts and the bounded chunk probe are the only facts read
    # back; no document-id set is ever materialized.
    assert scope.doc_exists is True
    assert scope.candidate_count == 1470
    assert scope.candidate_count_exact is True
    assert scope.filename_mode == "exact"
    stores.resolve_scope.assert_awaited_once()


async def test_metadata_retrieve_forwards_empty_scope() -> None:
    stores = AsyncMock()
    stores.resolve_scope.return_value = _scope(doc_exists=False, candidate_count=0)

    scope = await metadata_retrieve(
        stores=stores,
        filters=MetadataFilter(filename="missing.pdf"),
    )

    assert not scope
    assert scope.candidate_count == 0
    assert scope.candidate_count_exact is True
    stores.resolve_scope.assert_awaited_once()


async def test_metadata_retrieve_keeps_zero_chunk_match_active() -> None:
    """A matching document with zero chunks must still be an active scope."""
    stores = AsyncMock()
    stores.resolve_scope.return_value = _scope(doc_exists=True, candidate_count=0)

    scope = await metadata_retrieve(
        stores=stores,
        filters=MetadataFilter(filename="chunkless.pdf"),
    )

    assert bool(scope) is True
