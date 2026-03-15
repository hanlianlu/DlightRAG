# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for RAGService.areset() — 5-phase workspace reset."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dlightrag.core.service import RAGService


def _make_service(*, kv_storage: str = "JsonKVStorage", workspace: str = "test-ws") -> RAGService:
    """Create a RAGService with mocked config and storages."""
    config = MagicMock()
    config.kv_storage = kv_storage
    config.workspace = workspace
    config.working_dir = f"/tmp/dlightrag-test/{workspace}"

    service = RAGService(config=config)
    service._initialized = True

    # Mock LightRAG with all storage attrs
    lightrag = MagicMock()
    for attr in (
        "full_docs", "text_chunks", "full_entities", "full_relations",
        "entity_chunks", "relation_chunks", "entities_vdb", "relationships_vdb",
        "chunks_vdb", "chunk_entity_relation_graph", "llm_response_cache", "doc_status",
    ):
        storage = MagicMock()
        storage.drop = AsyncMock(return_value={"status": "success", "message": "dropped"})
        setattr(lightrag, attr, storage)

    service._lightrag = lightrag

    # Mock DlightRAG stores
    service._hash_index = MagicMock()
    service._hash_index.clear = AsyncMock()

    service._metadata_index = MagicMock()
    service._metadata_index.clear = AsyncMock()

    service._visual_chunks = MagicMock()
    service._visual_chunks.drop = AsyncMock(return_value={"status": "success", "message": "dropped"})

    return service


class TestAresetPhase1:
    """Phase 1: LightRAG stores."""

    async def test_drops_all_lightrag_stores(self) -> None:
        service = _make_service()
        result = await service.areset()

        for attr in (
            "full_docs", "text_chunks", "full_entities", "full_relations",
            "entity_chunks", "relation_chunks", "entities_vdb", "relationships_vdb",
            "chunks_vdb", "chunk_entity_relation_graph", "llm_response_cache", "doc_status",
        ):
            getattr(service._lightrag, attr).drop.assert_awaited_once()

        assert result["storages_dropped"] == 12


class TestAresetPhase2:
    """Phase 2: DlightRAG stores."""

    async def test_clears_hash_index(self) -> None:
        service = _make_service()
        await service.areset()
        service._hash_index.clear.assert_awaited_once()

    async def test_clears_metadata_index(self) -> None:
        service = _make_service()
        await service.areset()
        service._metadata_index.clear.assert_awaited_once()

    async def test_drops_visual_chunks(self) -> None:
        service = _make_service()
        await service.areset()
        service._visual_chunks.drop.assert_awaited_once()

    async def test_skips_none_metadata_index(self) -> None:
        service = _make_service()
        service._metadata_index = None
        result = await service.areset()
        assert "metadata_index" not in result["dlightrag_cleared"]

    async def test_skips_none_visual_chunks(self) -> None:
        service = _make_service()
        service._visual_chunks = None
        result = await service.areset()
        assert "visual_chunks" not in result["dlightrag_cleared"]


class TestAresetPhase3:
    """Phase 3: PG orphan table cleanup."""

    async def test_skips_on_non_pg_backend(self) -> None:
        service = _make_service(kv_storage="JsonKVStorage")
        result = await service.areset()
        assert result["orphan_tables_cleaned"] == 0

    async def test_runs_on_pg_backend(self) -> None:
        service = _make_service(kv_storage="PGKVStorage")
        with patch(
            "dlightrag.core.service.RAGService._clean_pg_orphan_tables",
            new_callable=AsyncMock,
            return_value=3,
        ):
            result = await service.areset()
        assert result["orphan_tables_cleaned"] == 3


class TestAresetPhase5:
    """Phase 5: Local files."""

    async def test_keep_files_skips_cleanup(self) -> None:
        service = _make_service()
        result = await service.areset(keep_files=True)
        assert result["local_files_removed"] == 0

    async def test_removes_local_files(self, tmp_path: Path) -> None:
        service = _make_service()
        service.config.working_dir = str(tmp_path)

        # Create test files
        (tmp_path / "cache.json").write_text("{}")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "data.bin").write_bytes(b"x" * 100)
        (tmp_path / "sources").mkdir()
        (tmp_path / "sources" / "keep.pdf").write_bytes(b"pdf")

        result = await service.areset()

        assert result["local_files_removed"] == 2
        assert (tmp_path / "sources" / "keep.pdf").exists()
        assert not (tmp_path / "cache.json").exists()
        assert not (tmp_path / "subdir").exists()


class TestAresetDryRun:
    """dry_run=True collects stats without executing."""

    async def test_dry_run_no_drops(self) -> None:
        service = _make_service()
        result = await service.areset(dry_run=True)

        # Stats reported but no actual drops
        assert result["storages_dropped"] == 12
        for attr in ("full_docs", "text_chunks"):
            getattr(service._lightrag, attr).drop.assert_not_awaited()
        service._hash_index.clear.assert_not_awaited()


class TestAresetErrorHandling:
    """Errors in one phase don't block subsequent phases."""

    async def test_phase1_error_continues_to_phase2(self) -> None:
        service = _make_service()
        service._lightrag.full_docs.drop = AsyncMock(side_effect=RuntimeError("boom"))

        result = await service.areset()

        # Phase 2 still ran
        service._hash_index.clear.assert_awaited_once()
        assert len(result["errors"]) >= 1

    async def test_phase2_error_continues(self) -> None:
        service = _make_service()
        service._hash_index.clear = AsyncMock(side_effect=RuntimeError("boom"))

        result = await service.areset()

        assert len(result["errors"]) >= 1
        # _initialized still set to False
        assert not service._initialized


class TestAresetState:
    """Service state after reset."""

    async def test_sets_initialized_false(self) -> None:
        service = _make_service()
        assert service._initialized is True
        await service.areset()
        assert service._initialized is False
