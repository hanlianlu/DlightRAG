# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for RAGService.areset() — 6-phase workspace reset via dlightrag.core.reset."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from dlightrag.core.service import _STORAGE_ATTRS, RAGService


class _FakeLightRAG:
    """Lightweight fake LightRAG for reset tests.

    Uses a plain object so ``vars()`` returns exactly the storage attributes
    we set, avoiding MagicMock internals leaking into dynamic discovery.
    """

    pass


def _make_service(*, kv_storage: str = "JsonKVStorage", workspace: str = "test_ws") -> RAGService:
    """Create a RAGService with mocked config and storages."""
    config = MagicMock()
    config.kv_storage = kv_storage
    config.workspace = workspace
    config.working_dir = f"/tmp/dlightrag-test/{workspace}"

    service = RAGService.__new__(RAGService)
    service.config = config
    service._initialized = True
    service.rag = None
    service.unified = None
    service.ingestion = None
    service.retrieval = None
    service._backend = None
    service._cancel_checker = None
    service.enable_vlm = False
    service._table_schema = None

    # Create fake LightRAG with all storage attrs (dynamic-discovery friendly)
    lightrag = _FakeLightRAG()
    for attr in _STORAGE_ATTRS:
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
    service._visual_chunks.drop = AsyncMock(
        return_value={"status": "success", "message": "dropped"}
    )

    return service


class TestAresetPhase0:
    """Phase 0: Cancel pending tasks."""

    async def test_cancels_worker_pools(self) -> None:
        service = _make_service()
        # Create a separate lightrag mock with worker pool shutdown
        lr = MagicMock()
        shutdown_mock = AsyncMock()
        inner_func = MagicMock()
        inner_func.shutdown = shutdown_mock
        embedding_func = MagicMock()
        embedding_func.func = inner_func
        lr.embedding_func = embedding_func
        lr.llm_model_func = MagicMock(spec=[])  # no shutdown
        # Give it a droppable storage so Phase 1 works
        storage = MagicMock()
        storage.drop = AsyncMock()
        lr.__dict__["chunks_vdb"] = storage
        service._lightrag = lr

        result = await service.areset()
        assert result["pending_tasks_cancelled"] >= 1
        shutdown_mock.assert_awaited_once()


class TestAresetPhase1:
    """Phase 1: LightRAG stores (dynamic discovery)."""

    async def test_drops_all_lightrag_stores(self) -> None:
        service = _make_service()
        result = await service.areset()

        for attr in _STORAGE_ATTRS:
            getattr(service._lightrag, attr).drop.assert_awaited_once()

        assert result["lightrag_storages_dropped"] == len(_STORAGE_ATTRS)


class TestAresetPhase2:
    """Phase 2: DlightRAG domain stores."""

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
        assert "metadata_index" not in result["domain_stores_dropped"]

    async def test_skips_none_visual_chunks(self) -> None:
        service = _make_service()
        service._visual_chunks = None
        result = await service.areset()
        assert "visual_chunks" not in result["domain_stores_dropped"]


class TestAresetPhase3:
    """Phase 3: PG orphan table cleanup."""

    async def test_skips_on_non_pg_backend(self) -> None:
        service = _make_service(kv_storage="JsonKVStorage")
        result = await service.areset()
        assert result["orphan_tables_cleaned"] == 0

    async def test_runs_on_pg_backend(self) -> None:
        service = _make_service(kv_storage="PGKVStorage")
        with (
            patch(
                "dlightrag.core.reset._clean_orphan_tables",
                new_callable=AsyncMock,
                return_value=3,
            ),
            patch(
                "dlightrag.core.reset._clean_workspace_meta",
                new_callable=AsyncMock,
            ),
            patch(
                "dlightrag.core.reset._drop_age_graphs",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            result = await service.areset()
        assert result["orphan_tables_cleaned"] == 3


class TestAresetPhase4:
    """Phase 4: AGE graph schema drop."""

    async def test_runs_on_pg_backend(self) -> None:
        service = _make_service(kv_storage="PGKVStorage")
        with (
            patch(
                "dlightrag.core.reset._clean_orphan_tables",
                new_callable=AsyncMock,
                return_value=0,
            ),
            patch(
                "dlightrag.core.reset._clean_workspace_meta",
                new_callable=AsyncMock,
            ),
            patch(
                "dlightrag.core.reset._drop_age_graphs",
                new_callable=AsyncMock,
                return_value=["test_ws_chunk_entity_relation"],
            ),
        ):
            result = await service.areset()
        assert result["graphs_dropped"] == ["test_ws_chunk_entity_relation"]

    async def test_skips_on_non_pg_backend(self) -> None:
        service = _make_service(kv_storage="JsonKVStorage")
        result = await service.areset()
        assert result["graphs_dropped"] == []


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

        assert result["local_files_removed"] == 3
        assert not (tmp_path / "sources").exists()
        assert not (tmp_path / "cache.json").exists()
        assert not (tmp_path / "subdir").exists()


class TestAresetDryRun:
    """dry_run=True collects stats without executing."""

    async def test_dry_run_no_drops(self) -> None:
        service = _make_service()
        result = await service.areset(dry_run=True)

        # Stats reported but no actual drops
        assert result["lightrag_storages_dropped"] == len(_STORAGE_ATTRS)
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
