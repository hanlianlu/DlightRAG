# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for RAGService.areset() — 6-phase workspace reset via dlightrag.core.reset."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dlightrag.core.service import _STORAGE_ATTRS, RAGService


class _FakeLightRAG:
    """Lightweight fake LightRAG for reset tests.

    Uses a plain object so ``vars()`` returns exactly the storage attributes
    we set, avoiding MagicMock internals leaking into dynamic discovery.
    """

    pass


def _make_service(*, workspace: str = "test_ws") -> RAGService:
    """Create a RAGService with mocked config and storages."""
    config = MagicMock()
    config.kv_storage = "PGKVStorage"
    config.workspace = workspace
    config.working_dir = f"/tmp/dlightrag-test/{workspace}"
    config.input_dir_path = Path(f"/tmp/dlightrag-test/{workspace}/inputs")

    service = RAGService.__new__(RAGService)
    service.config = config
    service._initialized = True
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

    # Mock DlightRAG-owned domain store.
    service._metadata_index = MagicMock()
    service._metadata_index.clear = AsyncMock()

    return service


@pytest.fixture(autouse=True)
def _pg_cleanup_patches():
    """Avoid real PostgreSQL cleanup calls in unit tests."""

    class FakeCheckpointStore:
        async def initialize(self) -> None:
            return None

        async def delete_sessions_by_workspace(self, workspace: str) -> int:
            return 0

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
            return_value=[],
        ),
        patch("dlightrag.storage.checkpoint_pg.PGCheckpointStore", FakeCheckpointStore),
    ):
        yield


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
        role_shutdown = AsyncMock()
        role_func = MagicMock()
        role_func.shutdown = role_shutdown
        lr.role_llm_funcs = {"query": role_func}
        # Give it a droppable storage so Phase 1 works
        storage = MagicMock()
        storage.drop = AsyncMock()
        lr.__dict__["chunks_vdb"] = storage
        service._lightrag = lr

        result = await service.areset()
        assert result["pending_tasks_cancelled"] >= 2
        shutdown_mock.assert_awaited_once()
        role_shutdown.assert_awaited_once()


class TestAresetPhase1:
    """Phase 1: LightRAG stores (dynamic discovery)."""

    async def test_drops_all_lightrag_stores(self) -> None:
        service = _make_service()
        result = await service.areset()

        for attr in _STORAGE_ATTRS:
            getattr(service._lightrag, attr).drop.assert_awaited_once()

        assert result["lightrag_storages_dropped"] == len(_STORAGE_ATTRS)

    async def test_skips_lightrag_storage_class_attributes(self) -> None:
        class StorageClass:
            async def drop(self):
                raise AssertionError("class drop must not be called")

        service = _make_service()
        service._lightrag.doc_status_storage_cls = StorageClass

        result = await service.areset()

        assert result["lightrag_storages_dropped"] == len(_STORAGE_ATTRS)
        assert not any("doc_status_storage_cls" in error for error in result["errors"])


class TestAresetPhase2:
    """Phase 2: DlightRAG domain stores."""

    async def test_clears_metadata_index(self) -> None:
        service = _make_service()
        await service.areset()
        cast(Any, service._metadata_index).clear.assert_awaited_once()

    async def test_skips_none_metadata_index(self) -> None:
        service = _make_service()
        service._metadata_index = None
        result = await service.areset()
        assert "metadata_index" not in result["domain_stores_dropped"]


class TestAresetPhase3:
    """Phase 3: PG orphan table cleanup."""

    async def test_runs_on_pg_backend(self) -> None:
        service = _make_service()
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
        service = _make_service()
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


class TestAresetPhase5:
    """Phase 5: Local files."""

    async def test_keep_files_skips_cleanup(self) -> None:
        service = _make_service()
        result = await service.areset(keep_files=True)
        assert result["local_files_removed"] == 0

    async def test_removes_local_files(self, tmp_path: Path) -> None:
        service = _make_service()
        service.config.working_dir = str(tmp_path)
        cast(Any, service.config).input_dir_path = tmp_path / "inputs"
        ws_dir = tmp_path / "inputs" / service.config.workspace

        # Workspace-scoped files under input_dir/<workspace>/
        ws_dir.mkdir(parents=True)
        (ws_dir / "parsed_doc.json").write_text("{}")
        (ws_dir / "subdir").mkdir()
        (ws_dir / "subdir" / "data.bin").write_bytes(b"x" * 100)

        result = await service.areset()

        # Only workspace-scoped files counted and removed
        assert result["local_files_removed"] == 2
        assert not ws_dir.exists()

    async def test_root_files_survive_reset(self, tmp_path: Path) -> None:
        """Shared files in working_dir root must NOT be deleted per-workspace."""
        import sqlite3

        service = _make_service()
        service.config.working_dir = str(tmp_path)
        cast(Any, service.config).input_dir_path = tmp_path / "inputs"

        # Valid SQLite database so Phase 5b doesn't choke
        db_path = tmp_path / "checkpoints.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, workspace TEXT)")
        conn.commit()
        conn.close()
        (tmp_path / "shared_config.json").write_text("{}")

        await service.areset()

        assert db_path.exists()
        assert (tmp_path / "shared_config.json").exists()


class TestAresetDryRun:
    """dry_run=True collects stats without executing."""

    async def test_dry_run_no_drops(self) -> None:
        service = _make_service()
        result = await service.areset(dry_run=True)

        # Stats reported but no actual drops
        assert result["lightrag_storages_dropped"] == len(_STORAGE_ATTRS)
        for attr in ("full_docs", "text_chunks"):
            getattr(service._lightrag, attr).drop.assert_not_awaited()
        cast(Any, service._metadata_index).clear.assert_not_awaited()


class TestAresetErrorHandling:
    """Errors in one phase don't block subsequent phases."""

    async def test_phase1_error_continues_to_phase2(self) -> None:
        service = _make_service()
        service._lightrag.full_docs.drop = AsyncMock(side_effect=RuntimeError("boom"))

        result = await service.areset()

        # Phase 2 still ran
        cast(Any, service._metadata_index).clear.assert_awaited_once()
        assert len(result["errors"]) >= 1

    async def test_phase2_error_continues(self) -> None:
        service = _make_service()
        cast(Any, service._metadata_index).clear = AsyncMock(side_effect=RuntimeError("boom"))

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
