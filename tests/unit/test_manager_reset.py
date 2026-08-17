# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for RAGServiceManager.areset()."""

import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dlightrag_rag.ingestion.jobs import IngestJobCoordinator
from dlightrag_rag.pool import WorkspaceUnavailableError

from dlightrag.core.servicemanager import RAGServiceManager


class _ResetJobStore:
    def __init__(self, deleted_count: int = 2) -> None:
        self.deleted_count = deleted_count
        self.deleted_workspaces: list[str] = []

    async def delete_for_workspace(self, workspace: str) -> int:
        self.deleted_workspaces.append(workspace)
        return self.deleted_count


@pytest.fixture
def manager(test_config) -> RAGServiceManager:
    manager = RAGServiceManager(config=test_config)
    manager._health.mark_ready()
    manager._corpus_maintenance = AsyncMock()
    manager._corpus_maintenance.list_workspaces.return_value = []
    manager._corpus_maintenance.clean_orphan_rows.return_value = 0
    manager._corpus_maintenance.delete_workspace_record.return_value = False
    manager._ingest_jobs = IngestJobCoordinator(
        lambda workspace: manager._workspace_pool.acquire(workspace),
        input_root=test_config.input_dir_path,
        store=cast(Any, _ResetJobStore()),
    )
    manager._ingest_jobs._store_started = True
    return manager


def _install_runtime(manager: RAGServiceManager, runtime: MagicMock) -> None:
    manager._workspace_pool.acquire = AsyncMock(  # type: ignore[method-assign]
        return_value=runtime
    )
    manager._workspace_pool.is_loaded = AsyncMock(  # type: ignore[method-assign]
        return_value=True
    )
    manager._workspace_pool.evict = AsyncMock()  # type: ignore[method-assign]


def _make_mock_service(workspace: str = "default") -> MagicMock:
    svc = MagicMock()
    svc.areset = AsyncMock(
        return_value={
            "workspace": workspace,
            "pending_tasks_cancelled": 0,
            "lightrag_storages_dropped": 12,
            "domain_stores_dropped": ["metadata_index"],
            "orphan_tables_cleaned": 0,
            "local_files_removed": 5,
            "errors": [],
        }
    )
    svc.aclose = AsyncMock()
    return svc


class TestManagerAresetSingleWorkspace:
    async def test_resets_single_workspace(self, manager: RAGServiceManager) -> None:
        svc = _make_mock_service()
        _install_runtime(manager, svc)

        result = await manager.areset(workspace="ws1")

        svc.areset.assert_awaited_once_with(keep_files=False, dry_run=False)
        cast(AsyncMock, manager._workspace_pool.evict).assert_awaited_once_with("ws1")
        assert "ws1" in result["workspaces"]
        assert result["total_errors"] == 0

    async def test_deletes_workspace_ingest_jobs(self, manager: RAGServiceManager) -> None:
        store = _ResetJobStore(deleted_count=4)
        manager._ingest_jobs._store = cast(Any, store)
        svc = _make_mock_service("project_a")
        _install_runtime(manager, svc)

        result = await manager.areset(workspace="Project A")

        assert store.deleted_workspaces == ["project_a"]
        assert result["workspaces"]["project_a"]["ingest_jobs_deleted"] == 4

    async def test_cancels_workspace_ingest_jobs_before_reset(
        self,
        manager: RAGServiceManager,
    ) -> None:
        svc = _make_mock_service("project_a")
        _install_runtime(manager, svc)
        cancelled = asyncio.Event()

        async def running_job() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        task = asyncio.create_task(running_job())
        manager._ingest_jobs._tasks["job-1"] = task
        manager._ingest_jobs._workspaces["job-1"] = "project_a"
        await asyncio.sleep(0)

        result = await manager.areset(workspace="project_a")

        assert cancelled.is_set()
        assert task.done()
        assert manager._ingest_jobs._tasks == {}
        assert manager._ingest_jobs._workspaces == {}
        assert result["workspaces"]["project_a"]["ingest_jobs_cancelled"] == 1

    async def test_passes_keep_files_and_dry_run(self, manager: RAGServiceManager) -> None:
        svc = _make_mock_service()
        _install_runtime(manager, svc)

        result = await manager.areset(workspace="ws1", keep_files=True, dry_run=True)

        svc.areset.assert_awaited_once_with(keep_files=True, dry_run=True)
        store = manager._ingest_jobs._store
        assert isinstance(store, _ResetJobStore)
        assert store.deleted_workspaces == []
        assert result["workspaces"]["ws1"]["ingest_jobs_cancelled"] == 0
        assert result["workspaces"]["ws1"]["ingest_jobs_deleted"] == 0
        cast(AsyncMock, manager._workspace_pool.evict).assert_not_awaited()


class TestManagerAresetAllWorkspaces:
    async def test_resets_all_workspaces(self, manager: RAGServiceManager) -> None:
        svc1 = _make_mock_service("ws1")
        svc2 = _make_mock_service("ws2")
        manager._workspace_pool.acquire = AsyncMock(  # type: ignore[method-assign]
            side_effect=[svc1, svc2]
        )
        manager._workspace_pool.evict = AsyncMock()  # type: ignore[method-assign]

        with patch.object(
            manager, "alist_workspaces", new_callable=AsyncMock, return_value=["ws1", "ws2"]
        ):
            result = await manager.areset()

        assert "ws1" in result["workspaces"]
        assert "ws2" in result["workspaces"]
        svc1.areset.assert_awaited_once()
        svc2.areset.assert_awaited_once()


class TestManagerAresetEviction:
    async def test_reset_completes_before_eviction(self, manager: RAGServiceManager) -> None:
        svc = _make_mock_service()
        _install_runtime(manager, svc)

        call_order = []
        original_areset = svc.areset

        async def track_areset(**kw):
            call_order.append("areset")
            return await original_areset(**kw)

        async def track_evict(_workspace: str) -> None:
            call_order.append("evict")

        svc.areset = AsyncMock(side_effect=track_areset)
        manager._workspace_pool.evict = AsyncMock(  # type: ignore[method-assign]
            side_effect=track_evict
        )

        await manager.areset(workspace="ws1")

        assert call_order == ["areset", "evict"]


class TestManagerAresetNonexistentWorkspace:
    async def test_nonexistent_workspace_triggers_orphan_cleanup(
        self,
        manager: RAGServiceManager,
    ) -> None:

        with patch.object(
            manager, "alist_workspaces", new_callable=AsyncMock, return_value=["ws1", "ws2"]
        ):
            result = await manager.areset(workspace="does-not-exist")

        assert "does_not_exist" in result["workspaces"]
        ws_result = result["workspaces"]["does_not_exist"]
        assert ws_result["workspace"] == "does_not_exist"
        assert "orphan_tables_cleaned" in ws_result
        assert "local_files_removed" in ws_result
        assert ws_result["ingest_jobs_cancelled"] == 0
        assert ws_result["ingest_jobs_deleted"] == 2
        store = manager._ingest_jobs._store
        assert isinstance(store, _ResetJobStore)
        assert store.deleted_workspaces == ["does_not_exist"]
        assert "errors" in ws_result


class TestManagerAresetErrorHandling:
    async def test_collects_errors_from_service(self, manager: RAGServiceManager) -> None:
        svc = _make_mock_service()
        svc.areset = AsyncMock(
            return_value={
                "workspace": "ws1",
                "lightrag_storages_dropped": 0,
                "domain_stores_dropped": [],
                "orphan_tables_cleaned": 0,
                "local_files_removed": 0,
                "errors": ["Phase 1 (full_docs): boom"],
            }
        )
        _install_runtime(manager, svc)

        result = await manager.areset(workspace="ws1")

        assert result["total_errors"] == 1

    async def test_acquire_failure_counts_error(self, manager: RAGServiceManager) -> None:
        manager._workspace_pool.acquire = AsyncMock(  # type: ignore[method-assign]
            side_effect=WorkspaceUnavailableError("down")
        )
        manager._workspace_pool.evict = AsyncMock()  # type: ignore[method-assign]

        with patch.object(
            manager, "alist_workspaces", new_callable=AsyncMock, return_value=["ws1"]
        ):
            result = await manager.areset()

        assert result["total_errors"] == 1
        assert "error" in result["workspaces"]["ws1"]

    async def test_evict_failure_does_not_replace_reset_result(
        self,
        manager: RAGServiceManager,
    ) -> None:
        svc = _make_mock_service()
        _install_runtime(manager, svc)
        manager._workspace_pool.evict = AsyncMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("close boom")
        )

        result = await manager.areset(workspace="ws1")

        assert result["total_errors"] == 0
