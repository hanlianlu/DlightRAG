# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for RAGServiceManager.areset()."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dlightrag.core.servicemanager import RAGServiceManager, RAGServiceUnavailableError


def _make_manager() -> RAGServiceManager:
    """Create a manager with mocked config."""
    config = MagicMock()
    config.workspace = "default"
    config.kv_storage = "JsonKVStorage"
    config.working_dir = "/tmp/dlightrag-test"
    config.request_timeout = 30
    manager = RAGServiceManager.__new__(RAGServiceManager)
    manager._config = config
    manager._services = {}
    manager._ready = True
    manager._degraded = False
    manager._startup_warnings = []
    manager._last_error = None
    manager._last_error_ts = None
    manager._retry_after = 30.0
    manager._answer_engine = None
    manager._lock = None
    return manager


def _make_mock_service(workspace: str = "default") -> MagicMock:
    svc = MagicMock()
    svc.areset = AsyncMock(return_value={
        "workspace": workspace,
        "storages_dropped": 12,
        "dlightrag_cleared": ["hash_index"],
        "orphan_tables_cleaned": 0,
        "local_files_removed": 5,
        "errors": [],
    })
    svc.close = AsyncMock()
    return svc


class TestManagerAresetSingleWorkspace:
    async def test_resets_single_workspace(self) -> None:
        manager = _make_manager()
        svc = _make_mock_service()
        manager._services["ws1"] = svc

        with patch.object(manager, "_get_service", new_callable=AsyncMock, return_value=svc):
            result = await manager.areset(workspace="ws1")

        svc.areset.assert_awaited_once_with(keep_files=False, dry_run=False)
        svc.close.assert_awaited_once()
        assert "ws1" not in manager._services
        assert "ws1" in result["workspaces"]
        assert result["total_errors"] == 0

    async def test_passes_keep_files_and_dry_run(self) -> None:
        manager = _make_manager()
        svc = _make_mock_service()
        manager._services["ws1"] = svc

        with patch.object(manager, "_get_service", new_callable=AsyncMock, return_value=svc):
            await manager.areset(workspace="ws1", keep_files=True, dry_run=True)

        svc.areset.assert_awaited_once_with(keep_files=True, dry_run=True)


class TestManagerAresetAllWorkspaces:
    async def test_resets_all_workspaces(self) -> None:
        manager = _make_manager()
        svc1 = _make_mock_service("ws1")
        svc2 = _make_mock_service("ws2")

        with (
            patch.object(
                manager, "list_workspaces", new_callable=AsyncMock, return_value=["ws1", "ws2"]
            ),
            patch.object(
                manager, "_get_service", new_callable=AsyncMock, side_effect=[svc1, svc2]
            ),
        ):
            result = await manager.areset()

        assert "ws1" in result["workspaces"]
        assert "ws2" in result["workspaces"]
        svc1.areset.assert_awaited_once()
        svc2.areset.assert_awaited_once()


class TestManagerAresetCacheEviction:
    async def test_evicts_service_from_cache(self) -> None:
        manager = _make_manager()
        svc = _make_mock_service()
        manager._services["ws1"] = svc

        with patch.object(manager, "_get_service", new_callable=AsyncMock, return_value=svc):
            await manager.areset(workspace="ws1")

        assert "ws1" not in manager._services

    async def test_close_called_before_eviction(self) -> None:
        manager = _make_manager()
        svc = _make_mock_service()
        manager._services["ws1"] = svc

        call_order = []
        original_areset = svc.areset

        async def track_areset(**kw):
            call_order.append("areset")
            return await original_areset(**kw)

        async def track_close():
            call_order.append("close")

        svc.areset = AsyncMock(side_effect=track_areset)
        svc.close = AsyncMock(side_effect=track_close)

        with patch.object(manager, "_get_service", new_callable=AsyncMock, return_value=svc):
            await manager.areset(workspace="ws1")

        assert call_order == ["areset", "close"]


class TestManagerAresetNonexistentWorkspace:
    async def test_nonexistent_workspace_is_noop(self) -> None:
        manager = _make_manager()

        with patch.object(
            manager, "list_workspaces", new_callable=AsyncMock, return_value=["ws1", "ws2"]
        ):
            result = await manager.areset(workspace="does-not-exist")

        assert result == {"workspaces": {}, "total_errors": 0}


class TestManagerAresetErrorHandling:
    async def test_collects_errors_from_service(self) -> None:
        manager = _make_manager()
        svc = _make_mock_service()
        svc.areset = AsyncMock(return_value={
            "workspace": "ws1",
            "storages_dropped": 0,
            "dlightrag_cleared": [],
            "orphan_tables_cleaned": 0,
            "local_files_removed": 0,
            "errors": ["Phase 1 (full_docs): boom"],
        })
        manager._services["ws1"] = svc

        with patch.object(manager, "_get_service", new_callable=AsyncMock, return_value=svc):
            result = await manager.areset(workspace="ws1")

        assert result["total_errors"] == 1

    async def test_get_service_failure_counts_error(self) -> None:
        manager = _make_manager()

        with (
            patch.object(
                manager, "list_workspaces", new_callable=AsyncMock, return_value=["ws1"]
            ),
            patch.object(
                manager, "_get_service",
                new_callable=AsyncMock,
                side_effect=RAGServiceUnavailableError("down"),
            ),
        ):
            result = await manager.areset()

        assert result["total_errors"] == 1
        assert "error" in result["workspaces"]["ws1"]

    async def test_close_failure_still_evicts(self) -> None:
        manager = _make_manager()
        svc = _make_mock_service()
        svc.close = AsyncMock(side_effect=RuntimeError("close boom"))
        manager._services["ws1"] = svc

        with patch.object(manager, "_get_service", new_callable=AsyncMock, return_value=svc):
            await manager.areset(workspace="ws1")

        assert "ws1" not in manager._services
