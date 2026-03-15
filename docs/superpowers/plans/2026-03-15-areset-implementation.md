# Workspace Reset Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move workspace reset logic from `scripts/reset.py` into `RAGService.areset()` + `RAGServiceManager.areset()`, expose via API, and simplify the script to a CLI wrapper.

**Architecture:** `RAGService.areset()` implements a 5-phase reset (LightRAG stores → DlightRAG stores → PG orphan cleanup → workspace metadata → local files). `RAGServiceManager.areset()` handles workspace routing and cache eviction. `scripts/reset.py` becomes a thin CLI wrapper.

**Tech Stack:** Python asyncio, FastAPI, asyncpg (PG-only phases), pytest

**Spec:** `docs/superpowers/specs/2026-03-15-areset-architecture-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/dlightrag/core/service.py` | `RAGService.areset()` — 5-phase core reset + `_STORAGE_ATTRS` constant |
| `src/dlightrag/core/servicemanager.py` | `RAGServiceManager.areset()` — workspace routing + cache eviction |
| `src/dlightrag/api/server.py` | `POST /reset` endpoint |
| `scripts/reset.py` | CLI wrapper calling `manager.areset()` |
| `tests/unit/test_service_reset.py` | Unit tests for `RAGService.areset()` |
| `tests/unit/test_manager_reset.py` | Unit tests for `RAGServiceManager.areset()` |

---

## Chunk 1: Core Implementation

### Task 1: RAGService.areset() — tests

**Files:**
- Create: `tests/unit/test_service_reset.py`

- [ ] **Step 1: Write test file with all areset unit tests**

```python
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for RAGService.areset() — 5-phase workspace reset."""

from __future__ import annotations

import shutil
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_service_reset.py -v --tb=short`
Expected: FAIL — `RAGService` has no `areset` method yet

- [ ] **Step 3: Commit test file**

```bash
git add tests/unit/test_service_reset.py
git commit -m "test: add unit tests for RAGService.areset()"
```

### Task 2: RAGService.areset() — implementation

**Files:**
- Modify: `src/dlightrag/core/service.py`

- [ ] **Step 1: Add `_STORAGE_ATTRS` constant and `areset()` method to RAGService**

Add near the top of `service.py` (after imports, before the class):

```python
# LightRAG storage attribute names — authoritative list for reset/drop.
_STORAGE_ATTRS = (
    "full_docs",
    "text_chunks",
    "full_entities",
    "full_relations",
    "entity_chunks",
    "relation_chunks",
    "entities_vdb",
    "relationships_vdb",
    "chunks_vdb",
    "chunk_entity_relation_graph",
    "llm_response_cache",
    "doc_status",
)
```

Add `areset()` method to `RAGService` class, after the `close()` method (around line 689):

```python
    async def areset(
        self,
        *,
        keep_files: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Reset all storage for this workspace.

        5-phase reset:
        1. LightRAG stores (drop)
        2. DlightRAG stores (clear/drop)
        3. PG orphan table cleanup (PG backend only)
        4. Workspace metadata (PG backend only)
        5. Local files (skip if keep_files=True)
        """
        workspace = self.config.workspace
        errors: list[str] = []
        stats: dict[str, Any] = {
            "workspace": workspace,
            "storages_dropped": 0,
            "dlightrag_cleared": [],
            "orphan_tables_cleaned": 0,
            "local_files_removed": 0,
            "errors": errors,
        }

        # Phase 1: LightRAG stores
        lr = self.lightrag  # property handles both modes
        if lr is not None:
            for attr in _STORAGE_ATTRS:
                storage = getattr(lr, attr, None)
                if storage is None:
                    continue
                try:
                    if not dry_run:
                        await storage.drop()
                    stats["storages_dropped"] += 1
                except Exception as exc:
                    errors.append(f"Phase 1 ({attr}): {exc}")
                    logger.warning("areset Phase 1 failed for %s: %s", attr, exc)

        # Phase 2: DlightRAG stores
        for name, store, method in (
            ("hash_index", self._hash_index, "clear"),
            ("metadata_index", self._metadata_index, "clear"),
            ("visual_chunks", self._visual_chunks, "drop"),
        ):
            if store is None:
                continue
            try:
                if not dry_run:
                    await getattr(store, method)()
                stats["dlightrag_cleared"].append(name)
            except Exception as exc:
                errors.append(f"Phase 2 ({name}): {exc}")
                logger.warning("areset Phase 2 failed for %s: %s", name, exc)

        # Phase 3 & 4: PG-specific cleanup
        is_pg = self.config.kv_storage.startswith("PG")
        if is_pg:
            try:
                orphans = await self._clean_pg_orphan_tables(workspace, dry_run=dry_run)
                stats["orphan_tables_cleaned"] = orphans
            except Exception as exc:
                errors.append(f"Phase 3 (orphan tables): {exc}")
                logger.warning("areset Phase 3 failed: %s", exc)

            try:
                if not dry_run:
                    await self._clean_workspace_meta(workspace)
            except Exception as exc:
                errors.append(f"Phase 4 (workspace meta): {exc}")
                logger.warning("areset Phase 4 failed: %s", exc)

        # Phase 5: Local files
        if not keep_files:
            try:
                working_dir = Path(self.config.working_dir)
                stats["local_files_removed"] = self._reset_local_files(
                    working_dir, dry_run=dry_run
                )
            except Exception as exc:
                errors.append(f"Phase 5 (local files): {exc}")
                logger.warning("areset Phase 5 failed: %s", exc)

        self._initialized = False
        logger.info("areset complete for workspace=%s: %s", workspace, stats)
        return stats
```

- [ ] **Step 2: Add the three helper methods to RAGService**

Add after `areset()`:

```python
    @staticmethod
    async def _clean_pg_orphan_tables(workspace: str, *, dry_run: bool) -> int:
        """Scan lightrag_*/dlightrag_* PG tables, delete rows for this workspace.

        Catches orphaned tables from old embedding models or mode switches
        that storage.drop() doesn't reach.
        """
        try:
            from lightrag.kg.postgres_impl import ClientManager

            db = await ClientManager.get_client()
            pool = db.pool
            if pool is None:
                return 0
        except Exception:
            return 0

        try:
            async with pool.acquire() as conn:
                table_rows = await conn.fetch(
                    "SELECT tablename FROM pg_tables "
                    "WHERE schemaname = 'public' "
                    "AND (tablename LIKE 'lightrag_%' OR tablename LIKE 'dlightrag_%') "
                    "ORDER BY tablename"
                )
                if not table_rows:
                    return 0

                cleaned = 0
                for row in table_rows:
                    table = row["tablename"]
                    col = await conn.fetchrow(
                        "SELECT 1 FROM information_schema.columns "
                        "WHERE table_name = $1 AND column_name = 'workspace'",
                        table,
                    )
                    if col is None:
                        continue

                    count_row = await conn.fetchrow(
                        f'SELECT COUNT(*) as count FROM "{table}" WHERE workspace = $1',
                        workspace,
                    )
                    count = count_row["count"] if count_row else 0

                    if count > 0:
                        if not dry_run:
                            await conn.execute(
                                f'DELETE FROM "{table}" WHERE workspace = $1',
                                workspace,
                            )
                        cleaned += 1

                    if not dry_run:
                        remaining = await conn.fetchrow(
                            f'SELECT EXISTS (SELECT 1 FROM "{table}") AS has_rows'
                        )
                        if not remaining["has_rows"]:
                            await conn.execute(f'DROP TABLE "{table}"')

                return cleaned
        except Exception as exc:
            logger.warning("PG orphan table cleanup failed: %s", exc)
            return 0

    @staticmethod
    async def _clean_workspace_meta(workspace: str) -> None:
        """Delete workspace record from dlightrag_workspace_meta."""
        from lightrag.kg.postgres_impl import ClientManager

        db = await ClientManager.get_client()
        pool = db.pool
        if pool is None:
            return
        async with pool.acquire() as conn:
            exists = await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                "WHERE table_name = 'dlightrag_workspace_meta')"
            )
            if exists:
                await conn.execute(
                    "DELETE FROM dlightrag_workspace_meta WHERE workspace = $1",
                    workspace,
                )

    @staticmethod
    def _reset_local_files(working_dir: Path, *, dry_run: bool) -> int:
        """Remove working_dir/* except sources/. Returns file count."""
        if not working_dir.exists():
            return 0

        file_count = 0
        for item in sorted(working_dir.iterdir()):
            if item.name == "sources":
                continue
            if item.is_file():
                file_count += 1
                if not dry_run:
                    item.unlink()
            elif item.is_dir():
                for f in item.rglob("*"):
                    if f.is_file():
                        file_count += 1
                if not dry_run:
                    shutil.rmtree(item, ignore_errors=True)
        return file_count
```

Also add `shutil` to the imports at the top of `service.py`:

```python
import shutil
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/unit/test_service_reset.py -v --tb=short`
Expected: ALL PASS

- [ ] **Step 4: Run full test suite to check for regressions**

Run: `uv run pytest tests/ --tb=short`
Expected: 781+ passed, 0 failed

- [ ] **Step 5: Commit**

```bash
git add src/dlightrag/core/service.py
git commit -m "feat: add RAGService.areset() — 5-phase workspace reset"
```

### Task 3: RAGServiceManager.areset()

**Files:**
- Create: `tests/unit/test_manager_reset.py`
- Modify: `src/dlightrag/core/servicemanager.py`

- [ ] **Step 1: Write tests**

```python
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for RAGServiceManager.areset()."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dlightrag.core.servicemanager import RAGServiceManager


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
    manager._retry_after = None
    manager._answer_engine = None
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
        svc.close = AsyncMock(side_effect=lambda: call_order.append("close"))
        original_areset = svc.areset
        svc.areset = AsyncMock(side_effect=lambda **kw: call_order.append("areset") or original_areset(**kw))

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

        with patch.object(manager, "_get_service", new_callable=AsyncMock, return_value=svc):
            result = await manager.areset(workspace="ws1")

        assert result["total_errors"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_manager_reset.py -v --tb=short`
Expected: FAIL — no `areset` method on `RAGServiceManager`

- [ ] **Step 3: Implement `RAGServiceManager.areset()`**

Add to `RAGServiceManager` class in `servicemanager.py`, after the `delete_files` method (around line 163):

```python
    async def areset(
        self,
        *,
        workspace: str | None = None,
        keep_files: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Reset one or all workspaces.

        Drops all storage backends, clears DlightRAG indexes, and optionally
        removes local files. After reset, the service is closed and evicted
        from cache.
        """
        if workspace is not None:
            workspaces = [workspace]
        else:
            workspaces = await self.list_workspaces()

        results: dict[str, Any] = {}
        total_errors = 0

        # If a specific workspace is requested, verify it exists
        if workspace is not None:
            known = await self.list_workspaces()
            if workspace not in known and workspace not in self._services:
                # No-op for nonexistent workspace (spec requirement)
                return {"workspaces": {}, "total_errors": 0}

        for ws in workspaces:
            try:
                svc = await self._get_service(ws)
                ws_result = await svc.areset(keep_files=keep_files, dry_run=dry_run)
                results[ws] = ws_result
                total_errors += len(ws_result.get("errors", []))
            except Exception as exc:
                results[ws] = {"error": str(exc)}
                total_errors += 1
                logger.warning("Failed to reset workspace '%s': %s", ws, exc)

            # Close and evict from cache (even on error — stale state)
            if ws in self._services:
                try:
                    await self._services[ws].close()
                except Exception:
                    logger.warning("Failed to close service for '%s'", ws, exc_info=True)
                del self._services[ws]

        return {"workspaces": results, "total_errors": total_errors}
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/test_manager_reset.py -v --tb=short`
Expected: ALL PASS

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/ --tb=short`
Expected: 781+ passed, 0 failed

- [ ] **Step 6: Commit**

```bash
git add tests/unit/test_manager_reset.py src/dlightrag/core/servicemanager.py
git commit -m "feat: add RAGServiceManager.areset() — workspace routing + cache eviction"
```

## Chunk 2: API Endpoint + Script Simplification

### Task 4: API `POST /reset` endpoint

**Files:**
- Modify: `src/dlightrag/api/server.py`

- [ ] **Step 1: Add request model and endpoint**

Add after the `DeleteRequest` model class:

```python
class ResetRequest(BaseModel):
    """Request to reset a workspace."""
    workspace: str | None = None
    keep_files: bool = False
    dry_run: bool = False
```

Add after the `DELETE /files` endpoint (around line 358):

```python
@app.post("/reset", dependencies=[Depends(_verify_auth)])
async def reset_workspace(body: ResetRequest, request: Request) -> dict[str, Any]:
    """Reset all RAG data for a workspace."""
    manager = _get_manager(request)
    ws = body.workspace or get_config().workspace
    result = await manager.areset(
        workspace=ws,
        keep_files=body.keep_files,
        dry_run=body.dry_run,
    )
    return result
```

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ --tb=short`
Expected: 781+ passed, 0 failed

- [ ] **Step 3: Commit**

```bash
git add src/dlightrag/api/server.py
git commit -m "feat: add POST /reset API endpoint"
```

### Task 5: Simplify `scripts/reset.py`

**Files:**
- Modify: `scripts/reset.py`

- [ ] **Step 1: Rewrite reset.py as a thin CLI wrapper**

Replace the entire file content with:

```python
#!/usr/bin/env python3
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Reset dlightrag RAG storage — drops all storage backends and local files.

Thin CLI wrapper around RAGServiceManager.areset(). All reset logic
lives in RAGService.areset() (5-phase reset).

Usage:
    uv run scripts/reset.py                      # reset default workspace (with confirmation)
    uv run scripts/reset.py --workspace project-a # reset specific workspace
    uv run scripts/reset.py --all                 # reset ALL workspaces
    uv run scripts/reset.py --dry-run             # preview what would be dropped
    uv run scripts/reset.py --keep-files          # drop storages, keep local files
    uv run scripts/reset.py -y                    # skip confirmation prompt
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)


def _print_workspace_result(ws: str, result: dict[str, Any]) -> None:
    """Pretty-print reset results for one workspace."""
    if "error" in result:
        print(f"\n  [{ws}] ERROR: {result['error']}")
        return

    print(f"\n  [{ws}]")
    print(f"    Storages dropped: {result.get('storages_dropped', 0)}")
    cleared = result.get("dlightrag_cleared", [])
    if cleared:
        print(f"    DlightRAG cleared: {', '.join(cleared)}")
    orphans = result.get("orphan_tables_cleaned", 0)
    if orphans:
        print(f"    Orphan tables cleaned: {orphans}")
    files = result.get("local_files_removed", 0)
    if files:
        print(f"    Local files removed: {files}")
    errors = result.get("errors", [])
    if errors:
        print(f"    Errors ({len(errors)}):")
        for err in errors:
            print(f"      - {err}")


async def _run(
    *,
    workspace: str | None,
    reset_all: bool,
    keep_files: bool,
    dry_run: bool,
) -> dict[str, Any]:
    from dlightrag.config import get_config
    from dlightrag.core.servicemanager import RAGServiceManager

    config = get_config()
    if workspace:
        config = config.model_copy(update={"workspace": workspace})

    # Show backend info
    print("\nStorage backends (from config):")
    print(f"  KV:         {config.kv_storage}")
    print(f"  Vector:     {config.vector_storage}")
    print(f"  Graph:      {config.graph_storage}")
    print(f"  Workspace:  {config.workspace}")

    manager = await RAGServiceManager.create(config)
    try:
        target_ws = None if reset_all else (workspace or config.workspace)
        result = await manager.areset(
            workspace=target_ws,
            keep_files=keep_files,
            dry_run=dry_run,
        )
        return result
    finally:
        await manager.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="dlightrag-reset",
        description="Reset dlightrag RAG storage (all configured backends)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without deleting")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--keep-files", action="store_true", help="Drop storages, keep local files")
    parser.add_argument("--workspace", default=None, help="Target workspace (default: from config)")
    parser.add_argument("--all", dest="reset_all", action="store_true", help="Reset ALL workspaces")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    if args.dry_run:
        print("\n(dry run — nothing will be deleted)")

    if not args.dry_run and not args.yes:
        scope = "ALL workspaces" if args.reset_all else (args.workspace or "default workspace")
        print(f"\nWARNING: This will permanently delete ALL RAG data in {scope}.")
        print("Type 'yes' to proceed: ", end="")
        try:
            if input().strip().lower() != "yes":
                print("Cancelled.")
                return 1
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return 1

    result = asyncio.run(
        _run(
            workspace=args.workspace,
            reset_all=args.reset_all,
            keep_files=args.keep_files,
            dry_run=args.dry_run,
        )
    )

    for ws, ws_result in result.get("workspaces", {}).items():
        _print_workspace_result(ws, ws_result)

    total_errors = result.get("total_errors", 0)
    print(f"\nDone. Total errors: {total_errors}")

    if args.dry_run:
        print("Run without --dry-run to actually delete.")

    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ --tb=short`
Expected: 781+ passed, 0 failed

- [ ] **Step 3: Verify script runs**

Run: `uv run scripts/reset.py --dry-run`
Expected: Shows backend info and dry-run preview without errors

- [ ] **Step 4: Commit**

```bash
git add scripts/reset.py
git commit -m "refactor: simplify reset.py to CLI wrapper around manager.areset()"
```
