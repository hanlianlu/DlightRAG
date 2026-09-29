# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Corpus Reset for one WorkspaceRag: clear corpus content; keep Workspace identity.

5-phase cleanup:
0. Cancel pending tasks
1. Drop LightRAG storages (dynamic discovery)
2. Drop DlightRAG domain stores (metadata index)
3. Clear remaining corpus rows, promotion jobs, and ingest counters
4. Remove filesystem artifacts
"""

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Any

from dlightrag.engine.ai.telemetry import safe_log_text
from dlightrag.engine.rag.corpus.ingestion.paths import RUN_STAGES_DIR_NAME
from dlightrag.engine.rag.corpus.metadata_index import MetadataIndexProtocol
from dlightrag.engine.rag.workspace.lifecycle import shutdown_lightrag_worker_pools
from dlightrag.engine.rag.workspace.ports import CorpusMaintenanceStore
from dlightrag.engine.rag.workspace.workspaces import require_canonical_workspace_id

logger = logging.getLogger(__name__)


# -- Public entry point --------------------------------------------------------


async def areset(
    *,
    workspace_id: str,
    input_root: Path,
    lightrag: Any,
    metadata_index: MetadataIndexProtocol | None,
    maintenance: CorpusMaintenanceStore,
) -> dict[str, Any]:
    """Run the module's five-phase cleanup for one workspace.

    Returns a stats dict with per-phase counts and any errors.
    """
    workspace = require_canonical_workspace_id(workspace_id)
    errors: list[str] = []
    stats: dict[str, Any] = {
        "workspace": workspace,
        "pending_tasks_cancelled": 0,
        "lightrag_storages_dropped": 0,
        "domain_stores_dropped": [],
        "orphan_tables_cleaned": 0,
        "local_files_removed": 0,
        "errors": errors,
    }

    # Phase 0: Cancel pending tasks (worker pools, background tasks)
    try:
        cancelled = await shutdown_lightrag_worker_pools(lightrag)
        stats["pending_tasks_cancelled"] = cancelled
    except Exception as exc:
        errors.append(f"Phase 0 (cancel tasks): {exc}")
        logger.warning("areset Phase 0 failed: %s", exc)

    # Phase 1: LightRAG storages -- dynamic discovery
    lr = lightrag
    if lr is not None:
        for attr in vars(lr):
            storage = getattr(lr, attr, None)
            if storage is None:
                continue
            if isinstance(storage, type):
                continue
            drop_fn = getattr(storage, "drop", None)
            if drop_fn is None or not callable(drop_fn):
                continue
            try:
                outcome = drop_fn()
                if outcome is not None:
                    outcome = await outcome  # type: ignore[misc]
                # LightRAG PostgreSQL storages swallow failures into
                # {"status": "error", ...} instead of raising, so an
                # unchecked drop would be miscounted as a success.
                if isinstance(outcome, dict) and outcome.get("status") == "error":
                    raise RuntimeError(str(outcome.get("message") or "drop reported an error"))
                stats["lightrag_storages_dropped"] += 1
            except Exception as exc:
                errors.append(f"Phase 1 ({attr}): {exc}")
                logger.warning("areset Phase 1 failed for %s: %s", attr, exc)

    # Phase 2: DlightRAG domain stores
    if metadata_index is not None:
        try:
            await metadata_index.clear()
            stats["domain_stores_dropped"].append("metadata_index")
        except Exception as exc:
            errors.append(f"Phase 2 (metadata_index): {exc}")
            logger.warning("areset Phase 2 failed for metadata_index: %s", exc)

    # Phase 3: Clear remaining corpus rows, promotion jobs, and ingest counters
    try:
        orphans = await maintenance.clean_orphan_rows(workspace)
        stats["orphan_tables_cleaned"] = orphans
    except Exception as exc:
        errors.append(f"Phase 3 (orphan tables): {exc}")
        logger.warning("areset Phase 3 failed: %s", exc)

    # Phase 4: File system cleanup — workspace-scoped only.
    # Each workspace owns <input_root>/<workspace>/ in the service's own corpus
    # directory; the root is shared and must never be wiped per-workspace.
    try:
        # A workspace tree can be large; the writer's event loop keeps serving
        # HTTP, SSE and Run leases while it is removed.
        stats["local_files_removed"] = await asyncio.to_thread(
            _reset_workspace_input, input_root, workspace
        )
    except Exception as exc:
        errors.append(f"Phase 4 (filesystem): {exc}")
        logger.warning("areset Phase 4 failed: %s", safe_log_text(exc))

    logger.info(
        "areset complete for workspace=%s: %s",
        safe_log_text(workspace),
        safe_log_text(stats),
    )
    return stats


# -- Internal helpers ----------------------------------------------------------


def _reset_workspace_input(input_root: Path, workspace: str) -> int:
    """Remove one workspace's corpus files; return how many files were removed."""
    input_ws_dir = _workspace_input_dir(input_root, workspace)
    if input_ws_dir is None or not input_ws_dir.is_dir():
        return 0
    return _reset_workspace_files(input_ws_dir)


def _reset_workspace_files(workspace_root: Path) -> int:
    """Remove a workspace's corpus files; its Run stages and the folder itself stay.

    Each Run stage belongs to one Run, which removes it once it ends, so a reset
    never enters them: an upload still being staged, or a Run queued behind the
    reset, keeps its files. Nothing else writes the rest of the folder while the
    reset holds the Workspace's mutation lane.
    """
    removed = 0
    for child in sorted(workspace_root.iterdir()):
        if child.name == RUN_STAGES_DIR_NAME:
            continue
        if child.is_symlink():
            raise ValueError("workspace path contains a symlink")
        if child.is_dir():
            removed += sum(1 for item in child.rglob("*") if item.is_file())
            shutil.rmtree(child)
        else:
            removed += 1
            child.unlink()
    return removed


def _workspace_input_dir(input_root: Path, workspace: str) -> Path | None:
    """Return the direct input-root child for a canonical workspace."""
    workspace_id = require_canonical_workspace_id(workspace)

    root = input_root.resolve()
    if not root.exists():
        return None
    for child in root.iterdir():
        if child.name != workspace_id:
            continue
        if child.is_symlink():
            raise ValueError("workspace path is a symlink")
        resolved = child.resolve()
        resolved.relative_to(root)
        return resolved
    return None
