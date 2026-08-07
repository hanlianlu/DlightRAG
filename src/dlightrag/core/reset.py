# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Workspace reset/cleanup logic extracted from RAGService.

All functions receive the service instance (or specific attributes) as
parameters to avoid circular imports. The public API remains
``service.areset()`` -- these are implementation helpers.

5-phase cleanup:
0. Cancel pending tasks
1. Drop LightRAG storages (dynamic discovery)
2. Drop DlightRAG domain stores (registry)
3. Orphan PG table scan (safety net)
4. Remove filesystem artifacts
"""

import logging
import shutil
from pathlib import Path
from typing import Any

from dlightrag.core.lightrag_lifecycle import shutdown_lightrag_worker_pools
from dlightrag.utils import log_safe, normalize_workspace

logger = logging.getLogger(__name__)


# -- Public entry point --------------------------------------------------------


async def areset(
    service: Any,
    *,
    keep_files: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Completely remove a workspace -- all data, graph schemas, and files.

    6-phase cleanup:
    0. Cancel pending tasks
    1. Drop LightRAG storages (dynamic discovery)
    2. Drop DlightRAG domain stores (registry)
    3. Orphan PG table scan (safety net)
    4. Remove filesystem artifacts

    Args:
        service: The RAGService whose workspace to reset.
        keep_files: If True, skip Phase 4 (filesystem cleanup).
        dry_run: If True, collect stats without executing any mutations.

    Returns:
        Stats dict with per-phase counts and any errors.
    """
    workspace = service.config.workspace
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
        cancelled = await _cancel_pending_tasks(service, dry_run=dry_run)
        stats["pending_tasks_cancelled"] = cancelled
    except Exception as exc:
        errors.append(f"Phase 0 (cancel tasks): {exc}")
        logger.warning("areset Phase 0 failed: %s", exc)

    # Phase 1: LightRAG storages -- dynamic discovery
    lr = service.lightrag  # property handles both modes
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
                if not dry_run:
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
    metadata_index = getattr(service, "_metadata_index", None)
    if metadata_index is not None:
        try:
            if not dry_run:
                await metadata_index.clear()
            stats["domain_stores_dropped"].append("metadata_index")
        except Exception as exc:
            errors.append(f"Phase 2 (metadata_index): {exc}")
            logger.warning("areset Phase 2 failed for metadata_index: %s", exc)

    # Phase 3: Orphan PG table scan (safety net)
    try:
        orphans = await _clean_orphan_tables(workspace, dry_run=dry_run)
        stats["orphan_tables_cleaned"] = orphans
    except Exception as exc:
        errors.append(f"Phase 3 (orphan tables): {exc}")
        logger.warning("areset Phase 3 failed: %s", exc)

    # Also clean workspace metadata
    try:
        if not dry_run:
            await _clean_workspace_meta(workspace, config=service.config)
    except Exception as exc:
        errors.append(f"Phase 3 (workspace meta): {exc}")
        logger.warning("areset Phase 3 workspace meta failed: %s", exc)

    # Phase 4: File system cleanup — workspace-scoped only.
    # Each workspace owns input_dir/<workspace>/; the working_dir root is shared
    # and must never be wiped per-workspace.
    if not keep_files:
        try:
            input_ws_dir = _workspace_input_dir(service.config.input_dir_path, workspace)
            if input_ws_dir is not None and input_ws_dir.is_dir():
                file_count = sum(1 for _ in input_ws_dir.rglob("*") if _.is_file())
                if not dry_run:
                    shutil.rmtree(input_ws_dir, ignore_errors=True)
                stats["local_files_removed"] = file_count
        except Exception as exc:
            errors.append(f"Phase 4 (filesystem): {exc}")
            logger.warning("areset Phase 4 failed: %s", log_safe(exc))

    # A dry run is a preview: never invalidate the live runtime.
    if not dry_run:
        service._initialized = False
    logger.info("areset complete for workspace=%s: %s", log_safe(workspace), log_safe(stats))
    return stats


# -- Internal helpers ----------------------------------------------------------


async def _quote_public_table(conn: Any, table: str) -> str:
    """Return a safely quoted public-table identifier for dynamic SQL."""
    quoted = await conn.fetchval("SELECT quote_ident($1)", table)
    return f"public.{quoted}"


def _workspace_input_dir(input_root: Path, workspace: str) -> Path | None:
    """Return the direct input_dir child for a normalized workspace."""
    workspace_id = normalize_workspace(workspace)
    if not workspace_id:
        raise ValueError("workspace name normalizes to empty")

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


async def _cancel_pending_tasks(service: Any, *, dry_run: bool) -> int:
    """Phase 0: Cancel pending async tasks (worker pools, etc.).

    Returns count of cancelled items.
    """
    return await shutdown_lightrag_worker_pools(service.lightrag, dry_run=dry_run)


async def _clean_orphan_tables(workspace: str, *, dry_run: bool) -> int:
    """Phase 3: Scan PG catalog for orphan lightrag_*/dlightrag_* tables and delete rows.

    Uses a direct ``asyncpg`` connection: LightRAG's process-wide pool is
    already bound to the default workspace's settings and rejects
    per-workspace reconfiguration.
    """
    import asyncpg

    from dlightrag.config import get_config

    config = get_config()

    try:
        conn = await asyncpg.connect(**config.pg_connection_kwargs())
    except Exception as exc:
        logger.warning("Failed to connect via asyncpg for orphan table cleanup: %s", exc)
        return 0

    try:
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
                "WHERE table_schema = 'public' "
                "AND table_name = $1 AND column_name = 'workspace'",
                table,
            )
            if col is None:
                continue

            qualified_table = await _quote_public_table(conn, table)
            count_row = await conn.fetchrow(
                f"SELECT COUNT(*) as count FROM {qualified_table} WHERE workspace = $1",  # noqa: S608
                workspace,
            )
            count = count_row["count"] if count_row else 0

            if count > 0:
                if not dry_run:
                    await conn.execute(
                        f"DELETE FROM {qualified_table} WHERE workspace = $1",  # noqa: S608
                        workspace,
                    )
                cleaned += 1

        # Do NOT drop emptied tables here. DlightRAG-owned metadata and ingest-job
        # tables are migration-managed global tables that carry a workspace column,
        # not per-workspace artifacts. Resetting the last workspace empties them;
        # dropping them orphans the migration ledger and breaks the running app.
        return cleaned
    except Exception as exc:
        logger.warning("PG orphan table cleanup failed: %s", exc)
        return 0
    finally:
        await conn.close()


async def _clean_workspace_meta(workspace: str, config: Any | None = None) -> None:
    """Delete workspace record from dlightrag_workspace_meta."""
    import asyncpg

    if config is None:
        from dlightrag.config import get_config

        config = get_config()

    conn = await asyncpg.connect(**config.pg_connection_kwargs())
    try:
        exists = await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = 'public' "
            "AND table_name = 'dlightrag_workspace_meta')"
        )
        if exists:
            await conn.execute(
                "DELETE FROM dlightrag_workspace_meta WHERE workspace = $1",
                workspace,
            )
    finally:
        await conn.close()


async def _list_all_workspaces(config: Any | None = None) -> list[str]:
    """Return all known workspace IDs from the database."""
    try:
        import asyncpg

        if config is None:
            from dlightrag.config import get_config

            config = get_config()

        conn = await asyncpg.connect(**config.pg_connection_kwargs())
        try:
            rows = await conn.fetch(
                "SELECT DISTINCT workspace FROM dlightrag_workspace_meta ORDER BY workspace"
            )
        finally:
            await conn.close()
        return [r["workspace"] for r in rows]
    except Exception:
        return []


# -- Orphaned workspace cleanup ------------------------------------------------


async def areset_orphaned_workspace(
    workspace: str,
    *,
    keep_files: bool = False,
    dry_run: bool = False,
    input_dir: str | None = None,
) -> dict[str, Any]:
    """Clean up orphaned workspace artifacts without a RAGService instance.

    For workspaces that no longer exist in ``dlightrag_workspace_meta`` but
    have leftover PG table rows or filesystem artifacts. This is a
    best-effort direct PG cleanup.
    """
    original_workspace = workspace
    workspace = normalize_workspace(workspace)
    errors: list[str] = []
    stats: dict[str, Any] = {
        "workspace": original_workspace,
        "orphan_tables_cleaned": 0,
        "local_files_removed": 0,
        "errors": errors,
    }

    # Clean orphan PG table rows
    try:
        orphans = await _clean_orphan_tables(workspace, dry_run=dry_run)
        stats["orphan_tables_cleaned"] = orphans
    except Exception as exc:
        errors.append(f"Orphan tables: {exc}")

    # Clean workspace metadata row (if any)
    try:
        if not dry_run:
            await _clean_workspace_meta(workspace)
    except Exception as exc:
        errors.append(f"Workspace meta: {exc}")

    # File system cleanup — workspace-scoped only (see areset Phase 4).
    if not keep_files:
        if input_dir:
            try:
                input_ws_dir = _workspace_input_dir(Path(input_dir), workspace)
                if input_ws_dir is not None and input_ws_dir.is_dir():
                    if not dry_run:
                        shutil.rmtree(input_ws_dir, ignore_errors=True)
                    stats["local_files_removed"] += 1
            except Exception as exc:
                errors.append(f"Filesystem (input_dir): {exc}")

    logger.info(
        "areset_orphaned complete for workspace=%s: %s",
        log_safe(workspace),
        log_safe(stats),
    )
    return stats
