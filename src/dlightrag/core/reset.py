# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Workspace reset/cleanup logic extracted from RAGService.

All functions receive the service instance (or specific attributes) as
parameters to avoid circular imports. The public API remains
``service.areset()`` -- these are implementation helpers.

6-phase cleanup:
0. Cancel pending tasks
1. Drop LightRAG storages (dynamic discovery)
2. Drop DlightRAG domain stores (registry)
3. Orphan PG table scan (safety net)
4. Drop AGE graph schemas
5. Remove filesystem artifacts
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dlightrag.core.service import RAGService

logger = logging.getLogger(__name__)


# -- Public entry point --------------------------------------------------------


async def areset(
    service: RAGService,
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
    4. Drop AGE graph schemas
    5. Remove filesystem artifacts

    Args:
        service: The RAGService whose workspace to reset.
        keep_files: If True, skip Phase 5 (filesystem cleanup).
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
        "graphs_dropped": [],
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
    collected_graph_names: list[str] = []
    lr = service.lightrag  # property handles both modes
    if lr is not None:
        for attr in vars(lr):
            storage = getattr(lr, attr, None)
            if storage is None:
                continue
            drop_fn = getattr(storage, "drop", None)
            if drop_fn is None or not callable(drop_fn):
                continue
            # Collect graph names for Phase 4
            graph_name = getattr(storage, "graph_name", None)
            if isinstance(graph_name, str) and graph_name:
                collected_graph_names.append(graph_name)
            try:
                if not dry_run:
                    result = drop_fn()
                    if result is not None:
                        await result  # type: ignore[misc]
                stats["lightrag_storages_dropped"] += 1
            except Exception as exc:
                errors.append(f"Phase 1 ({attr}): {exc}")
                logger.warning("areset Phase 1 failed for %s: %s", attr, exc)

    # Phase 2: DlightRAG domain stores -- registry
    for name, store, method in (
        ("hash_index", service._hash_index, "clear"),
        ("metadata_index", service._metadata_index, "clear"),
        ("visual_chunks", service._visual_chunks, "drop"),
    ):
        if store is None:
            continue
        try:
            if not dry_run:
                await getattr(store, method)()
            stats["domain_stores_dropped"].append(name)
        except Exception as exc:
            errors.append(f"Phase 2 ({name}): {exc}")
            logger.warning("areset Phase 2 failed for %s: %s", name, exc)

    # Phase 3: Orphan PG table scan (safety net)
    is_pg = service.config.kv_storage.startswith("PG")
    if is_pg:
        try:
            orphans = await _clean_orphan_tables(workspace, dry_run=dry_run)
            stats["orphan_tables_cleaned"] = orphans
        except Exception as exc:
            errors.append(f"Phase 3 (orphan tables): {exc}")
            logger.warning("areset Phase 3 failed: %s", exc)

        # Also clean workspace metadata
        try:
            if not dry_run:
                await _clean_workspace_meta(workspace)
        except Exception as exc:
            errors.append(f"Phase 3 (workspace meta): {exc}")
            logger.warning("areset Phase 3 workspace meta failed: %s", exc)

    # Phase 4: AGE graph schema drop
    if is_pg:
        try:
            dropped = await _drop_age_graphs(
                workspace,
                collected_graph_names,
                dry_run=dry_run,
            )
            stats["graphs_dropped"] = dropped
        except Exception as exc:
            errors.append(f"Phase 4 (AGE graphs): {exc}")
            logger.warning("areset Phase 4 failed: %s", exc)

    # Phase 5: File system cleanup
    if not keep_files:
        try:
            working_dir = Path(service.config.working_dir)
            stats["local_files_removed"] = _reset_local_files(working_dir, dry_run=dry_run)
        except Exception as exc:
            errors.append(f"Phase 5 (filesystem): {exc}")
            logger.warning("areset Phase 5 failed: %s", exc)

    service._initialized = False
    logger.info("areset complete for workspace=%s: %s", workspace, stats)
    return stats


# -- Internal helpers ----------------------------------------------------------


async def _cancel_pending_tasks(service: RAGService, *, dry_run: bool) -> int:
    """Phase 0: Cancel pending async tasks (worker pools, etc.).

    Returns count of cancelled items.
    """
    cancelled = 0

    # Shutdown LightRAG worker pools (embedding_func, llm_model_func)
    lr = service.lightrag
    if lr is not None:
        for attr in ("embedding_func", "llm_model_func"):
            try:
                obj = getattr(lr, attr, None)
                func = getattr(obj, "func", obj)
                shutdown = getattr(func, "shutdown", None)
                if shutdown is not None and not dry_run:
                    await shutdown()
                    cancelled += 1
            except Exception:  # noqa: BLE001
                logger.debug("Failed to shutdown %s worker pool", attr, exc_info=True)

    return cancelled


async def _clean_orphan_tables(workspace: str, *, dry_run: bool) -> int:
    """Phase 3: Scan PG catalog for orphan lightrag_*/dlightrag_* tables and delete rows.

    After workspace normalization, ``workspace`` is the same lowercase
    identifier used for both dlightrag_* and lightrag_* tables.
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

                # Check for workspace column
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

                # Drop empty tables
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


async def _list_all_workspaces() -> list[str]:
    """Return all known workspace IDs from the database."""
    try:
        from lightrag.kg.postgres_impl import ClientManager

        db = await ClientManager.get_client()
        pool = db.pool
        if pool is None:
            return []
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT DISTINCT workspace FROM dlightrag_file_hashes ORDER BY workspace"
            )
        return [r["workspace"] for r in rows]
    except Exception:
        return []


async def _drop_age_graphs(
    workspace: str,
    collected_names: list[str],
    *,
    dry_run: bool,
) -> list[str]:
    """Phase 4: Drop AGE graph schemas for this workspace.

    ``workspace`` is already the normalized lowercase identifier,
    so it is safe to use directly in graph name matching.
    """
    try:
        from lightrag.kg.postgres_impl import ClientManager

        db = await ClientManager.get_client()
        pool = db.pool
        if pool is None:
            return []
    except Exception:
        return []

    dropped: list[str] = []

    async with pool.acquire() as conn:
        # 1. Drop collected graph names from Phase 1
        for gname in collected_names:
            try:
                if not dry_run:
                    await conn.execute(f"SELECT * FROM ag_catalog.drop_graph('{gname}', true)")
                dropped.append(gname)
            except Exception as exc:
                logger.warning("Failed to drop collected graph %s: %s", gname, exc)

        # 2. Catalog scan fallback -- find additional graphs
        try:
            escaped = workspace.replace("_", r"\_")
            rows = await conn.fetch(
                "SELECT name FROM ag_catalog.ag_graph WHERE name ILIKE $1 ESCAPE '\\'",
                f"{escaped}\\_%",
            )

            if rows:
                known_workspaces = await _list_all_workspaces()
                for row in rows:
                    gname = row["name"]
                    if gname in dropped:
                        continue
                    # Cross-check: skip graphs belonging to other workspaces
                    belongs_to_other = any(
                        gname.startswith(f"{other}_") and other != workspace
                        for other in known_workspaces
                        if len(other) > len(workspace) and other.startswith(workspace)
                    )
                    if belongs_to_other:
                        continue
                    try:
                        if not dry_run:
                            await conn.execute(
                                f"SELECT * FROM ag_catalog.drop_graph('{gname}', true)"
                            )
                        dropped.append(gname)
                    except Exception as exc:
                        logger.warning("Failed to drop graph %s: %s", gname, exc)
        except Exception as exc:
            # ag_catalog may not exist if AGE is not installed
            logger.debug("AGE catalog scan skipped: %s", exc)

    return dropped


def _reset_local_files(working_dir: Path, *, dry_run: bool) -> int:
    """Phase 5: Remove all files under working_dir. Returns file count."""
    if not working_dir.exists():
        return 0

    file_count = 0
    for item in sorted(working_dir.iterdir()):
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
