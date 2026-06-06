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
from collections.abc import Mapping
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
            if isinstance(storage, type):
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
        ("metadata_index", getattr(service, "_metadata_index", None), "clear"),
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

    # Phase 4: AGE graph schema drop
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

    # Phase 5: File system cleanup — workspace-scoped only.
    # Each workspace owns input_dir/<workspace>/; the working_dir root
    # (checkpoints.db, etc.) is shared and must never be wiped per-workspace.
    if not keep_files:
        try:
            input_root = service.config.input_dir_path.resolve()
            input_ws_dir = (input_root / workspace).resolve()
            try:
                input_ws_dir.relative_to(input_root)
            except ValueError:
                errors.append("Phase 5 (filesystem): workspace path escapes input directory")
            else:
                if input_ws_dir.exists() and input_ws_dir.is_dir():
                    file_count = sum(1 for _ in input_ws_dir.rglob("*") if _.is_file())
                    if not dry_run:
                        shutil.rmtree(input_ws_dir, ignore_errors=True)
                    stats["local_files_removed"] = file_count
        except Exception as exc:
            errors.append(f"Phase 5 (filesystem): {exc}")
            logger.warning("areset Phase 5 failed: %s", exc)

    # Phase 5b: Remove checkpoint data for this workspace
    try:
        from dlightrag.core.checkpoint import ConversationCheckpoint

        cp = ConversationCheckpoint(Path(service.config.working_dir) / "checkpoints.db")
        removed = await cp.delete_sessions_by_workspace(workspace)
        if removed > 0:
            stats["checkpoint_sessions_removed"] = removed
    except Exception:
        logger.warning("Failed to clean checkpoint data for workspace %s", workspace, exc_info=True)

    service._initialized = False
    logger.info("areset complete for workspace=%s: %s", workspace, stats)
    return stats


# -- Internal helpers ----------------------------------------------------------


async def _quote_public_table(conn: Any, table: str) -> str:
    """Return a safely quoted public-table identifier for dynamic SQL."""
    quoted = await conn.fetchval("SELECT quote_ident($1)", table)
    return f"public.{quoted}"


async def _cancel_pending_tasks(service: RAGService, *, dry_run: bool) -> int:
    """Phase 0: Cancel pending async tasks (worker pools, etc.).

    Returns count of cancelled items.
    """
    cancelled = 0

    lr = service.lightrag
    if lr is not None:
        from dlightrag.utils.concurrency import shutdown_async_callable

        funcs: list[tuple[str, Any]] = []
        for attr in ("embedding_func", "llm_model_func", "rerank_model_func"):
            try:
                obj = getattr(lr, attr, None)
                funcs.append((attr, getattr(obj, "func", obj)))
            except Exception:  # noqa: BLE001
                logger.debug("Failed to collect %s worker pool", attr, exc_info=True)

        role_funcs = getattr(lr, "role_llm_funcs", None) or {}
        items = role_funcs.items() if isinstance(role_funcs, Mapping) else ()
        funcs.extend((f"role_llm_funcs.{role}", func) for role, func in items)

        seen: set[int] = set()
        for label, func in funcs:
            if func is None or id(func) in seen:
                continue
            seen.add(id(func))
            if not callable(getattr(func, "shutdown", None)):
                continue
            try:
                if not dry_run:
                    await shutdown_async_callable(func)
                cancelled += 1
            except Exception:  # noqa: BLE001
                logger.debug("Failed to shutdown %s worker pool", label, exc_info=True)

    return cancelled


async def _clean_orphan_tables(workspace: str, *, dry_run: bool) -> int:
    """Phase 3: Scan PG catalog for orphan lightrag_*/dlightrag_* tables and delete rows.

    Uses a direct ``asyncpg`` connection, bypassing LightRAG's process-wide pool
    (same rationale as ``_drop_age_graphs``).
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
                f"SELECT COUNT(*) as count FROM {qualified_table} WHERE workspace = $1",
                workspace,
            )
            count = count_row["count"] if count_row else 0

            if count > 0:
                if not dry_run:
                    await conn.execute(
                        f"DELETE FROM {qualified_table} WHERE workspace = $1",
                        workspace,
                    )
                cleaned += 1

            if not dry_run and table.startswith("dlightrag_"):
                remaining = await conn.fetchrow(
                    f"SELECT EXISTS (SELECT 1 FROM {qualified_table}) AS has_rows"
                )
                if not remaining["has_rows"]:
                    await conn.execute(f"DROP TABLE {qualified_table}")

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


async def _drop_age_graphs(
    workspace: str,
    collected_names: list[str],
    *,
    dry_run: bool,
) -> list[str]:
    """Phase 4: Drop AGE graph schemas for this workspace.

    Uses a direct ``asyncpg`` connection to run ``ag_catalog.drop_graph()``,
    bypassing LightRAG's process-wide pool — which is already initialized
    with the default workspace's PostgreSQL settings and rejects
    per-workspace reconfiguration.

    Falls back to direct ``DROP SCHEMA ... CASCADE`` for any non-AGE
    schemas that share the workspace prefix.
    """
    import asyncpg

    from dlightrag.config import get_config

    config = get_config()
    dropped: list[str] = []

    try:
        conn = await asyncpg.connect(**config.pg_connection_kwargs())
    except Exception as exc:
        logger.warning("Failed to connect via asyncpg for AGE graph cleanup: %s", exc)
        return dropped

    try:
        # 1. Drop collected graph names from Phase 1
        for gname in collected_names:
            if gname in dropped:
                continue
            try:
                if not dry_run:
                    await conn.execute("SELECT ag_catalog.drop_graph($1, true)", gname)
                dropped.append(gname)
            except Exception as exc:
                logger.warning("Failed to drop collected graph %s: %s", gname, exc)

        # 2. Catalog scan — find additional graphs matching workspace prefix
        try:
            known_workspaces = await _list_all_workspaces()
            escaped = workspace.replace("_", r"\_")
            rows = await conn.fetch(
                "SELECT name FROM ag_catalog.ag_graph WHERE name ILIKE $1 ESCAPE '\\'",
                f"{escaped}\\_%",
            )
            for row in rows:
                gname = row["name"]
                if gname in dropped:
                    continue
                belongs_to_other = any(
                    gname.startswith(f"{other}_") and other != workspace
                    for other in known_workspaces
                    if len(other) > len(workspace) and other.startswith(workspace)
                )
                if belongs_to_other:
                    continue
                try:
                    if not dry_run:
                        await conn.execute("SELECT ag_catalog.drop_graph($1, true)", gname)
                    dropped.append(gname)
                except Exception as exc:
                    logger.warning("Failed to drop graph %s: %s", gname, exc)
        except Exception as exc:
            logger.warning("AGE catalog scan failed: %s", exc)
    finally:
        await conn.close()

    # Fallback: catch any remaining non-AGE schemas via DROP SCHEMA CASCADE
    try:
        fallback_dropped = await _drop_workspace_schemas_via_pg(
            workspace, already_dropped=dropped, dry_run=dry_run
        )
        for schema_name in fallback_dropped:
            if schema_name not in dropped:
                dropped.append(schema_name)
    except Exception as exc:
        logger.warning("PG schema fallback failed for workspace %s: %s", workspace, exc)

    return dropped


async def _drop_workspace_schemas_via_pg(
    workspace: str,
    already_dropped: list[str],
    *,
    dry_run: bool,
) -> list[str]:
    """Fallback: drop PostgreSQL schemas matching ``{workspace}_%`` directly.

    Uses ``asyncpg`` (bypassing LightRAG's pool). Tries AGE-native
    ``ag_catalog.drop_graph()`` first, then falls back to ``DROP SCHEMA
    ... CASCADE`` for non-AGE schemas.
    """
    import asyncpg

    from dlightrag.config import get_config

    config = get_config()
    dropped: list[str] = []

    try:
        conn = await asyncpg.connect(**config.pg_connection_kwargs())
    except Exception as exc:
        logger.warning("Failed to connect via asyncpg for schema cleanup: %s", exc)
        return dropped

    try:
        escaped = workspace.replace("_", r"\_")
        rows = await conn.fetch(
            "SELECT schema_name FROM information_schema.schemata "
            "WHERE schema_name LIKE $1 ESCAPE '\\' "
            "AND schema_name NOT IN "
            "('public', 'information_schema', 'pg_catalog', 'pg_toast')",
            f"{escaped}\\_%",
        )
        for row in rows:
            schema_name = row["schema_name"]
            if schema_name in already_dropped:
                continue
            try:
                if not dry_run:
                    # AGE-native drop first — handles AGE label tables
                    await conn.execute("SELECT ag_catalog.drop_graph($1, true)", schema_name)
                dropped.append(schema_name)
                logger.info("Dropped schema %s (AGE-native)", schema_name)
            except Exception:
                # Not an AGE graph — try DROP SCHEMA CASCADE
                try:
                    if not dry_run:
                        safe = await conn.fetchval("SELECT quote_ident($1)", schema_name)
                        await conn.execute(f"DROP SCHEMA IF EXISTS {safe} CASCADE")
                    dropped.append(schema_name)
                    logger.info("Dropped schema %s (PG CASCADE)", schema_name)
                except Exception as exc:
                    logger.warning("Failed to drop schema %s: %s", schema_name, exc)
    finally:
        await conn.close()

    return dropped


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
    have leftover AGE graph schemas, PG table rows, or filesystem artifacts.
    This is a best-effort direct PG cleanup.
    """
    from dlightrag.utils import normalize_workspace

    original_workspace = workspace
    workspace = normalize_workspace(workspace)
    errors: list[str] = []
    stats: dict[str, Any] = {
        "workspace": original_workspace,
        "graphs_dropped": [],
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

    # Drop all AGE graph schemas matching this workspace prefix
    try:
        dropped = await _drop_age_graphs_for_workspace(workspace, dry_run=dry_run)
        stats["graphs_dropped"] = dropped
    except Exception as exc:
        errors.append(f"AGE graphs: {exc}")

    # File system cleanup — workspace-scoped only (see areset Phase 5).
    if not keep_files:
        if input_dir:
            try:
                input_root = Path(input_dir).resolve()
                input_ws_dir = (input_root / workspace).resolve()
                try:
                    input_ws_dir.relative_to(input_root)
                except ValueError:
                    errors.append("Filesystem (input_dir): workspace path escapes input directory")
                else:
                    if input_ws_dir.exists() and input_ws_dir.is_dir():
                        if not dry_run:
                            shutil.rmtree(input_ws_dir, ignore_errors=True)
                        stats["local_files_removed"] += 1
            except Exception as exc:
                errors.append(f"Filesystem (input_dir): {exc}")

    logger.info("areset_orphaned complete for workspace=%s: %s", workspace, stats)
    return stats


async def _drop_age_graphs_for_workspace(
    workspace: str,
    *,
    dry_run: bool,
) -> list[str]:
    """Drop ALL AGE graph schemas whose name starts with ``{workspace}_``.

    Unlike ``_drop_age_graphs``, this does NOT cross-check against known
    workspaces — it is intended for orphan cleanup where the workspace
    no longer exists in metadata.

    Uses a direct ``asyncpg`` connection, bypassing LightRAG's process-wide pool.
    """
    import asyncpg

    from dlightrag.config import get_config

    config = get_config()
    dropped: list[str] = []

    try:
        conn = await asyncpg.connect(**config.pg_connection_kwargs())
    except Exception as exc:
        logger.warning("Failed to connect via asyncpg for orphan graph cleanup: %s", exc)
        return dropped

    try:
        escaped = workspace.replace("_", r"\_")
        rows = await conn.fetch(
            "SELECT name FROM ag_catalog.ag_graph WHERE name ILIKE $1 ESCAPE '\\'",
            f"{escaped}\\_%",
        )
        for row in rows:
            gname = row["name"]
            try:
                if not dry_run:
                    await conn.execute("SELECT ag_catalog.drop_graph($1, true)", gname)
                dropped.append(gname)
            except Exception as exc:
                logger.warning("Failed to drop graph %s: %s", gname, exc)
    finally:
        await conn.close()

    # Fallback: catch any remaining non-AGE schemas
    try:
        fallback_dropped = await _drop_workspace_schemas_via_pg(
            workspace, already_dropped=dropped, dry_run=dry_run
        )
        for schema_name in fallback_dropped:
            if schema_name not in dropped:
                dropped.append(schema_name)
    except Exception as exc:
        logger.warning("PG schema fallback failed for orphan workspace %s: %s", workspace, exc)

    return dropped


async def drop_global_orphaned_age_graphs(
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Drop AGE graph schemas that do NOT belong to any known workspace.

    Returns stats dict with ``dropped`` list and ``errors`` list.
    """
    errors: list[str] = []
    known = await _list_all_workspaces()
    dropped: list[str] = []

    import asyncpg

    from dlightrag.config import get_config

    config = get_config()

    try:
        conn = await asyncpg.connect(**config.pg_connection_kwargs())
    except Exception as exc:
        return {"dropped": [], "errors": [str(exc)]}

    try:
        rows = await conn.fetch("SELECT name FROM ag_catalog.ag_graph")
        for row in rows:
            gname = row["name"]
            belongs = any(
                gname.startswith(f"{ws}_") or gname == f"{ws}_chunk_entity_relation" for ws in known
            )
            if belongs:
                continue
            try:
                if not dry_run:
                    await conn.execute("SELECT ag_catalog.drop_graph($1, true)", gname)
                dropped.append(gname)
            except Exception as exc:
                errors.append(f"Failed to drop {gname}: {exc}")
                logger.warning("Global orphan graph drop failed for %s: %s", gname, exc)
    finally:
        await conn.close()

    logger.info("Global orphan graph cleanup: dropped=%s errors=%s", dropped, errors)
    return {"dropped": dropped, "errors": errors}
