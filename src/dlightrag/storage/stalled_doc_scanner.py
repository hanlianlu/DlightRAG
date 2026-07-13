# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Recover documents stalled in intermediate LightRAG pipeline states.

LightRAG's ``_validate_and_fix_document_consistency`` unconditionally resets
all PARSING/ANALYZING/PROCESSING/FAILED documents to PENDING whenever the
pipeline starts. This has two gaps for DlightRAG's long-running deployments:

1. **No pipeline trigger → no recovery.** If no new file upload triggers
   ``apipeline_process_enqueue_documents()``, stalled documents stay stuck
   forever.
2. **Unconditional reset is too coarse.** It resets documents that are
   actively being processed by another worker, and wipes FAILED documents
   that operators may want to inspect before retrying.

This module provides time-based stalled-document detection as a complementary
layer: only documents whose ``updated_at`` exceeds the configured timeout are
reset.  It reads directly from ``LIGHTRAG_DOC_STATUS`` via a transient asyncpg
connection — no dependency on an initialized LightRAG instance.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

_STALLED_STATUSES: frozenset[str] = frozenset({"parsing", "analyzing", "processing"})


async def recover_stalled_docs(
    *,
    workspace: str,
    timeout_seconds: int = 3600,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Find and reset documents stalled in intermediate processing states.

    Queries ``LIGHTRAG_DOC_STATUS`` for documents stuck in PARSING, ANALYZING,
    or PROCESSING whose ``updated_at`` is older than *timeout_seconds*.  Resets
    them to PENDING so the next pipeline run will retry them.

    Args:
        workspace: The workspace to scan.
        timeout_seconds: Age in seconds after which a doc is considered stalled.
        dry_run: If True, collect stats without executing any mutations.

    Returns:
        Stats dict with ``stalled_count``, ``reset_count``, and ``stalled_ids``.
    """
    import asyncpg

    from dlightrag.config import get_config

    config = get_config()
    errors: list[str] = []
    stalled_rows: list[dict[str, Any]] = []

    conn: asyncpg.Connection | None = None
    try:
        conn = await asyncpg.connect(**config.pg_connection_kwargs())
    except Exception as exc:
        error_msg = f"Failed to connect to PostgreSQL for stalled doc scan: {exc}"
        logger.warning(error_msg)
        return {
            "workspace": workspace,
            "stalled_count": 0,
            "reset_count": 0,
            "stalled_ids": [],
            "errors": [error_msg],
        }

    try:
        # 1. Query for stalled documents
        stalled_rows = await _fetch_stalled_docs(
            conn,
            workspace=workspace,
            timeout_seconds=timeout_seconds,
        )

        if not stalled_rows:
            return {
                "workspace": workspace,
                "stalled_count": 0,
                "reset_count": 0,
                "stalled_ids": [],
                "errors": [],
            }

        stalled_ids = [str(row["id"]) for row in stalled_rows]

        # 2. Reset stalled documents to PENDING
        if not dry_run:
            reset_count = await _reset_stalled_docs(
                conn,
                workspace=workspace,
                doc_ids=stalled_ids,
                timeout_seconds=timeout_seconds,
            )
        else:
            reset_count = 0

        if stalled_rows:
            status_counts: dict[str, int] = {}
            for row in stalled_rows:
                status = str(row.get("status") or "?")
                status_counts[status] = status_counts.get(status, 0) + 1
            logger.info(
                "Stalled doc scan for workspace '%s': found %d (%s), reset %d. Stalled IDs: %s",
                workspace,
                len(stalled_rows),
                ", ".join(f"{s}={c}" for s, c in sorted(status_counts.items())),
                reset_count,
                stalled_ids,
            )

        return {
            "workspace": workspace,
            "stalled_count": len(stalled_rows),
            "reset_count": reset_count,
            "stalled_ids": stalled_ids,
            "errors": errors,
        }

    except Exception as exc:
        error_msg = f"Stalled doc recovery failed for workspace '{workspace}': {exc}"
        logger.warning(error_msg, exc_info=True)
        return {
            "workspace": workspace,
            "stalled_count": len(stalled_rows),
            "reset_count": 0,
            "stalled_ids": [str(row["id"]) for row in stalled_rows],
            "errors": [*errors, error_msg],
        }
    finally:
        if conn is not None:
            await conn.close()


async def _fetch_stalled_docs(
    conn: Any,
    *,
    workspace: str,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    return await conn.fetch(
        "SELECT id, status, updated_at, file_path "
        "FROM LIGHTRAG_DOC_STATUS "
        "WHERE workspace = $1 "
        "  AND status = ANY($2::text[]) "
        "  AND updated_at < NOW() - ($3 * INTERVAL '1 second') "
        "ORDER BY updated_at ASC",
        workspace,
        list(_STALLED_STATUSES),
        timeout_seconds,
    )


async def _reset_stalled_docs(
    conn: Any,
    *,
    workspace: str,
    doc_ids: list[str],
    timeout_seconds: int,
) -> int:
    """Reset stalled docs to PENDING. Returns count of rows updated.

    The scan's age/status predicates are re-applied in the UPDATE itself so a
    document another worker moved back into an active state (or that finished)
    between the SELECT and this write is left untouched -- closing the
    check-then-act window in shared multi-instance deployments.
    """
    result = await conn.execute(
        "UPDATE LIGHTRAG_DOC_STATUS "
        "SET status = 'pending', "
        "    error_msg = '', "
        "    updated_at = NOW() "
        "WHERE workspace = $1 "
        "  AND id = ANY($2::text[]) "
        "  AND status = ANY($3::text[]) "
        "  AND updated_at < NOW() - ($4 * INTERVAL '1 second')",
        workspace,
        doc_ids,
        list(_STALLED_STATUSES),
        timeout_seconds,
    )
    # asyncpg execute returns a string like "UPDATE N" for UPDATE statements
    count = _parse_update_count(result)
    return count


def _parse_update_count(result: str) -> int:
    """Parse the row count from asyncpg's UPDATE command tag."""
    parts = result.strip().split()
    if len(parts) >= 2:
        try:
            return int(parts[-1])
        except ValueError, IndexError:
            pass
    return 0


__all__ = ["recover_stalled_docs"]
