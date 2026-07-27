#!/usr/bin/env python3
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Reset dlightrag RAG storage — drops all storage backends and local files.

Thin CLI wrapper around RAGServiceManager.areset(). All reset logic
lives in RAGService.areset() (6-phase reset).

Usage:
    uv run scripts/reset.py                      # reset default workspace (with confirmation)
    uv run scripts/reset.py --workspace project-a # reset specific workspace
    uv run scripts/reset.py --all                 # reset ALL workspaces
    uv run scripts/reset.py --dry-run             # preview what would be dropped
    uv run scripts/reset.py --keep-files          # drop storages, keep local files
    uv run scripts/reset.py -y                    # skip confirmation prompt
"""

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
    if "lightrag_storages_dropped" in result:
        print(f"    LightRAG storages dropped: {result['lightrag_storages_dropped']}")
    cleared = result.get("domain_stores_dropped", [])
    if cleared:
        print(f"    DlightRAG stores cleared: {', '.join(cleared)}")
    graphs = result.get("graphs_dropped", [])
    if graphs:
        print(f"    AGE graphs dropped: {', '.join(graphs)}")
    pending_tasks = result.get("pending_tasks_cancelled", 0)
    if pending_tasks:
        print(f"    Pending tasks cancelled: {pending_tasks}")
    ingest_jobs = result.get("ingest_jobs_cancelled", 0)
    if ingest_jobs:
        print(f"    Ingest jobs cancelled: {ingest_jobs}")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dlightrag-reset",
        description="Reset dlightrag RAG storage (all configured backends)",
        suggest_on_error=True,
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without deleting")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--keep-files", action="store_true", help="Drop storages, keep local files")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--workspace", default=None, help="Target workspace (default: from config)")
    scope.add_argument("--all", dest="reset_all", action="store_true", help="Reset ALL workspaces")
    return parser


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

    manager = await RAGServiceManager.acreate(config)
    try:
        target_ws = None if reset_all else (workspace or config.workspace)
        result = await manager.areset(
            workspace=target_ws,
            keep_files=keep_files,
            dry_run=dry_run,
        )
        return result
    finally:
        await manager.aclose()


def main() -> int:
    args = build_parser().parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

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
        except EOFError, KeyboardInterrupt:
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
