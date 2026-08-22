#!/usr/bin/env python3
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Reset one or every authorized DlightRAG Corpus Workspace.

Usage:
    uv run scripts/reset_workspace.py
    uv run scripts/reset_workspace.py --workspace project-a
    uv run scripts/reset_workspace.py --all
    uv run scripts/reset_workspace.py --dry-run
    uv run scripts/reset_workspace.py --keep-files
    uv run scripts/reset_workspace.py -y
"""

import argparse
import asyncio
import logging
import sys
from typing import Any

from dlightrag.access import AccessAction, AccessGate, AllowAllAccessControl
from dlightrag.rag.workspaces import normalize_workspace
from dlightrag.services.corpora import CorpusAdmin, CorpusResetResult

logger = logging.getLogger(__name__)


def _print_workspace_result(workspace: str, result: dict[str, Any]) -> None:
    """Pretty-print reset results for one Corpus Workspace."""
    if "error" in result:
        print(f"\n  [{workspace}] ERROR: {result['error']}")
        return

    print(f"\n  [{workspace}]")
    if "lightrag_storages_dropped" in result:
        print(f"    LightRAG storages dropped: {result['lightrag_storages_dropped']}")
    cleared = result.get("domain_stores_dropped", [])
    if cleared:
        print(f"    DlightRAG stores cleared: {', '.join(cleared)}")
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
        for error in errors:
            print(f"      - {error}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dlightrag-reset-workspace",
        description="Reset DlightRAG Corpus Workspace data",
        suggest_on_error=True,
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without deleting")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--keep-files", action="store_true", help="Drop storages, keep local files")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--workspace", default=None, help="Target Corpus Workspace")
    scope.add_argument(
        "--all",
        dest="reset_all",
        action="store_true",
        help="Reset every authorized Corpus Workspace",
    )
    return parser


async def _authorized_reset_scope(
    corpora: CorpusAdmin,
    gate: AccessGate,
    *,
    workspace: str | None,
    reset_all: bool,
    default_workspace: str,
) -> tuple[str, ...]:
    """Expand the trusted host command into an explicit authorized scope."""
    if reset_all:
        records = await gate.filter_workspace_records(
            AccessAction.WORKSPACE_RESET,
            await corpora.alist_workspace_records(),
        )
        workspace_ids = tuple(record["workspace"] for record in records)
        if not workspace_ids:
            raise ValueError("No authorized Corpus Workspaces are available to reset")
        return workspace_ids

    raw_workspace = workspace if workspace is not None else default_workspace
    if not raw_workspace.strip() or raw_workspace.strip() == "*":
        raise ValueError("--workspace requires a concrete workspace name")
    workspace_id = normalize_workspace(raw_workspace)
    await gate.check(AccessAction.WORKSPACE_RESET, workspace=workspace_id)
    return (workspace_id,)


async def _reset_with_corpora(
    corpora: CorpusAdmin,
    gate: AccessGate,
    *,
    workspace: str | None,
    reset_all: bool,
    default_workspace: str,
    keep_files: bool,
    dry_run: bool,
) -> CorpusResetResult:
    workspace_ids = await _authorized_reset_scope(
        corpora,
        gate,
        workspace=workspace,
        reset_all=reset_all,
        default_workspace=default_workspace,
    )
    return await corpora.reset(
        workspace_ids=workspace_ids,
        keep_files=keep_files,
        dry_run=dry_run,
    )


async def _run(
    *,
    workspace: str | None,
    reset_all: bool,
    keep_files: bool,
    dry_run: bool,
) -> CorpusResetResult:
    from dlightrag.application import Application
    from dlightrag.config import get_config

    config = get_config()
    print("\nCorpus storage backends (from config):")
    print(f"  KV:         {config.kv_storage}")
    print(f"  Vector:     {config.vector_storage}")
    print(f"  Graph:      {config.graph_storage}")
    print(f"  Default:    {config.workspace}")

    application = await Application.acreate(config)
    try:
        return await _reset_with_corpora(
            application.corpora,
            AccessGate(AllowAllAccessControl(), None),
            workspace=workspace,
            reset_all=reset_all,
            default_workspace=config.workspace,
            keep_files=keep_files,
            dry_run=dry_run,
        )
    finally:
        await application.aclose()


def main() -> int:
    args = build_parser().parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if args.dry_run:
        print("\n(dry run - nothing will be deleted)")

    if not args.dry_run and not args.yes:
        scope = (
            "every authorized Corpus Workspace"
            if args.reset_all
            else (args.workspace or "the default Corpus Workspace")
        )
        print(f"\nWARNING: This will permanently delete Corpus data in {scope}.")
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

    for workspace, workspace_result in result.get("workspaces", {}).items():
        _print_workspace_result(workspace, workspace_result)

    total_errors = result.get("total_errors", 0)
    print(f"\nDone. Total errors: {total_errors}")
    if args.dry_run:
        print("Run without --dry-run to actually delete.")
    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
