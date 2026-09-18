# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Read-only Agent Workspace audit command: report orphans, delete nothing."""

import argparse
import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any

from dlightrag.adapters.postgres.core._pool import pg_pool
from dlightrag.application.config import DlightragConfig, get_config, load_config, set_config
from dlightrag.engine.answer.workspace import (
    RunWorkspaceAudit,
    audit_run_workspaces,
    resolve_workspace_root,
)

DEFAULT_SAMPLE = 20


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for `dlightrag-workspace-audit`."""
    parser = argparse.ArgumentParser(
        description=(
            "Report Agent Workspace roots whose Run row is gone. Read-only: this "
            "command never deletes; the runtime's orphan sweep is what removes them."
        ),
        suggest_on_error=True,
    )
    parser.add_argument("--env-file", help="Path to .env configuration file")
    parser.add_argument(
        "--root",
        help=(
            "Workspace root to audit. Defaults to the configured one; name a path "
            "directly to audit trees an earlier configuration left behind."
        ),
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=DEFAULT_SAMPLE,
        help=f"Orphan paths to list (default: {DEFAULT_SAMPLE}).",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate audit flags."""
    if args.sample < 0:
        raise SystemExit("--sample must be >= 0")
    if args.root is not None:
        root = Path(args.root).expanduser()
        if not root.is_absolute():
            raise SystemExit("--root must be an absolute path")


class _RunExistenceProbe:
    """The one read an audit needs, so it never composes the runtime stores."""

    async def get_run_global(self, *, run_id: str) -> object | None:
        async def probe(conn: Any) -> object | None:
            return await conn.fetchval(
                "SELECT 1 FROM dlightrag_runs WHERE run_id = $1::uuid", uuid.UUID(run_id)
            )

        return await pg_pool.run_once(probe)


async def run_audit(*, config: DlightragConfig, root: Path, sample: int) -> RunWorkspaceAudit:
    """Audit one workspace root against the operational store."""
    pg_pool.bind(config)
    try:
        return await audit_run_workspaces(
            workspace_root=root, store=_RunExistenceProbe(), sample=sample
        )
    finally:
        await pg_pool.close()


def main() -> None:
    """Entry point for `dlightrag-workspace-audit`."""
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    config = load_config(args.env_file) if args.env_file else get_config()
    set_config(config)
    logging.basicConfig(
        level=getattr(logging, config.observability.log_level.upper(), logging.INFO)
    )
    root = (
        Path(args.root).expanduser().resolve()
        if args.root
        else resolve_workspace_root(
            execution_environment=config.answer.agent.execution_environment,
            workspace_root=config.answer.agent.workspace_root,
        )
    )
    if root is None:
        print(
            "No Agent Workspace root: this deployment has execution disabled and no "
            "named root, so it owns no workspace path. Pass --root to audit one."
        )
        return
    report = asyncio.run(run_audit(config=config, root=root, sample=args.sample))
    print(
        f"Agent Workspace audit of {root}: {report.roots} run root(s), "
        f"{len(report.orphans)} without a Run row, {report.unreadable} unreadable. "
        "Nothing was deleted."
    )
    for path in report.orphans:
        print(f"  orphan: {path}")


__all__ = ["build_parser", "main", "run_audit", "validate_args"]
