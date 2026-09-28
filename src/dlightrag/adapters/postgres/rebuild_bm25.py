# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Offline workspace BM25 index and language-label rebuild command."""

import argparse
import asyncio
import logging

from dlightrag.adapters.postgres.core._pool import pg_pool
from dlightrag.adapters.postgres.corpus.corpus import build_pg_corpus_backend
from dlightrag.adapters.postgres.corpus.corpus_bm25 import (
    rebuild_postgres_bm25,
)
from dlightrag.application.config import DlightragConfig, get_config, load_config, set_config
from dlightrag.engine.rag.workspace.workspaces import (
    normalize_workspace,
    require_canonical_workspace_id,
)

DEFAULT_BATCH_SIZE = 500


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for `dlightrag-rebuild-bm25`."""
    parser = argparse.ArgumentParser(
        description="Offline DlightRAG workspace BM25 rebuild command",
        suggest_on_error=True,
    )
    parser.add_argument("--env-file", help="Path to .env configuration file")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm that DlightRAG and other writers are stopped.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Chunks per language-label update batch (default: {DEFAULT_BATCH_SIZE}).",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate offline rebuild flags."""
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if not args.yes:
        raise SystemExit("--yes is required; stop DlightRAG writers first")


def canonical_workspace_config(config: DlightragConfig) -> DlightragConfig:
    """Address the configured workspace by the canonical id the service stores it under.

    The service reaches a workspace only through the id ``normalize_workspace``
    derives from its label, so an offline command given ``My Space`` must address
    ``my_space`` rather than rows no service ever wrote. A label that yields no
    canonical id is refused before any storage is opened.
    """
    label = config.deployment.workspace
    try:
        workspace_id = require_canonical_workspace_id(normalize_workspace(label))
    except ValueError:
        raise SystemExit(
            f"deployment.workspace {label!r} does not normalize to a workspace id "
            "(1-64 letters, digits, or underscores)"
        ) from None
    if workspace_id == label:
        return config
    deployment = config.deployment.model_copy(update={"workspace": workspace_id})
    return config.model_copy(update={"deployment": deployment})


async def run_rebuild_bm25(
    *,
    config: DlightragConfig | None = None,
    assume_yes: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, int]:
    """Provision configured BM25 indexes and relabel all workspace chunks."""
    if not assume_yes:
        raise SystemExit("--yes is required; stop DlightRAG writers first")
    if batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    resolved_config = canonical_workspace_config(config or get_config())
    if not resolved_config.corpus.retrieval.bm25_enabled:
        raise SystemExit("Set bm25_enabled=true before rebuilding workspace BM25")
    if resolved_config.is_reader:
        raise SystemExit("BM25 rebuild requires the writer service role")

    pg_pool.bind(resolved_config)
    try:
        backend = build_pg_corpus_backend(resolved_config)
        async with backend.coordination.workspace_initialization():
            return await rebuild_postgres_bm25(
                resolved_config,
                batch_size=batch_size,
            )
    finally:
        await pg_pool.close()


def main() -> None:
    """Entry point for `dlightrag-rebuild-bm25`."""
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    config = load_config(args.env_file) if args.env_file else get_config()
    set_config(config)
    logging.basicConfig(
        level=getattr(logging, config.observability.log_level.upper(), logging.INFO)
    )
    stats = asyncio.run(
        run_rebuild_bm25(
            config=config,
            assume_yes=args.yes,
            batch_size=args.batch_size,
        )
    )
    print(
        "BM25 language labels: "
        f"{stats['processed_chunks']} scanned, {stats['updated_chunks']} updated"
    )


__all__ = [
    "build_parser",
    "canonical_workspace_config",
    "main",
    "run_rebuild_bm25",
    "validate_args",
]
