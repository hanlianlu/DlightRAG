# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Offline workspace BM25 index and language-label rebuild command."""

import argparse
import asyncio
import logging

import asyncpg

from dlightrag.config import DlightragConfig, get_config, load_config, set_config
from dlightrag.core.retrieval.bm25 import (
    BM25Profile,
    profiles_from_config,
    rebuild_postgres_bm25,
    required_postgres_extensions,
)
from dlightrag.storage.pool import pg_pool
from dlightrag.storage.postgres_version import ensure_postgres_extensions, ensure_postgres_major

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


async def _ensure_bm25_prerequisites(
    config: DlightragConfig,
    profiles: tuple[BM25Profile, ...],
) -> None:
    conn = await asyncpg.connect(**config.pg_connection_kwargs())
    try:
        await ensure_postgres_major(conn)
        await ensure_postgres_extensions(conn, required_postgres_extensions(profiles))
    finally:
        await conn.close()


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

    resolved_config = config or get_config()
    if not resolved_config.bm25_enabled:
        raise SystemExit("Set bm25_enabled=true before rebuilding workspace BM25")
    if resolved_config.is_reader:
        raise SystemExit("BM25 rebuild requires the writer service role")

    profiles = profiles_from_config(resolved_config.bm25_profiles)
    pg_pool.bind(resolved_config)
    try:
        await _ensure_bm25_prerequisites(resolved_config, profiles)
        return await rebuild_postgres_bm25(
            resolved_config,
            pool=pg_pool,
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
    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO))
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
    "main",
    "run_rebuild_bm25",
    "validate_args",
]
