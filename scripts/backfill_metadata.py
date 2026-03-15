# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Backfill PGMetadataIndex from existing hash_index entries.

Usage:
    uv run python scripts/backfill_metadata.py [--workspace default] [--dry-run]

Scans all documents in hash_index and upserts system metadata
(filename, stem, extension, rag_mode) into PGMetadataIndex.
Skips entries that already exist in the metadata index.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def backfill(workspace: str, dry_run: bool = False) -> None:
    from dlightrag.config import get_config
    from dlightrag.core.retrieval.metadata_index import PGMetadataIndex, extract_system_metadata

    config = get_config()

    if not config.kv_storage.startswith("PG"):
        logger.error("Metadata backfill requires PostgreSQL backend (kv_storage starts with 'PG')")
        sys.exit(1)

    # Create hash index (same logic as service layer)
    from dlightrag.core.ingestion.hash_index import PGHashIndex

    hash_index = PGHashIndex(workspace=workspace)
    await hash_index.initialize()

    # Create metadata index
    metadata_index = PGMetadataIndex(workspace=workspace)
    await metadata_index.initialize()

    # List all existing documents
    entries = await hash_index.list_all()
    logger.info("Found %d documents in hash_index (workspace=%s)", len(entries), workspace)

    if not entries:
        logger.info("Nothing to backfill.")
        return

    upserted = 0
    skipped = 0
    errors = 0

    for entry in entries:
        doc_id = entry.get("doc_id", "")
        file_path = entry.get("file_path", "")

        if not doc_id or not file_path:
            logger.warning("Skipping entry with missing doc_id or file_path: %s", entry)
            skipped += 1
            continue

        # Check if already exists
        existing = await metadata_index.get(doc_id)
        if existing:
            skipped += 1
            continue

        # Extract system metadata from file path
        meta = extract_system_metadata(file_path, rag_mode=config.rag_mode)

        if dry_run:
            logger.info("[DRY RUN] Would upsert: doc_id=%s filename=%s", doc_id, meta["filename"])
            upserted += 1
            continue

        try:
            await metadata_index.upsert(doc_id, meta)
            upserted += 1
            logger.info("Upserted: %s -> %s", meta["filename"], doc_id[:20])
        except Exception as e:
            logger.warning("Failed to upsert %s: %s", doc_id, e)
            errors += 1

    logger.info(
        "Backfill complete: %d upserted, %d skipped (already exist), %d errors",
        upserted,
        skipped,
        errors,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill PGMetadataIndex from hash_index")
    parser.add_argument("--workspace", default="default", help="Workspace name (default: default)")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done without writing"
    )
    args = parser.parse_args()
    asyncio.run(backfill(args.workspace, args.dry_run))


if __name__ == "__main__":
    main()
