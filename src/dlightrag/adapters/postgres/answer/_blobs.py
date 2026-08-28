# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Private PostgreSQL persistence for complete owner-scoped blobs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from dlightrag.engine.runtime.blob_chunks import plan_blob

_INSERT_BLOB_METADATA = """
INSERT INTO dlightrag_blobs (owner_id, digest, byte_size)
VALUES ($1, $2, $3)
ON CONFLICT (owner_id, digest) DO NOTHING
"""

_INSERT_BLOB_CHUNKS = """
WITH authoritative AS (
    SELECT byte_size
    FROM dlightrag_blobs
    WHERE owner_id = $1 AND digest = $2
), inserted AS (
    INSERT INTO dlightrag_blob_chunks (owner_id, digest, chunk_index, content)
    SELECT $1, $2, chunks.ordinality - 1, chunks.content
    FROM unnest($3::bytea[]) WITH ORDINALITY AS chunks(content, ordinality)
    CROSS JOIN authoritative
    WHERE authoritative.byte_size = $4
    ON CONFLICT (owner_id, digest, chunk_index) DO NOTHING
    RETURNING 1
)
SELECT byte_size FROM authoritative
"""


class BlobSizeConflict(Exception):
    """The owner already has this digest with a different authoritative size."""


async def write_blob_content(
    conn: Any,
    *,
    owner_id: str,
    digest: str,
    content: bytes,
) -> None:
    """Persist complete contiguous content without copying every chunk in Python."""
    plan = plan_blob(content)
    view = memoryview(content)
    chunks = tuple(view[start:end] for start, end in plan.chunk_ranges)
    await write_complete_blob(
        conn,
        owner_id=owner_id,
        digest=digest,
        total_bytes=plan.total_bytes,
        chunks=chunks,
    )


async def write_complete_blob(
    conn: Any,
    *,
    owner_id: str,
    digest: str,
    total_bytes: int,
    chunks: Sequence[bytes | memoryview],
) -> None:
    """Persist one complete blob in two set-based statements.

    ``DO NOTHING`` avoids holding update locks on deduplicated metadata rows.
    It also waits for a concurrent insert of the same identity to settle. The
    following statement receives a fresh Read Committed snapshot, gates every
    chunk on the authoritative size, and returns that size for conflict
    detection without a third round trip.
    """
    await conn.execute(
        _INSERT_BLOB_METADATA,
        owner_id,
        digest,
        total_bytes,
    )
    authoritative_size = await conn.fetchval(
        _INSERT_BLOB_CHUNKS,
        owner_id,
        digest,
        chunks,
        total_bytes,
    )
    if authoritative_size != total_bytes:
        raise BlobSizeConflict("blob digest collision with a different byte size")
