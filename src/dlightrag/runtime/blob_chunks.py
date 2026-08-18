# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Deterministic owner-scoped blob chunk planning.

Blobs are content-addressed: one digest over the complete byte string, chunked
into exactly 1,048,576 bytes per non-final chunk. The chunk plan is a pure
function of the content, so retries and re-uploads produce identical layouts
and dedupe on the same digest (M3-D15, M3-D21).
"""

import hashlib
from dataclasses import dataclass

BLOB_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class BlobChunkPlan:
    """One deterministic chunk layout plus the complete-content digest."""

    digest: str
    total_bytes: int
    chunk_ranges: tuple[tuple[int, int], ...]

    @property
    def chunk_count(self) -> int:
        return len(self.chunk_ranges)

    def chunk(self, content: bytes, index: int) -> bytes:
        """Return one chunk's exact bytes by zero-based index."""
        if index < 0 or index >= len(self.chunk_ranges):
            raise IndexError("blob chunk index out of range")
        start, end = self.chunk_ranges[index]
        return content[start:end]


def blob_digest(content: bytes) -> str:
    """Return the SHA-256 content address for one complete blob."""
    return hashlib.sha256(content).hexdigest()


def plan_blob(content: bytes) -> BlobChunkPlan:
    """Return the deterministic chunk plan for one complete blob."""
    total = len(content)
    if total == 0:
        return BlobChunkPlan(digest=blob_digest(content), total_bytes=0, chunk_ranges=())
    ranges = tuple(
        (start, min(start + BLOB_CHUNK_BYTES, total)) for start in range(0, total, BLOB_CHUNK_BYTES)
    )
    return BlobChunkPlan(
        digest=blob_digest(content),
        total_bytes=total,
        chunk_ranges=ranges,
    )


def verify_blob_chunks(content: bytes, plan: BlobChunkPlan) -> None:
    """Raise when content bytes do not reproduce the declared chunk plan."""
    rebuilt = b"".join(plan.chunk(content, index) for index in range(plan.chunk_count))
    if rebuilt != content or plan.total_bytes != len(content):
        raise ValueError("blob chunks do not reconstruct the complete content")
    if blob_digest(content) != plan.digest:
        raise ValueError("blob content does not match its declared digest")


__all__ = [
    "BLOB_CHUNK_BYTES",
    "BlobChunkPlan",
    "blob_digest",
    "plan_blob",
    "verify_blob_chunks",
]
