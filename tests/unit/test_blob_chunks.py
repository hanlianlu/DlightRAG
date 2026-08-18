# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for deterministic blob chunk planning and verification."""

import hashlib

import pytest

from dlightrag.runtime.blob_chunks import (
    BLOB_CHUNK_BYTES,
    blob_digest,
    plan_blob,
    verify_blob_chunks,
)


def test_one_mib_plus_one_byte_makes_exactly_two_chunks() -> None:
    content = b"x" * (BLOB_CHUNK_BYTES + 1)
    plan = plan_blob(content)

    assert plan.chunk_count == 2
    assert plan.total_bytes == BLOB_CHUNK_BYTES + 1
    assert plan.chunk_ranges == (
        (0, BLOB_CHUNK_BYTES),
        (BLOB_CHUNK_BYTES, BLOB_CHUNK_BYTES + 1),
    )
    assert plan.chunk(content, 0) == b"x" * BLOB_CHUNK_BYTES
    assert plan.chunk(content, 1) == b"x"
    assert plan.digest == hashlib.sha256(content).hexdigest()


def test_exact_chunk_multiple_has_no_empty_final_chunk() -> None:
    content = b"y" * (2 * BLOB_CHUNK_BYTES)
    plan = plan_blob(content)

    assert plan.chunk_count == 2
    assert plan.chunk_ranges == (
        (0, BLOB_CHUNK_BYTES),
        (BLOB_CHUNK_BYTES, 2 * BLOB_CHUNK_BYTES),
    )


def test_small_and_empty_blobs() -> None:
    small = plan_blob(b"short")
    assert small.chunk_count == 1
    assert small.chunk(b"short", 0) == b"short"

    empty = plan_blob(b"")
    assert empty.chunk_count == 0
    assert empty.total_bytes == 0
    assert empty.digest == blob_digest(b"")


def test_plan_is_pure_and_deterministic() -> None:
    content = bytes(range(256)) * 4096
    assert plan_blob(content) == plan_blob(content)
    assert blob_digest(content) == blob_digest(content)


def test_verification_accepts_intact_and_rejects_tampered_content() -> None:
    content = b"z" * (BLOB_CHUNK_BYTES + 7)
    plan = plan_blob(content)
    verify_blob_chunks(content, plan)

    tampered = content[:-1] + b"!"
    with pytest.raises(ValueError):
        verify_blob_chunks(tampered, plan)

    replaced = plan_blob(b"w" * len(content))
    with pytest.raises(ValueError):
        verify_blob_chunks(content, replaced)


def test_chunk_index_bounds() -> None:
    plan = plan_blob(b"abc")
    with pytest.raises(IndexError):
        plan.chunk(b"abc", 1)
