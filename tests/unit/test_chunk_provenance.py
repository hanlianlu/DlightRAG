# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for chunk provenance validation and PG store guards."""

from __future__ import annotations

import pytest

from dlightrag.storage.chunk_provenance import (
    PGChunkProvenance,
    validate_chunk_provenance_record,
)


def test_chunk_provenance_accepts_native_image() -> None:
    validate_chunk_provenance_record(
        {
            "chunk_id": "chunk-native-1",
            "full_doc_id": "doc-1",
            "embedding_input_kind": "image",
            "sidecar_type": "native_image",
        }
    )


def test_chunk_provenance_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="embedding_input_kind"):
        validate_chunk_provenance_record(
            {
                "chunk_id": "chunk-1",
                "full_doc_id": "doc-1",
                "embedding_input_kind": "audio",
            }
        )


def test_chunk_provenance_rejects_unknown_sidecar_type() -> None:
    with pytest.raises(ValueError, match="sidecar_type"):
        validate_chunk_provenance_record(
            {
                "chunk_id": "chunk-1",
                "full_doc_id": "doc-1",
                "embedding_input_kind": "image",
                "sidecar_type": "screenshot",
            }
        )


async def test_chunk_provenance_clear_requires_initialize() -> None:
    store = PGChunkProvenance(workspace="default")

    with pytest.raises(RuntimeError, match="not initialized"):
        await store.clear()
