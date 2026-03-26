# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for PGChunkMapping schema and SQL constants."""

from dlightrag.core.retrieval.chunk_mapping import _CREATE_TABLE, _INSERT, PGChunkMapping


class TestPGChunkMappingInit:
    def test_create_instance(self):
        cm = PGChunkMapping(workspace="test")
        assert cm._workspace == "test"

    def test_sql_constants_defined(self):
        assert "dlightrag_chunk_mapping" in _CREATE_TABLE
        assert "ON CONFLICT" in _INSERT

    def test_not_initialized_raises(self):
        cm = PGChunkMapping(workspace="test")
        import pytest

        with pytest.raises(RuntimeError, match="not initialized"):
            cm._get_pool()
