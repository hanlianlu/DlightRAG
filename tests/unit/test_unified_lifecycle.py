# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for extracted unified lifecycle helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from dlightrag.unifiedrepresent.lifecycle import unified_ingest


def _make_config() -> MagicMock:
    config = MagicMock()
    config.ingestion_replace_default = False
    config.blob_connection_string = "fake-conn-str"
    config.temp_dir = Path("/tmp/dlightrag_test")
    return config


def _make_engine() -> MagicMock:
    engine = MagicMock()
    engine.aingest = AsyncMock(return_value={
        "doc_id": "doc-001", "page_count": 2, "file_path": "/fake/test.pdf",
    })
    return engine


def _make_hash_index() -> MagicMock:
    idx = MagicMock()
    idx.should_skip_file = AsyncMock(return_value=(False, "abc123", None))
    idx.register = AsyncMock()
    return idx


class TestUnifiedIngestLocal:
    async def test_single_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.pdf"
        f.write_bytes(b"fake pdf")

        engine = _make_engine()
        result = await unified_ingest(
            engine=engine, config=_make_config(),
            hash_index=_make_hash_index(), source_type="local", path=str(f),
        )
        engine.aingest.assert_awaited_once()
        assert result["doc_id"] == "doc-001"

    async def test_single_file_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "test.pdf"
        f.write_bytes(b"fake pdf")
        hash_index = _make_hash_index()
        hash_index.should_skip_file = AsyncMock(return_value=(True, "abc123", "already ingested"))

        result = await unified_ingest(
            engine=_make_engine(), config=_make_config(),
            hash_index=hash_index, source_type="local", path=str(f),
        )
        assert result["status"] == "skipped"

    async def test_directory(self, tmp_path: Path) -> None:
        (tmp_path / "a.pdf").write_bytes(b"pdf1")
        (tmp_path / "b.pdf").write_bytes(b"pdf2")
        (tmp_path / "c.txt").write_bytes(b"ignored")

        engine = _make_engine()
        result = await unified_ingest(
            engine=engine, config=_make_config(),
            hash_index=_make_hash_index(), source_type="local", path=str(tmp_path),
        )
        assert result["status"] == "success"
        assert result["total_files"] == 2
        assert engine.aingest.await_count == 2

    async def test_path_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            await unified_ingest(
                engine=_make_engine(), config=_make_config(),
                hash_index=_make_hash_index(), source_type="local",
                path="/nonexistent/file.pdf",
            )
