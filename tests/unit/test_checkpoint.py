# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for ConversationCheckpoint."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from dlightrag.core.checkpoint import ConversationCheckpoint


class TestCheckpointInit:
    def test_db_created_on_first_write(self, tmp_path: Path) -> None:
        db_path = tmp_path / "checkpoints.db"
        cp = ConversationCheckpoint(db_path)
        assert not db_path.exists()

        cp._ensure_db_sync()
        assert db_path.exists()

    def test_schema_creates_all_tables(self, tmp_path: Path) -> None:
        db_path = tmp_path / "checkpoints.db"
        cp = ConversationCheckpoint(db_path)
        cp._ensure_db_sync()

        conn = sqlite3.connect(str(db_path))
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        assert "sessions" in tables
        assert "turns" in tables
        assert "context_anchors" in tables
        conn.close()

    def test_wal_mode_enabled(self, tmp_path: Path) -> None:
        db_path = tmp_path / "checkpoints.db"
        cp = ConversationCheckpoint(db_path)
        cp._ensure_db_sync()

        conn = sqlite3.connect(str(db_path))
        journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert journal.upper() == "WAL"
        conn.close()

    def test_ensure_db_idempotent(self, tmp_path: Path) -> None:
        db_path = tmp_path / "checkpoints.db"
        cp = ConversationCheckpoint(db_path)
        cp._ensure_db_sync()
        cp._ensure_db_sync()
        cp._ensure_db_sync()
        assert db_path.exists()
