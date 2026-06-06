# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for ConversationCheckpoint."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

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
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "sessions" in tables
        assert "turns" in tables
        assert "context_anchors" in tables
        conn.close()

    def test_wal_mode_enabled(self, tmp_path: Path) -> None:
        db_path = tmp_path / "checkpoints.db"
        cp = ConversationCheckpoint(db_path)
        conn = cp._ensure_db_sync()
        try:
            journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
            assert journal.upper() == "WAL"
        finally:
            conn.close()

    def test_connection_pragmas_are_configured(self, tmp_path: Path) -> None:
        db_path = tmp_path / "checkpoints.db"
        cp = ConversationCheckpoint(db_path)
        conn = cp._ensure_db_sync()
        try:
            assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
            assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 30000
            assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1
        finally:
            conn.close()

    def test_ensure_db_idempotent(self, tmp_path: Path) -> None:
        db_path = tmp_path / "checkpoints.db"
        cp = ConversationCheckpoint(db_path)
        cp._ensure_db_sync()
        cp._ensure_db_sync()
        cp._ensure_db_sync()
        assert db_path.exists()


@pytest.mark.asyncio
class TestCheckpointCRUD:
    async def test_ensure_session_and_save_turns(self, tmp_path: Path) -> None:
        db_path = tmp_path / "checkpoints.db"
        cp = ConversationCheckpoint(db_path)

        await cp.ensure_session("s1", workspace="default")
        await cp.save_turn("s1", 1, role="user", content="Hello")
        await cp.save_turn(
            "s1", 1, role="assistant", content="Hi there!", cited_chunk_ids=["c1", "c2"]
        )

        history = await cp.get_history("s1")
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "Hello"}
        assert history[1] == {"role": "assistant", "content": "Hi there!"}

    async def test_next_turn_number(self, tmp_path: Path) -> None:
        db_path = tmp_path / "checkpoints.db"
        cp = ConversationCheckpoint(db_path)

        assert await cp.next_turn_number("s1") == 1

        await cp.ensure_session("s1")
        await cp.save_turn("s1", 1, role="user", content="q1")
        await cp.save_turn("s1", 1, role="assistant", content="a1")

        assert await cp.next_turn_number("s1") == 2

    async def test_save_and_mark_anchors(self, tmp_path: Path) -> None:
        db_path = tmp_path / "checkpoints.db"
        cp = ConversationCheckpoint(db_path)

        chunks = [
            {
                "chunk_id": "c1",
                "file_path": "/docs/a.pdf",
                "sidecar": {"type": "drawing", "id": "im-1"},
                "relevance_score": 0.85,
            },
            {
                "chunk_id": "c2",
                "file_path": "/docs/b.pdf",
                "sidecar": None,
                "relevance_score": 0.42,
            },
        ]
        await cp.ensure_session("s1")
        await cp.save_anchors("s1", 1, chunks)
        await cp.mark_cited("s1", 1, ["c1"])

        anchors = await cp.get_previous_anchors("s1", last_n_turns=1)
        assert len(anchors) == 2
        cited = {a["chunk_id"] for a in anchors if a["was_cited"]}
        assert cited == {"c1"}

        cited_ids = await cp.get_cited_chunk_ids("s1")
        assert cited_ids == {"c1"}

    async def test_save_anchors_requires_existing_session(self, tmp_path: Path) -> None:
        db_path = tmp_path / "checkpoints.db"
        cp = ConversationCheckpoint(db_path)

        with pytest.raises(sqlite3.IntegrityError):
            await cp.save_anchors("missing-session", 1, [{"chunk_id": "c1"}])

    async def test_delete_session_cascades(self, tmp_path: Path) -> None:
        db_path = tmp_path / "checkpoints.db"
        cp = ConversationCheckpoint(db_path)

        await cp.ensure_session("s1")
        await cp.save_turn("s1", 1, role="user", content="q")
        await cp.save_anchors("s1", 1, [{"chunk_id": "c1"}])

        await cp.delete_session("s1")

        history = await cp.get_history("s1")
        assert history == []
        anchors = await cp.get_previous_anchors("s1", last_n_turns=1)
        assert anchors == []

    async def test_list_sessions(self, tmp_path: Path) -> None:
        db_path = tmp_path / "checkpoints.db"
        cp = ConversationCheckpoint(db_path)

        await cp.ensure_session("s1", workspace="ws-a")
        await cp.ensure_session("s2", workspace="ws-b")
        await cp.save_turn("s1", 1, role="user", content="q")

        all_sessions = await cp.list_sessions()
        assert len(all_sessions) == 2

        filtered = await cp.list_sessions(workspace="ws-a")
        assert len(filtered) == 1
        assert filtered[0]["session_id"] == "s1"
        assert filtered[0]["turn_count"] == 1

    async def test_delete_sessions_by_workspace(self, tmp_path: Path) -> None:
        db_path = tmp_path / "checkpoints.db"
        cp = ConversationCheckpoint(db_path)

        await cp.ensure_session("s1", workspace="ws-a")
        await cp.ensure_session("s2", workspace="ws-b")
        await cp.save_turn("s1", 1, role="user", content="q")
        await cp.save_turn("s2", 1, role="user", content="q")

        deleted = await cp.delete_sessions_by_workspace("ws-a")
        assert deleted == 1

        assert await cp.get_history("s1") == []
        assert len(await cp.get_history("s2")) == 1

    async def test_prune_old_sessions(self, tmp_path: Path) -> None:
        db_path = tmp_path / "checkpoints.db"
        cp = ConversationCheckpoint(db_path)

        await cp.ensure_session("s1")
        await cp.save_turn("s1", 1, role="user", content="q")

        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "UPDATE sessions SET updated_at = datetime('now', '-60 days') WHERE session_id = 's1'"
        )
        conn.commit()
        conn.close()

        deleted = await cp.prune_old_sessions(max_age_days=30)
        assert deleted == 1
        assert await cp.get_history("s1") == []

    async def test_prune_old_sessions_binds_age_modifier(self, monkeypatch, tmp_path: Path) -> None:
        cp = ConversationCheckpoint(tmp_path / "checkpoints.db")

        class Cursor:
            rowcount = 0

        class Conn:
            def __init__(self) -> None:
                self.calls: list[tuple[str, tuple[object, ...]]] = []
                self.committed = False
                self.closed = False

            def execute(self, query: str, params: tuple[object, ...] = ()) -> Cursor:
                self.calls.append((query, params))
                return Cursor()

            def commit(self) -> None:
                self.committed = True

            def close(self) -> None:
                self.closed = True

        conn = Conn()
        monkeypatch.setattr(cp, "_ensure_db_sync", lambda: conn)

        deleted = cp._prune_old_sessions_sync(30)

        assert deleted == 0
        assert conn.calls == [
            (
                "DELETE FROM sessions WHERE updated_at < datetime('now', ?)",
                ("-30 days",),
            ),
        ]
        assert conn.committed is True
        assert conn.closed is True

    async def test_idempotent_ensure_session(self, tmp_path: Path) -> None:
        db_path = tmp_path / "checkpoints.db"
        cp = ConversationCheckpoint(db_path)

        await cp.ensure_session("s1")
        await cp.ensure_session("s1")
        await cp.ensure_session("s1")

        sessions = await cp.list_sessions()
        assert len(sessions) == 1

    async def test_empty_chunks_no_error(self, tmp_path: Path) -> None:
        db_path = tmp_path / "checkpoints.db"
        cp = ConversationCheckpoint(db_path)

        await cp.save_anchors("s1", 1, [])
        await cp.mark_cited("s1", 1, [])

        anchors = await cp.get_previous_anchors("s1")
        assert anchors == []

    async def test_save_turn_pair_keeps_scoped_sessions_isolated(self, tmp_path: Path) -> None:
        cp = ConversationCheckpoint(tmp_path / "checkpoints.db")

        await cp.save_turn_pair(
            "jwt:alice:reports:same-session",
            workspace="reports",
            query="alice question",
            answer="alice answer",
            contexts={"chunks": []},
            cited_chunk_ids=[],
        )
        await cp.save_turn_pair(
            "jwt:bob:reports:same-session",
            workspace="reports",
            query="bob question",
            answer="bob answer",
            contexts={"chunks": []},
            cited_chunk_ids=[],
        )

        assert await cp.get_history("jwt:alice:reports:same-session") == [
            {"role": "user", "content": "alice question"},
            {"role": "assistant", "content": "alice answer"},
        ]
        assert await cp.get_history("jwt:bob:reports:same-session") == [
            {"role": "user", "content": "bob question"},
            {"role": "assistant", "content": "bob answer"},
        ]

    async def test_save_turn_pair_is_atomic_for_concurrent_same_session(
        self, tmp_path: Path
    ) -> None:
        cp = ConversationCheckpoint(tmp_path / "checkpoints.db")

        async def save(idx: int) -> None:
            await cp.save_turn_pair(
                "jwt:alice:reports:s1",
                workspace="reports",
                query=f"q{idx}",
                answer=f"a{idx}",
                contexts={"chunks": []},
                cited_chunk_ids=[],
            )

        await asyncio.gather(save(1), save(2))

        history = await cp.get_history("jwt:alice:reports:s1")
        assert len(history) == 4
        assert {item["content"] for item in history if item["role"] == "user"} == {"q1", "q2"}
        assert {item["content"] for item in history if item["role"] == "assistant"} == {
            "a1",
            "a2",
        }
