# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for PGCheckpointStore."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from dlightrag.storage.checkpoint_pg import PGCheckpointStore

# ── Mock asyncpg helpers ────────────────────────────────────────────────


class _Pool:
    def __init__(self, conn: _Conn) -> None:
        self._conn = conn

    def acquire(self) -> _Acquire:
        return _Acquire(self._conn)


class _Acquire:
    def __init__(self, conn: _Conn) -> None:
        self._conn = conn

    async def __aenter__(self) -> _Conn:
        return self._conn

    async def __aexit__(self, *_args: object) -> None:
        pass


class _Tx:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *_: object) -> None:
        pass


class _Conn:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.fetchrows: list[dict[str, Any] | None] = []
        self.fetches: list[list[dict[str, Any]]] = []
        self.fetchvals: list[int] = []

    async def execute(self, query: str, *args: Any) -> None:
        self.executed.append((query, args))

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        self.executed.append((query, args))
        return self.fetchrows.pop(0) if self.fetchrows else None

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.executed.append((query, args))
        return self.fetches.pop(0) if self.fetches else []

    async def fetchval(self, query: str, *args: Any) -> int:
        self.executed.append((query, args))
        return self.fetchvals.pop(0) if self.fetchvals else 0

    def transaction(self) -> _Tx:
        return _Tx()


def _make_store(conn: _Conn | None = None) -> PGCheckpointStore:
    c = conn or _Conn()
    pool = _Pool(c)
    store = PGCheckpointStore(pool=pool)
    store._initialized = True  # skip migration DDL in unit tests
    return store


# ── Tests ───────────────────────────────────────────────────────────────


class TestSaveTurnPair:
    async def test_save_and_get_history(self) -> None:
        conn = _Conn()
        conn.fetchrows = [{"turn_number": 1}]
        conn.fetches = [
            [
                {"turn_number": 1, "query": "Hello", "answer": "Hi there!"},
            ]
        ]
        store = _make_store(conn)

        turn = await store.save_turn_pair(
            "s1",
            workspace="default",
            query="Hello",
            answer="Hi there!",
            contexts={"chunks": []},
            cited_chunk_ids=["c1"],
        )
        assert turn == 1

        history = await store.get_history("s1")
        assert history == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

    async def test_turn_number_auto_increments(self) -> None:
        """Two sequential saves get turn numbers 1 and 2."""
        conn = _Conn()
        conn.fetchrows = [{"turn_number": 1}, {"turn_number": 2}]
        store = _make_store(conn)

        t1 = await store.save_turn_pair(
            "s1",
            workspace="default",
            query="q1",
            answer="a1",
            contexts={},
            cited_chunk_ids=[],
        )
        t2 = await store.save_turn_pair(
            "s1",
            workspace="default",
            query="q2",
            answer="a2",
            contexts={},
            cited_chunk_ids=[],
        )
        assert t1 == 1
        assert t2 == 2

    async def test_save_includes_context_chunks(self) -> None:
        conn = _Conn()
        conn.fetchrows = [{"turn_number": 1}]
        store = _make_store(conn)

        chunk = {"chunk_id": "c1", "file_path": "/a.pdf", "relevance_score": 0.9}
        await store.save_turn_pair(
            "s1",
            workspace="default",
            query="q",
            answer="a",
            contexts={"chunks": [chunk]},
            cited_chunk_ids=["c1"],
        )

        # Verify context_chunks JSONB was sent
        insert_calls = [c for c in conn.executed if "INSERT INTO" in str(c[0]).upper()]
        assert insert_calls
        args = insert_calls[0][1]
        context_json = args[5]
        assert isinstance(context_json, str)
        assert "c1" in context_json

    async def test_sessions_are_isolated(self) -> None:
        """Different session_ids don't leak into each other."""
        conn = _Conn()
        conn.fetchrows = [{"turn_number": 1}, {"turn_number": 1}]
        conn.fetches = [
            [{"turn_number": 1, "query": "alice-q", "answer": "alice-a"}],
            [{"turn_number": 1, "query": "bob-q", "answer": "bob-a"}],
        ]
        store = _make_store(conn)

        await store.save_turn_pair(
            "scoped:alice:s1",
            workspace="default",
            query="alice-q",
            answer="alice-a",
            contexts={},
            cited_chunk_ids=[],
        )
        await store.save_turn_pair(
            "scoped:bob:s1",
            workspace="default",
            query="bob-q",
            answer="bob-a",
            contexts={},
            cited_chunk_ids=[],
        )

        alice = await store.get_history("scoped:alice:s1")
        bob = await store.get_history("scoped:bob:s1")
        assert alice[0]["content"] == "alice-q"
        assert bob[0]["content"] == "bob-q"


class TestGetHistory:
    async def test_respects_max_turns(self) -> None:
        conn = _Conn()
        conn.fetches = [
            [
                {"turn_number": 1, "query": "q1", "answer": "a1"},
                {"turn_number": 2, "query": "q2", "answer": "a2"},
                {"turn_number": 3, "query": "q3", "answer": "a3"},
            ]
        ]
        store = _make_store(conn)

        history = await store.get_history("s1", max_turns=3)
        # 3 turns → 6 messages (user + assistant per turn)
        assert len(history) == 6

        # Verify LIMIT was sent
        limit_calls = [c for c in conn.executed if "LIMIT" in str(c[0])]
        assert limit_calls

    async def test_empty_session_returns_empty_list(self) -> None:
        conn = _Conn()
        conn.fetches = [[]]
        store = _make_store(conn)

        history = await store.get_history("nonexistent")
        assert history == []


class TestDeleteSessionsByWorkspace:
    async def test_deletes_correct_workspace(self) -> None:
        conn = _Conn()
        conn.fetchvals = [3]
        store = _make_store(conn)

        deleted = await store.delete_sessions_by_workspace("ws-a")
        assert deleted == 3

    async def test_no_rows_returns_zero(self) -> None:
        conn = _Conn()
        conn.fetchvals = [0]
        store = _make_store(conn)

        deleted = await store.delete_sessions_by_workspace("ws-empty")
        assert deleted == 0


class TestPruneOldSessions:
    async def test_prunes_by_age(self) -> None:
        conn = _Conn()
        conn.fetchvals = [5]
        store = _make_store(conn)

        deleted = await store.prune_old_sessions(max_age_days=30)
        assert deleted == 5

        delete_calls = [c for c in conn.executed if "DELETE FROM" in str(c[0]).upper()]
        assert delete_calls

    async def test_zero_deleted(self) -> None:
        conn = _Conn()
        conn.fetchvals = [0]
        store = _make_store(conn)

        deleted = await store.prune_old_sessions(max_age_days=30)
        assert deleted == 0


class TestInitialize:
    async def test_initialize_is_idempotent(self) -> None:
        """Calling initialize twice applies migration only once."""
        conn = _Conn()
        store = PGCheckpointStore(pool=_Pool(conn))
        assert not store._initialized

        await store.initialize()
        assert store._initialized

        # Second call should be a no-op
        await store.initialize()
        assert store._initialized

    async def test_public_methods_auto_initialize(self) -> None:
        """save_turn_pair initializes on first use."""
        conn = _Conn()
        conn.fetchrows = [{"turn_number": 1}]
        store = PGCheckpointStore(pool=_Pool(conn))
        assert not store._initialized

        await store.save_turn_pair(
            "s1",
            workspace="default",
            query="q",
            answer="a",
            contexts={},
            cited_chunk_ids=[],
        )
        assert store._initialized


class TestConcurrentWrites:
    async def test_same_session_concurrent_saves_get_distinct_turns(self) -> None:
        """Two concurrent saves to the same session get turn 1 and 2."""
        import asyncio

        conn = _Conn()
        # fetchrow is called during save, each returns the turn_number
        conn.fetchrows = [{"turn_number": 1}, {"turn_number": 2}]
        conn.fetches = [
            [
                {"turn_number": 1, "query": "q1", "answer": "a1"},
                {"turn_number": 2, "query": "q2", "answer": "a2"},
            ]
        ]
        store = _make_store(conn)

        async def save(content: str) -> int:
            return await store.save_turn_pair(
                "s1",
                workspace="default",
                query=content,
                answer=content,
                contexts={},
                cited_chunk_ids=[],
            )

        t1, t2 = await asyncio.gather(save("q1"), save("q2"))
        assert {t1, t2} == {1, 2}

        history = await store.get_history("s1")
        assert len(history) == 4


class TestSaveTurnCheckpointIntegration:
    """Integration tests for servicemanager.save_turn_checkpoint with PG store."""

    async def test_save_turn_checkpoint_calls_store(self) -> None:
        from dlightrag.core.servicemanager import RAGServiceManager

        cfg = MagicMock()
        cfg.workspace = "default"
        cfg.checkpoint_session_ttl_days = 30
        cfg.max_async = 8
        cfg.embedding.model = "test"
        cfg.working_dir_path = MagicMock()

        manager = RAGServiceManager.__new__(RAGServiceManager)
        manager._config = cfg

        mock_store = MagicMock()
        mock_store.save_turn_pair = AsyncMock(return_value=3)
        manager._checkpoint = mock_store

        await manager.save_turn_checkpoint(
            session_id="s1",
            query="test query",
            answer="test answer",
            contexts={"chunks": [{"chunk_id": "c1"}]},
            cited_chunk_ids=["c1"],
            workspace="default",
        )

        mock_store.save_turn_pair.assert_awaited_once()
        call_kwargs = mock_store.save_turn_pair.call_args.kwargs
        assert call_kwargs["query"] == "test query"
        assert call_kwargs["answer"] == "test answer"
        assert call_kwargs["cited_chunk_ids"] == ["c1"]

    async def test_save_turn_checkpoint_is_best_effort(self) -> None:
        from dlightrag.core.servicemanager import RAGServiceManager

        cfg = MagicMock()
        cfg.workspace = "default"
        cfg.checkpoint_session_ttl_days = 30
        cfg.max_async = 8
        cfg.embedding.model = "test"
        cfg.working_dir_path = MagicMock()

        manager = RAGServiceManager.__new__(RAGServiceManager)
        manager._config = cfg

        mock_store = MagicMock()
        mock_store.save_turn_pair = AsyncMock(side_effect=RuntimeError("PG gone"))
        manager._checkpoint = mock_store

        # Must not raise
        await manager.save_turn_checkpoint(
            session_id="s1",
            query="q",
            answer="a",
            contexts={},
            cited_chunk_ids=[],
        )
