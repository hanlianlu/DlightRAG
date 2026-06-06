# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Zero-dependency SQLite conversation persistence with provenance anchoring."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SQLITE_TIMEOUT_SECONDS = 30.0
_SQLITE_BUSY_TIMEOUT_MS = 30_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    workspace    TEXT NOT NULL DEFAULT 'default',
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS turns (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    turn_number   INTEGER NOT NULL,
    role          TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content       TEXT NOT NULL,
    cited_chunks  TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(session_id, turn_number, role)
);

CREATE TABLE IF NOT EXISTS context_anchors (
    session_id    TEXT NOT NULL,
    turn_number   INTEGER NOT NULL,
    chunk_id      TEXT NOT NULL,
    source_doc    TEXT,
    sidecar_type  TEXT,
    score         REAL,
    was_cited     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (session_id, turn_number, chunk_id),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, turn_number);
CREATE INDEX IF NOT EXISTS idx_anchors_session ON context_anchors(session_id);
CREATE INDEX IF NOT EXISTS idx_anchors_cited ON context_anchors(session_id, was_cited);
"""


class ConversationCheckpoint:
    """Zero-dependency SQLite conversation persistence.

    All I/O runs via ``asyncio.to_thread()`` with stdlib ``sqlite3`` in WAL mode.
    Database is created lazily on first write.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._schema_lock = threading.Lock()
        self._schema_initialized = False

    # ----------------------------------------------------------------
    # Public async API
    # ----------------------------------------------------------------

    async def ensure_session(self, session_id: str, *, workspace: str = "default") -> None:
        """Create session row if it doesn't exist."""
        await asyncio.to_thread(self._ensure_session_sync, session_id, workspace)

    async def next_turn_number(self, session_id: str) -> int:
        """Return the next turn number for this session (1-based)."""
        return await asyncio.to_thread(self._next_turn_number_sync, session_id)

    async def save_turn(
        self,
        session_id: str,
        turn_number: int,
        *,
        role: str,
        content: str,
        cited_chunk_ids: list[str] | None = None,
    ) -> None:
        """Save a single conversation turn."""
        await asyncio.to_thread(
            self._save_turn_sync,
            session_id,
            turn_number,
            role,
            content,
            cited_chunk_ids,
        )

    async def save_turn_pair(
        self,
        session_id: str,
        *,
        workspace: str,
        query: str,
        answer: str,
        contexts: dict[str, Any],
        cited_chunk_ids: list[str],
    ) -> int:
        """Atomically save one user/assistant exchange and its retrieval anchors.

        Returns the assigned 1-based turn number.
        """
        return await asyncio.to_thread(
            self._save_turn_pair_sync,
            session_id,
            workspace,
            query,
            answer,
            contexts,
            cited_chunk_ids,
        )

    async def save_anchors(
        self,
        session_id: str,
        turn_number: int,
        chunks: list[dict[str, Any]],
    ) -> None:
        """Save context anchor rows for retrieved chunks."""
        await asyncio.to_thread(self._save_anchors_sync, session_id, turn_number, chunks)

    async def mark_cited(
        self,
        session_id: str,
        turn_number: int,
        cited_chunk_ids: list[str],
    ) -> None:
        """Mark chunks as cited by the LLM in their answer."""
        if not cited_chunk_ids:
            return
        await asyncio.to_thread(self._mark_cited_sync, session_id, turn_number, cited_chunk_ids)

    async def get_history(self, session_id: str, *, max_turns: int = 50) -> list[dict[str, str]]:
        """Return conversation history as list of {role, content} dicts."""
        return await asyncio.to_thread(self._get_history_sync, session_id, max_turns)

    async def get_cited_chunk_ids(self, session_id: str) -> set[str]:
        """Return set of chunk_ids cited in this session."""
        return await asyncio.to_thread(self._get_cited_chunk_ids_sync, session_id)

    async def get_previous_anchors(
        self, session_id: str, *, last_n_turns: int = 3
    ) -> list[dict[str, Any]]:
        """Return context_anchors from the last N turns."""
        return await asyncio.to_thread(
            self._get_previous_anchors_sync,
            session_id,
            last_n_turns,
        )

    async def delete_session(self, session_id: str) -> None:
        """Delete a session and all its turns + anchors (CASCADE)."""
        await asyncio.to_thread(self._delete_session_sync, session_id)

    async def delete_sessions_by_workspace(self, workspace: str) -> int:
        """Delete all sessions for a workspace. Returns count deleted."""
        return await asyncio.to_thread(self._delete_sessions_by_workspace_sync, workspace)

    async def list_sessions(
        self, *, workspace: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        """List sessions, optionally filtered by workspace."""
        return await asyncio.to_thread(self._list_sessions_sync, workspace, limit)

    async def prune_old_sessions(self, *, max_age_days: int = 30) -> int:
        """Delete sessions older than max_age_days. Returns count deleted."""
        return await asyncio.to_thread(self._prune_old_sessions_sync, max_age_days)

    # ----------------------------------------------------------------
    # Sync internals — all called via asyncio.to_thread()
    # ----------------------------------------------------------------

    def _ensure_db_sync(self) -> sqlite3.Connection:
        """Open connection, create schema on first access."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(self._db_path),
            timeout=_SQLITE_TIMEOUT_SECONDS,
        )
        self._configure_connection_sync(conn)
        with self._schema_lock:
            if not self._schema_initialized:
                journal_mode = conn.execute("PRAGMA journal_mode = WAL").fetchone()
                if not journal_mode or str(journal_mode[0]).lower() != "wal":
                    logger.warning("SQLite checkpoint journal_mode is %s, not WAL", journal_mode)
                conn.executescript(_SCHEMA)
                conn.commit()
                conn.execute("PRAGMA optimize")
                self._schema_initialized = True
        return conn

    def _configure_connection_sync(self, conn: sqlite3.Connection) -> None:
        conn.execute(f"PRAGMA busy_timeout = {_SQLITE_BUSY_TIMEOUT_MS}")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA synchronous = NORMAL")

    def _ensure_session_sync(self, session_id: str, workspace: str) -> None:
        conn = self._ensure_db_sync()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id, workspace) VALUES (?, ?)",
                (session_id, workspace),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = datetime('now') WHERE session_id = ?",
                (session_id,),
            )
            conn.commit()
        finally:
            conn.close()

    def _next_turn_number_sync(self, session_id: str) -> int:
        conn = self._ensure_db_sync()
        try:
            row = conn.execute(
                "SELECT COALESCE(MAX(turn_number), 0) FROM turns WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return int(row[0]) + 1 if row else 1
        finally:
            conn.close()

    def _save_turn_sync(
        self,
        session_id: str,
        turn_number: int,
        role: str,
        content: str,
        cited_chunk_ids: list[str] | None,
    ) -> None:
        self._ensure_session_sync(session_id, "default")
        cited_json = json.dumps(cited_chunk_ids) if cited_chunk_ids else None
        conn = self._ensure_db_sync()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO turns
                   (session_id, turn_number, role, content, cited_chunks)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, turn_number, role, content, cited_json),
            )
            conn.commit()
        finally:
            conn.close()

    def _save_turn_pair_sync(
        self,
        session_id: str,
        workspace: str,
        query: str,
        answer: str,
        contexts: dict[str, Any],
        cited_chunk_ids: list[str],
    ) -> int:
        conn = self._ensure_db_sync()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id, workspace) VALUES (?, ?)",
                (session_id, workspace),
            )
            conn.execute(
                """UPDATE sessions
                   SET workspace = ?, updated_at = datetime('now')
                   WHERE session_id = ?""",
                (workspace, session_id),
            )
            row = conn.execute(
                "SELECT COALESCE(MAX(turn_number), 0) FROM turns WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            turn_number = int(row[0]) + 1 if row else 1
            cited_json = json.dumps(cited_chunk_ids) if cited_chunk_ids else None
            conn.executemany(
                """INSERT INTO turns
                   (session_id, turn_number, role, content, cited_chunks)
                   VALUES (?, ?, ?, ?, ?)""",
                [
                    (session_id, turn_number, "user", query, None),
                    (session_id, turn_number, "assistant", answer, cited_json),
                ],
            )

            chunks = contexts.get("chunks", [])
            anchor_chunks = chunks if isinstance(chunks, list) else []
            rows: list[tuple[str, int, str, str | None, str | None, float | None]] = []
            for chunk in anchor_chunks:
                if not isinstance(chunk, dict):
                    continue
                chunk_id = str(chunk.get("chunk_id") or "")
                if not chunk_id:
                    continue
                source_doc = str(chunk.get("file_path") or "") or None
                sidecar = chunk.get("sidecar")
                sidecar_type: str | None = None
                if isinstance(sidecar, dict):
                    sidecar_type = sidecar.get("type") or None
                score = chunk.get("relevance_score")
                if score is not None:
                    score = float(score)
                rows.append((session_id, turn_number, chunk_id, source_doc, sidecar_type, score))
            if rows:
                conn.executemany(
                    """INSERT OR REPLACE INTO context_anchors
                       (session_id, turn_number, chunk_id, source_doc, sidecar_type, score)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    rows,
                )
            if cited_chunk_ids:
                conn.executemany(
                    """UPDATE context_anchors SET was_cited = 1
                       WHERE session_id = ? AND turn_number = ? AND chunk_id = ?""",
                    [(session_id, turn_number, cid) for cid in cited_chunk_ids],
                )
            conn.commit()
            return turn_number
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _save_anchors_sync(
        self,
        session_id: str,
        turn_number: int,
        chunks: list[dict[str, Any]],
    ) -> None:
        if not chunks:
            return
        conn = self._ensure_db_sync()
        try:
            rows: list[tuple[str, int, str, str | None, str | None, float | None]] = []
            for chunk in chunks:
                chunk_id = str(chunk.get("chunk_id") or "")
                if not chunk_id:
                    continue
                source_doc = str(chunk.get("file_path") or "") or None
                sidecar = chunk.get("sidecar")
                sidecar_type: str | None = None
                if isinstance(sidecar, dict):
                    sidecar_type = sidecar.get("type") or None
                score = chunk.get("relevance_score")
                if score is not None:
                    score = float(score)
                rows.append((session_id, turn_number, chunk_id, source_doc, sidecar_type, score))
            if rows:
                conn.executemany(
                    """INSERT OR REPLACE INTO context_anchors
                       (session_id, turn_number, chunk_id, source_doc, sidecar_type, score)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    rows,
                )
                conn.commit()
        finally:
            conn.close()

    def _mark_cited_sync(
        self,
        session_id: str,
        turn_number: int,
        cited_chunk_ids: list[str],
    ) -> None:
        conn = self._ensure_db_sync()
        try:
            conn.executemany(
                """UPDATE context_anchors SET was_cited = 1
                   WHERE session_id = ? AND turn_number = ? AND chunk_id = ?""",
                [(session_id, turn_number, cid) for cid in cited_chunk_ids],
            )
            conn.commit()
        finally:
            conn.close()

    def _get_history_sync(self, session_id: str, max_turns: int) -> list[dict[str, str]]:
        conn = self._ensure_db_sync()
        try:
            rows = conn.execute(
                """SELECT role, content FROM turns
                   WHERE session_id = ?
                   ORDER BY turn_number ASC, id ASC
                   LIMIT ?""",
                (session_id, max_turns * 2),
            ).fetchall()
            return [{"role": r, "content": c} for r, c in rows]
        finally:
            conn.close()

    def _get_cited_chunk_ids_sync(self, session_id: str) -> set[str]:
        conn = self._ensure_db_sync()
        try:
            rows = conn.execute(
                "SELECT chunk_id FROM context_anchors WHERE session_id = ? AND was_cited = 1",
                (session_id,),
            ).fetchall()
            return {r[0] for r in rows}
        finally:
            conn.close()

    def _get_previous_anchors_sync(
        self, session_id: str, last_n_turns: int
    ) -> list[dict[str, Any]]:
        conn = self._ensure_db_sync()
        try:
            turn_rows = conn.execute(
                """SELECT DISTINCT turn_number FROM (
                       SELECT turn_number FROM turns WHERE session_id = ?
                       UNION
                       SELECT turn_number FROM context_anchors WHERE session_id = ?
                   )
                   ORDER BY turn_number DESC
                   LIMIT ?""",
                (session_id, session_id, last_n_turns),
            ).fetchall()
            if not turn_rows:
                return []
            turn_numbers = [r[0] for r in turn_rows]
            placeholders = ",".join("?" * len(turn_numbers))
            rows = conn.execute(
                f"""SELECT chunk_id, source_doc, sidecar_type, score, was_cited
                    FROM context_anchors
                    WHERE session_id = ? AND turn_number IN ({placeholders})""",
                [session_id, *turn_numbers],
            ).fetchall()
            keys = ["chunk_id", "source_doc", "sidecar_type", "score", "was_cited"]
            return [dict(zip(keys, r, strict=True)) for r in rows]
        finally:
            conn.close()

    def _delete_session_sync(self, session_id: str) -> None:
        conn = self._ensure_db_sync()
        try:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
        finally:
            conn.close()

    def _delete_sessions_by_workspace_sync(self, workspace: str) -> int:
        conn = self._ensure_db_sync()
        try:
            cursor = conn.execute("DELETE FROM sessions WHERE workspace = ?", (workspace,))
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def _list_sessions_sync(self, workspace: str | None, limit: int) -> list[dict[str, Any]]:
        conn = self._ensure_db_sync()
        try:
            if workspace:
                rows = conn.execute(
                    """SELECT session_id, workspace, created_at, updated_at,
                              (SELECT COUNT(*) FROM turns WHERE turns.session_id = sessions.session_id) AS turn_count
                       FROM sessions WHERE workspace = ?
                       ORDER BY updated_at DESC LIMIT ?""",
                    (workspace, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT session_id, workspace, created_at, updated_at,
                              (SELECT COUNT(*) FROM turns WHERE turns.session_id = sessions.session_id) AS turn_count
                       FROM sessions
                       ORDER BY updated_at DESC LIMIT ?""",
                    (limit,),
                ).fetchall()
            keys = ["session_id", "workspace", "created_at", "updated_at", "turn_count"]
            return [dict(zip(keys, r, strict=True)) for r in rows]
        finally:
            conn.close()

    def _prune_old_sessions_sync(self, max_age_days: int) -> int:
        conn = self._ensure_db_sync()
        try:
            age_modifier = f"-{int(max_age_days)} days"
            cursor = conn.execute(
                "DELETE FROM sessions WHERE updated_at < datetime('now', ?)",
                (age_modifier,),
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()
