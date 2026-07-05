# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL-backed conversation checkpoint store.

Replaces the SQLite ``ConversationCheckpoint`` (deleted) with a single-table
asyncpg store that reuses DlightRAG's existing ``pg_pool`` and
``migrations.py`` infrastructure — the same pattern as ``PGIngestJobStore``,
``PGMetadataIndex``, and ``PGWorkspaceRegistry``.
"""

import json
from typing import Any

from dlightrag.storage.migrations import Migration, apply_migrations

TABLE = "dlightrag_checkpoints"

_CREATE = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    id              BIGSERIAL PRIMARY KEY,
    session_id      TEXT NOT NULL,
    workspace       TEXT NOT NULL DEFAULT 'default',
    turn_number     INTEGER NOT NULL,
    query           TEXT NOT NULL,
    answer          TEXT NOT NULL,
    cited_chunk_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    context_chunks  JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

_CREATE_INDEXES = (
    f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{TABLE}_session_turn "
    f"ON {TABLE} (session_id, turn_number)",
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_workspace_updated "
    f"ON {TABLE} (workspace, updated_at DESC)",
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_session_updated "
    f"ON {TABLE} (session_id, updated_at DESC)",
)

_SAVE_TURN_PAIR = f"""
WITH next_turn AS (
    SELECT COALESCE(MAX(turn_number), 0) + 1 AS turn_number
    FROM {TABLE}
    WHERE session_id = $1
)
INSERT INTO {TABLE}
    (session_id, workspace, turn_number, query, answer,
     query_content, answer_content, cited_chunk_ids, context_chunks)
SELECT $1, $2, turn_number, $3, $4, $5::jsonb, $6::jsonb, $7::jsonb, $8::jsonb
FROM next_turn
ON CONFLICT (session_id, turn_number) DO UPDATE SET
    query = EXCLUDED.query,
    answer = EXCLUDED.answer,
    query_content = EXCLUDED.query_content,
    answer_content = EXCLUDED.answer_content,
    cited_chunk_ids = EXCLUDED.cited_chunk_ids,
    context_chunks = EXCLUDED.context_chunks,
    updated_at = NOW()
RETURNING turn_number
"""

_GET_HISTORY = f"""
SELECT turn_number, query, answer, query_content, answer_content
FROM {TABLE}
WHERE session_id = $1
ORDER BY turn_number ASC
LIMIT $2
"""

_DELETE_WORKSPACE = f"""
WITH deleted AS (
    DELETE FROM {TABLE}
    WHERE workspace = $1
    RETURNING 1
)
SELECT COUNT(*)::int FROM deleted
"""

_PRUNE_OLD = f"""
WITH deleted AS (
    DELETE FROM {TABLE}
    WHERE updated_at < NOW() - ($1 * INTERVAL '1 day')
    RETURNING 1
)
SELECT COUNT(*)::int FROM deleted
"""

_DELETE_SESSION = f"""
WITH deleted AS (
    DELETE FROM {TABLE}
    WHERE session_id = $1
    RETURNING 1
)
SELECT COUNT(*)::int FROM deleted
"""

_SCHEMA_MIGRATIONS = (
    Migration(
        "0001_checkpoints",
        "Create checkpoint sessions table",
        (_CREATE, *_CREATE_INDEXES),
    ),
    Migration(
        "0002_checkpoints_multimodal",
        "Add query_content and answer_content JSONB columns",
        (
            f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS query_content JSONB",
            f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS answer_content JSONB",
        ),
    ),
)


def _resolve_content(jsonb_val: Any, text_fallback: Any) -> str | list[dict[str, Any]]:
    """Resolve stored content to its canonical form.

    Prefers the JSONB value when it is a non-empty list; falls back to
    the plain-text column.  Handles ``None`` (missing column or NULL row),
    asyncpg's occasional ``str``-for-JSONB behaviour, and empty strings.
    """
    if jsonb_val is None:
        return str(text_fallback or "")
    if isinstance(jsonb_val, list):
        return jsonb_val
    if isinstance(jsonb_val, str):
        try:
            parsed = json.loads(jsonb_val)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError, TypeError:
            pass
        if jsonb_val.strip():
            return jsonb_val
    text = str(text_fallback or "")
    return text


class PGCheckpointStore:
    """PostgreSQL-backed conversation checkpoint store.

    One row per conversation turn (query + answer pair).  Context chunks and
    cited chunk IDs are stored as JSONB for potential future use; the public
    ``get_history`` API returns only ``{role, content}`` dicts, matching the
    old SQLite contract.

    Usage::

        store = PGCheckpointStore()
        await store.initialize()               # idempotent DDL
        turn = await store.save_turn_pair(...)  # returns turn_number
        history = await store.get_history(sid)  # list[dict[str, str]]
    """

    def __init__(self, *, pool: Any = None) -> None:
        self._pool = pool
        self._initialized = False

    # -- lifecycle -----------------------------------------------------------

    async def _run(self, operation):
        if self._pool is not None:
            async with self._pool.acquire() as conn:
                return await operation(conn)

        from dlightrag.storage.pool import pg_pool

        return await pg_pool.run(operation)

    async def initialize(self) -> None:
        """Apply schema migration.  Idempotent — safe to call repeatedly."""
        if self._initialized:
            return

        async def _operation(conn: Any) -> None:
            await apply_migrations(
                conn,
                scope="checkpoints",
                migrations=_SCHEMA_MIGRATIONS,
            )

        await self._run(_operation)
        self._initialized = True

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self.initialize()

    # -- public API ----------------------------------------------------------

    async def save_turn_pair(
        self,
        session_id: str,
        *,
        workspace: str,
        query: str,
        answer: str,
        query_content: list[dict[str, Any]] | None = None,
        answer_content: list[dict[str, Any]] | None = None,
        contexts: dict[str, Any],
        cited_chunk_ids: list[str],
    ) -> int:
        """Atomically save one user/assistant exchange.

        Returns the assigned 1-based turn number.
        """
        await self._ensure_initialized()

        chunks = contexts.get("chunks", []) if isinstance(contexts, dict) else []
        chunk_list = chunks if isinstance(chunks, list) else []

        async def _operation(conn: Any) -> int:
            row = await conn.fetchrow(
                _SAVE_TURN_PAIR,
                session_id,
                workspace,
                query,
                answer,
                json.dumps(query_content) if query_content is not None else None,
                json.dumps(answer_content) if answer_content is not None else None,
                json.dumps(cited_chunk_ids or []),
                json.dumps(chunk_list),
            )
            return int(row["turn_number"]) if row else 1

        return await self._run(_operation)

    async def get_history(
        self,
        session_id: str,
        *,
        max_turns: int = 50,
    ) -> list[dict[str, Any]]:
        """Return conversation history as list of {role, content} dicts.

        Prefers ``query_content`` / ``answer_content`` JSONB columns when
        present; falls back to the plain-text ``query`` / ``answer`` columns
        for backward compatibility with pre-multimodal rows.
        """
        await self._ensure_initialized()

        async def _operation(conn: Any) -> list[dict[str, Any]]:
            rows = await conn.fetch(_GET_HISTORY, session_id, max_turns)
            history: list[dict[str, Any]] = []
            for row in rows:
                # query → user
                qc = _resolve_content(row.get("query_content"), row["query"])
                history.append({"role": "user", "content": qc})
                # answer → assistant
                ac = _resolve_content(row.get("answer_content"), row["answer"])
                history.append({"role": "assistant", "content": ac})
            return history

        return await self._run(_operation)

    async def delete_sessions_by_workspace(self, workspace: str) -> int:
        """Delete all checkpoint rows for a workspace. Returns count deleted."""
        await self._ensure_initialized()

        async def _operation(conn: Any) -> int:
            deleted = await conn.fetchval(_DELETE_WORKSPACE, workspace)
            return int(deleted or 0)

        return await self._run(_operation)

    async def delete_session(self, session_id: str) -> int:
        """Delete all checkpoint rows for a single session. Returns count deleted."""
        await self._ensure_initialized()

        async def _operation(conn: Any) -> int:
            deleted = await conn.fetchval(_DELETE_SESSION, session_id)
            return int(deleted or 0)

        return await self._run(_operation)

    async def prune_old_sessions(self, *, max_age_days: int = 30) -> int:
        """Delete checkpoint rows older than *max_age_days*. Returns count deleted."""
        await self._ensure_initialized()

        async def _operation(conn: Any) -> int:
            deleted = await conn.fetchval(_PRUNE_OLD, max_age_days)
            return int(deleted or 0)

        return await self._run(_operation)

    async def close(self) -> None:
        """No-op — the pool is owned by ``pg_pool`` and closed at shutdown."""


__all__ = ["PGCheckpointStore"]
