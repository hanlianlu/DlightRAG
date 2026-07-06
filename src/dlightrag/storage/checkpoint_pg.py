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

_CREATE = """
CREATE TABLE IF NOT EXISTS dlightrag_checkpoints (
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

_SAVE_TURN_PAIR = """
WITH next_turn AS (
    SELECT COALESCE(MAX(turn_number), 0) + 1 AS turn_number
    FROM dlightrag_checkpoints
    WHERE session_id = $1
)
INSERT INTO dlightrag_checkpoints
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

_GET_HISTORY = """
SELECT turn_number, query_content, answer_content
FROM dlightrag_checkpoints
WHERE session_id = $1
  AND query_content IS NOT NULL
  AND answer_content IS NOT NULL
  AND jsonb_typeof(query_content) = 'array'
  AND jsonb_typeof(answer_content) = 'array'
ORDER BY turn_number ASC
LIMIT $2
"""

_DELETE_WORKSPACE = """
WITH deleted AS (
    DELETE FROM dlightrag_checkpoints
    WHERE workspace = $1
    RETURNING 1
)
SELECT COUNT(*)::int FROM deleted
"""

_PRUNE_OLD = """
WITH deleted AS (
    DELETE FROM dlightrag_checkpoints
    WHERE updated_at < NOW() - ($1 * INTERVAL '1 day')
    RETURNING 1
)
SELECT COUNT(*)::int FROM deleted
"""

_DELETE_SESSION = """
WITH deleted AS (
    DELETE FROM dlightrag_checkpoints
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


def _text_content(text: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": text}]


def _content_for_storage(
    content: list[dict[str, Any]] | None,
    text: str,
) -> list[dict[str, Any]]:
    return content if content is not None else _text_content(text)


def _content_from_jsonb(jsonb_val: Any) -> list[dict[str, Any]]:
    """Return checkpoint JSONB content as content blocks."""
    blocks = jsonb_val
    if isinstance(jsonb_val, str):
        try:
            blocks = json.loads(jsonb_val)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("checkpoint content must be a JSON array") from exc

    if not isinstance(blocks, list):
        raise ValueError("checkpoint content must be a JSON array")

    result: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict):
            raise ValueError("checkpoint content blocks must be JSON objects")
        result.append(block)
    return result


class PGCheckpointStore:
    """PostgreSQL-backed conversation checkpoint store.

    One row per conversation turn (query + answer pair). Content blocks,
    context chunks, and cited chunk IDs are stored as JSONB; the public
    ``get_history`` API returns ``{role, content}`` dicts.

    Usage::

        store = PGCheckpointStore()
        await store.initialize()               # idempotent DDL
        turn = await store.save_turn_pair(...)  # returns turn_number
        history = await store.get_history(sid)  # list[dict[str, Any]]
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
        stored_query_content = _content_for_storage(query_content, query)
        stored_answer_content = _content_for_storage(answer_content, answer)

        async def _operation(conn: Any) -> int:
            row = await conn.fetchrow(
                _SAVE_TURN_PAIR,
                session_id,
                workspace,
                query,
                answer,
                json.dumps(stored_query_content),
                json.dumps(stored_answer_content),
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

        Uses the canonical ``query_content`` / ``answer_content`` JSONB columns.
        Rows without structured content are ignored rather than reconstructed
        from legacy text columns.
        """
        await self._ensure_initialized()

        async def _operation(conn: Any) -> list[dict[str, Any]]:
            rows = await conn.fetch(_GET_HISTORY, session_id, max_turns)
            history: list[dict[str, Any]] = []
            for row in rows:
                qc = _content_from_jsonb(row["query_content"])
                history.append({"role": "user", "content": qc})
                ac = _content_from_jsonb(row["answer_content"])
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
