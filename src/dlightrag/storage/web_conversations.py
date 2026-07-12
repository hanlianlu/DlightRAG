# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL persistence for principal-scoped Web conversations."""

import datetime
import json
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from dlightrag.storage.migrations import Migration, apply_migrations

_CREATE_CONVERSATIONS = """
CREATE TABLE IF NOT EXISTS web_conversations (
    principal_id TEXT NOT NULL,
    conversation_id UUID NOT NULL,
    title TEXT,
    content_revision BIGINT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (principal_id, conversation_id),
    CHECK (title IS NULL OR char_length(title) BETWEEN 1 AND 120)
)
"""

_CREATE_TURNS = """
CREATE TABLE IF NOT EXISTS web_conversation_turns (
    turn_id UUID NOT NULL,
    principal_id TEXT NOT NULL,
    conversation_id UUID NOT NULL,
    turn_number INTEGER NOT NULL,
    submission_id UUID NOT NULL,
    user_text TEXT NOT NULL,
    assistant_text TEXT NOT NULL,
    answer_sources JSONB NOT NULL DEFAULT '{}'::jsonb,
    queried_workspaces JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (principal_id, conversation_id, turn_id),
    UNIQUE (principal_id, conversation_id, turn_number),
    FOREIGN KEY (principal_id, conversation_id)
      REFERENCES web_conversations (principal_id, conversation_id)
      ON DELETE CASCADE
)
"""

_CREATE_IMAGES = """
CREATE TABLE IF NOT EXISTS web_conversation_images (
    image_id UUID NOT NULL,
    principal_id TEXT NOT NULL,
    conversation_id UUID NOT NULL,
    turn_id UUID NOT NULL,
    ordinal INTEGER NOT NULL,
    mime_type TEXT NOT NULL,
    image_bytes BYTEA NOT NULL,
    byte_size INTEGER NOT NULL,
    content_sha256 TEXT NOT NULL,
    vlm_description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (principal_id, conversation_id, image_id),
    UNIQUE (principal_id, conversation_id, turn_id, ordinal),
    FOREIGN KEY (principal_id, conversation_id, turn_id)
      REFERENCES web_conversation_turns (principal_id, conversation_id, turn_id)
      ON DELETE CASCADE
)
"""

_CREATE_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_web_conversations_principal_updated "
    "ON web_conversations (principal_id, updated_at DESC, conversation_id DESC)",
    "CREATE INDEX IF NOT EXISTS idx_web_conversation_turns_principal_conversation "
    "ON web_conversation_turns (principal_id, conversation_id, turn_number DESC)",
)

WEB_CONVERSATION_MIGRATIONS = (
    Migration(
        "0001_web_conversations",
        "Replace legacy checkpoints with scoped Web conversations",
        (
            "DROP TABLE IF EXISTS dlightrag_checkpoints",
            "DELETE FROM dlightrag_schema_migrations WHERE scope = 'checkpoints'",
            _CREATE_CONVERSATIONS,
            _CREATE_TURNS,
            _CREATE_IMAGES,
            *_CREATE_INDEXES,
        ),
    ),
    Migration(
        "0002_web_conversation_submission_id",
        "Add scoped idempotency keys for Web answer submissions",
        (
            "ALTER TABLE web_conversation_turns ADD COLUMN IF NOT EXISTS submission_id UUID",
            "UPDATE web_conversation_turns SET submission_id = turn_id WHERE submission_id IS NULL",
            "ALTER TABLE web_conversation_turns ALTER COLUMN submission_id SET NOT NULL",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_web_conversation_turns_submission "
            "ON web_conversation_turns (principal_id, conversation_id, submission_id)",
        ),
    ),
)

_SUMMARY_COLUMNS = """
conversation_id::text AS conversation_id,
title,
content_revision,
created_at,
updated_at
"""

_CREATE_CONVERSATION = f"""
INSERT INTO web_conversations (principal_id, conversation_id)
VALUES ($1, $2::text::uuid)
RETURNING {_SUMMARY_COLUMNS}
"""

_PRUNE_PRINCIPAL = """
DELETE FROM web_conversations
WHERE principal_id = $1
  AND updated_at < NOW() - ($2 * INTERVAL '1 day')
"""

_LIST_CONVERSATIONS = f"""
SELECT {_SUMMARY_COLUMNS}
FROM web_conversations
WHERE principal_id = $1
  AND updated_at >= NOW() - ($2 * INTERVAL '1 day')
ORDER BY updated_at DESC, conversation_id DESC
"""

_RENAME_CONVERSATION = f"""
UPDATE web_conversations
SET title = $3,
    updated_at = NOW()
WHERE principal_id = $1
  AND conversation_id = $2::text::uuid
  AND updated_at >= NOW() - ($4 * INTERVAL '1 day')
RETURNING {_SUMMARY_COLUMNS}
"""

_DELETE_CONVERSATION = """
DELETE FROM web_conversations
WHERE principal_id = $1
  AND conversation_id = $2::text::uuid
  AND updated_at >= NOW() - ($3 * INTERVAL '1 day')
RETURNING 1
"""

_GET_CONVERSATION = """
SELECT
    principal_id,
    conversation_id::text AS conversation_id,
    content_revision,
    title,
    created_at,
    updated_at
FROM web_conversations
WHERE principal_id = $1
  AND conversation_id = $2::text::uuid
  AND updated_at >= NOW() - ($3 * INTERVAL '1 day')
"""

_GET_HISTORY = """
WITH recent_turns AS (
    SELECT
        t.turn_id,
        t.turn_number,
        t.submission_id,
        t.user_text,
        t.assistant_text,
        t.answer_sources,
        t.queried_workspaces,
        t.created_at
    FROM web_conversation_turns AS t
    JOIN web_conversations AS c
      ON c.principal_id = t.principal_id
     AND c.conversation_id = t.conversation_id
    WHERE t.principal_id = $1
      AND t.conversation_id = $2::text::uuid
      AND c.principal_id = $1
      AND c.updated_at >= NOW() - ($3 * INTERVAL '1 day')
    ORDER BY t.turn_number DESC
    LIMIT $4
)
SELECT
    recent_turns.turn_id::text AS turn_id,
    recent_turns.turn_number,
    recent_turns.submission_id,
    recent_turns.user_text,
    recent_turns.assistant_text,
    recent_turns.answer_sources,
    recent_turns.queried_workspaces,
    recent_turns.created_at,
    COALESCE(
        jsonb_agg(
            jsonb_build_object(
                'image_id', i.image_id::text,
                'ordinal', i.ordinal,
                'mime_type', i.mime_type,
                'byte_size', i.byte_size,
                'content_sha256', i.content_sha256,
                'vlm_description', i.vlm_description
            ) ORDER BY i.ordinal
        ) FILTER (WHERE i.image_id IS NOT NULL),
        '[]'::jsonb
    ) AS images
FROM recent_turns
LEFT JOIN web_conversation_images AS i
  ON i.principal_id = $1
 AND i.conversation_id = $2::text::uuid
 AND i.turn_id = recent_turns.turn_id
GROUP BY
    recent_turns.turn_id,
    recent_turns.turn_number,
    recent_turns.submission_id,
    recent_turns.user_text,
    recent_turns.assistant_text,
    recent_turns.answer_sources,
    recent_turns.queried_workspaces,
    recent_turns.created_at
ORDER BY recent_turns.turn_number ASC
"""

_GET_IMAGE = """
SELECT
    i.image_id::text AS image_id,
    i.mime_type,
    i.image_bytes
FROM web_conversation_images AS i
JOIN web_conversations AS c
  ON c.principal_id = i.principal_id
 AND c.conversation_id = i.conversation_id
WHERE i.principal_id = $1
  AND i.conversation_id = $2::text::uuid
  AND i.image_id = $3::text::uuid
  AND c.principal_id = $1
  AND c.updated_at >= NOW() - ($4 * INTERVAL '1 day')
"""

_GUARDED_UPDATE = f"""
UPDATE web_conversations
SET content_revision = content_revision + 1,
    title = COALESCE(title, $4),
    updated_at = NOW()
WHERE principal_id = $1
  AND conversation_id = $2::text::uuid
  AND content_revision = $3
RETURNING {_SUMMARY_COLUMNS}
"""

_INSERT_TURN = """
INSERT INTO web_conversation_turns (
    turn_id,
    principal_id,
    conversation_id,
    turn_number,
    submission_id,
    user_text,
    assistant_text,
    answer_sources,
    queried_workspaces
)
VALUES (
    $1::text::uuid,
    $2,
    $3::text::uuid,
    (
        SELECT COALESCE(MAX(turn_number), 0) + 1
        FROM web_conversation_turns
        WHERE principal_id = $2
          AND conversation_id = $3::text::uuid
    ),
    $4::text::uuid,
    $5,
    $6,
    $7::jsonb,
    $8::jsonb
)
RETURNING turn_id::text AS turn_id, turn_number
"""

_GET_COMMITTED_TURN = """
SELECT
    c.conversation_id::text AS conversation_id,
    c.title,
    c.content_revision,
    c.created_at,
    c.updated_at,
    t.turn_id::text AS turn_id,
    t.assistant_text,
    t.answer_sources,
    COALESCE(
        jsonb_agg(
            jsonb_build_object(
                'image_id', i.image_id::text,
                'ordinal', i.ordinal,
                'vlm_description', i.vlm_description
            ) ORDER BY i.ordinal
        ) FILTER (WHERE i.image_id IS NOT NULL),
        '[]'::jsonb
    ) AS images
FROM web_conversation_turns AS t
JOIN web_conversations AS c
  ON c.principal_id = t.principal_id
 AND c.conversation_id = t.conversation_id
LEFT JOIN web_conversation_images AS i
  ON i.principal_id = t.principal_id
 AND i.conversation_id = t.conversation_id
 AND i.turn_id = t.turn_id
WHERE t.principal_id = $1
  AND t.conversation_id = $2::text::uuid
  AND t.submission_id = $3::text::uuid
  AND c.updated_at >= NOW() - ($4 * INTERVAL '1 day')
GROUP BY
    c.conversation_id,
    c.title,
    c.content_revision,
    c.created_at,
    c.updated_at,
    t.turn_id,
    t.assistant_text,
    t.answer_sources
"""

_INSERT_IMAGE = """
INSERT INTO web_conversation_images (
    image_id,
    principal_id,
    conversation_id,
    turn_id,
    ordinal,
    mime_type,
    image_bytes,
    byte_size,
    content_sha256,
    vlm_description
)
VALUES (
    $1::text::uuid,
    $2,
    $3::text::uuid,
    $4::text::uuid,
    $5,
    $6,
    $7,
    $8,
    $9,
    $10
)
"""

_TRIM_TURNS = """
DELETE FROM web_conversation_turns
WHERE principal_id = $1
  AND conversation_id = $2::text::uuid
  AND turn_number <= $3
"""

_PRUNE_EXPIRED = """
WITH deleted AS (
    DELETE FROM web_conversations
    WHERE updated_at < NOW() - ($1 * INTERVAL '1 day')
    RETURNING 1
)
SELECT COUNT(*)::int AS count FROM deleted
"""


@dataclass(frozen=True, slots=True)
class ConversationSnapshot:
    principal_id: str
    conversation_id: str
    content_revision: int
    title: str | None
    created_at: datetime.datetime
    updated_at: datetime.datetime
    history: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class PendingConversationImage:
    image_id: str
    ordinal: int
    mime_type: str
    image_bytes: bytes
    content_sha256: str
    vlm_description: str | None


@dataclass(frozen=True, slots=True)
class CommitTurnResult:
    saved: bool
    reason: str | None
    summary: dict[str, Any] | None
    turn_id: str | None
    current_image_ids: tuple[str, ...] = ()
    assistant_text: str | None = None
    answer_sources: dict[str, Any] | None = None
    image_descriptions: dict[str, str] = field(default_factory=dict)
    replayed: bool = False


@dataclass(frozen=True, slots=True)
class StoredConversationImage:
    image_id: str
    mime_type: str
    image_bytes: bytes


def _row_dict(row: Any) -> dict[str, Any]:
    return dict(row)


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _history_row(row: Any) -> dict[str, Any]:
    result = _row_dict(row)
    for key in ("answer_sources", "queried_workspaces", "images"):
        result[key] = _json_value(result[key])
    return result


def _committed_result(row: Any, *, replayed: bool) -> CommitTurnResult:
    value = _row_dict(row)
    images = _json_value(value.get("images", []))
    answer_sources = _json_value(value.get("answer_sources", {}))
    summary = {
        key: value[key]
        for key in ("conversation_id", "title", "content_revision", "created_at", "updated_at")
        if key in value
    }
    descriptions = {
        str(image["ordinal"]): str(image["vlm_description"])
        for image in images
        if image.get("vlm_description")
    }
    return CommitTurnResult(
        saved=True,
        reason=None,
        summary=summary,
        turn_id=str(value["turn_id"]),
        current_image_ids=tuple(str(image["image_id"]) for image in images),
        assistant_text=str(value["assistant_text"]),
        answer_sources=answer_sources,
        image_descriptions=descriptions,
        replayed=replayed,
    )


class PGWebConversationStore:
    """Durable PostgreSQL store for server-owned Web conversations."""

    def __init__(self, *, pool: Any = None) -> None:
        self._pool = pool
        self._initialized = False

    async def _run_read(self, operation):
        if self._pool is not None:
            async with self._pool.acquire() as conn:
                return await operation(conn)

        from dlightrag.storage.pool import pg_pool

        return await pg_pool.run(operation)

    async def _run_write(self, operation):
        if self._pool is not None:
            async with self._pool.acquire() as conn:
                return await operation(conn)

        from dlightrag.storage.pool import pg_pool

        return await pg_pool.run_once(operation)

    async def initialize(self) -> None:
        """Create the Web conversation schema and remove legacy checkpoints."""
        if self._initialized:
            return

        async def _operation(conn: Any) -> None:
            await apply_migrations(
                conn,
                scope="web_conversations",
                migrations=WEB_CONVERSATION_MIGRATIONS,
            )

        await self._run_write(_operation)
        self._initialized = True

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self.initialize()

    async def create_conversation(self, principal_id: str) -> dict[str, Any]:
        """Create and return an empty conversation owned by ``principal_id``."""
        await self._ensure_initialized()
        conversation_id = str(uuid4())

        async def _operation(conn: Any) -> dict[str, Any]:
            row = await conn.fetchrow(_CREATE_CONVERSATION, principal_id, conversation_id)
            if row is None:
                raise RuntimeError("conversation insert returned no row")
            return _row_dict(row)

        return await self._run_write(_operation)

    async def list_conversations(
        self,
        principal_id: str,
        *,
        ttl_days: int,
    ) -> list[dict[str, Any]]:
        """Prune and list one principal's unexpired conversations."""
        await self._ensure_initialized()

        async def _prune(conn: Any) -> None:
            await conn.execute(_PRUNE_PRINCIPAL, principal_id, ttl_days)

        async def _select(conn: Any) -> list[dict[str, Any]]:
            rows = await conn.fetch(_LIST_CONVERSATIONS, principal_id, ttl_days)
            return [_row_dict(row) for row in rows]

        await self._run_write(_prune)
        return await self._run_read(_select)

    async def rename_conversation(
        self,
        principal_id: str,
        conversation_id: str,
        *,
        title: str,
        ttl_days: int,
    ) -> dict[str, Any] | None:
        """Rename an unexpired owned conversation."""
        await self._ensure_initialized()
        normalized_title = title.strip()
        if not normalized_title or len(normalized_title) > 120:
            raise ValueError("conversation title must contain 1 to 120 characters")

        async def _operation(conn: Any) -> dict[str, Any] | None:
            row = await conn.fetchrow(
                _RENAME_CONVERSATION,
                principal_id,
                conversation_id,
                normalized_title,
                ttl_days,
            )
            return _row_dict(row) if row is not None else None

        return await self._run_write(_operation)

    async def delete_conversation(
        self,
        principal_id: str,
        conversation_id: str,
        *,
        ttl_days: int,
    ) -> bool:
        """Delete an unexpired owned conversation and its cascaded children."""
        await self._ensure_initialized()

        async def _operation(conn: Any) -> bool:
            row = await conn.fetchrow(
                _DELETE_CONVERSATION,
                principal_id,
                conversation_id,
                ttl_days,
            )
            return row is not None

        return await self._run_write(_operation)

    async def snapshot(
        self,
        principal_id: str,
        conversation_id: str,
        *,
        ttl_days: int,
        max_turns: int = 100,
    ) -> ConversationSnapshot | None:
        """Load one unexpired conversation and its client-safe recent history."""
        await self._ensure_initialized()

        async def _operation(conn: Any) -> ConversationSnapshot | None:
            async with conn.transaction(isolation="repeatable_read", readonly=True):
                row = await conn.fetchrow(
                    _GET_CONVERSATION,
                    principal_id,
                    conversation_id,
                    ttl_days,
                )
                if row is None:
                    return None
                history_rows = await conn.fetch(
                    _GET_HISTORY,
                    principal_id,
                    conversation_id,
                    ttl_days,
                    max_turns,
                )
                return ConversationSnapshot(
                    principal_id=str(row["principal_id"]),
                    conversation_id=str(row["conversation_id"]),
                    content_revision=int(row["content_revision"]),
                    title=row["title"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    history=tuple(_history_row(history_row) for history_row in history_rows),
                )

        return await self._run_read(_operation)

    async def get_image(
        self,
        principal_id: str,
        conversation_id: str,
        image_id: str,
        *,
        ttl_days: int,
    ) -> StoredConversationImage | None:
        """Load image bytes only when all ownership and TTL constraints match."""
        await self._ensure_initialized()

        async def _operation(conn: Any) -> StoredConversationImage | None:
            row = await conn.fetchrow(
                _GET_IMAGE,
                principal_id,
                conversation_id,
                image_id,
                ttl_days,
            )
            if row is None:
                return None
            return StoredConversationImage(
                image_id=str(row["image_id"]),
                mime_type=str(row["mime_type"]),
                image_bytes=bytes(row["image_bytes"]),
            )

        return await self._run_read(_operation)

    async def find_committed_turn(
        self,
        principal_id: str,
        conversation_id: str,
        submission_id: str,
        *,
        ttl_days: int,
    ) -> CommitTurnResult | None:
        """Return the authoritative completed turn for one scoped submission key."""
        await self._ensure_initialized()

        async def _operation(conn: Any) -> CommitTurnResult | None:
            row = await conn.fetchrow(
                _GET_COMMITTED_TURN,
                principal_id,
                conversation_id,
                submission_id,
                ttl_days,
            )
            return _committed_result(row, replayed=True) if row is not None else None

        return await self._run_read(_operation)

    async def commit_turn(
        self,
        *,
        principal_id: str,
        conversation_id: str,
        submission_id: str,
        expected_revision: int,
        user_text: str,
        assistant_text: str,
        answer_sources: dict[str, Any],
        queried_workspaces: list[str],
        images: list[PendingConversationImage],
        max_turns: int,
        ttl_days: int,
    ) -> CommitTurnResult:
        """Atomically append one turn when the captured revision still matches."""
        await self._ensure_initialized()
        auto_title = " ".join(user_text.split())[:120] or None

        async def _operation(conn: Any) -> CommitTurnResult:
            async with conn.transaction():
                committed_row = await conn.fetchrow(
                    _GET_COMMITTED_TURN,
                    principal_id,
                    conversation_id,
                    submission_id,
                    ttl_days,
                )
                if committed_row is not None:
                    return _committed_result(committed_row, replayed=True)
                summary_row = await conn.fetchrow(
                    _GUARDED_UPDATE,
                    principal_id,
                    conversation_id,
                    expected_revision,
                    auto_title,
                )
                if summary_row is None:
                    committed_row = await conn.fetchrow(
                        _GET_COMMITTED_TURN,
                        principal_id,
                        conversation_id,
                        submission_id,
                        ttl_days,
                    )
                    if committed_row is not None:
                        return _committed_result(committed_row, replayed=True)
                    return CommitTurnResult(False, "conversation_changed", None, None)

                turn_id = str(uuid4())
                turn_row = await conn.fetchrow(
                    _INSERT_TURN,
                    turn_id,
                    principal_id,
                    conversation_id,
                    submission_id,
                    user_text,
                    assistant_text,
                    json.dumps(answer_sources),
                    json.dumps(queried_workspaces),
                )
                if turn_row is None:
                    raise RuntimeError("conversation turn insert returned no row")

                authoritative_turn_id = str(turn_row["turn_id"])
                for image in images:
                    await conn.execute(
                        _INSERT_IMAGE,
                        image.image_id,
                        principal_id,
                        conversation_id,
                        authoritative_turn_id,
                        image.ordinal,
                        image.mime_type,
                        image.image_bytes,
                        len(image.image_bytes),
                        image.content_sha256,
                        image.vlm_description,
                    )

                trim_before_or_at = int(turn_row["turn_number"]) - max_turns
                await conn.execute(
                    _TRIM_TURNS,
                    principal_id,
                    conversation_id,
                    trim_before_or_at,
                )
                return CommitTurnResult(
                    saved=True,
                    reason=None,
                    summary=_row_dict(summary_row),
                    turn_id=authoritative_turn_id,
                    current_image_ids=tuple(image.image_id for image in images),
                    assistant_text=assistant_text,
                    answer_sources=answer_sources,
                    image_descriptions={
                        str(image.ordinal): image.vlm_description
                        for image in images
                        if image.vlm_description is not None
                    },
                )

        return await self._run_write(_operation)

    async def prune_expired(self, *, ttl_days: int) -> int:
        """Delete expired Web conversations across principals at startup."""
        await self._ensure_initialized()

        async def _operation(conn: Any) -> int:
            row = await conn.fetchrow(_PRUNE_EXPIRED, ttl_days)
            return int(row["count"]) if row is not None else 0

        return await self._run_write(_operation)


__all__ = [
    "WEB_CONVERSATION_MIGRATIONS",
    "CommitTurnResult",
    "ConversationSnapshot",
    "PGWebConversationStore",
    "PendingConversationImage",
    "StoredConversationImage",
]
