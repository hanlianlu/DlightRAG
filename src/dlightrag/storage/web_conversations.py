# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL persistence for principal-scoped Web conversations.

Web answer attachments are stored as raw resources: original bytes, verified
MIME/suffix, and a content digest. There is no image table, no parse/chunk/
vector cache, and no stored VLM description; every request reads resources
locally instead of persisting derived artifacts.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass
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

_CREATE_ATTACHMENTS = """
CREATE TABLE IF NOT EXISTS web_conversation_attachments (
    attachment_id UUID NOT NULL,
    principal_id TEXT NOT NULL,
    conversation_id UUID NOT NULL,
    turn_id UUID NOT NULL,
    ordinal INTEGER NOT NULL,
    filename TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    suffix TEXT NOT NULL,
    attachment_bytes BYTEA NOT NULL,
    byte_size INTEGER NOT NULL,
    content_sha256 TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (principal_id, conversation_id, attachment_id),
    UNIQUE (principal_id, conversation_id, turn_id, ordinal),
    FOREIGN KEY (principal_id, conversation_id, turn_id)
      REFERENCES web_conversation_turns (principal_id, conversation_id, turn_id)
      ON DELETE CASCADE
)
"""

_CREATE_CONVERSATION_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_web_conversations_principal_updated "
    "ON web_conversations (principal_id, updated_at DESC, conversation_id DESC)",
    "CREATE INDEX IF NOT EXISTS idx_web_conversations_updated "
    "ON web_conversations (updated_at, principal_id, conversation_id)",
    "CREATE INDEX IF NOT EXISTS idx_web_conversation_turns_principal_conversation "
    "ON web_conversation_turns (principal_id, conversation_id, turn_number DESC)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_web_conversation_turns_submission "
    "ON web_conversation_turns (principal_id, conversation_id, submission_id)",
)

_CREATE_ATTACHMENT_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_web_conversation_attachments_catalog "
    "ON web_conversation_attachments (principal_id, conversation_id, created_at DESC)"
)

# Intentional one-time reset. Existing deployments applied an earlier schema
# with a separate image table and a derived parse/vector cache; this migration
# deletes every Web conversation row (cascading to turns and their children),
# drops the superseded tables, and recreates one raw-attachment table. The reset
# is deliberate: no committed turn may keep a source pointing at a dropped
# entity, and there is no compatibility view or renamed-column bridge.
_RESET_WEB_CONVERSATIONS = "DELETE FROM web_conversations"
_DROP_ATTACHMENT_CHUNKS = "DROP TABLE IF EXISTS web_conversation_attachment_chunks"
_DROP_IMAGES = "DROP TABLE IF EXISTS web_conversation_images"
_DROP_LEGACY_ATTACHMENTS = "DROP TABLE IF EXISTS web_conversation_attachments"

WEB_CONVERSATION_MIGRATIONS = (
    Migration(
        "0001_web_conversations",
        "Create scoped Web conversations and turns",
        (
            _CREATE_CONVERSATIONS,
            _CREATE_TURNS,
            *_CREATE_CONVERSATION_INDEXES,
        ),
    ),
    Migration(
        "0002_unified_web_conversation_attachments",
        "Reset Web conversations and store answer attachments as raw resources",
        (
            _RESET_WEB_CONVERSATIONS,
            _DROP_ATTACHMENT_CHUNKS,
            _DROP_IMAGES,
            _DROP_LEGACY_ATTACHMENTS,
            _CREATE_ATTACHMENTS,
            _CREATE_ATTACHMENT_INDEX,
        ),
    ),
    Migration(
        "0003_canonical_answer_sources",
        "Reset Web conversations before storing canonical answer source snapshots",
        (_RESET_WEB_CONVERSATIONS,),
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
"""  # noqa: S608 - interpolates only the trusted _SUMMARY_COLUMNS constant

_LIST_CONVERSATIONS = f"""
SELECT {_SUMMARY_COLUMNS}
FROM web_conversations
WHERE principal_id = $1
  AND updated_at >= NOW() - ($2 * INTERVAL '1 day')
ORDER BY updated_at DESC, conversation_id DESC
"""  # noqa: S608 - interpolates only the trusted _SUMMARY_COLUMNS constant

_RENAME_CONVERSATION = f"""
UPDATE web_conversations
SET title = $3,
    updated_at = NOW()
WHERE principal_id = $1
  AND conversation_id = $2::text::uuid
  AND updated_at >= NOW() - ($4 * INTERVAL '1 day')
RETURNING {_SUMMARY_COLUMNS}
"""  # noqa: S608 - interpolates only the trusted _SUMMARY_COLUMNS constant

_DELETE_CONVERSATION = """
DELETE FROM web_conversations
WHERE principal_id = $1
  AND conversation_id = $2::text::uuid
  AND updated_at >= NOW() - ($3 * INTERVAL '1 day')
RETURNING 1
"""

_DELETE_ALL_CONVERSATIONS = """
WITH deleted AS (
        DELETE FROM web_conversations
        WHERE principal_id = $1
        RETURNING 1
)
SELECT count(*)::int AS deleted_count
FROM deleted
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

_ATTACHMENT_MANIFEST_SUBQUERY = """
COALESCE(
    (
        SELECT jsonb_agg(
            jsonb_build_object(
                'attachment_id', a.attachment_id::text,
                'ordinal', a.ordinal,
                'filename', a.filename,
                'mime_type', a.mime_type,
                'byte_size', a.byte_size,
                'content_sha256', a.content_sha256
            ) ORDER BY a.ordinal
        )
        FROM web_conversation_attachments AS a
        WHERE a.principal_id = $1
          AND a.conversation_id = $2::text::uuid
          AND a.turn_id = {turn_id_expr}
    ),
    '[]'::jsonb
) AS attachments
"""

_GET_HISTORY = f"""
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
    {_ATTACHMENT_MANIFEST_SUBQUERY.format(turn_id_expr="recent_turns.turn_id")}
FROM recent_turns
ORDER BY recent_turns.turn_number ASC
"""  # noqa: S608 - interpolates only trusted attachment-manifest SQL constants

_GUARDED_UPDATE = f"""
UPDATE web_conversations
SET content_revision = content_revision + 1,
    title = COALESCE(title, $4),
    updated_at = NOW()
WHERE principal_id = $1
  AND conversation_id = $2::text::uuid
  AND content_revision = $3
RETURNING {_SUMMARY_COLUMNS}
"""  # noqa: S608 - interpolates only the trusted _SUMMARY_COLUMNS constant

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

_UPDATE_TURN_SOURCES = """
UPDATE web_conversation_turns
SET answer_sources = $4::jsonb
WHERE principal_id = $1
  AND conversation_id = $2::text::uuid
  AND submission_id = $3::text::uuid
"""

_GET_COMMITTED_TURN = f"""
SELECT
    c.conversation_id::text AS conversation_id,
    c.title,
    c.content_revision,
    c.created_at,
    c.updated_at,
    t.turn_id::text AS turn_id,
    t.assistant_text,
    t.answer_sources,
    t.queried_workspaces,
    {_ATTACHMENT_MANIFEST_SUBQUERY.format(turn_id_expr="t.turn_id")}
FROM web_conversation_turns AS t
JOIN web_conversations AS c
  ON c.principal_id = t.principal_id
 AND c.conversation_id = t.conversation_id
WHERE t.principal_id = $1
  AND t.conversation_id = $2::text::uuid
  AND t.submission_id = $3::text::uuid
  AND c.updated_at >= NOW() - ($4 * INTERVAL '1 day')
"""  # noqa: S608 - interpolates only trusted attachment-manifest SQL constants

_INSERT_ATTACHMENT = """
INSERT INTO web_conversation_attachments (
    attachment_id,
    principal_id,
    conversation_id,
    turn_id,
    ordinal,
    filename,
    mime_type,
    suffix,
    attachment_bytes,
    byte_size,
    content_sha256
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
    $10,
    $11
)
"""

_ATTACHMENT_COLUMNS = """
a.attachment_id::text AS attachment_id,
a.filename,
a.mime_type,
a.suffix,
a.attachment_bytes,
a.content_sha256
"""

_GET_ATTACHMENT = f"""
SELECT
{_ATTACHMENT_COLUMNS}
FROM web_conversation_attachments AS a
JOIN web_conversations AS c
  ON c.principal_id = a.principal_id
 AND c.conversation_id = a.conversation_id
WHERE a.principal_id = $1
  AND a.conversation_id = $2::text::uuid
  AND a.attachment_id = $3::text::uuid
  AND c.principal_id = $1
  AND c.updated_at >= NOW() - ($4 * INTERVAL '1 day')
"""  # noqa: S608 - interpolates only the trusted _ATTACHMENT_COLUMNS constant

_FETCH_ATTACHMENTS_BY_IDS = f"""
SELECT
{_ATTACHMENT_COLUMNS}
FROM web_conversation_attachments AS a
JOIN web_conversations AS c
  ON c.principal_id = a.principal_id
 AND c.conversation_id = a.conversation_id
WHERE a.principal_id = $1
  AND a.conversation_id = $2::text::uuid
  AND a.attachment_id = ANY($3::uuid[])
  AND c.updated_at >= NOW() - ($4 * INTERVAL '1 day')
"""  # noqa: S608 - interpolates only the trusted _ATTACHMENT_COLUMNS constant

_TRIM_TURNS = """
DELETE FROM web_conversation_turns
WHERE principal_id = $1
  AND conversation_id = $2::text::uuid
  AND turn_number <= $3
"""

_PRUNE_EXPIRED = """
WITH candidates AS (
    SELECT principal_id, conversation_id
    FROM web_conversations
    WHERE updated_at < NOW() - ($1 * INTERVAL '1 day')
    ORDER BY updated_at, principal_id, conversation_id
    LIMIT $2
    FOR UPDATE SKIP LOCKED
), deleted AS (
    DELETE FROM web_conversations AS conversations
    USING candidates
    WHERE conversations.principal_id = candidates.principal_id
      AND conversations.conversation_id = candidates.conversation_id
    RETURNING conversations.conversation_id
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
class PendingConversationAttachment:
    """One raw current-turn attachment to persist with a committed turn."""

    attachment_id: str
    ordinal: int
    filename: str
    mime_type: str
    suffix: str
    attachment_bytes: bytes
    content_sha256: str

    @property
    def byte_size(self) -> int:
        return len(self.attachment_bytes)


@dataclass(frozen=True, slots=True)
class CommitTurnResult:
    saved: bool
    reason: str | None
    summary: dict[str, Any] | None
    turn_id: str | None
    current_attachment_ids: tuple[str, ...] = ()
    assistant_text: str | None = None
    answer_sources: dict[str, Any] | None = None
    replayed: bool = False
    queried_workspaces: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StoredConversationAttachment:
    attachment_id: str
    filename: str
    mime_type: str
    suffix: str
    attachment_bytes: bytes
    content_sha256: str = ""


def _row_dict(row: Any) -> dict[str, Any]:
    return dict(row)


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _history_row(row: Any) -> dict[str, Any]:
    result = _row_dict(row)
    for key in ("answer_sources", "queried_workspaces", "attachments"):
        result[key] = _json_value(result[key])
    return result


def _committed_result(row: Any, *, replayed: bool) -> CommitTurnResult:
    value = _row_dict(row)
    attachments = _json_value(value.get("attachments", []))
    answer_sources = _json_value(value.get("answer_sources", {}))
    summary = {
        key: value[key]
        for key in ("conversation_id", "title", "content_revision", "created_at", "updated_at")
        if key in value
    }
    return CommitTurnResult(
        saved=True,
        reason=None,
        summary=summary,
        turn_id=str(value["turn_id"]),
        current_attachment_ids=tuple(str(item["attachment_id"]) for item in attachments),
        assistant_text=str(value["assistant_text"]),
        answer_sources=answer_sources,
        replayed=replayed,
        queried_workspaces=tuple(
            str(workspace) for workspace in _json_value(value.get("queried_workspaces", []))
        ),
    )


def _stored_attachment(row: Any) -> StoredConversationAttachment:
    return StoredConversationAttachment(
        attachment_id=str(row["attachment_id"]),
        filename=str(row["filename"]),
        mime_type=str(row["mime_type"]),
        suffix=str(row["suffix"]),
        attachment_bytes=bytes(row["attachment_bytes"]),
        content_sha256=str(row["content_sha256"]),
    )


class PGWebConversationStore:
    """Durable PostgreSQL store for server-owned Web conversations."""

    def __init__(self, *, pool: Any = None) -> None:
        self._pool = pool
        self._initialized = False

    async def _run_read(self, operation, *, retry: bool = True):
        if self._pool is not None:
            async with self._pool.acquire() as conn:
                return await operation(conn)

        from dlightrag.storage.pool import pg_pool

        return await (pg_pool.run(operation) if retry else pg_pool.run_once(operation))

    async def _run_write(self, operation):
        if self._pool is not None:
            async with self._pool.acquire() as conn:
                return await operation(conn)

        from dlightrag.storage.pool import pg_pool

        return await pg_pool.run_once(operation)

    async def initialize(self) -> None:
        """Create the Web conversation schema and apply the raw-attachment reset."""
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
        """List one principal's unexpired conversations."""
        await self._ensure_initialized()

        async def _select(conn: Any) -> list[dict[str, Any]]:
            rows = await conn.fetch(_LIST_CONVERSATIONS, principal_id, ttl_days)
            return [_row_dict(row) for row in rows]

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

    async def delete_all_conversations(self, principal_id: str) -> int:
        """Delete every conversation owned by one principal."""
        await self._ensure_initialized()

        async def _operation(conn: Any) -> int:
            row = await conn.fetchrow(_DELETE_ALL_CONVERSATIONS, principal_id)
            return int(row["deleted_count"]) if row is not None else 0

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

    async def get_attachment(
        self,
        principal_id: str,
        conversation_id: str,
        attachment_id: str,
        *,
        ttl_days: int,
    ) -> StoredConversationAttachment | None:
        """Load original attachment bytes only when ownership and TTL match."""
        await self._ensure_initialized()

        async def _operation(conn: Any) -> StoredConversationAttachment | None:
            row = await conn.fetchrow(
                _GET_ATTACHMENT,
                principal_id,
                conversation_id,
                attachment_id,
                ttl_days,
            )
            return _stored_attachment(row) if row is not None else None

        return await self._run_read(_operation)

    async def fetch_attachments_by_ids(
        self,
        principal_id: str,
        conversation_id: str,
        attachment_ids: list[str],
        *,
        ttl_days: int,
    ) -> list[StoredConversationAttachment]:
        """Return owned, unexpired original attachment bytes for the given ids."""
        if not attachment_ids:
            return []
        await self._ensure_initialized()

        async def _operation(conn: Any) -> list[StoredConversationAttachment]:
            rows = await conn.fetch(
                _FETCH_ATTACHMENTS_BY_IDS,
                principal_id,
                conversation_id,
                attachment_ids,
                ttl_days,
            )
            return [_stored_attachment(row) for row in rows]

        return await self._run_read(_operation)

    async def find_committed_turn(
        self,
        principal_id: str,
        conversation_id: str,
        submission_id: str,
        *,
        ttl_days: int,
        retry: bool = True,
    ) -> CommitTurnResult | None:
        """Return the authoritative completed turn for one scoped submission key.

        ``retry=False`` performs a single bounded-protocol lookup without the
        storage-layer retry budget, used by outcome reconciliation.
        """
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

        return await self._run_read(_operation, retry=retry)

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
        attachments: list[PendingConversationAttachment],
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
                for attachment in attachments:
                    await conn.execute(
                        _INSERT_ATTACHMENT,
                        attachment.attachment_id,
                        principal_id,
                        conversation_id,
                        authoritative_turn_id,
                        attachment.ordinal,
                        attachment.filename,
                        attachment.mime_type,
                        attachment.suffix,
                        attachment.attachment_bytes,
                        attachment.byte_size,
                        attachment.content_sha256,
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
                    current_attachment_ids=tuple(
                        attachment.attachment_id for attachment in attachments
                    ),
                    assistant_text=assistant_text,
                    answer_sources=answer_sources,
                    queried_workspaces=tuple(queried_workspaces),
                )

        return await self._run_write(_operation)

    async def update_turn_sources(
        self,
        *,
        principal_id: str,
        conversation_id: str,
        submission_id: str,
        answer_sources: dict[str, Any],
    ) -> bool:
        """Overwrite one committed turn's answer_sources snapshot.

        Used to fold post-answer semantic highlights into the stored turn so
        history and page reloads render the same highlighted sources the live
        answer showed. Returns whether a row matched the submission key.
        """
        await self._ensure_initialized()

        async def _operation(conn: Any) -> bool:
            status = await conn.execute(
                _UPDATE_TURN_SOURCES,
                principal_id,
                conversation_id,
                submission_id,
                json.dumps(answer_sources),
            )
            return status.rsplit(" ", 1)[-1] == "1"

        return await self._run_write(_operation)

    async def prune_expired(self, *, ttl_days: int, batch_size: int = 500) -> int:
        """Delete one skip-locked batch of expired conversations."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        await self._ensure_initialized()

        async def _operation(conn: Any) -> int:
            row = await conn.fetchrow(_PRUNE_EXPIRED, ttl_days, batch_size)
            return int(row["count"]) if row is not None else 0

        return await self._run_write(_operation)


__all__ = [
    "WEB_CONVERSATION_MIGRATIONS",
    "CommitTurnResult",
    "ConversationSnapshot",
    "PGWebConversationStore",
    "PendingConversationAttachment",
    "StoredConversationAttachment",
]
