# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL persistence for principal-scoped Web conversations."""

import datetime
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from dlightrag.storage.migrations import Migration, apply_migrations

if TYPE_CHECKING:
    from dlightrag.core.request.attachments import ParsedAttachmentBundle

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
    parse_summary TEXT,
    parse_catalog JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (principal_id, conversation_id, attachment_id),
    UNIQUE (principal_id, conversation_id, turn_id, ordinal),
    FOREIGN KEY (principal_id, conversation_id, turn_id)
      REFERENCES web_conversation_turns (principal_id, conversation_id, turn_id)
      ON DELETE CASCADE
)
"""

_CREATE_ATTACHMENT_CHUNKS = """
CREATE TABLE IF NOT EXISTS web_conversation_attachment_chunks (
    principal_id TEXT NOT NULL,
    conversation_id UUID NOT NULL,
    content_sha256 TEXT NOT NULL,
    parser_signature TEXT NOT NULL,
    chunk_signature TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    page_idx INTEGER,
    bbox JSONB,
    sidecar_type TEXT,
    image_bytes BYTEA,
    image_mime_type TEXT,
    token_estimate INTEGER NOT NULL DEFAULT 0,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding_signature TEXT,
    embedding_vector JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (
        principal_id,
        conversation_id,
        content_sha256,
        parser_signature,
        chunk_signature,
        chunk_id
    )
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
    Migration(
        "0003_web_conversation_attachments",
        "Add scoped Web document attachments and content-addressed parse cache",
        (
            _CREATE_ATTACHMENTS,
            _CREATE_ATTACHMENT_CHUNKS,
            "CREATE INDEX IF NOT EXISTS idx_web_conversation_attachments_catalog "
            "ON web_conversation_attachments (principal_id, conversation_id, created_at DESC)",
        ),
    ),
    Migration(
        "0004_web_conversation_attachment_chunks_cascade",
        "Reclaim the parse cache when its owning conversation is removed",
        (
            # Drop any pre-existing orphan cache rows (from deployments that ran
            # 0003 before this lifecycle fix) so the new FK validates cleanly.
            "DELETE FROM web_conversation_attachment_chunks AS c "
            "WHERE NOT EXISTS ("
            "SELECT 1 FROM web_conversations AS w "
            "WHERE w.principal_id = c.principal_id "
            "AND w.conversation_id = c.conversation_id)",
            # Cascade the content-addressed cache from its owning conversation.
            # The conversation row already exists when pre-commit parsing writes
            # the cache, and this references web_conversations (NOT the turn
            # rows), so it does not break pre-commit parsing. Conversation
            # deletion, principal prune, and TTL expiry all delete the
            # web_conversations row, so the cache is reclaimed on every path.
            # ADD CONSTRAINT is not idempotent (Postgres has no IF NOT EXISTS
            # form) and apply_migrations re-runs every statement on each startup,
            # so guard it with a catalog existence check to stay idempotent.
            """DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'web_conversation_attachment_chunks_conversation_fkey'
          AND conrelid = 'web_conversation_attachment_chunks'::regclass
    ) THEN
        ALTER TABLE web_conversation_attachment_chunks
        ADD CONSTRAINT web_conversation_attachment_chunks_conversation_fkey
        FOREIGN KEY (principal_id, conversation_id)
        REFERENCES web_conversations (principal_id, conversation_id)
        ON DELETE CASCADE;
    END IF;
END $$""",
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
"""  # noqa: S608 - interpolates only the trusted _SUMMARY_COLUMNS constant

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
    ) AS images,
    COALESCE(
        (
            SELECT jsonb_agg(
                jsonb_build_object(
                    'attachment_id', a.attachment_id::text,
                    'ordinal', a.ordinal,
                    'filename', a.filename,
                    'mime_type', a.mime_type,
                    'byte_size', a.byte_size,
                    'content_sha256', a.content_sha256,
                    'parse_summary', a.parse_summary
                ) ORDER BY a.ordinal
            )
            FROM web_conversation_attachments AS a
            WHERE a.principal_id = $1
              AND a.conversation_id = $2::text::uuid
              AND a.turn_id = recent_turns.turn_id
        ),
        '[]'::jsonb
    ) AS attachments
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
    ) AS images,
    COALESCE(
        (
            SELECT jsonb_agg(
                jsonb_build_object(
                    'attachment_id', a.attachment_id::text,
                    'ordinal', a.ordinal,
                    'filename', a.filename,
                    'mime_type', a.mime_type,
                    'byte_size', a.byte_size,
                    'content_sha256', a.content_sha256,
                    'parse_summary', a.parse_summary
                ) ORDER BY a.ordinal
            )
            FROM web_conversation_attachments AS a
            WHERE a.principal_id = $1
              AND a.conversation_id = $2::text::uuid
              AND a.turn_id = t.turn_id
        ),
        '[]'::jsonb
    ) AS attachments
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
    content_sha256,
    parse_summary,
    parse_catalog
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
    $11,
    $12,
    $13::jsonb
)
"""

_GET_ATTACHMENT = """
SELECT
    a.attachment_id::text AS attachment_id,
    a.filename,
    a.mime_type,
    a.suffix,
    a.attachment_bytes,
    a.content_sha256
FROM web_conversation_attachments AS a
JOIN web_conversations AS c
  ON c.principal_id = a.principal_id
 AND c.conversation_id = a.conversation_id
WHERE a.principal_id = $1
  AND a.conversation_id = $2::text::uuid
  AND a.attachment_id = $3::text::uuid
  AND c.principal_id = $1
  AND c.updated_at >= NOW() - ($4 * INTERVAL '1 day')
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

_LIST_IMAGE_CATALOG = """
SELECT
    i.image_id::text AS image_id,
    t.turn_number,
    i.ordinal,
    i.vlm_description
FROM web_conversation_images AS i
JOIN web_conversation_turns AS t
  ON t.principal_id = i.principal_id
 AND t.conversation_id = i.conversation_id
 AND t.turn_id = i.turn_id
JOIN web_conversations AS c
  ON c.principal_id = i.principal_id
 AND c.conversation_id = i.conversation_id
WHERE i.principal_id = $1
  AND i.conversation_id = $2::text::uuid
  AND c.updated_at >= NOW() - ($3 * INTERVAL '1 day')
ORDER BY t.turn_number DESC, i.ordinal ASC
LIMIT $4
"""

_FETCH_IMAGES_BY_IDS = """
SELECT
    i.image_id::text AS image_id,
    i.mime_type,
    i.image_bytes,
    i.vlm_description
FROM web_conversation_images AS i
JOIN web_conversations AS c
  ON c.principal_id = i.principal_id
 AND c.conversation_id = i.conversation_id
WHERE i.principal_id = $1
  AND i.conversation_id = $2::text::uuid
  AND i.image_id = ANY($3::uuid[])
  AND c.updated_at >= NOW() - ($4 * INTERVAL '1 day')
"""


_LOAD_ATTACHMENT_CHUNKS = """
SELECT
    c.chunk_id,
    c.chunk_index,
    c.content,
    c.page_idx,
    c.bbox,
    c.sidecar_type,
    c.image_bytes,
    c.image_mime_type,
    c.token_estimate,
    c.metadata,
    c.parser_signature,
    c.chunk_signature
FROM web_conversation_attachment_chunks AS c
WHERE c.principal_id = $1
  AND c.conversation_id = $2::text::uuid
  AND c.content_sha256 = $3
  AND c.parser_signature = $4
  AND c.chunk_signature = $5
ORDER BY c.chunk_index ASC
"""

_DELETE_ATTACHMENT_CHUNKS_FOR_SIGNATURE = """
DELETE FROM web_conversation_attachment_chunks
WHERE principal_id = $1
  AND conversation_id = $2::text::uuid
  AND content_sha256 = $3
  AND parser_signature = $4
  AND chunk_signature = $5
"""

_INSERT_ATTACHMENT_CHUNK = """
INSERT INTO web_conversation_attachment_chunks (
    principal_id, conversation_id, content_sha256, parser_signature, chunk_signature,
    chunk_id, chunk_index, content, page_idx, bbox, sidecar_type,
    image_bytes, image_mime_type, token_estimate, metadata
)
VALUES ($1, $2::text::uuid, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, $11, $12, $13, $14, $15::jsonb)
"""

_FETCH_DOCUMENTS_BY_IDS = """
SELECT
    a.attachment_id::text AS attachment_id,
    a.filename,
    a.mime_type,
    a.suffix,
    a.attachment_bytes,
    a.content_sha256
FROM web_conversation_attachments AS a
JOIN web_conversations AS c
  ON c.principal_id = a.principal_id
 AND c.conversation_id = a.conversation_id
WHERE a.principal_id = $1
  AND a.conversation_id = $2::text::uuid
  AND a.attachment_id = ANY($3::uuid[])
  AND c.updated_at >= NOW() - ($4 * INTERVAL '1 day')
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
class PendingConversationAttachment:
    attachment_id: str
    ordinal: int
    filename: str
    mime_type: str
    suffix: str
    attachment_bytes: bytes
    content_sha256: str
    parse_summary: str | None = None
    parse_catalog: dict[str, Any] = field(default_factory=dict)

    @property
    def byte_size(self) -> int:
        return len(self.attachment_bytes)


@dataclass(frozen=True, slots=True)
class CommitTurnResult:
    saved: bool
    reason: str | None
    summary: dict[str, Any] | None
    turn_id: str | None
    current_image_ids: tuple[str, ...] = ()
    current_attachment_ids: tuple[str, ...] = ()
    assistant_text: str | None = None
    answer_sources: dict[str, Any] | None = None
    image_descriptions: dict[str, str] = field(default_factory=dict)
    replayed: bool = False


@dataclass(frozen=True, slots=True)
class StoredConversationImage:
    image_id: str
    mime_type: str
    image_bytes: bytes
    vlm_description: str | None = None


@dataclass(frozen=True, slots=True)
class StoredConversationAttachment:
    attachment_id: str
    filename: str
    mime_type: str
    suffix: str
    attachment_bytes: bytes
    content_sha256: str = ""
    parse_summary: str | None = None
    parse_catalog: dict[str, Any] = field(default_factory=dict)

    @property
    def document_bytes(self) -> bytes:
        return self.attachment_bytes


def _row_dict(row: Any) -> dict[str, Any]:
    return dict(row)


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _history_row(row: Any) -> dict[str, Any]:
    result = _row_dict(row)
    for key in ("answer_sources", "queried_workspaces", "images", "attachments"):
        result[key] = _json_value(result[key])
    return result


def _committed_result(row: Any, *, replayed: bool) -> CommitTurnResult:
    value = _row_dict(row)
    images = _json_value(value.get("images", []))
    attachments = _json_value(value.get("attachments", []))
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
        current_attachment_ids=tuple(str(item["attachment_id"]) for item in attachments),
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

    async def list_image_catalog(
        self, principal_id: str, conversation_id: str, *, max_turns: int, ttl_days: int
    ) -> list[dict[str, Any]]:
        """Return scoped ``{image_id, turn_number, ordinal, vlm_description}`` rows.

        Captions only — never image bytes — so the web-variant planner can pick
        which prior images the follow-up refers to without loading blobs.
        """
        await self._ensure_initialized()

        async def _operation(conn: Any) -> list[dict[str, Any]]:
            rows = await conn.fetch(
                _LIST_IMAGE_CATALOG, principal_id, conversation_id, ttl_days, max_turns
            )
            return [dict(row) for row in rows]

        return await self._run_read(_operation)

    async def fetch_images_by_ids(
        self,
        principal_id: str,
        conversation_id: str,
        image_ids: list[str],
        *,
        ttl_days: int,
    ) -> list[StoredConversationImage]:
        """Return owned, unexpired image bytes + captions for the given ids.

        The caption travels with the bytes so the web layer can fold persisted
        descriptions into the retrieval query without re-running the VLM.
        """
        if not image_ids:
            return []
        await self._ensure_initialized()

        async def _operation(conn: Any) -> list[StoredConversationImage]:
            rows = await conn.fetch(
                _FETCH_IMAGES_BY_IDS, principal_id, conversation_id, image_ids, ttl_days
            )
            return [
                StoredConversationImage(
                    image_id=row["image_id"],
                    mime_type=row["mime_type"],
                    image_bytes=row["image_bytes"],
                    vlm_description=row["vlm_description"],
                )
                for row in rows
            ]

        return await self._run_read(_operation)

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

    async def get_attachment(
        self,
        principal_id: str,
        conversation_id: str,
        attachment_id: str,
        *,
        ttl_days: int,
    ) -> StoredConversationAttachment | None:
        """Load original document bytes only when ownership and TTL match."""
        await self._ensure_initialized()

        async def _operation(conn: Any) -> StoredConversationAttachment | None:
            row = await conn.fetchrow(
                _GET_ATTACHMENT,
                principal_id,
                conversation_id,
                attachment_id,
                ttl_days,
            )
            if row is None:
                return None
            return StoredConversationAttachment(
                attachment_id=str(row["attachment_id"]),
                filename=str(row["filename"]),
                mime_type=str(row["mime_type"]),
                suffix=str(row["suffix"]),
                attachment_bytes=bytes(row["attachment_bytes"]),
                content_sha256=str(row["content_sha256"]),
            )

        return await self._run_read(_operation)

    async def fetch_documents_by_ids(
        self,
        principal_id: str,
        conversation_id: str,
        attachment_ids: list[str],
        *,
        ttl_days: int,
    ) -> list[StoredConversationAttachment]:
        """Return owned, unexpired original document bytes for the given ids.

        Used by the planner-driven history-document path to rematerialize prior
        attachments the follow-up query refers to without a fresh upload.
        """
        if not attachment_ids:
            return []
        await self._ensure_initialized()

        async def _operation(conn: Any) -> list[StoredConversationAttachment]:
            rows = await conn.fetch(
                _FETCH_DOCUMENTS_BY_IDS,
                principal_id,
                conversation_id,
                attachment_ids,
                ttl_days,
            )
            return [
                StoredConversationAttachment(
                    attachment_id=str(row["attachment_id"]),
                    filename=str(row["filename"]),
                    mime_type=str(row["mime_type"]),
                    suffix=str(row["suffix"]),
                    attachment_bytes=bytes(row["attachment_bytes"]),
                    content_sha256=str(row["content_sha256"]),
                )
                for row in rows
            ]

        return await self._run_read(_operation)

    async def load_attachment_chunks(
        self,
        principal_id: str,
        conversation_id: str,
        attachment_id: str,
        filename: str,
        *,
        content_sha256: str,
        parser_signature: str,
        chunk_signature: str,
    ) -> ParsedAttachmentBundle | None:
        """Return the cached parse for one content+parser+chunk signature, if any.

        Cache rows are content-addressed and may be reused for a new attachment
        id that shares the same bytes; chunk ids are remapped to the requested
        attachment so citations never point at a prior attachment.
        """
        from dlightrag.core.request.attachments import (
            AttachmentContextChunk,
            ParsedAttachmentBundle,
        )

        await self._ensure_initialized()

        async def _operation(conn: Any) -> ParsedAttachmentBundle | None:
            rows = await conn.fetch(
                _LOAD_ATTACHMENT_CHUNKS,
                principal_id,
                conversation_id,
                content_sha256,
                parser_signature,
                chunk_signature,
            )
            if not rows:
                return None
            chunks = [
                AttachmentContextChunk(
                    chunk_id=f"att:{attachment_id}:{int(row['chunk_index']):04d}",
                    attachment_id=attachment_id,
                    filename=filename,
                    chunk_index=int(row["chunk_index"]),
                    content=str(row["content"]),
                    token_estimate=int(row["token_estimate"]),
                    page_idx=row["page_idx"],
                    bbox=_json_value(row["bbox"]) if row["bbox"] is not None else None,
                    sidecar_type=row["sidecar_type"],
                    image_bytes=bytes(row["image_bytes"])
                    if row["image_bytes"] is not None
                    else None,
                    image_mime_type=row["image_mime_type"],
                    metadata=_json_value(row["metadata"]),
                )
                for row in rows
            ]
            return ParsedAttachmentBundle(chunks=chunks, parser_signature=parser_signature)

        return await self._run_read(_operation)

    async def materialize_attachment_chunks(
        self,
        principal_id: str,
        conversation_id: str,
        *,
        content_sha256: str,
        parser_signature: str,
        chunk_signature: str,
        bundle: ParsedAttachmentBundle,
    ) -> None:
        """Replace the cached parse for one content+parser+chunk signature."""
        await self._ensure_initialized()

        async def _operation(conn: Any) -> None:
            async with conn.transaction():
                await conn.execute(
                    _DELETE_ATTACHMENT_CHUNKS_FOR_SIGNATURE,
                    principal_id,
                    conversation_id,
                    content_sha256,
                    parser_signature,
                    chunk_signature,
                )
                for chunk in bundle.chunks:
                    await conn.execute(
                        _INSERT_ATTACHMENT_CHUNK,
                        principal_id,
                        conversation_id,
                        content_sha256,
                        parser_signature,
                        chunk_signature,
                        chunk.chunk_id,
                        chunk.chunk_index,
                        chunk.content,
                        chunk.page_idx,
                        json.dumps(chunk.bbox) if chunk.bbox is not None else None,
                        chunk.sidecar_type,
                        chunk.image_bytes,
                        chunk.image_mime_type,
                        chunk.token_estimate,
                        json.dumps(chunk.metadata),
                    )

        await self._run_write(_operation)

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
        images: list[PendingConversationImage],
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
                        attachment.parse_summary,
                        json.dumps(attachment.parse_catalog),
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
                    current_attachment_ids=tuple(
                        attachment.attachment_id for attachment in attachments
                    ),
                    assistant_text=assistant_text,
                    answer_sources=answer_sources,
                    image_descriptions={
                        str(image.ordinal): image.vlm_description
                        for image in images
                        if image.vlm_description is not None
                    },
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
    "PendingConversationAttachment",
    "PendingConversationImage",
    "StoredConversationAttachment",
    "StoredConversationImage",
]
