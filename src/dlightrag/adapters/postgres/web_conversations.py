# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL persistence for principal-scoped Web conversations.

A conversation is navigation and history; it owns no execution state. Each turn
is one row pointing at the durable Answer run that owns the request input, the
uploaded bytes, the streamed events, and the canonical result. The turn is
created inside the run's own creation transaction and keyed by the browser's
submission id, so history exists before the 202 response and no subscriber,
finalizer, or reconnect has to commit it afterwards.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any
from uuid import uuid4

import asyncpg

from dlightrag.adapters.postgres._errors import is_postgres_unavailable
from dlightrag.adapters.postgres._migrations import (
    ForeignKeyRequirement,
    Migration,
    TableRequirement,
    apply_migrations,
    verify_migrations,
)
from dlightrag.adapters.postgres._operations import ConnectionPool, PostgresOperationRunner
from dlightrag.adapters.postgres.answer_runs import (
    ANSWER_RUN_MIGRATION_SCOPE,
    ANSWER_RUN_MIGRATIONS,
    ANSWER_RUN_SCHEMA_TABLES,
    PGAnswerRunStore,
    answer_run_columns,
    answer_run_record,
)
from dlightrag.runtime import (
    PendingArtifact,
    PendingArtifactReference,
    RunDeletion,
    RunSchemaError,
)
from dlightrag.web.conversation_models import (
    AnswerTurnCreation,
    ConversationSnapshot,
    ConversationSubmissionConflict,
    LinkedTurn,
    WebConversationSchemaError,
    WebConversationUnavailableError,
)

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

_CREATE_LEGACY_TURNS = """
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

# The turn carries conversation order and the run link only. Request content,
# answer text, sources, and uploaded bytes all live in the run it references, so
# nothing about one answer is stored twice. Deleting the run deletes the turn:
# pruning a failed or cancelled run removes its visible terminal entry, and
# deleting a conversation deletes its runs in the same transaction.
_CREATE_TURNS = """
CREATE TABLE IF NOT EXISTS web_conversation_turns (
    turn_id UUID NOT NULL,
    principal_id TEXT NOT NULL,
    conversation_id UUID NOT NULL,
    turn_number INTEGER NOT NULL,
    submission_id UUID NOT NULL,
    answer_run_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (principal_id, conversation_id, turn_id),
    UNIQUE (principal_id, conversation_id, turn_number),
    FOREIGN KEY (principal_id, conversation_id)
      REFERENCES web_conversations (principal_id, conversation_id)
      ON DELETE CASCADE,
    FOREIGN KEY (principal_id, answer_run_id)
      REFERENCES dlightrag_answer_runs (owner_id, run_id)
      ON DELETE CASCADE
)
"""

_CREATE_CONVERSATION_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_web_conversations_principal_updated "
    "ON web_conversations (principal_id, updated_at DESC, conversation_id DESC)",
    "CREATE INDEX IF NOT EXISTS idx_web_conversations_updated "
    "ON web_conversations (updated_at, principal_id, conversation_id)",
)

_CREATE_TURN_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_web_conversation_turns_principal_conversation "
    "ON web_conversation_turns (principal_id, conversation_id, turn_number DESC)",
    # The submission id is the owner-wide idempotency key of the run itself, so
    # its turn must be unique in the same namespace rather than per conversation.
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_web_conversation_turns_submission "
    "ON web_conversation_turns (principal_id, submission_id)",
    # One turn per run, and the reverse lookup run retention uses to recognize a
    # conversation-linked run.
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_web_conversation_turns_run "
    "ON web_conversation_turns (principal_id, answer_run_id)",
)

# Intentional one-time reset. Earlier schemas stored a second copy of each
# answer (assistant text plus a source snapshot) and the raw uploaded bytes in a
# Web-owned attachment table. Both are superseded by the durable run, so this
# migration deletes every Web conversation, drops those tables, and recreates
# turns as pure run links. There is no compatibility view, dual write, or
# backfill: no committed turn may keep a payload the run no longer owns.
_RESET_WEB_CONVERSATIONS = "DELETE FROM web_conversations"
_DROP_ATTACHMENT_CHUNKS = "DROP TABLE IF EXISTS web_conversation_attachment_chunks"
_DROP_IMAGES = "DROP TABLE IF EXISTS web_conversation_images"
_DROP_ATTACHMENTS = "DROP TABLE IF EXISTS web_conversation_attachments"
_DROP_TURNS = "DROP TABLE IF EXISTS web_conversation_turns"

WEB_CONVERSATION_MIGRATIONS = (
    Migration(
        "0001_web_conversations",
        "Create scoped Web conversations and turns",
        (
            _CREATE_CONVERSATIONS,
            _CREATE_LEGACY_TURNS,
            *_CREATE_CONVERSATION_INDEXES,
        ),
    ),
    Migration(
        "0004_answer_run_turns",
        "Reset Web conversations and link every turn to its durable Answer run",
        (
            _RESET_WEB_CONVERSATIONS,
            _DROP_ATTACHMENT_CHUNKS,
            _DROP_IMAGES,
            _DROP_ATTACHMENTS,
            _DROP_TURNS,
            _CREATE_TURNS,
            *_CREATE_TURN_INDEXES,
        ),
    ),
)

WEB_CONVERSATION_SCHEMA_TABLES = (
    TableRequirement(
        name="web_conversations",
        columns=(
            "principal_id",
            "conversation_id",
            "title",
            "content_revision",
            "created_at",
            "updated_at",
        ),
        primary_key=("principal_id", "conversation_id"),
        indexes=(
            "idx_web_conversations_principal_updated",
            "idx_web_conversations_updated",
        ),
    ),
    TableRequirement(
        name="web_conversation_turns",
        columns=(
            "turn_id",
            "principal_id",
            "conversation_id",
            "turn_number",
            "submission_id",
            "answer_run_id",
            "created_at",
        ),
        primary_key=("principal_id", "conversation_id", "turn_id"),
        unique=(("principal_id", "conversation_id", "turn_number"),),
        foreign_keys=(
            ForeignKeyRequirement(
                columns=("principal_id", "conversation_id"), references="web_conversations"
            ),
            ForeignKeyRequirement(
                columns=("principal_id", "answer_run_id"), references="dlightrag_answer_runs"
            ),
        ),
        indexes=("idx_web_conversation_turns_principal_conversation",),
        unique_indexes=(
            "idx_web_conversation_turns_submission",
            "idx_web_conversation_turns_run",
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

_LOCK_CONVERSATION = f"""
SELECT {_SUMMARY_COLUMNS}
FROM web_conversations
WHERE principal_id = $1
  AND conversation_id = $2::text::uuid
  AND updated_at >= NOW() - ($3 * INTERVAL '1 day')
FOR UPDATE
"""  # noqa: S608 - interpolates only the trusted _SUMMARY_COLUMNS constant

_SELECT_CONVERSATION_RUNS = """
SELECT answer_run_id::text AS answer_run_id
FROM web_conversation_turns
WHERE principal_id = $1 AND conversation_id = $2::text::uuid
"""

_SELECT_PRINCIPAL_RUNS = """
SELECT answer_run_id::text AS answer_run_id
FROM web_conversation_turns
WHERE principal_id = $1
"""

# Deletion snapshots run ids under this lock, so a submission that commits its
# own turn concurrently is either already visible or still waiting behind it.
# The order matches creation and retention: the conversation first, its runs
# second. Ordering by conversation_id keeps two owner-wide deletes deadlock-free.
_LOCK_PRINCIPAL_CONVERSATIONS = """
SELECT conversation_id
FROM web_conversations
WHERE principal_id = $1
ORDER BY conversation_id
FOR UPDATE
"""

_DELETE_CONVERSATION = """
DELETE FROM web_conversations
WHERE principal_id = $1
  AND conversation_id = $2::text::uuid
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

_TURN_COLUMNS = """
t.turn_id::text AS turn_id,
t.turn_number,
t.submission_id::text AS submission_id,
t.answer_run_id::text AS answer_run_id,
t.conversation_id::text AS turn_conversation_id,
t.created_at AS turn_created_at
"""

_GET_TURNS = f"""
SELECT
{_TURN_COLUMNS},
{answer_run_columns("r")}
FROM web_conversation_turns AS t
JOIN dlightrag_answer_runs AS r
  ON r.owner_id = t.principal_id
 AND r.run_id = t.answer_run_id
JOIN web_conversations AS c
  ON c.principal_id = t.principal_id
 AND c.conversation_id = t.conversation_id
WHERE t.principal_id = $1
  AND t.conversation_id = $2::text::uuid
  AND c.updated_at >= NOW() - ($3 * INTERVAL '1 day')
ORDER BY t.turn_number DESC
LIMIT $4
"""  # noqa: S608 - interpolates only trusted column-projection constants

_GET_TURN_BY_SUBMISSION = f"""
SELECT
{_TURN_COLUMNS},
r.request_fingerprint,
{answer_run_columns("r")}
FROM web_conversation_turns AS t
JOIN dlightrag_answer_runs AS r
  ON r.owner_id = t.principal_id
 AND r.run_id = t.answer_run_id
WHERE t.principal_id = $1
  AND t.submission_id = $2::text::uuid
"""  # noqa: S608 - interpolates only trusted column-projection constants

_GET_TURN_BY_RUN = f"""
SELECT
{_TURN_COLUMNS},
{answer_run_columns("r")}
FROM web_conversation_turns AS t
JOIN dlightrag_answer_runs AS r
  ON r.owner_id = t.principal_id
 AND r.run_id = t.answer_run_id
WHERE t.principal_id = $1
  AND t.answer_run_id = $2::text::uuid
"""  # noqa: S608 - interpolates only trusted column-projection constants

_TOUCH_CONVERSATION = f"""
UPDATE web_conversations
SET content_revision = content_revision + 1,
    title = COALESCE(title, $3),
    updated_at = NOW()
WHERE principal_id = $1
  AND conversation_id = $2::text::uuid
RETURNING {_SUMMARY_COLUMNS}
"""  # noqa: S608 - interpolates only the trusted _SUMMARY_COLUMNS constant

_INSERT_TURN = """
INSERT INTO web_conversation_turns (
    turn_id,
    principal_id,
    conversation_id,
    turn_number,
    submission_id,
    answer_run_id
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
    $5::text::uuid
)
RETURNING turn_id::text AS turn_id, turn_number
"""

_SELECT_TRIMMABLE_RUNS = """
SELECT answer_run_id::text AS answer_run_id
FROM web_conversation_turns
WHERE principal_id = $1
  AND conversation_id = $2::text::uuid
  AND turn_number <= $3
"""

_SELECT_EXPIRED_CONVERSATIONS = """
SELECT principal_id, conversation_id
FROM web_conversations
WHERE updated_at < NOW() - ($1 * INTERVAL '1 day')
ORDER BY updated_at, principal_id, conversation_id
LIMIT $2
FOR UPDATE SKIP LOCKED
"""

_SELECT_RUNS_FOR_CONVERSATIONS = """
SELECT t.principal_id, t.answer_run_id::text AS answer_run_id
FROM web_conversation_turns AS t
WHERE (t.principal_id, t.conversation_id)
      IN (SELECT * FROM unnest($1::text[], $2::uuid[]))
"""

_DELETE_CONVERSATIONS = """
WITH deleted AS (
    DELETE FROM web_conversations AS conversations
    WHERE (conversations.principal_id, conversations.conversation_id)
          IN (SELECT * FROM unnest($1::text[], $2::uuid[]))
    RETURNING 1
)
SELECT count(*)::int AS count FROM deleted
"""


#: The two indexes that make one submission id one turn in the owner's namespace.
#: Only these mean "this key is already accepted"; any other violation is a bug.
_SUBMISSION_KEY_INDEXES = frozenset(
    {"idx_web_conversation_turns_submission", "idx_web_conversation_turns_run"}
)


def _row_dict(row: Any) -> dict[str, Any]:
    return dict(row)


def _linked_turn(row: Any) -> LinkedTurn:
    return LinkedTurn(
        turn_id=str(row["turn_id"]),
        turn_number=int(row["turn_number"]),
        submission_id=str(row["submission_id"]),
        created_at=row["turn_created_at"],
        run=answer_run_record(row),
    )


class PGWebConversationStore(PostgresOperationRunner):
    """Durable PostgreSQL store for server-owned Web conversations."""

    def __init__(
        self,
        *,
        pool: ConnectionPool | None = None,
        run_store: PGAnswerRunStore | None = None,
    ) -> None:
        super().__init__(pool=pool)
        self._run_store = run_store or PGAnswerRunStore(pool=pool)
        self._initialized = False

    async def _run_read[T](self, operation: Callable[[Any], Awaitable[T]]) -> T:
        try:
            return await self._run(operation)
        except Exception as exc:
            if is_postgres_unavailable(exc):
                raise WebConversationUnavailableError from exc
            raise

    async def _run_write[T](self, operation: Callable[[Any], Awaitable[T]]) -> T:
        try:
            return await self._run_once(operation)
        except Exception as exc:
            if is_postgres_unavailable(exc):
                raise WebConversationUnavailableError from exc
            raise

    async def initialize(self, *, validate_only: bool = False) -> None:
        """Create the Web conversation schema, or validate it (reader).

        A turn's foreign key targets the durable Answer run table, so the run
        schema is established first in the same connection.
        """
        if self._initialized:
            return

        async def _operation(conn: Any) -> None:
            if validate_only:
                await verify_migrations(
                    conn,
                    scope=ANSWER_RUN_MIGRATION_SCOPE,
                    migrations=ANSWER_RUN_MIGRATIONS,
                    tables=ANSWER_RUN_SCHEMA_TABLES,
                    schema_error=RunSchemaError,
                )
                await verify_migrations(
                    conn,
                    scope="web_conversations",
                    migrations=WEB_CONVERSATION_MIGRATIONS,
                    tables=WEB_CONVERSATION_SCHEMA_TABLES,
                    schema_error=WebConversationSchemaError,
                )
                return
            await apply_migrations(
                conn,
                scope=ANSWER_RUN_MIGRATION_SCOPE,
                migrations=ANSWER_RUN_MIGRATIONS,
                schema_error=RunSchemaError,
            )
            await apply_migrations(
                conn,
                scope="web_conversations",
                migrations=WEB_CONVERSATION_MIGRATIONS,
                schema_error=WebConversationSchemaError,
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
        """Delete one owned conversation, its turns, and the runs they linked.

        The conversation is locked before its run ids are read, so a submission
        committing its own turn concurrently is either already visible here or
        still waiting behind this transaction; neither leaves an orphan run.
        Deleting the runs in the same transaction stops any lease-fenced worker
        from appending to state the conversation no longer owns, cascades the
        runs' events, turns, and artifact references, and releases blobs no
        surviving run still references. Runs are deleted before the conversation
        so this path takes the run lock before the turn lock, exactly as run
        retention does; the reverse order deadlocks against a concurrent prune.
        """
        await self._ensure_initialized()

        async def _operation(conn: Any) -> bool:
            async with conn.transaction():
                if (
                    await conn.fetchrow(_LOCK_CONVERSATION, principal_id, conversation_id, ttl_days)
                    is None
                ):
                    return False
                run_ids = [
                    str(row["answer_run_id"])
                    for row in await conn.fetch(
                        _SELECT_CONVERSATION_RUNS, principal_id, conversation_id
                    )
                ]
                await self._run_store.delete_runs_in(conn, owner_id=principal_id, run_ids=run_ids)
                await conn.execute(_DELETE_CONVERSATION, principal_id, conversation_id)
                return True

        return await self._run_write(_operation)

    async def delete_all_conversations(self, principal_id: str) -> int:
        """Delete every conversation owned by one principal and its linked runs."""
        await self._ensure_initialized()

        async def _operation(conn: Any) -> int:
            async with conn.transaction():
                await conn.fetch(_LOCK_PRINCIPAL_CONVERSATIONS, principal_id)
                run_ids = [
                    str(row["answer_run_id"])
                    for row in await conn.fetch(_SELECT_PRINCIPAL_RUNS, principal_id)
                ]
                await self._run_store.delete_runs_in(conn, owner_id=principal_id, run_ids=run_ids)
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
        """Load one unexpired conversation and its run-linked recent turns."""
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
                turn_rows = await conn.fetch(
                    _GET_TURNS,
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
                    turns=tuple(_linked_turn(turn) for turn in reversed(turn_rows)),
                )

        return await self._run_read(_operation)

    async def find_turn_by_run(self, principal_id: str, run_id: str) -> LinkedTurn | None:
        """Return the conversation entry one owned run belongs to, if any."""
        await self._ensure_initialized()

        async def _operation(conn: Any) -> LinkedTurn | None:
            row = await conn.fetchrow(_GET_TURN_BY_RUN, principal_id, run_id)
            return _linked_turn(row) if row is not None else None

        return await self._run_read(_operation)

    async def create_answer_turn(
        self,
        *,
        principal_id: str,
        conversation_id: str,
        submission_id: str,
        request: Mapping[str, Any],
        idempotency_fingerprint: str,
        artifacts: Sequence[PendingArtifact] = (),
        references: Sequence[PendingArtifactReference] = (),
        title_hint: str | None,
        max_turns: int,
        ttl_days: int,
    ) -> AnswerTurnCreation | None:
        """Accept one submission as a run plus its conversation entry, atomically.

        The owned conversation is locked, the run and its uploaded bytes are
        created or replayed under the submission id, and the linked turn is
        inserted before this transaction commits. Replaying the same submission
        returns the authoritative run and turn; reusing it for a different
        conversation or different normalized input is a conflict. ``None`` means
        the conversation does not exist for this principal or has expired.
        """
        await self._ensure_initialized()
        turn_id = str(uuid4())
        fingerprint = idempotency_fingerprint

        async def _operation(conn: Any) -> AnswerTurnCreation | None:
            async with conn.transaction():
                summary_row = await conn.fetchrow(
                    _LOCK_CONVERSATION, principal_id, conversation_id, ttl_days
                )
                if summary_row is None:
                    return None
                existing = await conn.fetchrow(_GET_TURN_BY_SUBMISSION, principal_id, submission_id)
                if existing is not None:
                    if str(existing["turn_conversation_id"]) != str(conversation_id):
                        raise ConversationSubmissionConflict(
                            "submission id was reused in a different conversation"
                        )
                    if str(existing["request_fingerprint"]) != fingerprint:
                        raise ConversationSubmissionConflict(
                            "submission id was reused with different request input"
                        )
                    return AnswerTurnCreation(
                        turn=_linked_turn(existing),
                        summary=_row_dict(summary_row),
                        replayed=True,
                    )
                creation = await self._run_store.create_run_in(
                    conn,
                    owner_id=principal_id,
                    request=request,
                    idempotency_fingerprint=fingerprint,
                    idempotency_key=submission_id,
                    artifacts=artifacts,
                    references=references,
                )
                try:
                    turn_row = await conn.fetchrow(
                        _INSERT_TURN,
                        turn_id,
                        principal_id,
                        conversation_id,
                        submission_id,
                        creation.run.run_id,
                    )
                except asyncpg.UniqueViolationError as exc:
                    # A concurrent submission committed this key first: both saw
                    # no turn and both replayed the one run the key owns, so the
                    # loser is a reuse conflict rather than a server fault. The
                    # transaction unwinds, leaving that run and its bytes alone.
                    if getattr(exc, "constraint_name", None) not in _SUBMISSION_KEY_INDEXES:
                        raise
                    raise ConversationSubmissionConflict(
                        "submission id was accepted concurrently for another conversation"
                    ) from None
                if turn_row is None:
                    raise RuntimeError("conversation turn insert returned no row")
                touched = await conn.fetchrow(
                    _TOUCH_CONVERSATION, principal_id, conversation_id, title_hint
                )
                turn_number = int(turn_row["turn_number"])
                await self._trim_turns(
                    conn,
                    principal_id=principal_id,
                    conversation_id=conversation_id,
                    before_or_at=turn_number - max_turns,
                )
                return AnswerTurnCreation(
                    turn=LinkedTurn(
                        turn_id=str(turn_row["turn_id"]),
                        turn_number=turn_number,
                        submission_id=submission_id,
                        created_at=creation.run.created_at,
                        run=creation.run,
                    ),
                    summary=_row_dict(touched if touched is not None else summary_row),
                    replayed=creation.replayed,
                )

        return await self._run_write(_operation)

    async def replay_answer_turn(
        self,
        *,
        principal_id: str,
        conversation_id: str,
        submission_id: str,
        idempotency_fingerprint: str,
    ) -> LinkedTurn | None:
        """Return an accepted browser submission before resolved input is rebuilt."""
        await self._ensure_initialized()

        async def _operation(conn: Any) -> LinkedTurn | None:
            existing = await conn.fetchrow(_GET_TURN_BY_SUBMISSION, principal_id, submission_id)
            if existing is None:
                return None
            if str(existing["turn_conversation_id"]) != str(conversation_id):
                raise ConversationSubmissionConflict(
                    "submission id was reused in a different conversation"
                )
            if str(existing["request_fingerprint"]) != idempotency_fingerprint:
                raise ConversationSubmissionConflict(
                    "submission id was reused with different request input"
                )
            return _linked_turn(existing)

        return await self._run_read(_operation)

    async def _trim_turns(
        self,
        conn: Any,
        *,
        principal_id: str,
        conversation_id: str,
        before_or_at: int,
    ) -> None:
        """Drop turns beyond the conversation window along with their runs."""
        if before_or_at <= 0:
            return
        rows = await conn.fetch(_SELECT_TRIMMABLE_RUNS, principal_id, conversation_id, before_or_at)
        await self._run_store.delete_runs_in(
            conn,
            owner_id=principal_id,
            run_ids=[str(row["answer_run_id"]) for row in rows],
        )

    async def prune_expired(self, *, ttl_days: int, batch_size: int = 500) -> int:
        """Delete one skip-locked batch of expired conversations and their runs."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        await self._ensure_initialized()

        async def _operation(conn: Any) -> int:
            async with conn.transaction():
                rows = await conn.fetch(_SELECT_EXPIRED_CONVERSATIONS, ttl_days, batch_size)
                if not rows:
                    return 0
                principals = [str(row["principal_id"]) for row in rows]
                conversation_ids = [row["conversation_id"] for row in rows]
                run_rows = await conn.fetch(
                    _SELECT_RUNS_FOR_CONVERSATIONS, principals, conversation_ids
                )
                await self._delete_runs_by_owner(conn, run_rows)
                deleted = await conn.fetchrow(_DELETE_CONVERSATIONS, principals, conversation_ids)
                return int(deleted["count"]) if deleted is not None else 0

        return await self._run_write(_operation)

    async def _delete_runs_by_owner(self, conn: Any, rows: Sequence[Any]) -> RunDeletion:
        """Delete linked runs one owner at a time, keeping every scope check."""
        by_owner: dict[str, list[str]] = {}
        for row in rows:
            by_owner.setdefault(str(row["principal_id"]), []).append(str(row["answer_run_id"]))
        runs = artifacts = 0
        for owner_id, run_ids in by_owner.items():
            outcome = await self._run_store.delete_runs_in(conn, owner_id=owner_id, run_ids=run_ids)
            runs += outcome.runs
            artifacts += outcome.artifacts
        return RunDeletion(runs=runs, artifacts=artifacts)


__all__ = [
    "WEB_CONVERSATION_MIGRATIONS",
    "WEB_CONVERSATION_SCHEMA_TABLES",
    "AnswerTurnCreation",
    "ConversationSnapshot",
    "ConversationSubmissionConflict",
    "LinkedTurn",
    "PGWebConversationStore",
]
