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

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any, cast
from uuid import UUID, uuid4

import asyncpg

from dlightrag.adapters.postgres.answer.session_repository import ensure_agent_session
from dlightrag.adapters.postgres.core._errors import is_postgres_unavailable
from dlightrag.adapters.postgres.core._migrations import (
    ForeignKeyRequirement,
    Migration,
    TableRequirement,
    apply_migrations,
    verify_migrations,
)
from dlightrag.adapters.postgres.core._operations import ConnectionPool, PostgresOperationRunner
from dlightrag.adapters.postgres.runtime._feedback import (
    answer_message_column,
    read_answer_message,
)
from dlightrag.adapters.postgres.runtime.run_store import (
    RUN_MIGRATION_SCOPE,
    RUN_MIGRATIONS,
    RUN_SCHEMA_TABLES,
    PGRunStore,
    run_columns,
    run_record,
)
from dlightrag.application.runs import RunView
from dlightrag.application.web_conversations.models import (
    RUN_ACTIVITY_MAX_LIMIT,
    STEERING_MESSAGES_PER_TURN,
    AgentspaceChange,
    AgentspaceFile,
    AgentspaceFileVersion,
    AgentspaceHead,
    AgentspaceSnapshot,
    AgentspaceSnapshotChanges,
    AnswerTurnCreation,
    CarriedAttachment,
    CommandTone,
    ConversationCommand,
    ConversationCursor,
    ConversationHead,
    ConversationHistoryCursor,
    ConversationHistoryPage,
    ConversationHistoryPageRequest,
    ConversationPageRequest,
    ConversationQueueConflict,
    ConversationRowPage,
    ConversationSubmissionConflict,
    GoalRoundGuard,
    GoalRoundRequest,
    LinkedTurn,
    LiveAnswerRun,
    QueuedDeliveryGuard,
    QueuedMessage,
    SubmissionSeed,
    WebConversationSchemaError,
    WebConversationUnavailableError,
    WebRunActivity,
    queued_delivery_digest,
)
from dlightrag.engine.answer.execution.connection_binding import RunConnectionBinding
from dlightrag.engine.answer.runs.envelope import accepted_input_envelope
from dlightrag.engine.answer.runs.routing import RoutingAcceptance
from dlightrag.engine.runtime.contracts import RunStatus
from dlightrag.engine.runtime.errors import RunSchemaError
from dlightrag.engine.runtime.records import (
    IdempotencyKeyConflict,
    PendingArtifact,
    PendingArtifactReference,
    PreparedRunEnvelope,
    RunDeletion,
    SessionWorkBusyError,
    parse_run_id,
)

_CREATE_CONVERSATIONS = """
CREATE TABLE IF NOT EXISTS web_conversations (
    principal_id TEXT NOT NULL,
    conversation_id UUID NOT NULL,
    agent_session_id UUID NOT NULL,
    agent_lane_id TEXT NOT NULL,
    title TEXT,
    content_revision BIGINT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    forked_from_conversation_id UUID,
    PRIMARY KEY (principal_id, conversation_id),
    CHECK (title IS NULL OR char_length(title) BETWEEN 1 AND 120)
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
      REFERENCES dlightrag_runs (owner_id, run_id)
      ON DELETE CASCADE
)
"""

# A command is a human-initiated transcript event with no Run behind it: the words and
# the server's own canonical result are all it carries (``result_data`` is the structured payload
# of that result, NULL for a command that has none). It allocates its ``sequence`` from
# the conversation's ``content_revision`` counter, exactly as a turn allocates
# ``turn_number``, so one counter orders both streams. Deleting the conversation deletes
# its commands with it.
_CREATE_COMMANDS = """
CREATE TABLE IF NOT EXISTS web_conversation_commands (
    principal_id TEXT NOT NULL,
    conversation_id UUID NOT NULL,
    command_id UUID NOT NULL,
    sequence BIGINT NOT NULL,
    name TEXT NOT NULL,
    input TEXT NOT NULL,
    result TEXT NOT NULL,
    tone TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    result_data JSONB,
    PRIMARY KEY (principal_id, conversation_id, command_id),
    UNIQUE (principal_id, conversation_id, sequence),
    FOREIGN KEY (principal_id, conversation_id)
      REFERENCES web_conversations (principal_id, conversation_id)
      ON DELETE CASCADE,
    CONSTRAINT web_conversation_commands_tone_check
      CHECK (tone IN ('success', 'error'))
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

_CREATE_COMMAND_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_web_conversation_commands_principal_conversation "
    "ON web_conversation_commands (principal_id, conversation_id, sequence DESC)",
)

# Messages a reader sent while a Run was live. A queued message is not a transcript
# event and starts no Run: it is coerced into one next-turn prompt after the live Run
# settles. Position is the conversation's queue order; the unique constraint is what
# makes a reorder a permutation rather than a duplicate or a loss.
_CREATE_GOAL_ROUNDS = """
CREATE TABLE IF NOT EXISTS web_conversation_goal_rounds (
    principal_id    TEXT        NOT NULL,
    conversation_id UUID        NOT NULL,
    request_id      UUID        NOT NULL,
    content         TEXT        NOT NULL,
    prompt_digest   TEXT        NOT NULL,
    workspaces      TEXT[]      NOT NULL DEFAULT '{}',
    goal_id         TEXT        NOT NULL DEFAULT '',
    goal_revision   BIGINT      NOT NULL DEFAULT 0,
    goal_round      INTEGER     NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (principal_id, conversation_id),
    FOREIGN KEY (principal_id, conversation_id)
        REFERENCES web_conversations (principal_id, conversation_id) ON DELETE CASCADE,
    CONSTRAINT web_conversation_goal_rounds_content_check
        CHECK (length(content) BETWEEN 1 AND 20000),
    CONSTRAINT web_conversation_goal_rounds_digest_check
        CHECK (prompt_digest ~ '^[0-9a-f]{64}$')
)
"""

_ENQUEUE_GOAL_ROUND = """
INSERT INTO web_conversation_goal_rounds
    (principal_id, conversation_id, request_id, content, prompt_digest, workspaces,
     goal_id, goal_revision, goal_round)
VALUES ($1, $2::text::uuid, $3::uuid, $4, $5, $6::text[], $7, $8, $9)
ON CONFLICT (principal_id, conversation_id) DO NOTHING
RETURNING request_id::text AS request_id
"""

_PENDING_GOAL_ROUND = """
SELECT request_id::text AS request_id, content, prompt_digest, workspaces,
       goal_id, goal_revision, goal_round, created_at
FROM web_conversation_goal_rounds
WHERE principal_id = $1 AND conversation_id = $2::text::uuid
"""

#: The workspaces the newest answer turn asked for: a goal round continues them. NULL when the
#: conversation has no turn yet (the caller falls back to its own default).
_NEWEST_ANSWER_WORKSPACES = """
SELECT ARRAY(
    SELECT jsonb_array_elements_text(
        COALESCE(r.prepared_input_json, r.accepted_input_json)->'workspaces'
    )
) AS workspaces
FROM web_conversation_turns AS t
JOIN dlightrag_runs AS r
  ON r.owner_id = t.principal_id
 AND r.run_id = t.answer_run_id
WHERE t.principal_id = $1
  AND t.conversation_id = $2::text::uuid
ORDER BY t.turn_number DESC
LIMIT 1
"""

#: The conversation that owns a Session. A conversation is its Session from creation, so this is
#: the one reverse lookup a startup recovery needs to start a round for a goal with no Run.
_CONVERSATION_FOR_SESSION = """
SELECT conversation_id::text AS conversation_id
FROM web_conversations
WHERE principal_id = $1 AND agent_session_id = $2::text::uuid
ORDER BY updated_at DESC
LIMIT 1
"""

#: Withdraw exactly the reservation the caller wrote. The request identity is part of the
#: predicate, not a post-read check: a stale attempt that lost its compare-and-set cannot delete
#: a newer attempt's row, and a row it does not own reports the same "already gone" as a consumed
#: one so the caller refunds nothing.
_CLEAR_GOAL_ROUND = """
DELETE FROM web_conversation_goal_rounds
WHERE principal_id = $1 AND conversation_id = $2::text::uuid
  AND request_id = $3::uuid AND prompt_digest = $4
RETURNING request_id::text AS request_id
"""

#: The delivery's own fence: delete exactly the continuation it read, in the same transaction
#: that creates the Run and its turn; a cleared or replaced row makes the delivery refuse.
_CONSUME_GOAL_ROUND = """
DELETE FROM web_conversation_goal_rounds
WHERE principal_id = $1 AND conversation_id = $2::text::uuid
  AND request_id = $3::uuid AND prompt_digest = $4
RETURNING request_id::text AS request_id
"""

#: The durable goal row a delivery must still find active, unchanged in identity and at the
#: round it reserved. Locked FOR UPDATE so a concurrent pause/clear/replace serializes with it.
_LOCK_GOAL_FOR_DELIVERY = """
SELECT goal_id, revision, phase, rounds_started
FROM dlightrag_answer_goals
WHERE owner_id = $1 AND session_id = $2::text::uuid
FOR UPDATE
"""

#: A delivery that finds a reserved-but-unadmitted round admits it here, in the same commit as
#: the turn it creates: the CAS is on the whole identity the row was reserved at.
_ADMIT_RESERVED_GOAL_ROUND = """
UPDATE dlightrag_answer_goals
SET rounds_started = rounds_started + 1,
    revision = revision + 1,
    updated_at = NOW()
WHERE owner_id = $1 AND session_id = $2::text::uuid
  AND goal_id = $3 AND revision = $4 AND phase = 'active'
  AND rounds_started = $5
  AND max_rounds > rounds_started
RETURNING revision
"""

_SUBMISSION_EXISTS = """
SELECT 1 FROM web_conversation_turns
WHERE principal_id = $1
  AND conversation_id = $2::text::uuid
  AND submission_id = $3::uuid
"""

_CREATE_QUEUE = """
CREATE TABLE IF NOT EXISTS web_conversation_queue (
    principal_id TEXT NOT NULL,
    conversation_id UUID NOT NULL,
    message_id UUID NOT NULL,
    position BIGINT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (principal_id, conversation_id, message_id),
    UNIQUE (principal_id, conversation_id, position),
    FOREIGN KEY (principal_id, conversation_id)
      REFERENCES web_conversations (principal_id, conversation_id)
      ON DELETE CASCADE,
    CONSTRAINT web_conversation_queue_content_check
      CHECK (char_length(content) BETWEEN 1 AND 20000),
    CONSTRAINT web_conversation_queue_position_check CHECK (position >= 1)
)
"""

WEB_CONVERSATION_MIGRATIONS = (
    Migration(
        "baseline",
        "Create Web conversations, turns, commands, queued messages, and goal continuations",
        (
            _CREATE_CONVERSATIONS,
            _CREATE_TURNS,
            *_CREATE_CONVERSATION_INDEXES,
            *_CREATE_TURN_INDEXES,
            _CREATE_COMMANDS,
            *_CREATE_COMMAND_INDEXES,
            _CREATE_QUEUE,
            _CREATE_GOAL_ROUNDS,
        ),
    ),
)

WEB_CONVERSATION_SCHEMA_TABLES = (
    TableRequirement(
        name="web_conversations",
        columns=(
            "principal_id",
            "conversation_id",
            "agent_session_id",
            "agent_lane_id",
            "title",
            "content_revision",
            "created_at",
            "updated_at",
            "forked_from_conversation_id",
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
                columns=("principal_id", "answer_run_id"), references="dlightrag_runs"
            ),
        ),
        indexes=("idx_web_conversation_turns_principal_conversation",),
        unique_indexes=(
            "idx_web_conversation_turns_submission",
            "idx_web_conversation_turns_run",
        ),
    ),
    TableRequirement(
        name="web_conversation_commands",
        columns=(
            "principal_id",
            "conversation_id",
            "command_id",
            "sequence",
            "name",
            "input",
            "result",
            "tone",
            "created_at",
            "result_data",
        ),
        primary_key=("principal_id", "conversation_id", "command_id"),
        unique=(("principal_id", "conversation_id", "sequence"),),
        checks=("web_conversation_commands_tone_check",),
        foreign_keys=(
            ForeignKeyRequirement(
                columns=("principal_id", "conversation_id"), references="web_conversations"
            ),
        ),
        indexes=("idx_web_conversation_commands_principal_conversation",),
    ),
    TableRequirement(
        name="web_conversation_queue",
        columns=(
            "principal_id",
            "conversation_id",
            "message_id",
            "position",
            "content",
            "created_at",
            "updated_at",
        ),
        primary_key=("principal_id", "conversation_id", "message_id"),
        unique=(("principal_id", "conversation_id", "position"),),
        checks=(
            "web_conversation_queue_content_check",
            "web_conversation_queue_position_check",
        ),
        foreign_keys=(
            ForeignKeyRequirement(
                columns=("principal_id", "conversation_id"), references="web_conversations"
            ),
        ),
    ),
    TableRequirement(
        name="web_conversation_goal_rounds",
        columns=(
            "principal_id",
            "conversation_id",
            "request_id",
            "content",
            "prompt_digest",
            "workspaces",
            "goal_id",
            "goal_revision",
            "goal_round",
            "created_at",
        ),
        primary_key=("principal_id", "conversation_id"),
        checks=(
            "web_conversation_goal_rounds_content_check",
            "web_conversation_goal_rounds_digest_check",
        ),
        foreign_keys=(
            ForeignKeyRequirement(
                columns=("principal_id", "conversation_id"), references="web_conversations"
            ),
        ),
    ),
)

# The queue is read and written under the conversation's row lock, so two appends get
# positions 1 and 2 rather than the same next number, and a reorder sees one stable set.
# A human send is delivered to the live Run instead of starting a competing one. The
# Session is the join: a goal round and a human turn are both answer Runs routed to the
# same agent_session_id, so the admission path never needs to know which started it.
_LIVE_ANSWER_RUN = """
SELECT r.run_id::text AS run_id, r.status,
       COALESCE(r.accepted_input_json, r.prepared_input_json)->>'continuation_kind'
         AS continuation_kind
FROM dlightrag_runs r
JOIN dlightrag_answer_run_routing routing
  ON routing.owner_id = r.owner_id AND routing.run_id = r.run_id
JOIN web_conversations c
  ON c.principal_id = $1
 AND c.conversation_id = $2::text::uuid
 AND c.agent_session_id = routing.agent_session_id
WHERE r.owner_id = $1
  AND r.run_kind = 'answer'
  AND r.status IN ('queued', 'running')
ORDER BY r.created_at DESC, r.run_id DESC
LIMIT 1
"""

#: One Session admits one live Run at a time. The claim runs in the same transaction
#: that links the conversation turn, so a settlement and an acceptance can never both
#: believe the Session is theirs; a slot already re-admitted to a successor is untouched.
_ADMIT_SESSION_WORK = """
INSERT INTO dlightrag_session_work_heads AS heads
    (owner_id, session_id, active_run_id, revision, updated_at)
VALUES ($1, $2::text::uuid, $3::uuid, 1, NOW())
ON CONFLICT (owner_id, session_id) DO UPDATE SET
    active_run_id = EXCLUDED.active_run_id,
    revision = heads.revision + 1,
    updated_at = NOW()
WHERE heads.active_run_id IS NULL
RETURNING revision
"""

_LOAD_SESSION_WORK_ACTIVE = """
SELECT active_run_id::text AS active_run_id
FROM dlightrag_session_work_heads
WHERE owner_id = $1 AND session_id = $2::text::uuid
"""

_RELEASE_SESSION_WORK_ROWS = """
DELETE FROM dlightrag_session_work_heads
WHERE owner_id = $1 AND session_id = $2::text::uuid
"""

_CONVERSATION_EXISTS = (
    "SELECT 1 FROM web_conversations WHERE principal_id = $1 AND conversation_id = $2::text::uuid"
)

_QUEUE_ROWS = (
    "SELECT message_id::text AS message_id, content, position, created_at "
    "FROM web_conversation_queue "
    "WHERE principal_id = $1 AND conversation_id = $2::text::uuid "
    "ORDER BY position"
)

_APPEND_QUEUED_MESSAGE = """
INSERT INTO web_conversation_queue
    (principal_id, conversation_id, message_id, position, content)
VALUES ($1, $2::text::uuid, $3::text::uuid,
        COALESCE((SELECT MAX(position) FROM web_conversation_queue
                  WHERE principal_id = $1 AND conversation_id = $2::text::uuid), 0) + 1,
        $4)
RETURNING message_id::text AS message_id, content, position, created_at
"""

_EDIT_QUEUED_MESSAGE = """
UPDATE web_conversation_queue
SET content = $4, updated_at = NOW()
WHERE principal_id = $1
  AND conversation_id = $2::text::uuid
  AND message_id = $3::text::uuid
RETURNING message_id::text AS message_id, content, position, created_at
"""

_REMOVE_QUEUED_MESSAGE = """
DELETE FROM web_conversation_queue
WHERE principal_id = $1
  AND conversation_id = $2::text::uuid
  AND message_id = $3::text::uuid
RETURNING message_id
"""

_REMOVE_QUEUED_MESSAGES = """
DELETE FROM web_conversation_queue
WHERE principal_id = $1
  AND conversation_id = $2::text::uuid
  AND message_id = ANY($3::uuid[])
RETURNING message_id
"""

_QUEUE_IDS_FOR_UPDATE = """
SELECT message_id::text AS message_id
FROM web_conversation_queue
WHERE principal_id = $1 AND conversation_id = $2::text::uuid
ORDER BY position
FOR UPDATE
"""

_SHIFT_QUEUE_POSITIONS = """
UPDATE web_conversation_queue
SET position = position + 1000000
WHERE principal_id = $1 AND conversation_id = $2::text::uuid
"""

_SET_QUEUE_POSITION = """
UPDATE web_conversation_queue
SET position = $4, updated_at = NOW()
WHERE principal_id = $1
  AND conversation_id = $2::text::uuid
  AND message_id = $3::text::uuid
"""

_SUMMARY_COLUMNS = """
conversation_id::text AS conversation_id,
agent_session_id::text AS agent_session_id,
agent_lane_id,
title,
content_revision,
created_at,
updated_at,
forked_from_conversation_id::text AS forked_from_conversation_id,
(
    SELECT parent.title FROM web_conversations parent
    WHERE parent.principal_id = web_conversations.principal_id
      AND parent.conversation_id = web_conversations.forked_from_conversation_id
) AS forked_from_title
"""

_CREATE_CONVERSATION = f"""
INSERT INTO web_conversations (
    principal_id, conversation_id, agent_session_id, agent_lane_id)
VALUES ($1, $2::text::uuid, $2::text::uuid, 'main')
RETURNING {_SUMMARY_COLUMNS}
"""  # noqa: S608 - interpolates only the trusted _SUMMARY_COLUMNS constant

_CREATE_CONVERSATION_IF_MISSING = f"""
INSERT INTO web_conversations (
    principal_id, conversation_id, forked_from_conversation_id,
    agent_session_id, agent_lane_id)
VALUES ($1, $2::text::uuid, $3::text::uuid, $4::text::uuid, $5)
ON CONFLICT (principal_id, conversation_id) DO NOTHING
RETURNING {_SUMMARY_COLUMNS}
"""  # noqa: S608 - interpolates only the trusted _SUMMARY_COLUMNS constant

_LIST_CONVERSATIONS_FIRST_PAGE = f"""
SELECT {_SUMMARY_COLUMNS}
FROM web_conversations
WHERE principal_id = $1
ORDER BY updated_at DESC, conversation_id DESC
LIMIT $2
"""  # noqa: S608 - interpolates only the trusted _SUMMARY_COLUMNS constant

_LIST_CONVERSATIONS_AFTER = f"""
SELECT {_SUMMARY_COLUMNS}
FROM web_conversations
WHERE principal_id = $1
  AND (updated_at, conversation_id) < ($2::timestamptz, $3::uuid)
ORDER BY updated_at DESC, conversation_id DESC
LIMIT $4
"""  # noqa: S608 - interpolates only the trusted _SUMMARY_COLUMNS constant

_RENAME_CONVERSATION = f"""
UPDATE web_conversations
SET title = $3,
    updated_at = NOW()
WHERE principal_id = $1
  AND conversation_id = $2::text::uuid
RETURNING {_SUMMARY_COLUMNS}
"""  # noqa: S608 - interpolates only the trusted _SUMMARY_COLUMNS constant

_LOCK_CONVERSATION = f"""
SELECT {_SUMMARY_COLUMNS}
FROM web_conversations
WHERE principal_id = $1
  AND conversation_id = $2::text::uuid
FOR UPDATE
"""  # noqa: S608 - interpolates only the trusted _SUMMARY_COLUMNS constant

_LOCK_AGENT_SESSION_IF_PRESENT = """
SELECT 1
FROM dlightrag_agent_sessions
WHERE owner_id = $1 AND session_id = $2::text::uuid
FOR UPDATE
"""

_REBASE_EMPTY_CONVERSATION = f"""
UPDATE web_conversations
SET agent_lane_id = 'main',
    content_revision = content_revision + 1,
    updated_at = NOW()
WHERE principal_id = $1
  AND conversation_id = $2::text::uuid
RETURNING {_SUMMARY_COLUMNS}
"""  # noqa: S608 - interpolates only the trusted _SUMMARY_COLUMNS constant

_DELETE_BATCH_SIZE = 128

_SELECT_CONVERSATION_RUN_BATCH = """
SELECT answer_run_id::text AS answer_run_id
FROM web_conversation_turns
WHERE principal_id = $1 AND conversation_id = $2::text::uuid
ORDER BY answer_run_id
LIMIT $3
"""

# A delete-all caller waits rather than skipping locked rows: returning while a
# pre-existing owned conversation survives would be a false success. Every
# caller locks the same UUID order, which also keeps two delete-all callers from
# deadlocking. Deleted rows fall out of the next first-page query, so no cursor,
# OFFSET, or transaction-wide identity set is needed.
_LOCK_PRINCIPAL_CONVERSATION_BATCH = """
SELECT conversations.conversation_id::text AS conversation_id,
       conversations.agent_session_id::text AS agent_session_id
FROM web_conversations AS conversations
WHERE conversations.principal_id = $1
ORDER BY conversations.conversation_id
LIMIT $2
FOR UPDATE OF conversations
"""

# This runs as a new READ COMMITTED statement after the conversation locks are
# acquired. It therefore sees a turn committed by a submitter that held a row
# lock before delete-all, unlike a projection in the blocking lock statement.
_SELECT_LINKED_CONVERSATION_BATCH = """
SELECT DISTINCT conversation_id::text AS conversation_id
FROM web_conversation_turns
WHERE principal_id = $1 AND conversation_id = ANY($2::uuid[])
ORDER BY conversation_id
LIMIT $3
"""

_DELETE_CONVERSATION = """
DELETE FROM web_conversations
WHERE principal_id = $1
  AND conversation_id = $2::text::uuid
RETURNING agent_session_id::text
"""

_DELETE_CONVERSATION_BATCH = """
WITH deleted AS (
    DELETE FROM web_conversations AS conversations
    WHERE conversations.principal_id = $1
      AND conversations.conversation_id = ANY($2::uuid[])
    RETURNING conversations.conversation_id, conversations.agent_session_id
)
SELECT conversation_id::text AS conversation_id,
       agent_session_id::text AS agent_session_id
FROM deleted
ORDER BY conversation_id
LIMIT $3
"""

_DELETE_AGENT_SESSION_IF_UNREFERENCED = """
DELETE FROM dlightrag_agent_sessions AS sessions
WHERE sessions.owner_id = $1 AND sessions.session_id = $2::text::uuid
  AND NOT EXISTS (
      SELECT 1 FROM dlightrag_answer_run_routing AS routing
      WHERE routing.owner_id = sessions.owner_id
        AND routing.agent_session_id = sessions.session_id
  )
"""

_DELETE_AGENT_SESSION_BATCH_IF_UNREFERENCED = """
DELETE FROM dlightrag_agent_sessions AS sessions
WHERE sessions.owner_id = $1 AND sessions.session_id = ANY($2::uuid[])
  AND NOT EXISTS (
      SELECT 1 FROM dlightrag_answer_run_routing AS routing
      WHERE routing.owner_id = sessions.owner_id
        AND routing.agent_session_id = sessions.session_id
  )
"""

_GET_CONVERSATION = """
SELECT
    principal_id,
    conversation_id::text AS conversation_id,
    agent_session_id::text AS agent_session_id,
    agent_lane_id,
    content_revision,
    title,
    created_at,
    updated_at
FROM web_conversations
WHERE principal_id = $1
  AND conversation_id = $2::text::uuid
"""

# What the Run answered and how its owner judged it. The main agent's Session holds every Run of
# its conversation, so the lookup is anchored on the Run's own entries (owner, run, sequence): an
# older turn would otherwise walk back past every newer one before finding its message.
_TURN_ANSWER_COLUMN = answer_message_column(
    alias="answer_message",
    entries="""entry.owner_id = t.principal_id AND entry.run_id = t.answer_run_id
      AND entry.session_id = (
          SELECT routing.agent_session_id
          FROM dlightrag_answer_run_routing AS routing
          WHERE routing.owner_id = t.principal_id AND routing.run_id = t.answer_run_id)""",
    owner="t.principal_id",
    run="t.answer_run_id",
    agent="'main'",
)

_TURN_COLUMNS = f"""
t.turn_id::text AS turn_id,
t.turn_number,
t.submission_id::text AS submission_id,
t.answer_run_id::text AS answer_run_id,
t.conversation_id::text AS turn_conversation_id,
t.created_at AS turn_created_at,
(SELECT count(*)::int
 FROM dlightrag_answer_child_sessions AS child
 WHERE child.owner_id = t.principal_id AND child.run_id = t.answer_run_id
) AS child_count,
{_TURN_ANSWER_COLUMN}
"""  # noqa: S608 - interpolates only the trusted _TURN_ANSWER_COLUMN constant

#: The user's own steering of one run, oldest first: the parent's inbox entries, not
#: a Child's targeted control nor a message the parent wrote itself.
_STEERING_COLUMN = f"""
(
    SELECT array_agg(s.content ORDER BY s.control_sequence)
    FROM (
        SELECT c.content, c.control_sequence
        FROM dlightrag_agent_controls AS c
        WHERE c.owner_id = r.owner_id
          AND c.run_id = r.run_id
          AND c.kind = 'steer'
          AND c.origin = 'user'
          AND c.target_session_id IS NULL
        ORDER BY c.control_sequence DESC
        LIMIT {STEERING_MESSAGES_PER_TURN}
    ) AS s
) AS steering_messages
"""  # noqa: S608 - interpolates only a trusted integer constant

_TURN_CONVERSATION_SUMMARY_COLUMNS = """
c.conversation_id::text AS conversation_id,
c.title,
c.content_revision,
c.created_at,
c.updated_at
"""

_GET_TURNS_PAGE = f"""
WITH selected_turns AS (
    SELECT t.*
    FROM web_conversation_turns AS t
    WHERE t.principal_id = $1
      AND t.conversation_id = $2::text::uuid
      AND ($3::bigint IS NULL OR t.turn_number < $3)
    ORDER BY t.turn_number DESC
    LIMIT $4
)
SELECT
{_TURN_COLUMNS},
{run_columns("r")},
{_STEERING_COLUMN}
FROM selected_turns AS t
JOIN dlightrag_runs AS r
  ON r.owner_id = t.principal_id
 AND r.run_id = t.answer_run_id
ORDER BY t.turn_number DESC
"""  # noqa: S608 - interpolates only trusted column-projection constants

# One command stream page under the same ordering key turns use: a conversation's
# content_revision allocates ``turn_number`` and command ``sequence`` from one counter,
# so "older than this sequence" means the same thing for both.
_GET_COMMANDS_PAGE = """
SELECT
    command_id::text AS command_id,
    sequence,
    name,
    input,
    result,
    tone,
    created_at,
    result_data
FROM web_conversation_commands
WHERE principal_id = $1
  AND conversation_id = $2::text::uuid
  AND ($3::bigint IS NULL OR sequence < $3)
ORDER BY sequence DESC
LIMIT $4
"""

# The conversation's own counter allocates both command sequences and turn numbers, and
# the bump returns the value the insert uses, so two concurrent command appends cannot
# share one sequence. Updating ``updated_at`` here is also what moves the conversation to
# the top of the owner's list, exactly as a turn does.
_APPEND_COMMAND = """
WITH allocated AS (
    UPDATE web_conversations
    SET content_revision = content_revision + 1,
        updated_at = NOW()
    WHERE principal_id = $1
      AND conversation_id = $2::text::uuid
    RETURNING content_revision
)
INSERT INTO web_conversation_commands (
    command_id,
    principal_id,
    conversation_id,
    sequence,
    name,
    input,
    result,
    tone,
    result_data
)
SELECT
    $3::text::uuid,
    $1,
    $2::text::uuid,
    allocated.content_revision,
    $4,
    $5,
    $6,
    $7,
    $8::jsonb
FROM allocated
RETURNING
    command_id::text AS command_id,
    sequence,
    name,
    input,
    result,
    tone,
    created_at,
    result_data
"""

_GET_TIP_RUN = """
SELECT t.answer_run_id::text AS answer_run_id
FROM web_conversation_turns AS t
WHERE t.principal_id = $1
  AND t.conversation_id = $2::text::uuid
ORDER BY t.turn_number DESC
LIMIT 1
"""

_GET_CARRIED_ATTACHMENTS = """
SELECT
    t.answer_run_id::text AS run_id,
    (attachment.value->>'ordinal')::integer AS source_ordinal,
    attachment.value->>'digest' AS digest,
    attachment.value->>'filename' AS filename,
    attachment.value->>'mime_type' AS mime_type,
    COALESCE((attachment.value->>'byte_size')::bigint, 0) AS byte_size
FROM web_conversation_turns AS t
JOIN dlightrag_runs AS r
  ON r.owner_id = t.principal_id
 AND r.run_id = t.answer_run_id
CROSS JOIN LATERAL jsonb_array_elements(
    COALESCE(r.prepared_input_json, r.accepted_input_json)->'attachments'
) AS attachment(value)
WHERE t.principal_id = $1
  AND t.conversation_id = $2::text::uuid
  AND r.status = 'succeeded'
ORDER BY t.turn_number DESC, source_ordinal ASC
LIMIT $3
"""

_GET_TURN_BY_SUBMISSION = f"""
SELECT
{_TURN_COLUMNS},
r.request_fingerprint,
{run_columns("r")},
{_STEERING_COLUMN},
{_TURN_CONVERSATION_SUMMARY_COLUMNS}
FROM web_conversation_turns AS t
JOIN dlightrag_runs AS r
  ON r.owner_id = t.principal_id
 AND r.run_id = t.answer_run_id
JOIN web_conversations AS c
  ON c.principal_id = t.principal_id
 AND c.conversation_id = t.conversation_id
WHERE t.principal_id = $1
  AND t.submission_id = $2::text::uuid
"""  # noqa: S608 - interpolates only trusted column-projection constants

_GET_TURN_BY_RUN = f"""
SELECT
{_TURN_COLUMNS},
{run_columns("r")},
{_STEERING_COLUMN}
FROM web_conversation_turns AS t
JOIN dlightrag_runs AS r
  ON r.owner_id = t.principal_id
 AND r.run_id = t.answer_run_id
WHERE t.principal_id = $1
  AND t.answer_run_id = $2::text::uuid
"""  # noqa: S608 - interpolates only trusted column-projection constants

# The reader's recent run activity: one hard page of the Answer runs a Web turn linked
# to a conversation, newest first. The turn is the conversation link, so a run accepted
# through REST/MCP (no turn) is absent; the owner predicate starts from the turn, whose
# (principal_id, answer_run_id) unique index covers the read, and the runs PK serves the
# join. The title is the conversation's own and stays null for an untitled one. A goal
# round's accepted envelope carries ``continuation_kind`` whether or not it has a parent Run
# (a ``/goal``-created round has none), so the projection states exactly the goal-continuation
# fact and no raw request field.
_RECENT_ANSWER_RUNS = """
SELECT
    r.run_id::text AS run_id,
    t.conversation_id::text AS conversation_id,
    c.title AS conversation_title,
    r.status,
    r.phase,
    r.created_at,
    r.started_at,
    r.finished_at,
    COALESCE(r.accepted_input_json ->> 'continuation_kind', '') = 'goal_round' AS goal_round
FROM web_conversation_turns AS t
JOIN dlightrag_runs AS r
  ON r.owner_id = t.principal_id
 AND r.run_id = t.answer_run_id
LEFT JOIN web_conversations AS c
  ON c.principal_id = t.principal_id
 AND c.conversation_id = t.conversation_id
WHERE t.principal_id = $1
  AND r.run_kind = 'answer'
ORDER BY r.created_at DESC, r.run_id DESC
LIMIT $2
"""

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
    $4,
    $5::text::uuid,
    $6::text::uuid
)
RETURNING turn_id::text AS turn_id, turn_number
"""

_SELECT_EMPTY_CONVERSATIONS = """
SELECT principal_id, conversation_id, agent_session_id
FROM web_conversations AS conversations
WHERE NOT EXISTS (
    SELECT 1
    FROM web_conversation_turns AS turns
    WHERE turns.principal_id = conversations.principal_id
      AND turns.conversation_id = conversations.conversation_id
)
ORDER BY updated_at, principal_id, conversation_id
LIMIT $1
FOR UPDATE SKIP LOCKED
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


def _conversation_head(row: Any) -> ConversationHead:
    return ConversationHead(
        principal_id=str(row["principal_id"]),
        conversation_id=str(row["conversation_id"]),
        content_revision=int(row["content_revision"]),
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        agent_session_id=str(row["agent_session_id"]),
        agent_lane_id=str(row["agent_lane_id"]),
    )


def _linked_turn(row: Any) -> LinkedTurn:
    answer_sequence, feedback = read_answer_message(row["answer_message"])
    return LinkedTurn(
        turn_id=str(row["turn_id"]),
        turn_number=int(row["turn_number"]),
        submission_id=str(row["submission_id"]),
        created_at=row["turn_created_at"],
        run=RunView.from_runtime(run_record(row)),
        conversation_id=str(row["turn_conversation_id"]),
        child_count=int(row["child_count"]),
        answer_sequence=answer_sequence,
        feedback=feedback,
        steering_messages=tuple(row["steering_messages"] or ()),
    )


def _command(row: Any) -> ConversationCommand:
    return ConversationCommand(
        command_id=str(row["command_id"]),
        sequence=int(row["sequence"]),
        name=str(row["name"]),
        input=str(row["input"]),
        result=str(row["result"]),
        tone=_command_tone(str(row["tone"])),
        created_at=row["created_at"],
        result_data=_command_result_data(row["result_data"]),
    )


def _command_result_data(value: Any) -> dict[str, Any] | None:
    """Decode the stored JSONB payload; asyncpg hands JSON columns back as text."""
    if value is None:
        return None
    loaded = json.loads(value) if isinstance(value, str) else value
    return dict(loaded) if isinstance(loaded, Mapping) else None


def _command_tone(value: str) -> CommandTone:
    """Narrow a stored ``tone`` to the two values the schema check admits."""
    if value == "success":
        return "success"
    if value == "error":
        return "error"
    raise RuntimeError("conversation command row carried an unknown tone")


def _queued_message(row: Any) -> QueuedMessage:
    """Map one queue row; its order is the conversation's, never a run's."""
    return QueuedMessage(
        message_id=str(row["message_id"]),
        content=str(row["content"]),
        position=int(row["position"]),
        created_at=row["created_at"],
    )


def _run_activity(row: Any) -> WebRunActivity:
    """Map one run-activity row, decoded by its explicit ``_RECENT_ANSWER_RUNS`` columns."""
    return WebRunActivity(
        run_id=str(row["run_id"]),
        conversation_id=str(row["conversation_id"]),
        conversation_title=(
            str(row["conversation_title"]) if row["conversation_title"] is not None else None
        ),
        status=cast(RunStatus, str(row["status"])),
        phase=str(row["phase"]) if row["phase"] is not None else None,
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        goal_round=bool(row["goal_round"]),
    )


class PGWebConversationStore(PostgresOperationRunner):
    """Durable PostgreSQL store for server-owned Web conversations."""

    def __init__(
        self,
        *,
        pool: ConnectionPool | None = None,
        run_store: PGRunStore | None = None,
    ) -> None:
        super().__init__(pool=pool)
        self._run_store = run_store or PGRunStore(pool=pool)
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
                    scope=RUN_MIGRATION_SCOPE,
                    migrations=RUN_MIGRATIONS,
                    tables=RUN_SCHEMA_TABLES,
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
                scope=RUN_MIGRATION_SCOPE,
                migrations=RUN_MIGRATIONS,
                tables=RUN_SCHEMA_TABLES,
                schema_error=RunSchemaError,
            )
            await apply_migrations(
                conn,
                scope="web_conversations",
                migrations=WEB_CONVERSATION_MIGRATIONS,
                tables=WEB_CONVERSATION_SCHEMA_TABLES,
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
            await self._ensure_session_for(conn, principal_id, conversation_id)
            return _row_dict(row)

        return await self._run_write(_operation)

    async def _ensure_session_for(self, conn: Any, principal_id: str, session_id: str) -> None:
        await ensure_agent_session(
            conn, owner_id=principal_id, session_id=session_id, lease_run_id=str(uuid4())
        )

    async def list_conversations(
        self,
        principal_id: str,
        *,
        page: ConversationPageRequest,
    ) -> ConversationRowPage:
        """Fetch one hard-bounded principal page through the covering order index."""
        if not isinstance(page, ConversationPageRequest):
            raise ValueError("conversation page request is required")
        validated_page = ConversationPageRequest(limit=page.limit, cursor=page.cursor)
        cursor = validated_page.cursor
        if cursor is not None and not isinstance(cursor, ConversationCursor):
            raise ValueError("conversation cursor must contain paired ordering fields")
        await self._ensure_initialized()
        fetch_limit = validated_page.limit + 1

        async def _select(conn: Any) -> ConversationRowPage:
            if cursor is None:
                rows = await conn.fetch(
                    _LIST_CONVERSATIONS_FIRST_PAGE,
                    principal_id,
                    fetch_limit,
                )
            else:
                rows = await conn.fetch(
                    _LIST_CONVERSATIONS_AFTER,
                    principal_id,
                    cursor.updated_at,
                    cursor.conversation_id,
                    fetch_limit,
                )
            fetched_rows = len(rows)
            return ConversationRowPage(
                items=tuple(_row_dict(row) for row in rows[: validated_page.limit]),
                has_more=fetched_rows > validated_page.limit,
                fetched_rows=fetched_rows,
            )

        return await self._run_read(_select)

    async def rename_conversation(
        self,
        principal_id: str,
        conversation_id: str,
        *,
        title: str,
    ) -> dict[str, Any] | None:
        """Rename an owned conversation."""
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
            )
            return _row_dict(row) if row is not None else None

        return await self._run_write(_operation)

    @staticmethod
    def _canonical_delete_id(value: Any, *, kind: str) -> str:
        if not isinstance(value, str):
            raise RuntimeError(f"{kind} deletion row did not contain a text UUID")
        parsed = parse_run_id(value)
        if parsed is None or str(parsed) != value:
            raise RuntimeError(f"{kind} deletion row contained a non-canonical UUID")
        return value

    async def _delete_linked_runs_in_batches(
        self,
        conn: Any,
        *,
        principal_id: str,
        conversation_id: str,
    ) -> None:
        """Delete one locked conversation's run links with bounded metadata.

        Run deletion cascades each selected turn, so repeatedly reading the
        first ordered page advances without OFFSET. Re-reading a selected id is
        a fail-closed progress violation: it catches a malformed or partial run
        store result before the conversation row could cascade a leftover turn.
        """
        previous_ids: frozenset[str] = frozenset()
        while True:
            rows = await conn.fetch(
                _SELECT_CONVERSATION_RUN_BATCH,
                principal_id,
                conversation_id,
                _DELETE_BATCH_SIZE,
            )
            if len(rows) > _DELETE_BATCH_SIZE:
                raise RuntimeError("conversation run deletion exceeded its batch bound")
            run_ids = [self._canonical_delete_id(row["answer_run_id"], kind="run") for row in rows]
            if not run_ids:
                return
            if run_ids != sorted(run_ids) or len(set(run_ids)) != len(run_ids):
                raise RuntimeError("conversation run deletion returned unstable identities")
            if previous_ids.intersection(run_ids):
                raise RuntimeError("conversation run deletion made no forward progress")

            deletion = await self._run_store.delete_runs_in(
                conn,
                owner_id=principal_id,
                run_ids=run_ids,
            )
            if (
                not isinstance(deletion, RunDeletion)
                or type(deletion.runs) is not int
                or type(deletion.artifacts) is not int
                or not 0 <= deletion.runs <= len(run_ids)
                or deletion.artifacts < 0
            ):
                raise RuntimeError("run store returned an invalid deletion result")
            previous_ids = frozenset(run_ids)

    @staticmethod
    async def _conversation_session(
        conn: Any, principal_id: str, conversation_id: str
    ) -> str | None:
        row = await conn.fetchval(
            "SELECT agent_session_id::text FROM web_conversations"
            " WHERE principal_id = $1 AND conversation_id = $2::text::uuid",
            principal_id,
            conversation_id,
        )
        return None if row is None else str(row)

    @staticmethod
    def _manifest_files(manifest: Any) -> tuple[AgentspaceFile, ...]:
        entries = manifest
        if isinstance(entries, str):
            entries = json.loads(entries)
        if not isinstance(entries, list):
            return ()
        return tuple(
            AgentspaceFile(
                relative_path=str(entry["path"]),
                content_digest=str(entry["digest"]),
                size_bytes=int(entry["size"]),
            )
            for entry in entries
            if isinstance(entry, dict) and {"path", "digest", "size"} <= set(entry)
        )

    @staticmethod
    def _manifest_diff(
        manifest: Any, parent_manifest: Any
    ) -> tuple[tuple[AgentspaceFile, ...], AgentspaceSnapshotChanges]:
        """Mark one snapshot's files against its parent's manifest and count the difference.

        A missing parent manifest is the empty one: the first snapshot of a Session, or a
        newer snapshot whose parent row the retained history no longer holds, has nothing
        earlier it can be the same as or have lost, so every file is ``added`` and nothing is
        ``removed``. The snapshot list and the snapshot detail both resolve their difference
        through this one helper, so the list's counts and the detail's per-file markers can
        never disagree about the same snapshot.
        """
        files = PGWebConversationStore._manifest_files(manifest)
        parent_digests = {
            file.relative_path: file.content_digest
            for file in PGWebConversationStore._manifest_files(parent_manifest)
        }
        current_paths: set[str] = set()
        added = 0
        changed = 0
        annotated: list[AgentspaceFile] = []
        for file in files:
            current_paths.add(file.relative_path)
            parent_digest = parent_digests.get(file.relative_path)
            change: AgentspaceChange
            if parent_digest is None:
                change = "added"
                added += 1
            elif parent_digest == file.content_digest:
                change = "unchanged"
            else:
                change = "changed"
                changed += 1
            annotated.append(replace(file, change=change))
        removed = sum(1 for path in parent_digests if path not in current_paths)
        return (
            tuple(annotated),
            AgentspaceSnapshotChanges(added=added, changed=changed, removed=removed),
        )

    async def agentspace_heads(
        self, principal_id: str, conversation_id: str
    ) -> tuple[AgentspaceHead, ...] | None:
        """Every branch head of this conversation's Session, or None when it does not exist."""
        await self._ensure_initialized()

        async def _operation(conn: Any) -> tuple[AgentspaceHead, ...] | None:
            session_id = await self._conversation_session(conn, principal_id, conversation_id)
            if session_id is None:
                return None
            rows = await conn.fetch(
                "SELECT lane_id, head_run_id::text AS head_run_id, manifest_digest,"
                "       revision, committed_at"
                " FROM dlightrag_session_workspace_heads"
                " WHERE owner_id = $1 AND session_id = $2::text::uuid"
                " ORDER BY lane_id",
                principal_id,
                session_id,
            )
            return tuple(
                AgentspaceHead(
                    lane_id=str(row["lane_id"]),
                    head_run_id=str(row["head_run_id"]),
                    manifest_digest=str(row["manifest_digest"]),
                    revision=int(row["revision"]),
                    committed_at=row["committed_at"],
                )
                for row in rows
            )

        return await self._run_read(_operation)

    async def agentspace_snapshots(
        self, principal_id: str, conversation_id: str, *, limit: int
    ) -> tuple[AgentspaceSnapshot, ...] | None:
        """The newest bounded page of this Session's restorable snapshots."""
        await self._ensure_initialized()
        cap = min(100, max(1, int(limit)))

        async def _operation(conn: Any) -> tuple[AgentspaceSnapshot, ...] | None:
            session_id = await self._conversation_session(conn, principal_id, conversation_id)
            if session_id is None:
                return None
            rows = await conn.fetch(
                "SELECT run_id::text AS run_id, parent_run_id::text AS parent_run_id,"
                "       manifest_digest, jsonb_array_length(manifest_json) AS file_count,"
                "       created_at, manifest_json"
                " FROM dlightrag_session_workspace_snapshots"
                " WHERE owner_id = $1 AND session_id = $2::text::uuid"
                " ORDER BY created_at DESC, run_id DESC LIMIT $3",
                principal_id,
                session_id,
                cap,
            )
            parent_run_ids = sorted(
                {str(row["parent_run_id"]) for row in rows if row["parent_run_id"] is not None}
            )
            parent_manifests: dict[str, Any] = {}
            if parent_run_ids:
                # The page's parents are read in one bounded statement, never one per
                # snapshot. A parent the retained history has pruned is simply absent, and
                # the diff reads that absence as the empty manifest.
                parent_rows = await conn.fetch(
                    "SELECT run_id::text AS run_id, manifest_json"
                    " FROM dlightrag_session_workspace_snapshots"
                    " WHERE owner_id = $1 AND session_id = $2::text::uuid"
                    "   AND run_id = ANY($3::uuid[])",
                    principal_id,
                    session_id,
                    [UUID(parent_run_id) for parent_run_id in parent_run_ids],
                )
                parent_manifests = {
                    str(parent_row["run_id"]): parent_row["manifest_json"]
                    for parent_row in parent_rows
                }
            snapshots: list[AgentspaceSnapshot] = []
            for row in rows:
                parent_run_id = (
                    str(row["parent_run_id"]) if row["parent_run_id"] is not None else None
                )
                _files, changes = self._manifest_diff(
                    row["manifest_json"],
                    parent_manifests.get(parent_run_id) if parent_run_id is not None else None,
                )
                snapshots.append(
                    AgentspaceSnapshot(
                        run_id=str(row["run_id"]),
                        parent_run_id=parent_run_id,
                        manifest_digest=str(row["manifest_digest"]),
                        file_count=int(row["file_count"]),
                        created_at=row["created_at"],
                        changes=changes,
                    )
                )
            return tuple(snapshots)

        return await self._run_read(_operation)

    async def agentspace_snapshot_files(
        self, principal_id: str, conversation_id: str, run_id: str
    ) -> tuple[AgentspaceFile, ...] | None:
        """The files one Run's snapshot carried, marked by how that snapshot changed them."""
        await self._ensure_initialized()
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return None

        async def _operation(conn: Any) -> tuple[AgentspaceFile, ...] | None:
            session_id = await self._conversation_session(conn, principal_id, conversation_id)
            if session_id is None:
                return None
            # The parent is joined beside its child in the same bounded read: a null parent
            # reference and a parent row pruning removed both surface as a null manifest, so
            # the snapshot is diffed against the empty one.
            row = await conn.fetchrow(
                "SELECT snapshot.manifest_json,"
                "       parent.manifest_json AS parent_manifest_json"
                " FROM dlightrag_session_workspace_snapshots AS snapshot"
                " LEFT JOIN dlightrag_session_workspace_snapshots AS parent"
                "   ON parent.owner_id = snapshot.owner_id"
                "  AND parent.session_id = snapshot.session_id"
                "  AND parent.run_id = snapshot.parent_run_id"
                " WHERE snapshot.owner_id = $1 AND snapshot.session_id = $2::text::uuid"
                "   AND snapshot.run_id = $3",
                principal_id,
                session_id,
                run_uuid,
            )
            if row is None:
                return None
            files, _changes = self._manifest_diff(row["manifest_json"], row["parent_manifest_json"])
            return files

        return await self._run_read(_operation)

    async def agentspace_file_versions(
        self, principal_id: str, conversation_id: str, relative_path: str
    ) -> tuple[AgentspaceFileVersion, ...] | None:
        """Every snapshot that carried a path, oldest first, and which one changed it."""
        await self._ensure_initialized()

        async def _operation(conn: Any) -> tuple[AgentspaceFileVersion, ...] | None:
            session_id = await self._conversation_session(conn, principal_id, conversation_id)
            if session_id is None:
                return None
            rows = await conn.fetch(
                "SELECT run_id::text AS run_id, manifest_json, created_at"
                " FROM dlightrag_session_workspace_snapshots"
                " WHERE owner_id = $1 AND session_id = $2::text::uuid"
                " ORDER BY created_at, run_id",
                principal_id,
                session_id,
            )
            versions: list[AgentspaceFileVersion] = []
            previous_digest: str | None = None
            for row in rows:
                for file in self._manifest_files(row["manifest_json"]):
                    if file.relative_path != relative_path:
                        continue
                    change: Any = (
                        "added"
                        if previous_digest is None
                        else ("changed" if previous_digest != file.content_digest else "unchanged")
                    )
                    previous_digest = file.content_digest
                    versions.append(
                        AgentspaceFileVersion(
                            run_id=str(row["run_id"]),
                            content_digest=file.content_digest,
                            size_bytes=file.size_bytes,
                            created_at=row["created_at"],
                            change=change,
                        )
                    )
                    break
            return tuple(versions)

        return await self._run_read(_operation)

    async def reset_session_agentspace(self, principal_id: str, conversation_id: str) -> str | None:
        """Forget this conversation's workspace: heads, snapshots and the disk trees' owner id.

        Refused while the Session is working: reset is a human control for a line of work that
        is at rest, and deleting state out from under a live Run would make the next phase seed
        from nothing.
        """
        await self._ensure_initialized()

        async def _operation(conn: Any) -> str | None:
            async with conn.transaction():
                summary = await conn.fetchrow(_LOCK_CONVERSATION, principal_id, conversation_id)
                if summary is None:
                    return None
                session_id = str(summary["agent_session_id"])
                active = await conn.fetchval(
                    "SELECT active_run_id::text AS active_run_id"
                    " FROM dlightrag_session_work_heads"
                    " WHERE owner_id = $1 AND session_id = $2::text::uuid FOR UPDATE",
                    principal_id,
                    session_id,
                )
                if active is not None:
                    raise SessionWorkBusyError(str(active))
                await conn.execute(
                    "DELETE FROM dlightrag_session_workspace_heads"
                    " WHERE owner_id = $1 AND session_id = $2::text::uuid",
                    principal_id,
                    session_id,
                )
                await conn.execute(
                    "DELETE FROM dlightrag_session_workspace_snapshots"
                    " WHERE owner_id = $1 AND session_id = $2::text::uuid",
                    principal_id,
                    session_id,
                )
                await conn.execute(
                    "DELETE FROM dlightrag_session_work_heads"
                    " WHERE owner_id = $1 AND session_id = $2::text::uuid",
                    principal_id,
                    session_id,
                )
                return session_id

        return await self._run_write(_operation)

    async def conversation_agent_session(
        self, principal_id: str, conversation_id: str
    ) -> str | None:
        """The Session a conversation owns, read before its row and trees are deleted."""
        await self._ensure_initialized()

        async def _operation(conn: Any) -> str | None:
            row = await conn.fetchval(
                "SELECT agent_session_id::text FROM web_conversations"
                " WHERE principal_id = $1 AND conversation_id = $2::text::uuid",
                principal_id,
                conversation_id,
            )
            return None if row is None else str(row)

        return await self._run_read(_operation)

    async def conversation_for_session(self, principal_id: str, session_id: str) -> str | None:
        """The conversation that owns a Session, or None when this owner has none.

        A conversation is its Session from creation, so startup recovery needs this reverse
        lookup to start a round for an active goal whose Session has no answer Run yet.
        """
        await self._ensure_initialized()

        async def _operation(conn: Any) -> str | None:
            row = await conn.fetchval(_CONVERSATION_FOR_SESSION, principal_id, session_id)
            return None if row is None else str(row)

        return await self._run_read(_operation)

    async def newest_answer_workspaces(
        self, principal_id: str, conversation_id: str
    ) -> tuple[str, ...] | None:
        """The workspaces the newest answer turn asked for; None when it has no turn.

        A goal round continues the Session the conversation already is, and the selected
        workspaces are part of that request. No turn means the caller's own default stands.
        """
        await self._ensure_initialized()

        async def _operation(conn: Any) -> tuple[str, ...] | None:
            rows = await conn.fetch(_NEWEST_ANSWER_WORKSPACES, principal_id, conversation_id)
            if not rows:
                return None
            return tuple(str(item) for item in (rows[0]["workspaces"] or ()))

        return await self._run_read(_operation)

    async def agent_sessions(self, principal_id: str) -> tuple[str, ...]:
        """Every Session this owner's conversations own, read before a delete-all."""
        await self._ensure_initialized()

        async def _operation(conn: Any) -> tuple[str, ...]:
            rows = await conn.fetch(
                "SELECT DISTINCT agent_session_id::text AS session_id"
                " FROM web_conversations WHERE principal_id = $1",
                principal_id,
            )
            return tuple(str(row["session_id"]) for row in rows)

        return await self._run_read(_operation)

    async def delete_conversation(
        self,
        principal_id: str,
        conversation_id: str,
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
                summary_row = await conn.fetchrow(_LOCK_CONVERSATION, principal_id, conversation_id)
                if summary_row is None:
                    return False
                agent_session_id = self._canonical_delete_id(
                    summary_row["agent_session_id"], kind="Agent Session"
                )
                await self._delete_linked_runs_in_batches(
                    conn,
                    principal_id=principal_id,
                    conversation_id=conversation_id,
                )
                deleted_session_id = await conn.fetchval(
                    _DELETE_CONVERSATION, principal_id, conversation_id
                )
                if str(deleted_session_id) != agent_session_id:
                    raise RuntimeError("locked conversation was not deleted exactly once")
                await conn.execute(
                    _DELETE_AGENT_SESSION_IF_UNREFERENCED,
                    principal_id,
                    agent_session_id,
                )
                await conn.execute(
                    _RELEASE_SESSION_WORK_ROWS,
                    principal_id,
                    str(deleted_session_id),
                )
                return True

        return await self._run_write(_operation)

    async def delete_all_conversations(self, principal_id: str) -> int:
        """Delete every owned conversation in bounded locked batches, atomically.

        The outer transaction deliberately spans every batch. A submission to
        an existing conversation takes the same conversation row lock, so it is
        either included before that row is drained or observes its deletion;
        no accepted turn is orphaned. Concurrent owner-wide deleters wait on the
        first UUID rather than skipping it and all acquire UUIDs in one order.
        """
        await self._ensure_initialized()

        async def _operation(conn: Any) -> int:
            async with conn.transaction():
                deleted_count = 0
                while True:
                    rows = await conn.fetch(
                        _LOCK_PRINCIPAL_CONVERSATION_BATCH,
                        principal_id,
                        _DELETE_BATCH_SIZE,
                    )
                    if len(rows) > _DELETE_BATCH_SIZE:
                        raise RuntimeError("conversation deletion exceeded its batch bound")
                    conversations = [
                        (
                            self._canonical_delete_id(row["conversation_id"], kind="conversation"),
                            self._canonical_delete_id(
                                row["agent_session_id"], kind="Agent Session"
                            ),
                        )
                        for row in rows
                    ]
                    if not conversations:
                        return deleted_count
                    conversation_ids = [conversation_id for conversation_id, _ in conversations]
                    if conversation_ids != sorted(conversation_ids) or len(
                        set(conversation_ids)
                    ) != len(conversation_ids):
                        raise RuntimeError("conversation deletion returned unstable identities")
                    linked_rows = await conn.fetch(
                        _SELECT_LINKED_CONVERSATION_BATCH,
                        principal_id,
                        conversation_ids,
                        _DELETE_BATCH_SIZE,
                    )
                    if len(linked_rows) > _DELETE_BATCH_SIZE:
                        raise RuntimeError("linked conversation scan exceeded its batch bound")
                    linked_conversation_ids = [
                        self._canonical_delete_id(row["conversation_id"], kind="conversation")
                        for row in linked_rows
                    ]
                    if (
                        linked_conversation_ids != sorted(linked_conversation_ids)
                        or len(set(linked_conversation_ids)) != len(linked_conversation_ids)
                        or not set(linked_conversation_ids).issubset(conversation_ids)
                    ):
                        raise RuntimeError("linked conversation scan returned unstable identities")

                    for conversation_id in linked_conversation_ids:
                        await self._delete_linked_runs_in_batches(
                            conn,
                            principal_id=principal_id,
                            conversation_id=conversation_id,
                        )
                    deleted_rows = await conn.fetch(
                        _DELETE_CONVERSATION_BATCH,
                        principal_id,
                        conversation_ids,
                        _DELETE_BATCH_SIZE,
                    )
                    if len(deleted_rows) > _DELETE_BATCH_SIZE:
                        raise RuntimeError("conversation deletion exceeded its batch bound")
                    deleted_conversations = [
                        (
                            self._canonical_delete_id(row["conversation_id"], kind="conversation"),
                            self._canonical_delete_id(
                                row["agent_session_id"], kind="Agent Session"
                            ),
                        )
                        for row in deleted_rows
                    ]
                    expected_conversations = conversations
                    if deleted_conversations != expected_conversations:
                        raise RuntimeError("locked conversations were not deleted exactly once")
                    await conn.execute(
                        _DELETE_AGENT_SESSION_BATCH_IF_UNREFERENCED,
                        principal_id,
                        [session_id for _, session_id in deleted_conversations],
                    )
                    deleted_count += len(deleted_conversations)

        return await self._run_write(_operation)

    async def history_page(
        self,
        principal_id: str,
        conversation_id: str,
        *,
        page: ConversationHistoryPageRequest,
    ) -> ConversationHistoryPage | None:
        """Read one chronological limit+1 page of turns and commands under a coherent head.

        Commands and turns allocate their ordering number from the conversation's one
        counter, so a page fetches up to ``limit + 1`` of each before the cursor, merges
        them by that shared sequence descending, and keeps the newest ``limit`` together.
        The continuation is the oldest retained sequence, whichever stream it came from.
        """
        validated = ConversationHistoryPageRequest(limit=page.limit, cursor=page.cursor)
        if (
            validated.cursor is not None
            and str(validated.cursor.conversation_id) != conversation_id
        ):
            raise ValueError("conversation history cursor belongs to another conversation")
        before_sequence = validated.cursor.before_sequence if validated.cursor is not None else None
        await self._ensure_initialized()

        async def _operation(conn: Any) -> ConversationHistoryPage | None:
            async with conn.transaction(isolation="repeatable_read", readonly=True):
                row = await conn.fetchrow(_GET_CONVERSATION, principal_id, conversation_id)
                if row is None:
                    return None
                turn_rows = await conn.fetch(
                    _GET_TURNS_PAGE,
                    principal_id,
                    conversation_id,
                    before_sequence,
                    validated.limit + 1,
                )
                command_rows = await conn.fetch(
                    _GET_COMMANDS_PAGE,
                    principal_id,
                    conversation_id,
                    before_sequence,
                    validated.limit + 1,
                )
                merged: list[tuple[int, LinkedTurn | ConversationCommand]] = [
                    (int(turn_row["turn_number"]), _linked_turn(turn_row)) for turn_row in turn_rows
                ]
                merged.extend(
                    (int(command_row["sequence"]), _command(command_row))
                    for command_row in command_rows
                )
                merged.sort(key=lambda entry: entry[0], reverse=True)
                retained = merged[: validated.limit]
                next_cursor = None
                if len(merged) > validated.limit:
                    if not retained:
                        raise RuntimeError("history page reported more rows after an empty page")
                    next_cursor = ConversationHistoryCursor(
                        conversation_id=validated.cursor.conversation_id
                        if validated.cursor is not None
                        else UUID(conversation_id),
                        before_sequence=retained[-1][0],
                    )
                chronological = list(reversed(retained))
                return ConversationHistoryPage(
                    conversation=_conversation_head(row),
                    turns=tuple(
                        item for _sequence, item in chronological if isinstance(item, LinkedTurn)
                    ),
                    commands=tuple(
                        item
                        for _sequence, item in chronological
                        if isinstance(item, ConversationCommand)
                    ),
                    next_cursor=next_cursor,
                    fetched_rows=len(turn_rows) + len(command_rows),
                )

        return await self._run_read(_operation)

    async def recent_answer_runs(
        self, principal_id: str, *, limit: int
    ) -> tuple[WebRunActivity, ...]:
        """Read the owner's newest conversation-linked Answer runs, newest first.

        One hard-bounded page with no continuation: the caller replaces its snapshot on
        the next poll. Only a run a Web conversation turn links is activity here.
        """
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError("run activity limit must be an integer")
        if not 1 <= limit <= RUN_ACTIVITY_MAX_LIMIT:
            raise ValueError(f"run activity limit must be between 1 and {RUN_ACTIVITY_MAX_LIMIT}")
        await self._ensure_initialized()

        async def _operation(conn: Any) -> tuple[WebRunActivity, ...]:
            rows = await conn.fetch(_RECENT_ANSWER_RUNS, principal_id, limit)
            return tuple(_run_activity(row) for row in rows)

        return await self._run_read(_operation)

    async def append_command(
        self,
        principal_id: str,
        conversation_id: str,
        *,
        name: str,
        command_input: str,
        result: str,
        tone: CommandTone,
        result_data: Mapping[str, Any] | None = None,
    ) -> ConversationCommand | None:
        """Append one command event, allocating its sequence from the shared counter.

        The conversation's row lock is taken by the counter bump, so a concurrent
        conversation delete either sees this command or the command sees no conversation.
        ``None`` means the conversation is not this principal's or no longer exists.
        """
        await self._ensure_initialized()
        command_id = str(uuid4())
        result_data_json = (
            None
            if result_data is None
            else json.dumps(dict(result_data), ensure_ascii=False, sort_keys=True)
        )

        async def _operation(conn: Any) -> ConversationCommand | None:
            row = await conn.fetchrow(
                _APPEND_COMMAND,
                principal_id,
                conversation_id,
                command_id,
                name,
                command_input,
                result,
                tone,
                result_data_json,
            )
            return None if row is None else _command(row)

        return await self._run_write(_operation)

    async def live_answer_run(
        self, principal_id: str, conversation_id: str
    ) -> LiveAnswerRun | None:
        """Return the one non-terminal Answer Run this conversation is executing."""
        await self._ensure_initialized()

        async def _operation(conn: Any) -> LiveAnswerRun | None:
            row = await conn.fetchrow(_LIVE_ANSWER_RUN, principal_id, conversation_id)
            if row is None:
                return None
            kind = row["continuation_kind"]
            return LiveAnswerRun(
                run_id=str(row["run_id"]),
                status=str(row["status"]),
                continuation_kind=None if kind is None else str(kind),
            )

        return await self._run_read(_operation)

    async def enqueue_goal_round(
        self,
        principal_id: str,
        conversation_id: str,
        *,
        request_id: str,
        content: str,
        prompt_digest: str,
        workspaces: Sequence[str] = (),
        goal_id: str,
        goal_revision: int,
        goal_round: int,
    ) -> bool:
        """Hold one goal continuation for the conversation; a conversation must exist.

        The row names the goal and the admitted round it belongs to, so the delivery path can
        refuse a continuation whose goal was cleared, replaced, or has already moved past it.
        """
        await self._ensure_initialized()

        async def _operation(conn: Any) -> bool:
            async with conn.transaction():
                exists = await conn.fetchval(_CONVERSATION_EXISTS, principal_id, conversation_id)
                if exists is None:
                    return False
                claimed = await conn.fetchval(
                    _ENQUEUE_GOAL_ROUND,
                    principal_id,
                    conversation_id,
                    UUID(request_id),
                    content,
                    prompt_digest,
                    list(workspaces),
                    goal_id,
                    goal_revision,
                    goal_round,
                )
                return claimed is not None

        return await self._run_write(_operation)

    async def pending_goal_round(
        self, principal_id: str, conversation_id: str
    ) -> GoalRoundRequest | None:
        """The goal continuation waiting to be delivered, if one is."""
        await self._ensure_initialized()

        async def _operation(conn: Any) -> GoalRoundRequest | None:
            row = await conn.fetchrow(_PENDING_GOAL_ROUND, principal_id, conversation_id)
            if row is None:
                return None
            return GoalRoundRequest(
                request_id=str(row["request_id"]),
                content=str(row["content"]),
                prompt_digest=str(row["prompt_digest"]),
                workspaces=tuple(str(item) for item in (row["workspaces"] or ())),
                goal_id=str(row["goal_id"]),
                goal_revision=int(row["goal_revision"]),
                goal_round=int(row["goal_round"]),
                created_at=row["created_at"],
            )

        return await self._run_read(_operation)

    async def clear_goal_round(
        self,
        principal_id: str,
        conversation_id: str,
        *,
        request_id: str,
        prompt_digest: str,
    ) -> bool:
        """Drop this attempt's pending goal continuation; True only when its own row was deleted."""
        await self._ensure_initialized()

        async def _operation(conn: Any) -> bool:
            deleted = await conn.fetchval(
                _CLEAR_GOAL_ROUND,
                principal_id,
                conversation_id,
                UUID(request_id),
                prompt_digest,
            )
            return deleted is not None

        return await self._run_write(_operation)

    async def submission_exists(
        self, principal_id: str, conversation_id: str, submission_id: str
    ) -> bool:
        """Whether this conversation already committed the turn one submission key names.

        The dropped-pending refund reads it: a row that is gone because its turn committed must
        not release the round that turn is running.
        """
        await self._ensure_initialized()

        async def _operation(conn: Any) -> bool:
            row = await conn.fetchval(
                _SUBMISSION_EXISTS, principal_id, conversation_id, UUID(submission_id)
            )
            return row is not None

        return await self._run_read(_operation)

    async def queue_messages(
        self, principal_id: str, conversation_id: str
    ) -> tuple[QueuedMessage, ...] | None:
        """Return the conversation's queued messages in order; None when it is not owned."""
        await self._ensure_initialized()

        async def _operation(conn: Any) -> tuple[QueuedMessage, ...] | None:
            owned = await conn.fetchval(_CONVERSATION_EXISTS, principal_id, conversation_id)
            if owned is None:
                return None
            rows = await conn.fetch(_QUEUE_ROWS, principal_id, conversation_id)
            return tuple(_queued_message(item) for item in rows)

        return await self._run_read(_operation)

    async def append_queued_message(
        self, principal_id: str, conversation_id: str, *, content: str
    ) -> QueuedMessage | None:
        """Append one message under the conversation's row lock; None when it is not owned."""
        await self._ensure_initialized()
        message_id = str(uuid4())

        async def _operation(conn: Any) -> QueuedMessage | None:
            async with conn.transaction():
                owned = await conn.fetchrow(_LOCK_CONVERSATION, principal_id, conversation_id)
                if owned is None:
                    return None
                row = await conn.fetchrow(
                    _APPEND_QUEUED_MESSAGE,
                    principal_id,
                    conversation_id,
                    message_id,
                    content,
                )
                if row is None:
                    raise RuntimeError("queued message insert returned no row")
                return _queued_message(row)

        return await self._run_write(_operation)

    async def edit_queued_message(
        self,
        principal_id: str,
        conversation_id: str,
        message_id: str,
        *,
        content: str,
    ) -> QueuedMessage | None:
        """Rewrite one queued message; None when the message or conversation is gone."""
        await self._ensure_initialized()

        async def _operation(conn: Any) -> QueuedMessage | None:
            row = await conn.fetchrow(
                _EDIT_QUEUED_MESSAGE,
                principal_id,
                conversation_id,
                message_id,
                content,
            )
            return None if row is None else _queued_message(row)

        return await self._run_write(_operation)

    async def remove_queued_message(
        self, principal_id: str, conversation_id: str, message_id: str
    ) -> bool:
        """Delete one queued message; False when the message or conversation is gone."""
        await self._ensure_initialized()

        async def _operation(conn: Any) -> bool:
            row = await conn.fetchrow(
                _REMOVE_QUEUED_MESSAGE, principal_id, conversation_id, message_id
            )
            return row is not None

        return await self._run_write(_operation)

    async def reorder_queued_messages(
        self,
        principal_id: str,
        conversation_id: str,
        *,
        message_ids: Sequence[str],
    ) -> tuple[QueuedMessage, ...] | None:
        """Rewrite the queue order to ``message_ids`` under the conversation's row lock.

        The set must be exactly the current set; a caller reordering a stale read gets a
        conflict instead of a queue that silently dropped an arrival. Positions move to
        an offset first so the unique constraint never collides mid-permutation.
        """
        await self._ensure_initialized()

        async def _operation(conn: Any) -> tuple[QueuedMessage, ...] | None:
            async with conn.transaction():
                owned = await conn.fetchrow(_LOCK_CONVERSATION, principal_id, conversation_id)
                if owned is None:
                    return None
                rows = await conn.fetch(_QUEUE_IDS_FOR_UPDATE, principal_id, conversation_id)
                current_ids = [str(item["message_id"]) for item in rows]
                if sorted(current_ids) != sorted(message_ids) or len(set(message_ids)) != len(
                    message_ids
                ):
                    raise ConversationQueueConflict()
                await conn.execute(_SHIFT_QUEUE_POSITIONS, principal_id, conversation_id)
                for position, message_id in enumerate(message_ids, start=1):
                    await conn.execute(
                        _SET_QUEUE_POSITION,
                        principal_id,
                        conversation_id,
                        message_id,
                        position,
                    )
                ordered = await conn.fetch(_QUEUE_ROWS, principal_id, conversation_id)
                return tuple(_queued_message(item) for item in ordered)

        return await self._run_write(_operation)

    async def submission_seed(
        self,
        principal_id: str,
        conversation_id: str,
        *,
        attachment_limit: int,
    ) -> SubmissionSeed | None:
        """Read the execution mapping and newest successful attachment metadata."""
        if isinstance(attachment_limit, bool) or attachment_limit < 0:
            raise ValueError("attachment limit must be non-negative")
        await self._ensure_initialized()

        async def _operation(conn: Any) -> SubmissionSeed | None:
            async with conn.transaction(isolation="repeatable_read", readonly=True):
                row = await conn.fetchrow(_GET_CONVERSATION, principal_id, conversation_id)
                if row is None:
                    return None
                tip_run_id = await conn.fetchval(_GET_TIP_RUN, principal_id, conversation_id)
                attachment_rows = (
                    await conn.fetch(
                        _GET_CARRIED_ATTACHMENTS,
                        principal_id,
                        conversation_id,
                        attachment_limit,
                    )
                    if attachment_limit
                    else ()
                )
                attachments = tuple(
                    CarriedAttachment(
                        run_id=str(item["run_id"]),
                        source_ordinal=int(item["source_ordinal"]),
                        digest=str(item["digest"]),
                        filename=str(item["filename"]),
                        mime_type=str(item["mime_type"]),
                        byte_size=int(item["byte_size"]),
                    )
                    for item in attachment_rows
                )
                return SubmissionSeed(
                    head=_conversation_head(row),
                    attachments=attachments,
                    parent_run_id=(str(tip_run_id) if tip_run_id is not None else None),
                )

        return await self._run_read(_operation)

    async def find_turn_by_run(self, principal_id: str, run_id: str) -> LinkedTurn | None:
        """Return the conversation entry one owned run belongs to, if any."""
        await self._ensure_initialized()

        async def _operation(conn: Any) -> LinkedTurn | None:
            row = await conn.fetchrow(_GET_TURN_BY_RUN, principal_id, run_id)
            return _linked_turn(row) if row is not None else None

        return await self._run_read(_operation)

    async def find_answer_turn_by_submission(
        self, principal_id: str, submission_id: str
    ) -> AnswerTurnCreation | None:
        """Return one owner-scoped accepted submission without needing its conversation."""
        await self._ensure_initialized()

        async def _operation(conn: Any) -> AnswerTurnCreation | None:
            row = await conn.fetchrow(_GET_TURN_BY_SUBMISSION, principal_id, submission_id)
            if row is None:
                return None
            return AnswerTurnCreation(
                turn=_linked_turn(row),
                summary=_row_dict(row),
                replayed=True,
            )

        return await self._run_read(_operation)

    async def create_answer_turn(
        self,
        *,
        principal_id: str,
        conversation_id: str,
        submission_id: str,
        envelope: PreparedRunEnvelope,
        run_id: str,
        artifacts: Sequence[PendingArtifact] = (),
        references: Sequence[PendingArtifactReference] = (),
        title_hint: str | None,
        routing: RoutingAcceptance | None = None,
        connection_bindings: tuple[RunConnectionBinding, ...] = (),
        create_conversation: bool = False,
        forked_from_conversation_id: str | None = None,
        queue_guard: QueuedDeliveryGuard | None = None,
        goal_round_guard: GoalRoundGuard | None = None,
    ) -> AnswerTurnCreation | None:
        """Accept one submission as a run plus its conversation entry, atomically.

        The owned conversation is locked, or inserted when this is its first
        submission. The run, uploaded bytes, and linked turn are committed in
        that same transaction. Replaying the same submission returns the
        authoritative run and turn; reusing it for a different conversation or
        different normalized input is a conflict. ``None`` means a required
        existing conversation does not exist for this principal.
        """
        await self._ensure_initialized()
        turn_id = str(uuid4())
        request = envelope.payload
        fingerprint = envelope.request_fingerprint

        async def _operation(conn: Any) -> AnswerTurnCreation | None:
            async with conn.transaction():
                summary_row = await conn.fetchrow(_LOCK_CONVERSATION, principal_id, conversation_id)
                if summary_row is None and create_conversation:
                    summary_row = await conn.fetchrow(
                        _CREATE_CONVERSATION_IF_MISSING,
                        principal_id,
                        conversation_id,
                        forked_from_conversation_id,
                        str(request["agent_session_id"]),
                        str(request.get("agent_lane_id") or "main"),
                    )
                    if summary_row is None:
                        summary_row = await conn.fetchrow(
                            _LOCK_CONVERSATION,
                            principal_id,
                            conversation_id,
                        )
                if summary_row is None:
                    return None
                if str(summary_row["agent_session_id"]) != str(
                    request.get("agent_session_id") or ""
                ) or str(summary_row["agent_lane_id"]) != str(
                    request.get("agent_lane_id") or "main"
                ):
                    raise ConversationSubmissionConflict(
                        "conversation Agent Session/Lane mapping changed"
                    )
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
                    if goal_round_guard is not None:
                        await conn.execute(
                            _CONSUME_GOAL_ROUND,
                            principal_id,
                            conversation_id,
                            UUID(goal_round_guard.request_id),
                            goal_round_guard.prompt_digest,
                        )
                    if queue_guard is not None:
                        # The original delivery already removed these rows in its own
                        # transaction; a replay removes any that a retry re-read.
                        await conn.fetch(
                            _REMOVE_QUEUED_MESSAGES,
                            principal_id,
                            conversation_id,
                            list(queue_guard.message_ids),
                        )
                    return AnswerTurnCreation(
                        turn=_linked_turn(existing),
                        summary=_row_dict(existing),
                        replayed=True,
                    )
                accepted_request = request
                accepted_routing = routing
                session_exists = await conn.fetchval(
                    _LOCK_AGENT_SESSION_IF_PRESENT,
                    principal_id,
                    str(summary_row["agent_session_id"]),
                )
                if session_exists is None:
                    accepted_request = {
                        **request,
                        "agent_lane_id": "main",
                        "source_lane_id": None,
                    }
                    accepted_routing = replace(
                        routing or RoutingAcceptance.fallback(request),
                        agent_lane_id="main",
                        source_lane_id=None,
                    )
                    if str(summary_row["agent_lane_id"]) != "main":
                        rebased = await conn.fetchrow(
                            _REBASE_EMPTY_CONVERSATION,
                            principal_id,
                            conversation_id,
                        )
                        if rebased is None:
                            raise ConversationSubmissionConflict(
                                "conversation Agent Session/Lane mapping changed"
                            )
                        summary_row = rebased
                if queue_guard is not None:
                    # The delivery read the queue outside this transaction; re-read it
                    # under the conversation lock, and refuse a prompt built from rows
                    # the reader has since changed or removed.
                    queued_rows = await conn.fetch(_QUEUE_ROWS, principal_id, conversation_id)
                    queued_now = tuple(_queued_message(row) for row in queued_rows)
                    if not queued_now or queued_delivery_digest(queued_now) != queue_guard.digest:
                        return None
                if goal_round_guard is not None:
                    # The continuation and the goal were read outside this transaction; re-read
                    # both under the conversation's row lock, and refuse a continuation whose
                    # goal was paused, cleared, replaced or moved past the round it reserved. A
                    # reservation written before admission is admitted here, in the same commit.
                    pending = await conn.fetchrow(
                        _PENDING_GOAL_ROUND, principal_id, conversation_id
                    )
                    if (
                        pending is None
                        or str(pending["request_id"]) != goal_round_guard.request_id
                        or str(pending["prompt_digest"]) != goal_round_guard.prompt_digest
                    ):
                        return None
                    session_id = str(summary_row["agent_session_id"])
                    goal_row = await conn.fetchrow(
                        _LOCK_GOAL_FOR_DELIVERY, principal_id, session_id
                    )
                    if goal_row is None:
                        return None
                    rounds_started = int(goal_row["rounds_started"])
                    revision = int(goal_row["revision"])
                    # The two legal states are exact pairs, not a cross product:
                    # reserved-but-unadmitted (revision/round one behind the guard) or admitted
                    # (revision/round equal to it). A mixed pair means a different writer moved
                    # one half, so the continuation no longer describes this goal state.
                    reserved = (
                        revision == goal_round_guard.goal_revision - 1
                        and rounds_started == goal_round_guard.goal_round - 1
                    )
                    admitted_before = (
                        revision == goal_round_guard.goal_revision
                        and rounds_started == goal_round_guard.goal_round
                    )
                    if (
                        str(goal_row["goal_id"]) != goal_round_guard.goal_id
                        or str(goal_row["phase"]) != "active"
                        or not (reserved or admitted_before)
                    ):
                        return None
                    if reserved:
                        if rounds_started + 1 != goal_round_guard.goal_round:
                            return None
                        admitted = await conn.fetchval(
                            _ADMIT_RESERVED_GOAL_ROUND,
                            principal_id,
                            session_id,
                            goal_round_guard.goal_id,
                            int(goal_row["revision"]),
                            rounds_started,
                        )
                        if admitted is None:
                            return None
                    consumed = await conn.fetchval(
                        _CONSUME_GOAL_ROUND,
                        principal_id,
                        conversation_id,
                        UUID(goal_round_guard.request_id),
                        goal_round_guard.prompt_digest,
                    )
                    if consumed is None:
                        return None
                claimed = await conn.fetchval(
                    _ADMIT_SESSION_WORK,
                    principal_id,
                    str(summary_row["agent_session_id"]),
                    UUID(run_id),
                )
                if claimed is None:
                    active = await conn.fetchval(
                        _LOAD_SESSION_WORK_ACTIVE,
                        principal_id,
                        str(summary_row["agent_session_id"]),
                    )
                    raise SessionWorkBusyError(None if active is None else str(active))
                try:
                    creation = await self._run_store.accept_run_in(
                        conn,
                        envelope=replace(
                            envelope,
                            payload=accepted_request,
                            accepted_input=accepted_input_envelope(accepted_request),
                        ),
                        run_id=run_id,
                        artifacts=artifacts,
                        references=references,
                        routing=accepted_routing,
                        connection_bindings=connection_bindings,
                    )
                except IdempotencyKeyConflict as exc:
                    accepted = await conn.fetchrow(
                        _GET_TURN_BY_SUBMISSION,
                        principal_id,
                        submission_id,
                    )
                    if accepted is not None and str(accepted["turn_conversation_id"]) != str(
                        conversation_id
                    ):
                        raise ConversationSubmissionConflict(
                            "submission id was reused in a different conversation"
                        ) from exc
                    raise
                touched = await conn.fetchrow(
                    _TOUCH_CONVERSATION, principal_id, conversation_id, title_hint
                )
                if touched is None:
                    raise RuntimeError("conversation touch returned no row")
                turn_number = int(touched["content_revision"])
                try:
                    turn_row = await conn.fetchrow(
                        _INSERT_TURN,
                        turn_id,
                        principal_id,
                        conversation_id,
                        turn_number,
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
                if int(turn_row["turn_number"]) != turn_number:
                    raise RuntimeError("conversation turn insert changed its allocated number")
                if queue_guard is not None:
                    # Delete inside the same commit that linked the turn: the queued
                    # words and the Run that answers them are durable together.
                    await conn.fetch(
                        _REMOVE_QUEUED_MESSAGES,
                        principal_id,
                        conversation_id,
                        list(queue_guard.message_ids),
                    )
                return AnswerTurnCreation(
                    turn=LinkedTurn(
                        turn_id=str(turn_row["turn_id"]),
                        turn_number=turn_number,
                        submission_id=submission_id,
                        created_at=creation.run.created_at,
                        run=RunView.from_runtime(creation.run),
                        conversation_id=conversation_id,
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
    ) -> AnswerTurnCreation | None:
        """Return an accepted browser submission before resolved input is rebuilt."""
        await self._ensure_initialized()

        async def _operation(conn: Any) -> AnswerTurnCreation | None:
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
            return AnswerTurnCreation(
                turn=_linked_turn(existing),
                summary=_row_dict(existing),
                replayed=True,
            )

        return await self._run_read(_operation)

    async def prune_empty_conversations(self, *, batch_size: int = 500) -> int:
        """Delete one skip-locked batch of conversations with no turns left.

        Turns live and die with their Answer runs: run retention deletes a
        terminal run after its retention floor and the turn cascade empties the
        conversation. A conversation row with no turns carries no content, so
        this sweep reclaims it.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        await self._ensure_initialized()

        async def _operation(conn: Any) -> int:
            async with conn.transaction():
                rows = await conn.fetch(_SELECT_EMPTY_CONVERSATIONS, batch_size)
                if not rows:
                    return 0
                principals = [str(row["principal_id"]) for row in rows]
                conversation_ids = [row["conversation_id"] for row in rows]
                session_mappings = {
                    (str(row["principal_id"]), str(row["agent_session_id"])) for row in rows
                }
                deleted = await conn.fetchrow(_DELETE_CONVERSATIONS, principals, conversation_ids)
                for principal, session_id in session_mappings:
                    await conn.execute(
                        _DELETE_AGENT_SESSION_IF_UNREFERENCED,
                        principal,
                        session_id,
                    )
                    await conn.execute(_RELEASE_SESSION_WORK_ROWS, principal, session_id)
                return int(deleted["count"]) if deleted is not None else 0

        return await self._run_write(_operation)


__all__ = [
    "WEB_CONVERSATION_MIGRATIONS",
    "WEB_CONVERSATION_SCHEMA_TABLES",
    "AnswerTurnCreation",
    "ConversationHead",
    "ConversationHistoryPage",
    "ConversationSubmissionConflict",
    "LinkedTurn",
    "PGWebConversationStore",
]
