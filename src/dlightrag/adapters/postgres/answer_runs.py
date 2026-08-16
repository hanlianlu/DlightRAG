# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL state for durable Answer runs.

PostgreSQL is authoritative for the whole run lifecycle: status, phase, completed
control turns, cancellation, lease ownership, durable events, the latest
checkpoint, the canonical result, and immutable input/fetched artifacts. Every
method is owner-scoped, and every worker write is predicated on the worker's
lease owner plus fencing epoch so a process that lost its lease can never mutate
state owned by the current worker.

Lease duration, batch sizes, retention, and the crash-recovery bound are fixed
internal constants; this module deliberately exposes no operator knobs.
"""

import json
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

import asyncpg

from dlightrag.adapters.postgres._migrations import (
    ForeignKeyRequirement,
    Migration,
    TableRequirement,
    apply_migrations,
    verify_migrations,
)
from dlightrag.adapters.postgres._operations import ConnectionPool, PostgresOperationRunner
from dlightrag.runtime import (
    ANSWER_RUN_LEASE_SECONDS,
    MAX_CONSECUTIVE_RECOVERIES,
    RUN_ABANDONED_ERROR_KIND,
    RUN_RETENTION_SECONDS,
    AnswerRunEvent,
    AnswerRunEventType,
    AnswerRunPhase,
    AnswerRunRecord,
    AnswerRunStatus,
    ArtifactAttachOutcome,
    CancellationOutcome,
    CheckpointCommit,
    ClaimedRun,
    IdempotencyKeyConflict,
    LeaseRenewal,
    PendingArtifact,
    PendingArtifactReference,
    RunArtifactReference,
    RunCheckpoint,
    RunCreation,
    RunDeletion,
    RunSchemaError,
    ShutdownOutcome,
    SweepOutcome,
    TerminalOutcome,
    canonical_run_request_json,
    parse_run_id,
)

ANSWER_RUN_MIGRATION_SCOPE = "answer_runs"

_ABANDONED_ERROR_MESSAGE = "Answer run exceeded its crash-recovery bound."
_BATCH_LIMIT = 200
_EVENT_PAGE_LIMIT = 500


_CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS dlightrag_answer_runs (
    owner_id            TEXT        NOT NULL,
    run_id              UUID        NOT NULL,
    idempotency_key     TEXT,
    request_json        JSONB       NOT NULL,
    request_fingerprint TEXT        NOT NULL,
    status              TEXT        NOT NULL DEFAULT 'queued',
    phase               TEXT,
    stop_reason         TEXT,
    completed_turns     INTEGER     NOT NULL DEFAULT 0,
    cancel_requested_at TIMESTAMPTZ,
    lease_owner         TEXT,
    lease_expires_at    TIMESTAMPTZ,
    fencing_epoch       BIGINT      NOT NULL DEFAULT 0,
    recovery_count      INTEGER     NOT NULL DEFAULT 0,
    next_event_sequence BIGINT      NOT NULL DEFAULT 1,
    events_trimmed_at   TIMESTAMPTZ,
    checkpoint_json     JSONB,
    result_json         JSONB,
    error_kind          TEXT,
    error_message       TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at          TIMESTAMPTZ,
    finished_at         TIMESTAMPTZ,
    PRIMARY KEY (owner_id, run_id),
    CONSTRAINT dlightrag_answer_runs_status_check
        CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
    CONSTRAINT dlightrag_answer_runs_phase_check
        CHECK (phase IS NULL OR phase IN ('planning', 'searching', 'researching', 'generating')),
    CONSTRAINT dlightrag_answer_runs_counter_check
        CHECK (completed_turns >= 0 AND fencing_epoch >= 0 AND recovery_count >= 0
               AND next_event_sequence >= 1),
    CONSTRAINT dlightrag_answer_runs_lease_check
        CHECK ((lease_owner IS NULL) = (lease_expires_at IS NULL)),
    CONSTRAINT dlightrag_answer_runs_terminal_check
        CHECK ((status IN ('succeeded', 'failed', 'cancelled')) = (finished_at IS NOT NULL)),
    CONSTRAINT dlightrag_answer_runs_result_check
        CHECK (status <> 'succeeded' OR result_json IS NOT NULL),
    CONSTRAINT dlightrag_answer_runs_error_check
        CHECK ((status = 'failed') = (error_kind IS NOT NULL))
)
"""

_CREATE_EVENTS = """
CREATE TABLE IF NOT EXISTS dlightrag_answer_run_events (
    owner_id       TEXT        NOT NULL,
    run_id         UUID        NOT NULL,
    event_sequence BIGINT      NOT NULL,
    event_type     TEXT        NOT NULL,
    payload        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id, run_id, event_sequence),
    FOREIGN KEY (owner_id, run_id)
        REFERENCES dlightrag_answer_runs (owner_id, run_id) ON DELETE CASCADE,
    CONSTRAINT dlightrag_answer_run_events_type_check
        CHECK (event_type IN ('progress', 'token', 'reset', 'done', 'error')),
    CONSTRAINT dlightrag_answer_run_events_sequence_check
        CHECK (event_sequence >= 1)
)
"""

_CREATE_ARTIFACTS = """
CREATE TABLE IF NOT EXISTS dlightrag_answer_artifacts (
    owner_id   TEXT        NOT NULL,
    digest     TEXT        NOT NULL,
    byte_size  BIGINT      NOT NULL,
    content    BYTEA       NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id, digest),
    CONSTRAINT dlightrag_answer_artifacts_digest_check
        CHECK (digest ~ '^[0-9a-f]{64}$'),
    CONSTRAINT dlightrag_answer_artifacts_size_check
        CHECK (byte_size = octet_length(content))
)
"""

_CREATE_RUN_ARTIFACTS = """
CREATE TABLE IF NOT EXISTS dlightrag_answer_run_artifacts (
    owner_id          TEXT        NOT NULL,
    run_id            UUID        NOT NULL,
    resource_id       TEXT        NOT NULL,
    reference_kind    TEXT        NOT NULL,
    ordinal           INTEGER     NOT NULL,
    digest            TEXT        NOT NULL,
    filename          TEXT        NOT NULL,
    mime_type         TEXT        NOT NULL,
    transform_locator JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id, run_id, resource_id),
    UNIQUE (owner_id, run_id, reference_kind, ordinal),
    FOREIGN KEY (owner_id, run_id)
        REFERENCES dlightrag_answer_runs (owner_id, run_id) ON DELETE CASCADE,
    FOREIGN KEY (owner_id, digest)
        REFERENCES dlightrag_answer_artifacts (owner_id, digest) ON DELETE RESTRICT,
    CONSTRAINT dlightrag_answer_run_artifacts_kind_check
        CHECK (reference_kind IN
               ('current_attachment', 'history_attachment', 'fetched_resource')),
    CONSTRAINT dlightrag_answer_run_artifacts_ordinal_check
        CHECK (ordinal >= 0)
)
"""

_CREATE_INDEXES = (
    # Claim and sweep scan nonterminal rows oldest-first across every owner.
    "CREATE INDEX IF NOT EXISTS idx_dlightrag_answer_runs_claim "
    "ON dlightrag_answer_runs (created_at, run_id) "
    "WHERE status IN ('queued', 'running')",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_dlightrag_answer_runs_idempotency "
    "ON dlightrag_answer_runs (owner_id, idempotency_key) "
    "WHERE idempotency_key IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS idx_dlightrag_answer_runs_retention "
    "ON dlightrag_answer_runs (finished_at) "
    "WHERE finished_at IS NOT NULL",
    # Exactly one terminal event per run, enforced durably rather than by convention.
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_dlightrag_answer_run_events_terminal "
    "ON dlightrag_answer_run_events (owner_id, run_id) "
    "WHERE event_type IN ('done', 'error')",
    # Reverse lookup for ownership-safe blob cleanup and the RESTRICT foreign key.
    "CREATE INDEX IF NOT EXISTS idx_dlightrag_answer_run_artifacts_digest "
    "ON dlightrag_answer_run_artifacts (owner_id, digest)",
)

ANSWER_RUN_MIGRATIONS = (
    Migration(
        "0001_answer_runs",
        "Create durable Answer run, event, and artifact state",
        (
            _CREATE_RUNS,
            _CREATE_EVENTS,
            _CREATE_ARTIFACTS,
            _CREATE_RUN_ARTIFACTS,
            *_CREATE_INDEXES,
        ),
    ),
)

ANSWER_RUN_SCHEMA_TABLES = (
    TableRequirement(
        name="dlightrag_answer_runs",
        columns=(
            "owner_id",
            "run_id",
            "idempotency_key",
            "request_json",
            "request_fingerprint",
            "status",
            "phase",
            "stop_reason",
            "completed_turns",
            "cancel_requested_at",
            "lease_owner",
            "lease_expires_at",
            "fencing_epoch",
            "recovery_count",
            "next_event_sequence",
            "events_trimmed_at",
            "checkpoint_json",
            "result_json",
            "error_kind",
            "error_message",
            "created_at",
            "updated_at",
            "started_at",
            "finished_at",
        ),
        primary_key=("owner_id", "run_id"),
        checks=(
            "dlightrag_answer_runs_status_check",
            "dlightrag_answer_runs_phase_check",
            "dlightrag_answer_runs_counter_check",
            "dlightrag_answer_runs_lease_check",
            "dlightrag_answer_runs_terminal_check",
            "dlightrag_answer_runs_result_check",
            "dlightrag_answer_runs_error_check",
        ),
        indexes=(
            "idx_dlightrag_answer_runs_claim",
            "idx_dlightrag_answer_runs_retention",
        ),
        unique_indexes=("idx_dlightrag_answer_runs_idempotency",),
    ),
    TableRequirement(
        name="dlightrag_answer_run_events",
        columns=(
            "owner_id",
            "run_id",
            "event_sequence",
            "event_type",
            "payload",
            "created_at",
        ),
        primary_key=("owner_id", "run_id", "event_sequence"),
        foreign_keys=(
            ForeignKeyRequirement(
                columns=("owner_id", "run_id"), references="dlightrag_answer_runs"
            ),
        ),
        checks=(
            "dlightrag_answer_run_events_type_check",
            "dlightrag_answer_run_events_sequence_check",
        ),
        unique_indexes=("idx_dlightrag_answer_run_events_terminal",),
    ),
    TableRequirement(
        name="dlightrag_answer_artifacts",
        columns=("owner_id", "digest", "byte_size", "content", "created_at"),
        primary_key=("owner_id", "digest"),
        checks=(
            "dlightrag_answer_artifacts_digest_check",
            "dlightrag_answer_artifacts_size_check",
        ),
    ),
    TableRequirement(
        name="dlightrag_answer_run_artifacts",
        columns=(
            "owner_id",
            "run_id",
            "resource_id",
            "reference_kind",
            "ordinal",
            "digest",
            "filename",
            "mime_type",
            "transform_locator",
            "created_at",
        ),
        primary_key=("owner_id", "run_id", "resource_id"),
        unique=(("owner_id", "run_id", "reference_kind", "ordinal"),),
        foreign_keys=(
            ForeignKeyRequirement(
                columns=("owner_id", "run_id"), references="dlightrag_answer_runs"
            ),
            ForeignKeyRequirement(
                columns=("owner_id", "digest"), references="dlightrag_answer_artifacts"
            ),
        ),
        checks=(
            "dlightrag_answer_run_artifacts_kind_check",
            "dlightrag_answer_run_artifacts_ordinal_check",
        ),
        indexes=("idx_dlightrag_answer_run_artifacts_digest",),
    ),
)

#: ``(expression, output name)`` for every column :func:`answer_run_record` reads.
#: Joined queries project them through :func:`answer_run_columns` so a table alias
#: disambiguates the names other durable tables also use.
_RUN_COLUMN_SPECS: tuple[tuple[str, str], ...] = (
    ("owner_id", "owner_id"),
    ("run_id::text", "run_id"),
    ("idempotency_key", "idempotency_key"),
    ("request_json", "request_json"),
    ("status", "status"),
    ("phase", "phase"),
    ("stop_reason", "stop_reason"),
    ("completed_turns", "completed_turns"),
    ("cancel_requested_at", "cancel_requested_at"),
    ("lease_owner", "lease_owner"),
    ("lease_expires_at", "lease_expires_at"),
    ("fencing_epoch", "fencing_epoch"),
    ("recovery_count", "recovery_count"),
    ("next_event_sequence", "next_event_sequence"),
    ("events_trimmed_at", "events_trimmed_at"),
    ("result_json", "result_json"),
    ("error_kind", "error_kind"),
    ("error_message", "error_message"),
    ("created_at", "created_at"),
    ("updated_at", "updated_at"),
    ("started_at", "started_at"),
    ("finished_at", "finished_at"),
)


def answer_run_columns(alias: str = "") -> str:
    """Project one run row's columns, optionally through a join alias."""
    prefix = f"{alias}." if alias else ""
    return ",\n".join(f"{prefix}{expression} AS {name}" for expression, name in _RUN_COLUMN_SPECS)


_RUN_COLUMNS = answer_run_columns()

_INSERT_RUN = f"""
INSERT INTO dlightrag_answer_runs (
    owner_id, run_id, idempotency_key, request_json, request_fingerprint
)
VALUES ($1, $2, $3, $4::jsonb, $5)
ON CONFLICT (owner_id, idempotency_key) WHERE idempotency_key IS NOT NULL DO NOTHING
RETURNING {_RUN_COLUMNS}
"""  # noqa: S608 - interpolates only the trusted _RUN_COLUMNS constant

_SELECT_RUN_BY_KEY = f"""
SELECT {_RUN_COLUMNS}, request_fingerprint
FROM dlightrag_answer_runs
WHERE owner_id = $1 AND idempotency_key = $2
"""  # noqa: S608 - interpolates only the trusted _RUN_COLUMNS constant

_SELECT_ACTIVE_RUN_REQUIREMENTS = """
SELECT DISTINCT
    request_json->>'context_policy_revision' AS context_policy_revision,
    request_json->'pinned_models' AS pinned_models
FROM dlightrag_answer_runs
WHERE status IN ('queued', 'running')
    AND cancel_requested_at IS NULL
    AND NOT (
            status = 'running'
            AND lease_expires_at < NOW()
            AND recovery_count >= $1
    )
"""

_SELECT_RUN = f"""
SELECT {_RUN_COLUMNS}
FROM dlightrag_answer_runs
WHERE owner_id = $1 AND run_id = $2
"""  # noqa: S608 - interpolates only the trusted _RUN_COLUMNS constant

_SELECT_RUN_FOR_UPDATE = f"""
SELECT {_RUN_COLUMNS}
FROM dlightrag_answer_runs
WHERE owner_id = $1 AND run_id = $2
FOR UPDATE
"""  # noqa: S608 - interpolates only the trusted _RUN_COLUMNS constant

_SELECT_EVENTS = """
SELECT event_sequence, event_type, payload, created_at
FROM dlightrag_answer_run_events
WHERE owner_id = $1 AND run_id = $2 AND event_sequence > $3
ORDER BY event_sequence
LIMIT $4
"""

# Oldest eligible work across every owner. Cancel-pending rows and rows past the
# recovery bound are left to the sweeper, which finalizes them without holding a
# local execution slot.
_SELECT_CLAIM_CANDIDATE = """
SELECT owner_id, run_id
FROM dlightrag_answer_runs
WHERE cancel_requested_at IS NULL
  AND (
      status = 'queued'
      OR (status = 'running' AND lease_expires_at < NOW() AND recovery_count < $1)
  )
ORDER BY created_at, run_id
LIMIT 1
FOR UPDATE SKIP LOCKED
"""

_CLAIM_RUN = f"""
UPDATE dlightrag_answer_runs
SET status = 'running',
    lease_owner = $3,
    lease_expires_at = NOW() + ($4 * INTERVAL '1 second'),
    fencing_epoch = fencing_epoch + 1,
    recovery_count = CASE WHEN status = 'running' THEN recovery_count + 1 ELSE recovery_count END,
    started_at = COALESCE(started_at, NOW()),
    updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2
  AND status IN ('queued', 'running')
RETURNING {_RUN_COLUMNS}, checkpoint_json
"""  # noqa: S608 - interpolates only the trusted _RUN_COLUMNS constant

_HEARTBEAT = """
UPDATE dlightrag_answer_runs
SET lease_expires_at = NOW() + ($5 * INTERVAL '1 second'),
    updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2
  AND lease_owner = $3 AND fencing_epoch = $4
  AND status = 'running' AND lease_expires_at > NOW()
RETURNING (cancel_requested_at IS NOT NULL) AS cancel_requested
"""

# The UPDATE is the row lock: concurrent appenders serialize on it and each one
# consumes exactly one sequence, so the per-run event stream is gap-free.
_APPEND_EVENT = """
WITH bumped AS (
    UPDATE dlightrag_answer_runs
    SET next_event_sequence = next_event_sequence + 1,
        phase = COALESCE($5::text, phase),
        lease_expires_at = NOW() + ($8 * INTERVAL '1 second'),
        updated_at = NOW()
    WHERE owner_id = $1 AND run_id = $2
      AND lease_owner = $3 AND fencing_epoch = $4
      AND status = 'running' AND lease_expires_at > NOW()
    RETURNING next_event_sequence - 1 AS event_sequence
), inserted AS (
    INSERT INTO dlightrag_answer_run_events (
        owner_id, run_id, event_sequence, event_type, payload
    )
    SELECT $1, $2, event_sequence, $6::text, $7::jsonb FROM bumped
    RETURNING event_sequence
)
SELECT event_sequence FROM inserted
"""

_COMMIT_CHECKPOINT = """
UPDATE dlightrag_answer_runs
SET checkpoint_json = $6::jsonb,
    completed_turns = completed_turns + 1,
    recovery_count = 0,
    lease_expires_at = NOW() + ($7 * INTERVAL '1 second'),
    updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2
  AND lease_owner = $3 AND fencing_epoch = $4
  AND status = 'running' AND lease_expires_at > NOW()
  AND completed_turns = $5
RETURNING completed_turns
"""

_LOCK_RUN_STATE = """
SELECT status,
       lease_owner,
       fencing_epoch,
       completed_turns,
       (lease_expires_at > NOW()) AS lease_live,
       checkpoint_json ->> 'completed_turns' AS checkpoint_turn
FROM dlightrag_answer_runs
WHERE owner_id = $1 AND run_id = $2
FOR UPDATE
"""

# One fenced terminal transition that also appends the run's single terminal
# event. $12 withholds success while a cancellation is pending. A terminal run
# never resumes, so its checkpoint is dropped rather than retained for the
# lifetime of a conversation-linked row.
_FINISH_RUN = """
WITH bumped AS (
    UPDATE dlightrag_answer_runs
    SET status = $5::text,
        stop_reason = $6::text,
        result_json = $7::jsonb,
        error_kind = $8::text,
        error_message = $9::text,
        phase = NULL,
        checkpoint_json = NULL,
        lease_owner = NULL,
        lease_expires_at = NULL,
        finished_at = NOW(),
        updated_at = NOW(),
        next_event_sequence = next_event_sequence + 1
    WHERE owner_id = $1 AND run_id = $2
      AND lease_owner = $3 AND fencing_epoch = $4
      AND status = 'running' AND lease_expires_at > NOW()
      AND (NOT $12::boolean OR cancel_requested_at IS NULL)
    RETURNING next_event_sequence - 1 AS event_sequence
), inserted AS (
    INSERT INTO dlightrag_answer_run_events (
        owner_id, run_id, event_sequence, event_type, payload
    )
    SELECT $1, $2, event_sequence, $10::text, $11::jsonb FROM bumped
    RETURNING event_sequence
)
SELECT event_sequence FROM inserted
"""

# Terminal transition for rows with no live lease: queued cancellation, sweeper
# cancellation, and recovery-bound abandonment. Callers must already hold the row
# lock (explicit FOR UPDATE or FOR UPDATE SKIP LOCKED candidate selection).
_FINALIZE_UNLEASED = """
WITH bumped AS (
    UPDATE dlightrag_answer_runs AS r
    SET status = $3::text,
        cancel_requested_at = CASE
            WHEN $3::text = 'cancelled' THEN COALESCE(r.cancel_requested_at, NOW())
            ELSE r.cancel_requested_at
        END,
        stop_reason = NULL,
        error_kind = $4::text,
        error_message = $5::text,
        phase = NULL,
        checkpoint_json = NULL,
        lease_owner = NULL,
        lease_expires_at = NULL,
        finished_at = NOW(),
        updated_at = NOW(),
        next_event_sequence = r.next_event_sequence + 1
    WHERE (r.owner_id, r.run_id) IN (SELECT * FROM unnest($1::text[], $2::uuid[]))
      AND r.status IN ('queued', 'running')
      AND (r.lease_expires_at IS NULL OR r.lease_expires_at < NOW())
    RETURNING r.owner_id, r.run_id, r.next_event_sequence - 1 AS event_sequence
), inserted AS (
    INSERT INTO dlightrag_answer_run_events (
        owner_id, run_id, event_sequence, event_type, payload
    )
    SELECT owner_id, run_id, event_sequence, $6::text, $7::jsonb FROM bumped
    RETURNING event_sequence
)
SELECT count(*)::int FROM inserted
"""

_REQUEST_CANCELLATION = """
UPDATE dlightrag_answer_runs
SET cancel_requested_at = COALESCE(cancel_requested_at, NOW()),
    updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2
"""

_REQUEUE_RUN = """
UPDATE dlightrag_answer_runs
SET status = 'queued',
    lease_owner = NULL,
    lease_expires_at = NULL,
    updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2
  AND lease_owner = $3 AND fencing_epoch = $4
  AND status = 'running' AND lease_expires_at > NOW()
  AND cancel_requested_at IS NULL
RETURNING 1
"""

_SELECT_CANCEL_PENDING = """
SELECT owner_id, run_id
FROM dlightrag_answer_runs
WHERE cancel_requested_at IS NOT NULL
  AND status IN ('queued', 'running')
  AND (lease_expires_at IS NULL OR lease_expires_at < NOW())
ORDER BY updated_at
LIMIT $1
FOR UPDATE SKIP LOCKED
"""

_SELECT_ABANDONED = """
SELECT owner_id, run_id
FROM dlightrag_answer_runs
WHERE status = 'running'
  AND lease_expires_at < NOW()
  AND recovery_count >= $1
  AND cancel_requested_at IS NULL
ORDER BY updated_at
LIMIT $2
FOR UPDATE SKIP LOCKED
"""

_INSERT_ARTIFACT = """
INSERT INTO dlightrag_answer_artifacts (owner_id, digest, byte_size, content)
VALUES ($1, $2, $3, $4)
ON CONFLICT (owner_id, digest) DO NOTHING
"""

_INSERT_RUN_ARTIFACT = """
INSERT INTO dlightrag_answer_run_artifacts (
    owner_id, run_id, resource_id, reference_kind, ordinal, digest,
    filename, mime_type, transform_locator
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb)
ON CONFLICT (owner_id, run_id, resource_id) DO NOTHING
"""

# An uncheckpointed read-only tool batch may execute again and fetch different
# bytes under a new resource id, so (reference_kind, ordinal) is the logical
# replay slot. Releasing it — and any row already holding the incoming resource
# id — lets the rebind below replace the whole reference atomically instead of
# colliding on the ordinal unique index. Only fetched rows are released; an
# accepted input reference is never rewritten.
_RELEASE_FETCHED_SLOT = """
DELETE FROM dlightrag_answer_run_artifacts
WHERE owner_id = $1 AND run_id = $2 AND reference_kind = 'fetched_resource'
  AND (ordinal = $3 OR resource_id = $4)
RETURNING digest
"""

_INSERT_FETCHED_REFERENCE = """
INSERT INTO dlightrag_answer_run_artifacts (
    owner_id, run_id, resource_id, reference_kind, ordinal, digest,
    filename, mime_type, transform_locator
)
VALUES ($1, $2, $3, 'fetched_resource', $4, $5, $6, $7, $8::jsonb)
"""

# The artifact write fence: it must hold the run row while the blobs and
# references are written, so a worker that lost its lease mid-write persists
# nothing.
_LOCK_RUN_FOR_WORKER = """
SELECT completed_turns
FROM dlightrag_answer_runs
WHERE owner_id = $1 AND run_id = $2
  AND lease_owner = $3 AND fencing_epoch = $4
  AND status = 'running' AND lease_expires_at > NOW()
FOR UPDATE
"""

_SELECT_ARTIFACT = """
SELECT content
FROM dlightrag_answer_artifacts
WHERE owner_id = $1 AND digest = $2
"""

_SELECT_RUN_ARTIFACTS = """
SELECT resource_id, reference_kind, ordinal, digest, filename, mime_type,
       transform_locator, created_at
FROM dlightrag_answer_run_artifacts
WHERE owner_id = $1 AND run_id = $2
ORDER BY reference_kind, ordinal
"""

_SELECT_RUN_DIGESTS = """
SELECT DISTINCT owner_id, digest
FROM dlightrag_answer_run_artifacts
WHERE (owner_id, run_id) IN (SELECT * FROM unnest($1::text[], $2::uuid[]))
"""

_DELETE_RUNS = """
WITH deleted AS (
    DELETE FROM dlightrag_answer_runs
    WHERE (owner_id, run_id) IN (SELECT * FROM unnest($1::text[], $2::uuid[]))
    RETURNING 1
)
SELECT count(*)::int FROM deleted
"""

# SKIP LOCKED yields to any transaction already holding the artifact's key-share
# lock for a new reference, so an adopting run always wins over cleanup.
_DELETE_UNREFERENCED_ARTIFACTS = """
WITH candidates AS (
    SELECT a.owner_id, a.digest
    FROM dlightrag_answer_artifacts AS a
    WHERE (a.owner_id, a.digest) IN (SELECT * FROM unnest($1::text[], $2::text[]))
      AND NOT EXISTS (
          SELECT 1
          FROM dlightrag_answer_run_artifacts AS r
          WHERE r.owner_id = a.owner_id AND r.digest = a.digest
      )
    FOR UPDATE SKIP LOCKED
), deleted AS (
    DELETE FROM dlightrag_answer_artifacts AS a
    USING candidates AS c
    WHERE a.owner_id = c.owner_id AND a.digest = c.digest
    RETURNING 1
)
SELECT count(*)::int FROM deleted
"""

# A succeeded run a Web conversation still links to is owned by that
# conversation's lifetime, so retention skips it until the conversation (or the
# turn itself) is gone. Failed and cancelled runs always prune, and their
# visible terminal turns cascade away with them.
_SELECT_EXPIRED_RUNS = """
SELECT runs.owner_id, runs.run_id
FROM dlightrag_answer_runs AS runs
WHERE runs.status IN ('succeeded', 'failed', 'cancelled')
  AND runs.finished_at < NOW() - ($1 * INTERVAL '1 second')
  AND (
      runs.status <> 'succeeded'
      OR NOT EXISTS (
          SELECT 1
          FROM web_conversation_turns AS turns
          WHERE turns.principal_id = runs.owner_id
            AND turns.answer_run_id = runs.run_id
      )
  )
ORDER BY runs.finished_at
LIMIT $2
FOR UPDATE OF runs SKIP LOCKED
"""

_SELECT_TRIMMABLE_RUNS = """
SELECT owner_id, run_id
FROM dlightrag_answer_runs
WHERE status IN ('succeeded', 'failed', 'cancelled')
  AND finished_at < NOW() - ($1 * INTERVAL '1 second')
  AND events_trimmed_at IS NULL
ORDER BY finished_at
LIMIT $2
FOR UPDATE SKIP LOCKED
"""

_DELETE_EVENTS_FOR_RUNS = """
DELETE FROM dlightrag_answer_run_events
WHERE (owner_id, run_id) IN (SELECT * FROM unnest($1::text[], $2::uuid[]))
"""

_MARK_EVENTS_TRIMMED = """
UPDATE dlightrag_answer_runs
SET events_trimmed_at = NOW(),
    updated_at = NOW()
WHERE (owner_id, run_id) IN (SELECT * FROM unnest($1::text[], $2::uuid[]))
"""


def _new_run_id() -> uuid.UUID:
    """Return a fresh time-ordered UUIDv7 run identifier."""
    return uuid.uuid7()


class PGAnswerRunStore(PostgresOperationRunner):
    """Owner-scoped durable Answer run state backed by PostgreSQL."""

    def __init__(self, *, pool: ConnectionPool | None = None) -> None:
        super().__init__(pool=pool)
        self._initialized = False

    async def _run_read[T](self, operation: Callable[[Any], Awaitable[T]]) -> T:
        return await self._run(operation)

    async def _run_write[T](self, operation: Callable[[Any], Awaitable[T]]) -> T:
        return await self._run_once(operation)

    async def initialize(self, *, validate_only: bool = False) -> None:
        """Create the durable Answer run schema, or validate it (reader)."""
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
                return
            await apply_migrations(
                conn,
                scope=ANSWER_RUN_MIGRATION_SCOPE,
                migrations=ANSWER_RUN_MIGRATIONS,
                schema_error=RunSchemaError,
            )

        await self._run_write(_operation)
        self._initialized = True

    # ------------------------------------------------------------------
    # Creation and owner-scoped reads
    # ------------------------------------------------------------------

    async def replay_run(
        self,
        *,
        owner_id: str,
        idempotency_key: str,
        idempotency_fingerprint: str,
    ) -> RunCreation | None:
        """Return an accepted keyed run before any request enrichment is repeated."""
        owner = _require_owner(owner_id)
        if not idempotency_key:
            raise ValueError("idempotency_key must be non-empty")
        if not idempotency_fingerprint:
            raise ValueError("idempotency_fingerprint must be non-empty")

        async def _operation(conn: Any) -> RunCreation | None:
            row = await conn.fetchrow(_SELECT_RUN_BY_KEY, owner, idempotency_key)
            if row is None:
                return None
            if str(row["request_fingerprint"]) != idempotency_fingerprint:
                raise IdempotencyKeyConflict(
                    "idempotency key was reused with different request input"
                )
            return RunCreation(run=answer_run_record(row), replayed=True)

        return await self._run_read(_operation)

    async def list_active_run_requirements(self) -> tuple[Mapping[str, Any], ...]:
        """Return distinct requirements for active runs that may execute."""

        async def _operation(conn: Any) -> tuple[Mapping[str, Any], ...]:
            rows = await conn.fetch(
                _SELECT_ACTIVE_RUN_REQUIREMENTS,
                MAX_CONSECUTIVE_RECOVERIES,
            )
            return tuple(
                {
                    "context_policy_revision": row["context_policy_revision"],
                    "pinned_models": _json_array(row["pinned_models"]),
                }
                for row in rows
            )

        return await self._run_read(_operation)

    async def create_run(
        self,
        *,
        owner_id: str,
        request: Mapping[str, Any],
        idempotency_fingerprint: str,
        idempotency_key: str | None = None,
        artifacts: Sequence[PendingArtifact] = (),
        references: Sequence[PendingArtifactReference] = (),
    ) -> RunCreation:
        """Accept one run, its input blobs, and its references in one transaction.

        Replaying an idempotency key with the same normalized request returns the
        existing run; reusing it with different input raises
        :class:`IdempotencyKeyConflict`.
        """

        # An unusable owner or request is a caller fault, so both are rejected
        # before a connection is acquired rather than inside a transaction that
        # would then have to unwind.
        _require_owner(owner_id)
        canonical_run_request_json(request)

        async def _operation(conn: Any) -> RunCreation:
            async with conn.transaction():
                return await self.create_run_in(
                    conn,
                    owner_id=owner_id,
                    request=request,
                    idempotency_fingerprint=idempotency_fingerprint,
                    idempotency_key=idempotency_key,
                    artifacts=artifacts,
                    references=references,
                )

        return await self._run_write(_operation)

    async def create_run_in(
        self,
        conn: Any,
        *,
        owner_id: str,
        request: Mapping[str, Any],
        idempotency_fingerprint: str,
        idempotency_key: str | None = None,
        artifacts: Sequence[PendingArtifact] = (),
        references: Sequence[PendingArtifactReference] = (),
    ) -> RunCreation:
        """Create or replay one run inside a transaction the caller already owns.

        This is the composition seam another durable table uses to link its own
        row to the accepted run atomically. It performs no transaction control of
        its own, so the caller's commit is what makes the run and its link
        durable together.
        """
        owner = _require_owner(owner_id)
        if any(reference.reference_kind == "fetched_resource" for reference in references):
            # A fetched resource is worker-fenced run state, never accepted input.
            raise ValueError("fetched_resource references cannot be run creation inputs")
        payload = canonical_run_request_json(request)
        if not idempotency_fingerprint:
            raise ValueError("idempotency_fingerprint must be non-empty")
        fingerprint = idempotency_fingerprint
        run_uuid = _new_run_id()
        row = await conn.fetchrow(
            _INSERT_RUN, owner, run_uuid, idempotency_key, payload, fingerprint
        )
        if row is None:
            existing = await conn.fetchrow(_SELECT_RUN_BY_KEY, owner, idempotency_key)
            if existing is None:
                raise RuntimeError("answer run insert reported a vanished conflict")
            if str(existing["request_fingerprint"]) != fingerprint:
                raise IdempotencyKeyConflict(
                    "idempotency key was reused with different request input"
                )
            return RunCreation(run=answer_run_record(existing), replayed=True)
        await self._write_artifacts(
            conn,
            owner_id=owner,
            run_uuid=run_uuid,
            artifacts=artifacts,
            references=references,
        )
        return RunCreation(run=answer_run_record(row), replayed=False)

    async def get_run(self, *, owner_id: str, run_id: str) -> AnswerRunRecord | None:
        """Load one owned run; unknown and foreign identifiers both return ``None``."""
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return None

        async def _operation(conn: Any) -> AnswerRunRecord | None:
            row = await conn.fetchrow(_SELECT_RUN, owner, run_uuid)
            return answer_run_record(row) if row is not None else None

        return await self._run_read(_operation)

    async def read_event_page(
        self,
        *,
        owner_id: str,
        run_id: str,
        after_sequence: int = 0,
    ) -> tuple[AnswerRunEvent, ...]:
        """Replay at most one bounded page of committed events after ``after_sequence``.

        This never returns a complete replay by itself. A subscriber MUST loop,
        passing the last returned sequence as the next cursor, until it receives
        an empty page; only then has it seen every committed event, including the
        terminal one. The page size is a fixed internal bound.
        """
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return ()
        cursor = max(0, int(after_sequence))

        async def _operation(conn: Any) -> tuple[AnswerRunEvent, ...]:
            rows = await conn.fetch(_SELECT_EVENTS, owner, run_uuid, cursor, _EVENT_PAGE_LIMIT)
            return tuple(_event_record(row) for row in rows)

        return await self._run_read(_operation)

    async def request_cancellation(self, *, owner_id: str, run_id: str) -> CancellationOutcome:
        """Cancel a queued run outright or record the request for a running one."""
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return CancellationOutcome(outcome="unknown", run=None)

        async def _operation(conn: Any) -> CancellationOutcome:
            async with conn.transaction():
                row = await conn.fetchrow(_SELECT_RUN_FOR_UPDATE, owner, run_uuid)
                if row is None:
                    return CancellationOutcome(outcome="unknown", run=None)
                record = answer_run_record(row)
                if record.terminal:
                    return CancellationOutcome(outcome="already_terminal", run=record)
                if record.status == "queued":
                    await conn.fetchval(
                        _FINALIZE_UNLEASED,
                        [owner],
                        [run_uuid],
                        "cancelled",
                        None,
                        None,
                        "done",
                        json.dumps({"status": "cancelled"}),
                    )
                    updated = await conn.fetchrow(_SELECT_RUN, owner, run_uuid)
                    return CancellationOutcome(
                        outcome="cancelled",
                        run=answer_run_record(updated) if updated is not None else record,
                    )
                await conn.execute(_REQUEST_CANCELLATION, owner, run_uuid)
                updated = await conn.fetchrow(_SELECT_RUN, owner, run_uuid)
                return CancellationOutcome(
                    outcome="pending",
                    run=answer_run_record(updated) if updated is not None else record,
                )

        return await self._run_write(_operation)

    # ------------------------------------------------------------------
    # Worker lifecycle
    # ------------------------------------------------------------------

    async def claim_next(self, *, worker_id: str) -> ClaimedRun | None:
        """Claim the oldest eligible run for a worker that already holds a slot."""
        worker = str(worker_id).strip()
        if not worker:
            raise ValueError("worker_id cannot be empty")

        async def _operation(conn: Any) -> ClaimedRun | None:
            async with conn.transaction():
                candidate = await conn.fetchrow(_SELECT_CLAIM_CANDIDATE, MAX_CONSECUTIVE_RECOVERIES)
                if candidate is None:
                    return None
                row = await conn.fetchrow(
                    _CLAIM_RUN,
                    candidate["owner_id"],
                    candidate["run_id"],
                    worker,
                    ANSWER_RUN_LEASE_SECONDS,
                )
                if row is None:
                    return None
                return ClaimedRun(
                    run=answer_run_record(row),
                    checkpoint=_checkpoint_record(row["checkpoint_json"]),
                )

        return await self._run_write(_operation)

    async def heartbeat(
        self, *, owner_id: str, run_id: str, worker_id: str, fencing_epoch: int
    ) -> LeaseRenewal:
        """Renew an unexpired fenced lease and report pending cancellation."""
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return LeaseRenewal(renewed=False, cancel_requested=False)

        async def _operation(conn: Any) -> LeaseRenewal:
            row = await conn.fetchrow(
                _HEARTBEAT,
                owner,
                run_uuid,
                worker_id,
                fencing_epoch,
                ANSWER_RUN_LEASE_SECONDS,
            )
            if row is None:
                return LeaseRenewal(renewed=False, cancel_requested=False)
            return LeaseRenewal(renewed=True, cancel_requested=bool(row["cancel_requested"]))

        return await self._run_write(_operation)

    async def record_phase(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        phase: AnswerRunPhase,
    ) -> int | None:
        """Advance the run's phase and append its ``progress`` event atomically."""
        return await self._append_event(
            owner_id=owner_id,
            run_id=run_id,
            worker_id=worker_id,
            fencing_epoch=fencing_epoch,
            phase=phase,
            event_type="progress",
            payload={"phase": phase},
        )

    async def append_token_batch(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        text: str,
    ) -> int | None:
        """Append one coalesced batch of generated text."""
        return await self._append_event(
            owner_id=owner_id,
            run_id=run_id,
            worker_id=worker_id,
            fencing_epoch=fencing_epoch,
            phase=None,
            event_type="token",
            payload={"text": text},
        )

    async def append_reset(
        self, *, owner_id: str, run_id: str, worker_id: str, fencing_epoch: int
    ) -> int | None:
        """Append the event that clears a partial draft before regeneration."""
        return await self._append_event(
            owner_id=owner_id,
            run_id=run_id,
            worker_id=worker_id,
            fencing_epoch=fencing_epoch,
            phase=None,
            event_type="reset",
            payload={},
        )

    async def commit_checkpoint(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        expected_completed_turns: int,
        version: int,
        state: Mapping[str, Any],
    ) -> CheckpointCommit:
        """Advance one control turn and its checkpoint under one fenced predicate.

        A zero-row compare-and-set is resolved through a locked reread instead of
        being read as lease loss, so an indeterminate commit result never fails a
        run that actually committed.
        """
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        expected = int(expected_completed_turns)
        committed_turn = expected + 1
        envelope = json.dumps(
            {"version": int(version), "completed_turns": committed_turn, "state": dict(state)},
            ensure_ascii=False,
            allow_nan=False,
        )
        if run_uuid is None:
            return CheckpointCommit(outcome="lease_lost", completed_turns=expected)

        async def _operation(conn: Any) -> CheckpointCommit:
            async with conn.transaction():
                turns = await conn.fetchval(
                    _COMMIT_CHECKPOINT,
                    owner,
                    run_uuid,
                    worker_id,
                    fencing_epoch,
                    expected,
                    envelope,
                    ANSWER_RUN_LEASE_SECONDS,
                )
                if turns is not None:
                    return CheckpointCommit(outcome="committed", completed_turns=int(turns))
                return await self._resolve_checkpoint(
                    conn,
                    owner_id=owner,
                    run_uuid=run_uuid,
                    worker_id=worker_id,
                    fencing_epoch=fencing_epoch,
                    expected=expected,
                    envelope=envelope,
                )

        return await self._run_write(_operation)

    async def finish_success(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        result: Mapping[str, Any],
        stop_reason: str | None = None,
    ) -> TerminalOutcome:
        """Store the canonical result, append ``done``, and succeed the run.

        A cancellation that won the row first commits ``cancelled`` instead.
        """
        payload = dict(result)
        outcome = await self._finish(
            owner_id=owner_id,
            run_id=run_id,
            worker_id=worker_id,
            fencing_epoch=fencing_epoch,
            status="succeeded",
            stop_reason=stop_reason,
            result=payload,
            error_kind=None,
            error_message=None,
            event_type="done",
            event_payload={"status": "succeeded", "result": payload},
            require_uncancelled=True,
        )
        if outcome.committed:
            return outcome
        return await self.finish_cancelled(
            owner_id=owner_id,
            run_id=run_id,
            worker_id=worker_id,
            fencing_epoch=fencing_epoch,
        )

    async def finish_failure(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        error_kind: str,
        error_message: str,
    ) -> TerminalOutcome:
        """Fail the run and append its single ``error`` event."""
        return await self._finish(
            owner_id=owner_id,
            run_id=run_id,
            worker_id=worker_id,
            fencing_epoch=fencing_epoch,
            status="failed",
            stop_reason=None,
            result=None,
            error_kind=error_kind,
            error_message=error_message,
            event_type="error",
            event_payload={"kind": error_kind, "message": error_message},
            require_uncancelled=False,
        )

    async def finish_cancelled(
        self, *, owner_id: str, run_id: str, worker_id: str, fencing_epoch: int
    ) -> TerminalOutcome:
        """Commit the cancellation a running worker observed."""
        return await self._finish(
            owner_id=owner_id,
            run_id=run_id,
            worker_id=worker_id,
            fencing_epoch=fencing_epoch,
            status="cancelled",
            stop_reason=None,
            result=None,
            error_kind=None,
            error_message=None,
            event_type="done",
            event_payload={"status": "cancelled"},
            require_uncancelled=False,
        )

    async def release_for_shutdown(
        self, *, owner_id: str, run_id: str, worker_id: str, fencing_epoch: int
    ) -> ShutdownOutcome:
        """Requeue an owned run on graceful shutdown, or finalize its cancellation.

        The requeue preserves the checkpoint, completed-turn count, cancellation
        field, and recovery count, so an orderly restart is not crash recovery.
        """
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return "lease_lost"

        async def _operation(conn: Any) -> bool:
            return (
                await conn.fetchval(_REQUEUE_RUN, owner, run_uuid, worker_id, fencing_epoch)
                is not None
            )

        if await self._run_write(_operation):
            return "requeued"
        cancelled = await self.finish_cancelled(
            owner_id=owner_id,
            run_id=run_id,
            worker_id=worker_id,
            fencing_epoch=fencing_epoch,
        )
        return "cancelled" if cancelled.committed else "lease_lost"

    async def sweep_once(self) -> SweepOutcome:
        """Finalize unleased cancellations and runs past the recovery bound.

        This path holds no execution slot: it only terminates rows whose owner is
        gone, so a busy process still recovers another host's abandoned work.
        """

        async def _operation(conn: Any) -> SweepOutcome:
            async with conn.transaction():
                cancelled = await _finalize_batch(
                    conn,
                    await conn.fetch(_SELECT_CANCEL_PENDING, _BATCH_LIMIT),
                    status="cancelled",
                    error_kind=None,
                    error_message=None,
                    event_type="done",
                    event_payload={"status": "cancelled"},
                )
                abandoned = await _finalize_batch(
                    conn,
                    await conn.fetch(_SELECT_ABANDONED, MAX_CONSECUTIVE_RECOVERIES, _BATCH_LIMIT),
                    status="failed",
                    error_kind=RUN_ABANDONED_ERROR_KIND,
                    error_message=_ABANDONED_ERROR_MESSAGE,
                    event_type="error",
                    event_payload={
                        "kind": RUN_ABANDONED_ERROR_KIND,
                        "message": _ABANDONED_ERROR_MESSAGE,
                    },
                )
                return SweepOutcome(cancelled=cancelled, abandoned=abandoned)

        return await self._run_write(_operation)

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

    async def attach_artifacts(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        expected_completed_turns: int,
        artifacts: Sequence[PendingArtifact] = (),
        references: Sequence[PendingArtifactReference] = (),
    ) -> ArtifactAttachOutcome:
        """Persist validated bytes and their references under the worker's fence.

        The run row is locked and its owner, running status, unexpired lease,
        fencing epoch, and completed-turn count are all validated before anything
        is written, so a stale, expired, or wrong-turn worker stores no blob and
        no reference. A ``fetched_resource`` reference rebinds its
        ``(reference_kind, ordinal)`` replay slot, and bytes the rebind orphaned
        are cleaned up ownership-safely in the same transaction.

        Accepted input artifacts belong to :meth:`create_run` and need no fence.
        """
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return "lease_lost"
        expected = int(expected_completed_turns)

        async def _operation(conn: Any) -> ArtifactAttachOutcome:
            async with conn.transaction():
                turns = await conn.fetchval(
                    _LOCK_RUN_FOR_WORKER, owner, run_uuid, worker_id, fencing_epoch
                )
                if turns is None:
                    return "lease_lost"
                if int(turns) != expected:
                    return "turn_mismatch"
                displaced = await self._write_artifacts(
                    conn,
                    owner_id=owner,
                    run_uuid=run_uuid,
                    artifacts=artifacts,
                    references=references,
                )
                if displaced:
                    await _delete_unreferenced(conn, [owner] * len(displaced), displaced)
                return "attached"

        return await self._run_write(_operation)

    async def load_artifact(self, *, owner_id: str, digest: str) -> bytes | None:
        """Read one owner's stored bytes by content address."""
        owner = _require_owner(owner_id)

        async def _operation(conn: Any) -> bytes | None:
            content = await conn.fetchval(_SELECT_ARTIFACT, owner, digest)
            return bytes(content) if content is not None else None

        return await self._run_read(_operation)

    async def list_run_artifacts(
        self, *, owner_id: str, run_id: str
    ) -> tuple[RunArtifactReference, ...]:
        """List one owned run's ordered artifact references."""
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return ()

        async def _operation(conn: Any) -> tuple[RunArtifactReference, ...]:
            rows = await conn.fetch(_SELECT_RUN_ARTIFACTS, owner, run_uuid)
            return tuple(_reference_record(row) for row in rows)

        return await self._run_read(_operation)

    async def delete_runs(self, *, owner_id: str, run_ids: Sequence[str]) -> RunDeletion:
        """Delete owned runs and any blobs no reference keeps alive."""

        async def _operation(conn: Any) -> RunDeletion:
            async with conn.transaction():
                return await self.delete_runs_in(conn, owner_id=owner_id, run_ids=run_ids)

        return await self._run_write(_operation)

    async def delete_runs_in(
        self, conn: Any, *, owner_id: str, run_ids: Sequence[str]
    ) -> RunDeletion:
        """Delete owned runs and orphaned blobs inside a caller-owned transaction.

        The composition seam a linked table uses so deleting its own rows and the
        runs they referenced is one atomic act; a lease-fenced worker can no
        longer append to a run whose row disappeared.
        """
        owner = _require_owner(owner_id)
        run_uuids = [parsed for parsed in (parse_run_id(value) for value in run_ids) if parsed]
        if not run_uuids:
            return RunDeletion(runs=0, artifacts=0)
        owners = [owner] * len(run_uuids)
        pairs = await conn.fetch(_SELECT_RUN_DIGESTS, owners, run_uuids)
        deleted = await conn.fetchval(_DELETE_RUNS, owners, run_uuids)
        artifacts = await _delete_unreferenced(conn, *_digest_pairs(pairs))
        return RunDeletion(runs=int(deleted or 0), artifacts=artifacts)

    # ------------------------------------------------------------------
    # Retention
    # ------------------------------------------------------------------

    async def trim_expired_event_logs(self) -> int:
        """Delete expired terminal runs' events and mark them trimmed.

        The canonical result stays on the run row; only its event endpoint becomes
        gone once ``events_trimmed_at`` is set.
        """

        async def _operation(conn: Any) -> int:
            async with conn.transaction():
                rows = await conn.fetch(_SELECT_TRIMMABLE_RUNS, RUN_RETENTION_SECONDS, _BATCH_LIMIT)
                if not rows:
                    return 0
                owners = [str(row["owner_id"]) for row in rows]
                run_uuids = [row["run_id"] for row in rows]
                await conn.execute(_DELETE_EVENTS_FOR_RUNS, owners, run_uuids)
                await conn.execute(_MARK_EVENTS_TRIMMED, owners, run_uuids)
                return len(rows)

        return await self._run_write(_operation)

    async def prune_expired_runs(self) -> RunDeletion:
        """Delete expired terminal runs in one bounded ``SKIP LOCKED`` batch."""

        async def _operation(conn: Any) -> RunDeletion:
            async with conn.transaction():
                rows = await conn.fetch(_SELECT_EXPIRED_RUNS, RUN_RETENTION_SECONDS, _BATCH_LIMIT)
                if not rows:
                    return RunDeletion(runs=0, artifacts=0)
                owners = [str(row["owner_id"]) for row in rows]
                run_uuids = [row["run_id"] for row in rows]
                pairs = await conn.fetch(_SELECT_RUN_DIGESTS, owners, run_uuids)
                deleted = await conn.fetchval(_DELETE_RUNS, owners, run_uuids)
                artifacts = await _delete_unreferenced(conn, *_digest_pairs(pairs))
                return RunDeletion(runs=int(deleted or 0), artifacts=artifacts)

        return await self._run_write(_operation)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _append_event(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        phase: AnswerRunPhase | None,
        event_type: AnswerRunEventType,
        payload: Mapping[str, Any],
    ) -> int | None:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return None
        encoded = json.dumps(dict(payload), ensure_ascii=False, allow_nan=False)

        async def _operation(conn: Any) -> int | None:
            sequence = await conn.fetchval(
                _APPEND_EVENT,
                owner,
                run_uuid,
                worker_id,
                fencing_epoch,
                phase,
                event_type,
                encoded,
                ANSWER_RUN_LEASE_SECONDS,
            )
            return int(sequence) if sequence is not None else None

        return await self._run_write(_operation)

    async def _finish(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        status: AnswerRunStatus,
        stop_reason: str | None,
        result: Mapping[str, Any] | None,
        error_kind: str | None,
        error_message: str | None,
        event_type: AnswerRunEventType,
        event_payload: Mapping[str, Any],
        require_uncancelled: bool,
    ) -> TerminalOutcome:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return TerminalOutcome(committed=False, status=None, event_sequence=None)
        result_json = (
            None
            if result is None
            else json.dumps(dict(result), ensure_ascii=False, allow_nan=False)
        )
        encoded = json.dumps(dict(event_payload), ensure_ascii=False, allow_nan=False)

        async def _operation(conn: Any) -> int | None:
            sequence = await conn.fetchval(
                _FINISH_RUN,
                owner,
                run_uuid,
                worker_id,
                fencing_epoch,
                status,
                stop_reason,
                result_json,
                error_kind,
                error_message,
                event_type,
                encoded,
                require_uncancelled,
            )
            return int(sequence) if sequence is not None else None

        sequence = await self._run_write(_operation)
        if sequence is None:
            return TerminalOutcome(committed=False, status=None, event_sequence=None)
        return TerminalOutcome(committed=True, status=status, event_sequence=sequence)

    async def _resolve_checkpoint(
        self,
        conn: Any,
        *,
        owner_id: str,
        run_uuid: uuid.UUID,
        worker_id: str,
        fencing_epoch: int,
        expected: int,
        envelope: str,
    ) -> CheckpointCommit:
        state = await conn.fetchrow(_LOCK_RUN_STATE, owner_id, run_uuid)
        if state is None:
            return CheckpointCommit(outcome="lease_lost", completed_turns=expected)
        turns = int(state["completed_turns"])
        checkpoint_turn = state["checkpoint_turn"]
        if turns == expected + 1 and checkpoint_turn == str(expected + 1):
            return CheckpointCommit(outcome="committed", completed_turns=turns)
        owns_lease = (
            state["lease_owner"] == worker_id
            and int(state["fencing_epoch"]) == fencing_epoch
            and bool(state["lease_live"])
            and state["status"] == "running"
        )
        if not owns_lease:
            return CheckpointCommit(outcome="lease_lost", completed_turns=turns)
        if turns != expected:
            return CheckpointCommit(outcome="corrupt", completed_turns=turns)
        retried = await conn.fetchval(
            _COMMIT_CHECKPOINT,
            owner_id,
            run_uuid,
            worker_id,
            fencing_epoch,
            expected,
            envelope,
            ANSWER_RUN_LEASE_SECONDS,
        )
        if retried is None:
            return CheckpointCommit(outcome="corrupt", completed_turns=turns)
        return CheckpointCommit(outcome="committed", completed_turns=int(retried))

    async def _write_artifacts(
        self,
        conn: Any,
        *,
        owner_id: str,
        run_uuid: uuid.UUID,
        artifacts: Sequence[PendingArtifact],
        references: Sequence[PendingArtifactReference],
    ) -> list[str]:
        """Write blobs and references, returning digests a replay slot released."""
        for artifact in artifacts:
            await conn.execute(
                _INSERT_ARTIFACT,
                owner_id,
                artifact.digest,
                len(artifact.content),
                artifact.content,
            )
        displaced: list[str] = []
        for reference in references:
            locator = json.dumps(dict(reference.transform_locator), ensure_ascii=False)
            if reference.reference_kind == "fetched_resource":
                released = await conn.fetch(
                    _RELEASE_FETCHED_SLOT,
                    owner_id,
                    run_uuid,
                    reference.ordinal,
                    reference.resource_id,
                )
                displaced.extend(str(row["digest"]) for row in released)
                await conn.execute(
                    _INSERT_FETCHED_REFERENCE,
                    owner_id,
                    run_uuid,
                    reference.resource_id,
                    reference.ordinal,
                    reference.digest,
                    reference.filename,
                    reference.mime_type,
                    locator,
                )
                continue
            await conn.execute(
                _INSERT_RUN_ARTIFACT,
                owner_id,
                run_uuid,
                reference.resource_id,
                reference.reference_kind,
                reference.ordinal,
                reference.digest,
                reference.filename,
                reference.mime_type,
                locator,
            )
        return displaced


async def _finalize_batch(
    conn: Any,
    rows: Sequence[Any],
    *,
    status: AnswerRunStatus,
    error_kind: str | None,
    error_message: str | None,
    event_type: AnswerRunEventType,
    event_payload: Mapping[str, Any],
) -> int:
    if not rows:
        return 0
    finalized = await conn.fetchval(
        _FINALIZE_UNLEASED,
        [str(row["owner_id"]) for row in rows],
        [row["run_id"] for row in rows],
        status,
        error_kind,
        error_message,
        event_type,
        json.dumps(dict(event_payload), ensure_ascii=False),
    )
    return int(finalized or 0)


def _digest_pairs(rows: Sequence[Any]) -> tuple[list[str], list[str]]:
    return [str(row["owner_id"]) for row in rows], [str(row["digest"]) for row in rows]


async def _delete_unreferenced(conn: Any, owners: Sequence[str], digests: Sequence[str]) -> int:
    """Delete blobs no run still references, yielding to concurrent adopters.

    Each delete runs inside its own savepoint. A RESTRICT raised by an adoption
    that beat the reference check must not abort the caller's transaction, or the
    run deletion it already performed would silently roll back and retention would
    never advance past a contended batch. One contended blob must not shield the
    rest either, so a failed batch is retried digest by digest; the adopting run
    now owns that blob and releases it when it is itself deleted.
    """
    if not owners:
        return 0
    deleted = await _try_delete_unreferenced(conn, owners, digests)
    if deleted is not None:
        return deleted
    if len(owners) == 1:
        return 0
    survivors = 0
    for owner, digest in zip(owners, digests, strict=True):
        survivors += await _try_delete_unreferenced(conn, [owner], [digest]) or 0
    return survivors


async def _try_delete_unreferenced(
    conn: Any, owners: Sequence[str], digests: Sequence[str]
) -> int | None:
    """Delete one savepointed batch, or ``None`` when an adopter refused it."""
    try:
        async with conn.transaction():
            deleted = await conn.fetchval(_DELETE_UNREFERENCED_ARTIFACTS, owners, digests)
    except (
        asyncpg.exceptions.RestrictViolationError,
        asyncpg.exceptions.ForeignKeyViolationError,
    ):
        return None
    return int(deleted or 0)


def _require_owner(owner_id: str) -> str:
    owner = str(owner_id).strip()
    if not owner:
        raise ValueError("owner_id cannot be empty")
    return owner


def _json_object(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        loaded = json.loads(value)
        return dict(loaded) if isinstance(loaded, dict) else {}
    return dict(value) if isinstance(value, Mapping) else {}


def _json_array(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        loaded = json.loads(value)
        return list(loaded) if isinstance(loaded, list) else []
    return list(value) if isinstance(value, list | tuple) else []


def answer_run_record(row: Any) -> AnswerRunRecord:
    return AnswerRunRecord(
        owner_id=str(row["owner_id"]),
        run_id=str(row["run_id"]),
        idempotency_key=row["idempotency_key"],
        request=_json_object(row["request_json"]),
        status=row["status"],
        phase=row["phase"],
        stop_reason=row["stop_reason"],
        completed_turns=int(row["completed_turns"]),
        cancel_requested_at=row["cancel_requested_at"],
        lease_owner=row["lease_owner"],
        lease_expires_at=row["lease_expires_at"],
        fencing_epoch=int(row["fencing_epoch"]),
        recovery_count=int(row["recovery_count"]),
        next_event_sequence=int(row["next_event_sequence"]),
        events_trimmed_at=row["events_trimmed_at"],
        result=None if row["result_json"] is None else _json_object(row["result_json"]),
        error_kind=row["error_kind"],
        error_message=row["error_message"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
    )


def _checkpoint_record(value: Any) -> RunCheckpoint | None:
    envelope = _json_object(value) if value is not None else {}
    version = envelope.get("version")
    completed_turns = envelope.get("completed_turns")
    state = envelope.get("state")
    if not isinstance(version, int) or not isinstance(completed_turns, int):
        return None
    if not isinstance(state, dict):
        return None
    return RunCheckpoint(version=version, completed_turns=completed_turns, state=state)


def _event_record(row: Any) -> AnswerRunEvent:
    return AnswerRunEvent(
        sequence=int(row["event_sequence"]),
        event_type=row["event_type"],
        payload=_json_object(row["payload"]),
        created_at=row["created_at"],
    )


def _reference_record(row: Any) -> RunArtifactReference:
    return RunArtifactReference(
        resource_id=str(row["resource_id"]),
        reference_kind=row["reference_kind"],
        ordinal=int(row["ordinal"]),
        digest=str(row["digest"]),
        filename=str(row["filename"]),
        mime_type=str(row["mime_type"]),
        transform_locator=_json_object(row["transform_locator"]),
        created_at=row["created_at"],
    )


__all__ = [
    "ANSWER_RUN_MIGRATIONS",
    "ANSWER_RUN_MIGRATION_SCOPE",
    "ANSWER_RUN_SCHEMA_TABLES",
    "PGAnswerRunStore",
    "answer_run_columns",
    "answer_run_record",
]
