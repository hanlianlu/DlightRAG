# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL adapters for the M3 durable Answer run, journal, and blob state.

This module owns every concrete PostgreSQL implementation for the M3 run
lifecycle: the rewritten baseline schema (``answer_runs:0001_answer_runs``),
claim-bound journal/progress store construction, acceptance, events, terminal
transitions, sweeping, retention, and blob-backed artifacts.

There is no checkpoint column, single-row artifact ``content`` table, dual
write, or compatibility decoder anywhere in this adapter (M3-D5).
"""

from __future__ import annotations

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
from dlightrag.adapters.postgres._pool import pg_pool
from dlightrag.adapters.postgres.session_journal import PGJournalStore, PGProgressStore
from dlightrag.adapters.postgres.workspace import PGWorkspaceStore
from dlightrag.runtime.cancellation import RunCancellationListener, cancellation_notify_key
from dlightrag.runtime.contracts import AnswerRunPhase
from dlightrag.runtime.errors import RunSchemaError
from dlightrag.runtime.policy import (
    ANSWER_RUN_LEASE_SECONDS,
    MAX_RECLAIMS_WITHOUT_PROGRESS,
    RUN_ABANDONED_ERROR_KIND,
    RUN_RETENTION_SECONDS,
)
from dlightrag.runtime.records import (
    AnswerRunEvent,
    AnswerRunRecord,
    CancellationOutcome,
    ClaimedRun,
    IdempotencyKeyConflict,
    LeaseRenewal,
    PendingArtifact,
    PendingArtifactReference,
    ReclaimDecision,
    ReclaimState,
    RunArtifactReference,
    RunCreation,
    RunDeletion,
    RunExecutionContext,
    ShutdownOutcome,
    SweepOutcome,
    TerminalOutcome,
    advance_reclaim,
    parse_run_id,
)

ANSWER_RUN_MIGRATION_SCOPE = "answer_runs"

_ABANDONED_ERROR_MESSAGE = "Answer run exceeded its reclaim-without-progress bound."
_BATCH_LIMIT = 200
_EVENT_PAGE_LIMIT = 500

# ─────────────────────────────────────────────────────────────────
# Final M3 baseline schema
# ─────────────────────────────────────────────────────────────────

_CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS dlightrag_answer_runs (
    owner_id            TEXT        NOT NULL,
    run_id              UUID        NOT NULL,
    idempotency_key     TEXT,
    prepared_input_json JSONB,
    request_fingerprint TEXT        NOT NULL,
    status              TEXT        NOT NULL DEFAULT 'queued',
    phase               TEXT,
    stop_reason         TEXT,
    cancel_requested_at TIMESTAMPTZ,
    lease_owner         TEXT,
    lease_expires_at    TIMESTAMPTZ,
    fencing_epoch       BIGINT      NOT NULL DEFAULT 0,
    durable_progress_version       BIGINT  NOT NULL DEFAULT 0,
    last_reclaim_progress_version  BIGINT  NOT NULL DEFAULT 0,
    reclaims_without_progress      INTEGER NOT NULL DEFAULT 0,
    next_event_sequence BIGINT      NOT NULL DEFAULT 1,
    events_trimmed_at   TIMESTAMPTZ,
    result_json         JSONB,
    error_kind          TEXT,
    error_message       TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at          TIMESTAMPTZ,
    finished_at         TIMESTAMPTZ,
    workspace_epoch     BIGINT,
    PRIMARY KEY (owner_id, run_id),
    CONSTRAINT dlightrag_answer_runs_status_check
        CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
    CONSTRAINT dlightrag_answer_runs_phase_check
        CHECK (phase IS NULL OR phase IN ('planning', 'searching', 'researching', 'generating')),
    CONSTRAINT dlightrag_answer_runs_counter_check
        CHECK (fencing_epoch >= 0 AND next_event_sequence >= 1
               AND durable_progress_version >= 0
               AND last_reclaim_progress_version >= 0
               AND reclaims_without_progress >= 0
               AND (workspace_epoch IS NULL OR workspace_epoch >= 1)),
    CONSTRAINT dlightrag_answer_runs_lease_check
        CHECK ((lease_owner IS NULL) = (lease_expires_at IS NULL)),
    CONSTRAINT dlightrag_answer_runs_terminal_check
        CHECK ((status IN ('succeeded', 'failed', 'cancelled')) = (finished_at IS NOT NULL)),
    CONSTRAINT dlightrag_answer_runs_result_check
        CHECK (status <> 'succeeded' OR result_json IS NOT NULL),
    CONSTRAINT dlightrag_answer_runs_error_check
        CHECK ((status = 'failed') = (error_kind IS NOT NULL)),
    CONSTRAINT dlightrag_answer_runs_prepared_input_check
        CHECK ((status IN ('queued', 'running')) = (prepared_input_json IS NOT NULL)),
    CONSTRAINT dlightrag_answer_runs_workspace_epoch_check
        CHECK (workspace_epoch IS NULL OR workspace_epoch >= 1)
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

_CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS dlightrag_agent_sessions (
    owner_id             TEXT        NOT NULL,
    run_id               UUID        NOT NULL,
    session_id           UUID        NOT NULL,
    version              BIGINT      NOT NULL DEFAULT 0,
    last_sequence        BIGINT      NOT NULL DEFAULT 0,
    active_projection_id UUID,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id, run_id, session_id),
    FOREIGN KEY (owner_id, run_id)
        REFERENCES dlightrag_answer_runs (owner_id, run_id) ON DELETE CASCADE,
    CONSTRAINT dlightrag_agent_sessions_version_check CHECK (version >= 0),
    CONSTRAINT dlightrag_agent_sessions_sequence_check CHECK (last_sequence >= 0)
)
"""

# The session's active projection pointer is a deferrable composite foreign key
# (M3-D40): both tables exist before this constraint is added, and the deferred
# check lets the initial session and projection commit in one transaction.
_ADD_SESSION_PROJECTION_FK = """
ALTER TABLE dlightrag_agent_sessions
ADD CONSTRAINT dlightrag_agent_sessions_projection_fkey
FOREIGN KEY (owner_id, run_id, session_id, active_projection_id)
REFERENCES dlightrag_agent_context_projections
    (owner_id, run_id, session_id, projection_id)
DEFERRABLE INITIALLY DEFERRED
"""

_CREATE_ENTRIES = """
CREATE TABLE IF NOT EXISTS dlightrag_agent_session_entries (
    owner_id       TEXT        NOT NULL,
    run_id         UUID        NOT NULL,
    session_id     UUID        NOT NULL,
    sequence       BIGINT      NOT NULL,
    entry_id       UUID        NOT NULL,
    entry_type     TEXT        NOT NULL,
    schema_version INTEGER     NOT NULL,
    timestamp      TIMESTAMPTZ NOT NULL,
    payload_json   JSONB       NOT NULL,
    PRIMARY KEY (owner_id, run_id, session_id, sequence),
    UNIQUE (entry_id),
    FOREIGN KEY (owner_id, run_id, session_id)
        REFERENCES dlightrag_agent_sessions (owner_id, run_id, session_id) ON DELETE CASCADE,
    CONSTRAINT dlightrag_agent_session_entries_sequence_check CHECK (sequence >= 1),
    CONSTRAINT dlightrag_agent_session_entries_type_check CHECK (entry_type IN (
        'user_message', 'assistant_message', 'effect_intent', 'effect_result',
        'context_injection', 'compaction', 'profile_fact', 'session_terminal'
    )),
    CONSTRAINT dlightrag_agent_session_entries_version_check CHECK (schema_version >= 1)
)
"""

_CREATE_PROJECTIONS = """
CREATE TABLE IF NOT EXISTS dlightrag_agent_context_projections (
    owner_id                  TEXT        NOT NULL,
    run_id                    UUID        NOT NULL,
    session_id                UUID        NOT NULL,
    projection_id             UUID        NOT NULL,
    first_retained_sequence   BIGINT      NOT NULL,
    covered_through_sequence  BIGINT      NOT NULL,
    summary                   TEXT,
    token_anchors             JSONB       NOT NULL DEFAULT '[]'::jsonb,
    schema_version            INTEGER     NOT NULL,
    created_at                TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id, run_id, session_id, projection_id),
    FOREIGN KEY (owner_id, run_id, session_id)
        REFERENCES dlightrag_agent_sessions (owner_id, run_id, session_id) ON DELETE CASCADE,
    CONSTRAINT dlightrag_agent_context_projections_range_check
        CHECK (covered_through_sequence >= 0
               AND first_retained_sequence > covered_through_sequence),
    CONSTRAINT dlightrag_agent_context_projections_summary_check
        CHECK ((covered_through_sequence = 0) = (summary IS NULL)),
    CONSTRAINT dlightrag_agent_context_projections_version_check CHECK (schema_version >= 1)
)
"""

_CREATE_EFFECTS = """
CREATE TABLE IF NOT EXISTS dlightrag_agent_effects (
    owner_id            TEXT        NOT NULL,
    run_id              UUID        NOT NULL,
    session_id          UUID        NOT NULL,
    intent_id           UUID        NOT NULL,
    tool_name           TEXT        NOT NULL,
    replay_policy       TEXT        NOT NULL,
    contract_version    INTEGER     NOT NULL,
    input_schema_digest TEXT        NOT NULL,
    canonical_input     JSONB       NOT NULL,
    source_call_id      TEXT,
    outcome             TEXT,
    result_entry_sequence BIGINT,
    result_digest       TEXT,
    host_update_digest  TEXT,
    settled_at          TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id, run_id, session_id, intent_id),
    FOREIGN KEY (owner_id, run_id, session_id)
        REFERENCES dlightrag_agent_sessions (owner_id, run_id, session_id) ON DELETE CASCADE,
    CONSTRAINT dlightrag_agent_effects_policy_check
        CHECK (replay_policy IN ('safe', 'never')),
    CONSTRAINT dlightrag_agent_effects_contract_check
        CHECK (contract_version >= 1),
    CONSTRAINT dlightrag_agent_effects_digest_check
        CHECK (input_schema_digest ~ '^[0-9a-f]{64}$'),
    CONSTRAINT dlightrag_agent_effects_settlement_check
        CHECK ((settled_at IS NULL) = (outcome IS NULL)
               AND (settled_at IS NULL) = (result_digest IS NULL)),
    CONSTRAINT dlightrag_agent_effects_outcome_check
        CHECK (outcome IS NULL OR outcome IN
               ('succeeded', 'interrupted', 'tool_contract_changed'))
)
"""

_CREATE_STAGES = """
CREATE TABLE IF NOT EXISTS dlightrag_answer_run_stages (
    owner_id          TEXT        NOT NULL,
    run_id            UUID        NOT NULL,
    stage_intent_id   UUID        NOT NULL,
    stage_name        TEXT        NOT NULL,
    progress_version  BIGINT      NOT NULL,
    state             JSONB       NOT NULL,
    state_digest      TEXT        NOT NULL,
    settled_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id, run_id, stage_intent_id),
    FOREIGN KEY (owner_id, run_id)
        REFERENCES dlightrag_answer_runs (owner_id, run_id) ON DELETE CASCADE,
    CONSTRAINT dlightrag_answer_run_stages_name_check
        CHECK (stage_name IN ('planner', 'retrieval', 'final_generation')),
    CONSTRAINT dlightrag_answer_run_stages_digest_check
        CHECK (state_digest ~ '^[0-9a-f]{64}$')
)
"""

_CREATE_EVIDENCE = """
CREATE TABLE IF NOT EXISTS dlightrag_answer_evidence (
    owner_id        TEXT        NOT NULL,
    run_id          UUID        NOT NULL,
    session_id      UUID        NOT NULL,
    intent_id       UUID        NOT NULL,
    result_ordinal  INTEGER     NOT NULL,
    content_digest  TEXT        NOT NULL,
    locator_digest  TEXT        NOT NULL,
    content         BYTEA       NOT NULL,
    locator         BYTEA       NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id, run_id, session_id, intent_id, result_ordinal),
    FOREIGN KEY (owner_id, run_id)
        REFERENCES dlightrag_answer_runs (owner_id, run_id) ON DELETE CASCADE,
    CONSTRAINT dlightrag_answer_evidence_ordinal_check CHECK (result_ordinal >= 0),
    CONSTRAINT dlightrag_answer_evidence_digest_check
        CHECK (content_digest ~ '^[0-9a-f]{64}$' AND locator_digest ~ '^[0-9a-f]{64}$')
)
"""

_CREATE_RESOURCES = """
CREATE TABLE IF NOT EXISTS dlightrag_answer_resources (
    owner_id       TEXT        NOT NULL,
    run_id         UUID        NOT NULL,
    resource_id    TEXT        NOT NULL,
    kind           TEXT        NOT NULL,
    safe_name      TEXT        NOT NULL,
    media_type     TEXT        NOT NULL,
    capabilities   JSONB       NOT NULL DEFAULT '{}'::jsonb,
    ordinal        INTEGER,
    blob_digest    TEXT,
    locator_digest TEXT,
    session_id     UUID,
    intent_id      UUID,
    result_ordinal INTEGER,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id, run_id, resource_id),
    FOREIGN KEY (owner_id, run_id)
        REFERENCES dlightrag_answer_runs (owner_id, run_id) ON DELETE CASCADE,
    CONSTRAINT dlightrag_answer_resources_kind_check
        CHECK (kind IN ('accepted_blob', 'evidence', 'fetched_blob', 'committed_spill')),
    CONSTRAINT dlightrag_answer_resources_blob_link_check
        CHECK ((kind = 'accepted_blob' AND blob_digest IS NOT NULL)
               OR (kind = 'fetched_blob'
                   AND blob_digest IS NOT NULL AND locator_digest IS NOT NULL)
               OR (kind = 'evidence' AND locator_digest IS NOT NULL)
               OR (kind = 'committed_spill'
                   AND blob_digest IS NULL AND locator_digest IS NULL))
)
"""

_CREATE_BLOBS = """
CREATE TABLE IF NOT EXISTS dlightrag_blobs (
    owner_id   TEXT        NOT NULL,
    digest     TEXT        NOT NULL,
    byte_size  BIGINT      NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id, digest),
    CONSTRAINT dlightrag_blobs_digest_check CHECK (digest ~ '^[0-9a-f]{64}$'),
    CONSTRAINT dlightrag_blobs_size_check CHECK (byte_size >= 0)
)
"""

_CREATE_BLOB_CHUNKS = """
CREATE TABLE IF NOT EXISTS dlightrag_blob_chunks (
    owner_id    TEXT    NOT NULL,
    digest      TEXT    NOT NULL,
    chunk_index INTEGER NOT NULL,
    content     BYTEA   NOT NULL,
    PRIMARY KEY (owner_id, digest, chunk_index),
    FOREIGN KEY (owner_id, digest)
        REFERENCES dlightrag_blobs (owner_id, digest) ON DELETE CASCADE,
    CONSTRAINT dlightrag_blob_chunks_index_check CHECK (chunk_index >= 0)
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
        REFERENCES dlightrag_blobs (owner_id, digest) ON DELETE RESTRICT,
    CONSTRAINT dlightrag_answer_run_artifacts_kind_check
        CHECK (reference_kind IN
               ('current_attachment', 'history_attachment', 'fetched_resource',
                'primary_report', 'published_artifact')),
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
    "CREATE INDEX IF NOT EXISTS idx_dlightrag_answer_evidence_run "
    "ON dlightrag_answer_evidence (owner_id, run_id)",
    "CREATE INDEX IF NOT EXISTS idx_dlightrag_answer_resources_run "
    "ON dlightrag_answer_resources (owner_id, run_id)",
    "CREATE INDEX IF NOT EXISTS idx_dlightrag_answer_resources_blob "
    "ON dlightrag_answer_resources (owner_id, blob_digest) "
    "WHERE blob_digest IS NOT NULL",
)

_M4_WORKSPACE_DDL = (
    "ALTER TABLE dlightrag_answer_runs ADD COLUMN IF NOT EXISTS workspace_epoch BIGINT",
    "ALTER TABLE dlightrag_answer_runs "
    "DROP CONSTRAINT IF EXISTS dlightrag_answer_runs_workspace_epoch_check",
    "ALTER TABLE dlightrag_answer_runs "
    "ADD CONSTRAINT dlightrag_answer_runs_workspace_epoch_check "
    "CHECK (workspace_epoch IS NULL OR workspace_epoch >= 1)",
    """
CREATE TABLE IF NOT EXISTS dlightrag_answer_workspace_inventory (
    owner_id        TEXT        NOT NULL,
    run_id          UUID        NOT NULL,
    relative_path   TEXT        NOT NULL,
    entry_type      TEXT        NOT NULL,
    mode            INTEGER,
    size_bytes      BIGINT      NOT NULL,
    content_digest  TEXT,
    PRIMARY KEY (owner_id, run_id, relative_path),
    FOREIGN KEY (owner_id, run_id)
        REFERENCES dlightrag_answer_runs (owner_id, run_id) ON DELETE CASCADE
)
""",
    """
CREATE TABLE IF NOT EXISTS dlightrag_answer_committed_spills (
    owner_id        TEXT        NOT NULL,
    run_id          UUID        NOT NULL,
    resource_id     TEXT        NOT NULL,
    content_digest  TEXT        NOT NULL,
    size_bytes      BIGINT      NOT NULL,
    session_id      UUID        NOT NULL,
    intent_id       UUID        NOT NULL,
    PRIMARY KEY (owner_id, run_id, resource_id),
    FOREIGN KEY (owner_id, run_id)
        REFERENCES dlightrag_answer_runs (owner_id, run_id) ON DELETE CASCADE
)
""",
    "ALTER TABLE dlightrag_answer_resources "
    "DROP CONSTRAINT IF EXISTS dlightrag_answer_resources_kind_check",
    "ALTER TABLE dlightrag_answer_resources "
    "ADD CONSTRAINT dlightrag_answer_resources_kind_check "
    "CHECK (kind IN ('accepted_blob', 'evidence', 'fetched_blob', 'committed_spill'))",
    "ALTER TABLE dlightrag_answer_resources "
    "DROP CONSTRAINT IF EXISTS dlightrag_answer_resources_blob_link_check",
    "ALTER TABLE dlightrag_answer_resources "
    "ADD CONSTRAINT dlightrag_answer_resources_blob_link_check "
    "CHECK ((kind = 'accepted_blob' AND blob_digest IS NOT NULL) "
    "OR (kind = 'fetched_blob' AND blob_digest IS NOT NULL AND locator_digest IS NOT NULL) "
    "OR (kind = 'evidence' AND locator_digest IS NOT NULL) "
    "OR (kind = 'committed_spill' AND blob_digest IS NULL AND locator_digest IS NULL))",
)

_M5_PUBLICATION_DDL = (
    "ALTER TABLE dlightrag_answer_run_artifacts "
    "DROP CONSTRAINT IF EXISTS dlightrag_answer_run_artifacts_kind_check",
    "ALTER TABLE dlightrag_answer_run_artifacts "
    "ADD CONSTRAINT dlightrag_answer_run_artifacts_kind_check "
    "CHECK (reference_kind IN ("
    "'current_attachment', 'history_attachment', 'fetched_resource', "
    "'primary_report', 'published_artifact'))",
)

ANSWER_RUN_MIGRATIONS = (
    Migration(
        "0001_answer_runs",
        "Create the final M3 Answer run, journal, evidence, resource, and blob state",
        (
            _CREATE_RUNS,
            _CREATE_EVENTS,
            _CREATE_SESSIONS,
            _CREATE_ENTRIES,
            _CREATE_PROJECTIONS,
            _CREATE_EFFECTS,
            _CREATE_STAGES,
            _CREATE_EVIDENCE,
            _CREATE_RESOURCES,
            _CREATE_BLOBS,
            _CREATE_BLOB_CHUNKS,
            _CREATE_RUN_ARTIFACTS,
            _ADD_SESSION_PROJECTION_FK,
            *_CREATE_INDEXES,
            _M4_WORKSPACE_DDL[3],
            _M4_WORKSPACE_DDL[4],
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
            "prepared_input_json",
            "request_fingerprint",
            "status",
            "phase",
            "stop_reason",
            "cancel_requested_at",
            "lease_owner",
            "lease_expires_at",
            "fencing_epoch",
            "durable_progress_version",
            "last_reclaim_progress_version",
            "reclaims_without_progress",
            "next_event_sequence",
            "events_trimmed_at",
            "result_json",
            "error_kind",
            "error_message",
            "created_at",
            "updated_at",
            "started_at",
            "finished_at",
            "workspace_epoch",
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
            "dlightrag_answer_runs_prepared_input_check",
            "dlightrag_answer_runs_workspace_epoch_check",
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
        name="dlightrag_agent_sessions",
        columns=(
            "owner_id",
            "run_id",
            "session_id",
            "version",
            "last_sequence",
            "active_projection_id",
            "created_at",
            "updated_at",
        ),
        primary_key=("owner_id", "run_id", "session_id"),
        foreign_keys=(
            ForeignKeyRequirement(
                columns=("owner_id", "run_id"), references="dlightrag_answer_runs"
            ),
        ),
        checks=(
            "dlightrag_agent_sessions_version_check",
            "dlightrag_agent_sessions_sequence_check",
        ),
    ),
    TableRequirement(
        name="dlightrag_agent_session_entries",
        columns=(
            "owner_id",
            "run_id",
            "session_id",
            "sequence",
            "entry_id",
            "entry_type",
            "schema_version",
            "timestamp",
            "payload_json",
        ),
        primary_key=("owner_id", "run_id", "session_id", "sequence"),
        unique=(("entry_id",),),
        foreign_keys=(
            ForeignKeyRequirement(
                columns=("owner_id", "run_id", "session_id"),
                references="dlightrag_agent_sessions",
            ),
        ),
        checks=(
            "dlightrag_agent_session_entries_sequence_check",
            "dlightrag_agent_session_entries_type_check",
            "dlightrag_agent_session_entries_version_check",
        ),
    ),
    TableRequirement(
        name="dlightrag_agent_context_projections",
        columns=(
            "owner_id",
            "run_id",
            "session_id",
            "projection_id",
            "first_retained_sequence",
            "covered_through_sequence",
            "summary",
            "token_anchors",
            "schema_version",
            "created_at",
        ),
        primary_key=("owner_id", "run_id", "session_id", "projection_id"),
        foreign_keys=(
            ForeignKeyRequirement(
                columns=("owner_id", "run_id", "session_id"),
                references="dlightrag_agent_sessions",
            ),
        ),
        checks=(
            "dlightrag_agent_context_projections_range_check",
            "dlightrag_agent_context_projections_summary_check",
            "dlightrag_agent_context_projections_version_check",
        ),
    ),
    TableRequirement(
        name="dlightrag_agent_effects",
        columns=(
            "owner_id",
            "run_id",
            "session_id",
            "intent_id",
            "tool_name",
            "replay_policy",
            "contract_version",
            "input_schema_digest",
            "canonical_input",
            "source_call_id",
            "outcome",
            "result_entry_sequence",
            "result_digest",
            "host_update_digest",
            "settled_at",
            "created_at",
        ),
        primary_key=("owner_id", "run_id", "session_id", "intent_id"),
        foreign_keys=(
            ForeignKeyRequirement(
                columns=("owner_id", "run_id", "session_id"),
                references="dlightrag_agent_sessions",
            ),
        ),
        checks=(
            "dlightrag_agent_effects_policy_check",
            "dlightrag_agent_effects_contract_check",
            "dlightrag_agent_effects_digest_check",
            "dlightrag_agent_effects_settlement_check",
            "dlightrag_agent_effects_outcome_check",
        ),
    ),
    TableRequirement(
        name="dlightrag_answer_run_stages",
        columns=(
            "owner_id",
            "run_id",
            "stage_intent_id",
            "stage_name",
            "progress_version",
            "state",
            "state_digest",
            "settled_at",
        ),
        primary_key=("owner_id", "run_id", "stage_intent_id"),
        foreign_keys=(
            ForeignKeyRequirement(
                columns=("owner_id", "run_id"), references="dlightrag_answer_runs"
            ),
        ),
        checks=(
            "dlightrag_answer_run_stages_name_check",
            "dlightrag_answer_run_stages_digest_check",
        ),
    ),
    TableRequirement(
        name="dlightrag_answer_evidence",
        columns=(
            "owner_id",
            "run_id",
            "session_id",
            "intent_id",
            "result_ordinal",
            "content_digest",
            "locator_digest",
            "content",
            "locator",
            "created_at",
        ),
        primary_key=("owner_id", "run_id", "session_id", "intent_id", "result_ordinal"),
        foreign_keys=(
            ForeignKeyRequirement(
                columns=("owner_id", "run_id"), references="dlightrag_answer_runs"
            ),
        ),
        checks=(
            "dlightrag_answer_evidence_ordinal_check",
            "dlightrag_answer_evidence_digest_check",
        ),
        indexes=("idx_dlightrag_answer_evidence_run",),
    ),
    TableRequirement(
        name="dlightrag_answer_resources",
        columns=(
            "owner_id",
            "run_id",
            "resource_id",
            "kind",
            "safe_name",
            "media_type",
            "capabilities",
            "ordinal",
            "blob_digest",
            "locator_digest",
            "session_id",
            "intent_id",
            "result_ordinal",
            "created_at",
        ),
        primary_key=("owner_id", "run_id", "resource_id"),
        foreign_keys=(
            ForeignKeyRequirement(
                columns=("owner_id", "run_id"), references="dlightrag_answer_runs"
            ),
        ),
        checks=(
            "dlightrag_answer_resources_kind_check",
            "dlightrag_answer_resources_blob_link_check",
        ),
        indexes=(
            "idx_dlightrag_answer_resources_run",
            "idx_dlightrag_answer_resources_blob",
        ),
    ),
    TableRequirement(
        name="dlightrag_blobs",
        columns=("owner_id", "digest", "byte_size", "created_at"),
        primary_key=("owner_id", "digest"),
        checks=(
            "dlightrag_blobs_digest_check",
            "dlightrag_blobs_size_check",
        ),
    ),
    TableRequirement(
        name="dlightrag_blob_chunks",
        columns=("owner_id", "digest", "chunk_index", "content"),
        primary_key=("owner_id", "digest", "chunk_index"),
        foreign_keys=(
            ForeignKeyRequirement(columns=("owner_id", "digest"), references="dlightrag_blobs"),
        ),
        checks=("dlightrag_blob_chunks_index_check",),
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
            ForeignKeyRequirement(columns=("owner_id", "digest"), references="dlightrag_blobs"),
        ),
        checks=(
            "dlightrag_answer_run_artifacts_kind_check",
            "dlightrag_answer_run_artifacts_ordinal_check",
        ),
        indexes=("idx_dlightrag_answer_run_artifacts_digest",),
    ),
    TableRequirement(
        name="dlightrag_answer_workspace_inventory",
        columns=(
            "owner_id",
            "run_id",
            "relative_path",
            "entry_type",
            "mode",
            "size_bytes",
            "content_digest",
        ),
        primary_key=("owner_id", "run_id", "relative_path"),
        foreign_keys=(
            ForeignKeyRequirement(
                columns=("owner_id", "run_id"), references="dlightrag_answer_runs"
            ),
        ),
    ),
    TableRequirement(
        name="dlightrag_answer_committed_spills",
        columns=(
            "owner_id",
            "run_id",
            "resource_id",
            "content_digest",
            "size_bytes",
            "session_id",
            "intent_id",
        ),
        primary_key=("owner_id", "run_id", "resource_id"),
        foreign_keys=(
            ForeignKeyRequirement(
                columns=("owner_id", "run_id"), references="dlightrag_answer_runs"
            ),
        ),
    ),
)

#: ``(expression, output name)`` for every column :func:`answer_run_record` reads.
_RUN_COLUMN_SPECS: tuple[tuple[str, str], ...] = (
    ("owner_id", "owner_id"),
    ("run_id::text", "run_id"),
    ("idempotency_key", "idempotency_key"),
    ("prepared_input_json", "prepared_input"),
    ("status", "status"),
    ("phase", "phase"),
    ("stop_reason", "stop_reason"),
    ("cancel_requested_at", "cancel_requested_at"),
    ("lease_owner", "lease_owner"),
    ("lease_expires_at", "lease_expires_at"),
    ("fencing_epoch", "fencing_epoch"),
    ("durable_progress_version", "durable_progress_version"),
    ("last_reclaim_progress_version", "last_reclaim_progress_version"),
    ("reclaims_without_progress", "reclaims_without_progress"),
    ("next_event_sequence", "next_event_sequence"),
    ("events_trimmed_at", "events_trimmed_at"),
    ("result_json", "result_json"),
    ("error_kind", "error_kind"),
    ("error_message", "error_message"),
    ("created_at", "created_at"),
    ("updated_at", "updated_at"),
    ("started_at", "started_at"),
    ("finished_at", "finished_at"),
    ("workspace_epoch", "workspace_epoch"),
)


def answer_run_columns(alias: str = "") -> str:
    """Project one run row's columns, optionally through a join alias."""
    prefix = f"{alias}." if alias else ""
    return ",\n".join(f"{prefix}{expression} AS {name}" for expression, name in _RUN_COLUMN_SPECS)


_RUN_COLUMNS = answer_run_columns()
_LIST_RUNS = f"""
SELECT {_RUN_COLUMNS}
FROM dlightrag_answer_runs
WHERE owner_id = $1 ORDER BY created_at, run_id LIMIT $2
"""  # noqa: S608 - interpolates only the trusted _RUN_COLUMNS constant
_LIST_RUNS_AFTER = f"""
SELECT {_RUN_COLUMNS}
FROM dlightrag_answer_runs
WHERE owner_id = $1 AND (created_at, run_id) > (
 SELECT created_at, run_id FROM dlightrag_answer_runs
 WHERE owner_id = $1 AND run_id = $2)
ORDER BY created_at, run_id LIMIT $3
"""  # noqa: S608 - interpolates only the trusted _RUN_COLUMNS constant

_INSERT_RUN = f"""
INSERT INTO dlightrag_answer_runs (
    owner_id, run_id, idempotency_key, prepared_input_json, request_fingerprint
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

_SELECT_RUN = f"""
SELECT {_RUN_COLUMNS}
FROM dlightrag_answer_runs
WHERE owner_id = $1 AND run_id = $2
"""  # noqa: S608 - interpolates only the trusted _RUN_COLUMNS constant

_SELECT_EVENTS = """
SELECT event_sequence, event_type, payload, created_at
FROM dlightrag_answer_run_events
WHERE owner_id = $1 AND run_id = $2 AND event_sequence > $3
ORDER BY event_sequence
LIMIT $4
"""

_SELECT_CLAIM_CANDIDATE = """
SELECT owner_id, run_id
FROM dlightrag_answer_runs
WHERE cancel_requested_at IS NULL
  AND (
      status = 'queued'
      OR (status = 'running' AND lease_expires_at < NOW()
          AND reclaims_without_progress < $1)
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
    reclaims_without_progress = $5,
    last_reclaim_progress_version = $6,
    started_at = COALESCE(started_at, NOW()),
    updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2
  AND status IN ('queued', 'running')
RETURNING {_RUN_COLUMNS}
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

# One fenced terminal transition that also appends the run's single terminal
# event. Prepared input is cleared on every terminal transition (M3-D5).
_FINISH_RUN = """
WITH bumped AS (
    UPDATE dlightrag_answer_runs
    SET status = $5::text,
        stop_reason = $6::text,
        result_json = $7::jsonb,
        error_kind = $8::text,
        error_message = $9::text,
        phase = NULL,
        prepared_input_json = NULL,
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
        prepared_input_json = NULL,
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
WITH updated AS (
    UPDATE dlightrag_answer_runs
    SET cancel_requested_at = COALESCE(cancel_requested_at, NOW()),
        updated_at = NOW()
    WHERE owner_id = $1 AND run_id = $2
      AND status = 'running'
    RETURNING 1
), notified AS (
    SELECT pg_notify('dlightrag_answer_run_cancel', $3) FROM updated
)
SELECT count(*)::int FROM notified
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

_INSERT_BLOB_METADATA = """
INSERT INTO dlightrag_blobs (owner_id, digest, byte_size)
VALUES ($1, $2, $3)
ON CONFLICT (owner_id, digest) DO NOTHING
"""

_INSERT_BLOB_CHUNK = """
INSERT INTO dlightrag_blob_chunks (owner_id, digest, chunk_index, content)
VALUES ($1, $2, $3, $4)
ON CONFLICT (owner_id, digest, chunk_index) DO NOTHING
"""

_INSERT_RESOURCE = """
INSERT INTO dlightrag_answer_resources (
    owner_id, run_id, resource_id, kind, safe_name, media_type, capabilities,
    ordinal, blob_digest, locator_digest, session_id, intent_id, result_ordinal
)
VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10, $11, $12, $13)
ON CONFLICT (owner_id, run_id, resource_id) DO NOTHING
"""

_INSERT_RUN_ARTIFACT = """
INSERT INTO dlightrag_answer_run_artifacts (
    owner_id, run_id, resource_id, reference_kind, ordinal, digest,
    filename, mime_type, transform_locator
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb)
ON CONFLICT (owner_id, run_id, resource_id) DO NOTHING
"""

_SELECT_BLOB_CHUNKS = """
SELECT content FROM dlightrag_blob_chunks
WHERE owner_id = $1 AND digest = $2
ORDER BY chunk_index
"""

_SELECT_BLOB_SIZE = """
SELECT byte_size FROM dlightrag_blobs WHERE owner_id = $1 AND digest = $2
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

# Retention order: references first, then blob chunks/metadata only when no
# run/resource reference remains (M3 blob store contract). Resources cascade
# with their run; orphan reference rows are gone with the run row too.
_DELETE_UNREFERENCED_BLOBS = """
WITH referenced AS (
    SELECT DISTINCT owner_id, digest FROM dlightrag_answer_run_artifacts
    UNION
    SELECT DISTINCT owner_id, blob_digest AS digest FROM dlightrag_answer_resources
        WHERE blob_digest IS NOT NULL
), candidates AS (
    SELECT b.owner_id, b.digest
    FROM dlightrag_blobs AS b
    WHERE (b.owner_id, b.digest) IN (SELECT * FROM unnest($1::text[], $2::text[]))
      AND NOT EXISTS (
          SELECT 1 FROM referenced AS r
          WHERE r.owner_id = b.owner_id AND r.digest = b.digest
      )
    FOR UPDATE SKIP LOCKED
), deleted AS (
    DELETE FROM dlightrag_blobs AS b
    USING candidates AS c
    WHERE b.owner_id = c.owner_id AND b.digest = c.digest
    RETURNING 1
)
SELECT count(*)::int FROM deleted
"""

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


def _digest_pairs(rows: Sequence[Any]) -> tuple[list[str], list[str]]:
    """Split reference rows into parallel owner and digest lists."""
    owners: list[str] = []
    digests: list[str] = []
    for row in rows:
        owners.append(str(row["owner_id"]))
        digests.append(str(row["digest"]))
    return owners, digests


async def _try_delete_unreferenced(
    conn: Any, owners: Sequence[str], digests: Sequence[str]
) -> int | None:
    """Delete one savepointed blob batch, or None when an adopter refused it."""
    try:
        async with conn.transaction():
            deleted = await conn.fetchval(_DELETE_UNREFERENCED_BLOBS, owners, digests)
    except asyncpg.RestrictViolationError:
        return None
    except asyncpg.PostgresError:
        raise
    return int(deleted or 0)


async def _delete_unreferenced(conn: Any, owners: Sequence[str], digests: Sequence[str]) -> int:
    """Delete blobs no run or resource still references, yielding to adopters.

    Each delete runs inside its own savepoint. A RESTRICT raised by an adoption
    that beat the reference check must not abort the caller's transaction, or the
    run deletion it already performed would silently roll back and retention would
    never advance past a contended batch. One contended blob must not shield the
    rest either, so a failed batch is retried digest by digest.
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


def _new_run_id() -> uuid.UUID:
    """Return a fresh time-ordered UUIDv7 run identifier."""
    return uuid.uuid7()


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


def _optional_int(row: Any, name: str) -> int | None:
    try:
        value = row[name]
    except KeyError, TypeError:
        return None
    return int(value) if value is not None else None


def answer_run_record(row: Any) -> AnswerRunRecord:
    """Project one stored run row into the storage-neutral M3 record."""
    prepared = row["prepared_input"]
    return AnswerRunRecord(
        owner_id=str(row["owner_id"]),
        run_id=str(row["run_id"]),
        idempotency_key=row["idempotency_key"],
        prepared_input=_json_object(prepared) if prepared is not None else None,
        status=row["status"],
        phase=row["phase"],
        stop_reason=row["stop_reason"],
        cancel_requested_at=row["cancel_requested_at"],
        lease_owner=row["lease_owner"],
        lease_expires_at=row["lease_expires_at"],
        fencing_epoch=int(row["fencing_epoch"]),
        durable_progress_version=int(row["durable_progress_version"]),
        last_reclaim_progress_version=int(row["last_reclaim_progress_version"]),
        reclaims_without_progress=int(row["reclaims_without_progress"]),
        next_event_sequence=int(row["next_event_sequence"]),
        events_trimmed_at=row["events_trimmed_at"],
        result=_json_object(row["result_json"]) if row["result_json"] is not None else None,
        error_kind=row["error_kind"],
        error_message=row["error_message"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        workspace_epoch=_optional_int(row, "workspace_epoch"),
    )


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
        """Create the durable M3 Answer run schema, or validate it (reader)."""
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
            for statement in _M4_WORKSPACE_DDL:
                await conn.execute(statement)
            for statement in _M5_PUBLICATION_DDL:
                await conn.execute(statement)

        await self._run(_operation)
        self._initialized = True

    # -- acceptance ---------------------------------------------------
    async def replay_run(
        self, *, owner_id: str, idempotency_key: str, idempotency_fingerprint: str
    ) -> RunCreation | None:
        """Replay a matching idempotency key before any preparation happens."""
        owner = _require_owner(owner_id)

        async def _operation(conn: Any) -> RunCreation | None:
            row = await conn.fetchrow(_SELECT_RUN_BY_KEY, owner, idempotency_key)
            if row is None:
                return None
            return _require_replay_match(
                owner=owner,
                key=idempotency_key,
                stored_fingerprint=str(row["request_fingerprint"]),
                fingerprint=idempotency_fingerprint,
                row=row,
            )

        return await self._run_read(_operation)

    async def create_run(
        self,
        *,
        owner_id: str,
        prepared_input: Mapping[str, object],
        idempotency_fingerprint: str,
        idempotency_key: str | None = None,
        resources: Sequence[Mapping[str, object]] = (),
        artifacts: Sequence[PendingArtifact] = (),
        references: Sequence[PendingArtifactReference] = (),
    ) -> RunCreation:
        """Accept one M3 run with its bounded prepared input (3E)."""
        return await self.accept_run(
            owner_id=owner_id,
            run_id=str(_new_run_id()),
            idempotency_key=idempotency_key,
            fingerprint=idempotency_fingerprint,
            prepared_input=prepared_input,
            resources=resources,
            blobs=artifacts,
            references=references,
        )

    async def accept_run(
        self,
        *,
        owner_id: str,
        run_id: str,
        idempotency_key: str | None,
        fingerprint: str,
        prepared_input: Mapping[str, object],
        resources: Sequence[Mapping[str, object]] = (),
        blobs: Sequence[PendingArtifact] = (),
        references: Sequence[PendingArtifactReference] = (),
    ) -> RunCreation:
        """Atomically accept one run: blobs, resources, references, run row.

        The public request fingerprint is computed before enrichment and is
        compared against any idempotent replay; a mismatch is an
        :class:`IdempotencyKeyConflict` (M3 acceptance ordering).
        """
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            raise ValueError("run_id must be a canonical UUID")
        prepared_json = json.dumps(dict(prepared_input), ensure_ascii=False, sort_keys=True)

        async def _operation(conn: Any) -> RunCreation:
            async with conn.transaction():
                if idempotency_key:
                    replayed = await conn.fetchrow(_SELECT_RUN_BY_KEY, owner, idempotency_key)
                    if replayed is not None:
                        return _require_replay_match(
                            owner=owner,
                            key=idempotency_key,
                            stored_fingerprint=str(replayed["request_fingerprint"]),
                            fingerprint=fingerprint,
                            row=replayed,
                        )
                for blob in blobs:
                    await self._write_blob(conn, owner, blob)
                row = await conn.fetchrow(
                    _INSERT_RUN,
                    owner,
                    run_uuid,
                    idempotency_key,
                    prepared_json,
                    fingerprint,
                )
                if row is None:
                    replayed = await conn.fetchrow(_SELECT_RUN_BY_KEY, owner, idempotency_key)
                    return RunCreation(run=answer_run_record(replayed), replayed=True)
                for resource in resources:
                    await conn.execute(
                        _INSERT_RESOURCE,
                        owner,
                        run_uuid,
                        str(resource["resource_id"]),
                        "accepted_blob",
                        str(resource["safe_name"]),
                        str(resource["media_type"]),
                        json.dumps(resource.get("capabilities") or {}, ensure_ascii=False),
                        int(str(resource["ordinal"])),
                        str(resource["blob_digest"]),
                        None,
                        None,
                        None,
                        None,
                    )
                for reference in references:
                    await conn.execute(
                        _INSERT_RUN_ARTIFACT,
                        owner,
                        run_uuid,
                        reference.resource_id,
                        reference.reference_kind,
                        reference.ordinal,
                        reference.digest,
                        reference.filename,
                        reference.mime_type,
                        json.dumps(dict(reference.transform_locator), ensure_ascii=False),
                    )
                return RunCreation(run=answer_run_record(row), replayed=False)

        return await self._run_write(_operation)

    async def _write_publications(
        self, conn: Any, owner: str, run_uuid: uuid.UUID, publications: Sequence[Any]
    ) -> None:
        for index, item in enumerate(publications):
            blob = PendingArtifact(content=item.content)
            await self._write_blob(conn, owner, blob)
            await conn.execute(
                _INSERT_RUN_ARTIFACT,
                owner,
                run_uuid,
                item.resource_id,
                item.reference_kind,
                index,
                blob.digest,
                item.filename,
                item.mime_type,
                "{}",
            )

    async def _write_blob(self, conn: Any, owner: str, blob: PendingArtifact) -> None:
        plan = _plan_blob(blob.content)
        await conn.execute(_INSERT_BLOB_METADATA, owner, blob.digest, plan.total_bytes)
        existing = await conn.fetchval(_SELECT_BLOB_SIZE, owner, blob.digest)
        if existing != plan.total_bytes:
            raise ValueError("blob digest collision with a different byte size")
        for index in range(plan.chunk_count):
            await conn.execute(
                _INSERT_BLOB_CHUNK, owner, blob.digest, index, plan.chunk(blob.content, index)
            )

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
        row to the accepted run atomically. It performs no transaction control
        of its own, so the caller's commit is what makes the run and its link
        durable together. ``request`` is the bounded M3 prepared input.
        """
        owner = _require_owner(owner_id)
        if any(reference.reference_kind == "fetched_resource" for reference in references):
            # A fetched resource is worker-fenced run state, never accepted input.
            raise ValueError("fetched_resource references cannot be run creation inputs")
        if not idempotency_fingerprint:
            raise ValueError("idempotency_fingerprint must be non-empty")
        payload = json.dumps(dict(request), ensure_ascii=False, sort_keys=True)
        run_uuid = _new_run_id()
        for blob in artifacts:
            await self._write_blob(conn, owner, blob)
        row = await conn.fetchrow(
            _INSERT_RUN, owner, run_uuid, idempotency_key, payload, idempotency_fingerprint
        )
        if row is None:
            existing = await conn.fetchrow(_SELECT_RUN_BY_KEY, owner, idempotency_key)
            if existing is None:
                raise RuntimeError("answer run insert reported a vanished conflict")
            if str(existing["request_fingerprint"]) != idempotency_fingerprint:
                raise IdempotencyKeyConflict(
                    "idempotency key was reused with different request input"
                )
            return RunCreation(run=answer_run_record(existing), replayed=True)
        for reference in references:
            await conn.execute(
                _INSERT_RUN_ARTIFACT,
                owner,
                run_uuid,
                reference.resource_id,
                reference.reference_kind,
                reference.ordinal,
                reference.digest,
                reference.filename,
                reference.mime_type,
                json.dumps(dict(reference.transform_locator), ensure_ascii=False),
            )
        return RunCreation(run=answer_run_record(row), replayed=False)

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

    async def list_active_run_requirements(self) -> tuple[Mapping[str, Any], ...]:
        """Return pinned profile facts of active runs, read from prepared inputs."""

        async def _operation(conn: Any) -> tuple[Mapping[str, Any], ...]:
            rows = await conn.fetch(
                "SELECT prepared_input_json FROM dlightrag_answer_runs"
                " WHERE status IN ('queued', 'running')"
                " AND cancel_requested_at IS NULL"
                " AND NOT (status = 'running' AND lease_expires_at < NOW()"
                "          AND reclaims_without_progress >= $1)",
                MAX_RECLAIMS_WITHOUT_PROGRESS,
            )
            requirements: list[Mapping[str, Any]] = []
            for row in rows:
                prepared = _json_object(row["prepared_input_json"])
                requirements.append(
                    {
                        "context_policy_revision": prepared.get("context_policy_revision"),
                        "pinned_models": prepared.get("pinned_models"),
                    }
                )
            return tuple(requirements)

        return await self._run_read(_operation)

    async def delete_runs(self, *, owner_id: str, run_ids: Sequence[str]) -> RunDeletion:
        """Delete owned runs and orphaned blobs in one dedicated transaction."""

        async def _operation(conn: Any) -> RunDeletion:
            async with conn.transaction():
                return await self.delete_runs_in(conn, owner_id=owner_id, run_ids=run_ids)

        return await self._run_write(_operation)

    # -- reads --------------------------------------------------------
    async def get_run(self, *, owner_id: str, run_id: str) -> AnswerRunRecord | None:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return None

        async def _operation(conn: Any) -> AnswerRunRecord | None:
            row = await conn.fetchrow(_SELECT_RUN, owner, run_uuid)
            return answer_run_record(row) if row is not None else None

        return await self._run_read(_operation)

    async def list_runs(
        self, *, owner_id: str, after_run_id: str | None = None, limit: int = 50
    ) -> tuple[AnswerRunRecord, ...]:
        owner = _require_owner(owner_id)
        cap = max(1, min(int(limit), 100))
        after = parse_run_id(after_run_id) if after_run_id else None

        async def _operation(conn: Any) -> tuple[AnswerRunRecord, ...]:
            if after is None:
                rows = await conn.fetch(_LIST_RUNS, owner, cap)
            else:
                rows = await conn.fetch(_LIST_RUNS_AFTER, owner, after, cap)
            return tuple(answer_run_record(row) for row in rows)

        return await self._run_read(_operation)

    async def list_run_artifacts(
        self, *, owner_id: str, run_id: str
    ) -> tuple[RunArtifactReference, ...]:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return ()

        async def _operation(conn: Any) -> tuple[RunArtifactReference, ...]:
            rows = await conn.fetch(_SELECT_RUN_ARTIFACTS, owner, run_uuid)
            return tuple(_reference_record(row) for row in rows)

        return await self._run_read(_operation)

    async def read_event_page(
        self, *, owner_id: str, run_id: str, after_sequence: int = 0
    ) -> tuple[AnswerRunEvent, ...]:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return ()

        async def _operation(conn: Any) -> tuple[AnswerRunEvent, ...]:
            rows = await conn.fetch(
                _SELECT_EVENTS,
                owner,
                run_uuid,
                max(0, int(after_sequence)),
                _EVENT_PAGE_LIMIT,
            )
            return tuple(_event_record(row) for row in rows)

        return await self._run_read(_operation)

    async def load_artifact(self, *, owner_id: str, digest: str) -> bytes | None:
        """Reassemble one complete blob from its 1 MiB chunks."""
        owner = _require_owner(owner_id)

        async def _operation(conn: Any) -> bytes | None:
            size = await conn.fetchval(_SELECT_BLOB_SIZE, owner, digest)
            if size is None:
                return None
            chunks = await conn.fetch(_SELECT_BLOB_CHUNKS, owner, digest)
            return b"".join(bytes(row["content"]) for row in chunks)

        return await self._run_read(_operation)

    # -- cancellation -------------------------------------------------
    async def rescan_cancel_pending(self, *, worker_id: str) -> list[tuple[str, str]]:
        """Return locally leased cancel-pending runs for this worker.

        The listener calls this on every connect/reconnect and on every wake
        notification; the payload itself never cancels anything (M3-D19).
        """

        async def _operation(conn: Any) -> list[tuple[str, str]]:
            rows = await conn.fetch(
                "SELECT owner_id, run_id FROM dlightrag_answer_runs"
                " WHERE cancel_requested_at IS NOT NULL"
                " AND status = 'running' AND lease_owner = $1"
                " AND lease_expires_at > NOW()",
                worker_id,
            )
            return [(str(row["owner_id"]), str(row["run_id"])) for row in rows]

        return await self._run_read(_operation)

    def build_cancellation_listener(
        self,
        *,
        worker_id: str,
        on_cancel: Callable[[str, str], Awaitable[None]],
    ) -> RunCancellationListener:
        """Build the dedicated reconnecting LISTEN listener for this store."""

        async def _open_connection() -> Any:
            if self._operation_pool is not None:
                return await self._operation_pool.acquire().__aenter__()
            pool = await pg_pool.get()
            return await pool.acquire().__aenter__()

        async def _rescan() -> list[tuple[str, str]]:
            return await self.rescan_cancel_pending(worker_id=worker_id)

        return RunCancellationListener(
            open_connection=_open_connection,
            rescan=_rescan,
            on_cancel=on_cancel,
        )

    async def request_cancellation(self, *, owner_id: str, run_id: str) -> CancellationOutcome:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return CancellationOutcome(outcome="unknown", run=None)

        async def _operation(conn: Any) -> CancellationOutcome:
            async with conn.transaction():
                row = await conn.fetchrow(_SELECT_RUN, owner, run_uuid)
                if row is None:
                    return CancellationOutcome(outcome="unknown", run=None)
                run = answer_run_record(row)
                if run.terminal:
                    return CancellationOutcome(outcome="already_terminal", run=run)
                if run.status == "queued":
                    await self._finalize_cancelled_queued(conn, owner, run_uuid)
                    return CancellationOutcome(
                        outcome="cancelled",
                        run=answer_run_record(await conn.fetchrow(_SELECT_RUN, owner, run_uuid)),
                    )
                await conn.execute(
                    _REQUEST_CANCELLATION,
                    owner,
                    run_uuid,
                    cancellation_notify_key(owner_id=owner, run_id=str(run_uuid)),
                )
                return CancellationOutcome(
                    outcome="pending",
                    run=answer_run_record(await conn.fetchrow(_SELECT_RUN, owner, run_uuid)),
                )

        return await self._run_write(_operation)

    async def _finalize_cancelled_queued(self, conn: Any, owner: str, run_uuid: uuid.UUID) -> None:
        await conn.execute(
            _FINALIZE_UNLEASED,
            [owner],
            [run_uuid],
            "cancelled",
            None,
            None,
            "done",
            "{}",
        )

    # -- claim and writes ---------------------------------------------
    async def claim_next(self, *, worker_id: str) -> ClaimedRun | None:
        """Claim the oldest eligible run and build its claim-bound execution surface."""
        worker = str(worker_id).strip()
        if not worker:
            raise ValueError("worker_id cannot be empty")

        async def _operation(conn: Any) -> ClaimedRun | None:
            while True:
                candidate = await conn.fetchrow(
                    _SELECT_CLAIM_CANDIDATE, MAX_RECLAIMS_WITHOUT_PROGRESS
                )
                if candidate is None:
                    return None
                locked = await conn.fetchrow(
                    _SELECT_RUN, candidate["owner_id"], candidate["run_id"]
                )
                if locked is None:
                    continue
                if locked["status"] == "queued":
                    decision = ReclaimDecision(
                        abandoned=False,
                        reclaims_without_progress=int(locked["reclaims_without_progress"]),
                        last_reclaim_progress_version=int(locked["durable_progress_version"]),
                    )
                else:
                    decision = advance_reclaim(
                        ReclaimState(
                            durable_progress_version=int(locked["durable_progress_version"]),
                            last_reclaim_progress_version=int(
                                locked["last_reclaim_progress_version"]
                            ),
                            reclaims_without_progress=int(locked["reclaims_without_progress"]),
                        )
                    )
                if decision.abandoned:
                    await conn.execute(
                        _FINALIZE_UNLEASED,
                        [candidate["owner_id"]],
                        [candidate["run_id"]],
                        "failed",
                        RUN_ABANDONED_ERROR_KIND,
                        _ABANDONED_ERROR_MESSAGE,
                        "error",
                        json.dumps({"error_kind": RUN_ABANDONED_ERROR_KIND}),
                    )
                    continue
                row = await conn.fetchrow(
                    _CLAIM_RUN,
                    candidate["owner_id"],
                    candidate["run_id"],
                    worker,
                    ANSWER_RUN_LEASE_SECONDS,
                    decision.reclaims_without_progress,
                    decision.last_reclaim_progress_version,
                )
                if row is None:
                    continue
                return self._claim_from_row(row, worker)

        async def _wrapped(conn: Any) -> ClaimedRun | None:
            async with conn.transaction():
                return await _operation(conn)

        return await self._run_write(_wrapped)

    def _claim_from_row(self, row: Any, worker: str) -> ClaimedRun:
        run = answer_run_record(row)
        owner = run.owner_id
        run_uuid = parse_run_id(run.run_id)
        if run_uuid is None:
            raise RuntimeError("claimed run id is not a canonical UUID")
        execution = RunExecutionContext(
            owner_id=owner,
            run_id=run.run_id,
            worker_id=worker,
            lease_owner=worker,
            fencing_epoch=run.fencing_epoch,
            session_store=PGJournalStore(
                pool=self._operation_pool,
                owner_id=owner,
                run_id=run_uuid,
                worker_id=worker,
                lease_owner=worker,
                fencing_epoch=run.fencing_epoch,
            ),
            progress_store=PGProgressStore(
                pool=self._operation_pool,
                owner_id=owner,
                run_id=run_uuid,
                worker_id=worker,
                lease_owner=worker,
                fencing_epoch=run.fencing_epoch,
            ),
            workspace_store=PGWorkspaceStore(
                pool=self._operation_pool,
                owner_id=owner,
                run_id=run_uuid,
                worker_id=worker,
                lease_owner=worker,
                fencing_epoch=run.fencing_epoch,
            ),
        )
        return ClaimedRun(run=run, execution=execution)

    async def heartbeat(
        self, *, owner_id: str, run_id: str, worker_id: str, fencing_epoch: int
    ) -> LeaseRenewal:
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
        return await self._append_event(
            owner_id=owner_id,
            run_id=run_id,
            worker_id=worker_id,
            fencing_epoch=fencing_epoch,
            phase=None,
            event_type="reset",
            payload={},
        )

    async def _append_event(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        phase: AnswerRunPhase | None,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> int | None:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return None

        async def _operation(conn: Any) -> int | None:
            sequence = await conn.fetchval(
                _APPEND_EVENT,
                owner,
                run_uuid,
                worker_id,
                fencing_epoch,
                phase,
                event_type,
                json.dumps(dict(payload), ensure_ascii=False),
                ANSWER_RUN_LEASE_SECONDS,
            )
            return int(sequence) if sequence is not None else None

        return await self._run_write(_operation)

    # -- terminal transitions -----------------------------------------
    async def finish_success(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        result: Mapping[str, object],
        stop_reason: str | None = None,
        publications: Sequence[Any] = (),
    ) -> TerminalOutcome:
        return await self._finish_run(
            owner_id=owner_id,
            run_id=run_id,
            worker_id=worker_id,
            fencing_epoch=fencing_epoch,
            status="succeeded",
            stop_reason=stop_reason,
            result=result,
            error_kind=None,
            error_message=None,
            event_type="done",
            payload={"status": "succeeded", "result": result},
            withhold_on_cancel=True,
            publications=publications,
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
        return await self._finish_run(
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
            payload={"error_kind": error_kind},
            withhold_on_cancel=True,
        )

    async def finish_cancelled(
        self, *, owner_id: str, run_id: str, worker_id: str, fencing_epoch: int
    ) -> TerminalOutcome:
        return await self._finish_run(
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
            payload={"status": "cancelled"},
            withhold_on_cancel=False,
        )

    async def _finish_run(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        status: str,
        stop_reason: str | None,
        result: Mapping[str, object] | None,
        error_kind: str | None,
        error_message: str | None,
        event_type: str,
        payload: Mapping[str, Any],
        withhold_on_cancel: bool,
        publications: Sequence[Any] = (),
    ) -> TerminalOutcome:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return TerminalOutcome(committed=False, status=None, event_sequence=None)

        async def _operation(conn: Any) -> TerminalOutcome:
            async with conn.transaction():
                sequence = await conn.fetchval(
                    _FINISH_RUN,
                    owner,
                    run_uuid,
                    worker_id,
                    fencing_epoch,
                    status,
                    stop_reason,
                    json.dumps(result, ensure_ascii=False) if result is not None else None,
                    error_kind,
                    error_message,
                    event_type,
                    json.dumps(dict(payload), ensure_ascii=False),
                    withhold_on_cancel,
                )
                if sequence is not None:
                    if status == "succeeded" and publications:
                        await self._write_publications(conn, owner, run_uuid, publications)
                    await conn.execute(
                        "DELETE FROM dlightrag_answer_committed_spills"
                        " WHERE owner_id = $1 AND run_id = $2",
                        owner,
                        run_uuid,
                    )
                    await conn.execute(
                        "DELETE FROM dlightrag_answer_resources"
                        " WHERE owner_id = $1 AND run_id = $2 AND kind = 'committed_spill'",
                        owner,
                        run_uuid,
                    )
                    return TerminalOutcome(
                        committed=True,
                        status=status,  # type: ignore[arg-type]
                        event_sequence=int(sequence),
                    )
                if status == "succeeded":
                    # A pending cancellation won the row: commit the cancelled
                    # terminal transition instead of leaving the run running.
                    current = await conn.fetchrow(_SELECT_RUN, owner, run_uuid)
                    if current is not None and current["cancel_requested_at"] is not None:
                        cancelled = await conn.fetchval(
                            _FINISH_RUN,
                            owner,
                            run_uuid,
                            worker_id,
                            fencing_epoch,
                            "cancelled",
                            None,
                            None,
                            None,
                            None,
                            "done",
                            json.dumps({"status": "cancelled"}),
                            False,
                        )
                        if cancelled is not None:
                            return TerminalOutcome(
                                committed=True,
                                status="cancelled",
                                event_sequence=int(cancelled),
                            )
                return TerminalOutcome(committed=False, status=None, event_sequence=None)

        return await self._run_write(_operation)

    async def release_for_shutdown(
        self, *, owner_id: str, run_id: str, worker_id: str, fencing_epoch: int
    ) -> ShutdownOutcome:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return "lease_lost"

        async def _operation(conn: Any) -> ShutdownOutcome:
            async with conn.transaction():
                row = await conn.fetchrow(_REQUEUE_RUN, owner, run_uuid, worker_id, fencing_epoch)
                if row is not None:
                    return "requeued"
                current = await conn.fetchrow(_SELECT_RUN, owner, run_uuid)
                if current is not None and current["cancel_requested_at"] is not None:
                    sequence = await conn.fetchval(
                        _FINISH_RUN,
                        owner,
                        run_uuid,
                        worker_id,
                        fencing_epoch,
                        "cancelled",
                        None,
                        None,
                        None,
                        None,
                        "done",
                        json.dumps({"status": "cancelled"}),
                        False,
                    )
                    return "cancelled" if sequence is not None else "lease_lost"
                return "lease_lost"

        return await self._run_write(_operation)

    # -- sweep and retention ------------------------------------------
    async def sweep_once(self) -> SweepOutcome:
        """Finalize cancel-pending and reclaim-poisoned rows without a slot."""

        async def _operation(conn: Any) -> SweepOutcome:
            async with conn.transaction():
                pending = await conn.fetch(_SELECT_CANCEL_PENDING, _BATCH_LIMIT)
                cancelled = 0
                if pending:
                    owners = [row["owner_id"] for row in pending]
                    run_ids = [row["run_id"] for row in pending]
                    cancelled = await conn.fetchval(
                        _FINALIZE_UNLEASED,
                        owners,
                        run_ids,
                        "cancelled",
                        None,
                        None,
                        "done",
                        "{}",
                    )
                poisoned = await conn.fetch(
                    "SELECT owner_id, run_id FROM dlightrag_answer_runs"
                    " WHERE status = 'running' AND lease_expires_at < NOW()"
                    " AND reclaims_without_progress >= $1"
                    " AND cancel_requested_at IS NULL"
                    " ORDER BY updated_at LIMIT $2"
                    " FOR UPDATE SKIP LOCKED",
                    MAX_RECLAIMS_WITHOUT_PROGRESS,
                    _BATCH_LIMIT,
                )
                abandoned = 0
                if poisoned:
                    owners = [row["owner_id"] for row in poisoned]
                    run_ids = [row["run_id"] for row in poisoned]
                    abandoned = await conn.fetchval(
                        _FINALIZE_UNLEASED,
                        owners,
                        run_ids,
                        "failed",
                        RUN_ABANDONED_ERROR_KIND,
                        _ABANDONED_ERROR_MESSAGE,
                        "error",
                        "{}",
                    )
                return SweepOutcome(cancelled=int(cancelled), abandoned=int(abandoned))

        return await self._run_write(_operation)

    async def trim_expired_event_logs(self) -> int:
        async def _operation(conn: Any) -> int:
            async with conn.transaction():
                rows = await conn.fetch(_SELECT_TRIMMABLE_RUNS, RUN_RETENTION_SECONDS, _BATCH_LIMIT)
                if not rows:
                    return 0
                owners = [row["owner_id"] for row in rows]
                run_ids = [row["run_id"] for row in rows]
                await conn.execute(_DELETE_EVENTS_FOR_RUNS, owners, run_ids)
                await conn.execute(_MARK_EVENTS_TRIMMED, owners, run_ids)
                return len(rows)

        return await self._run_write(_operation)

    async def prune_expired_runs(self) -> RunDeletion:
        """Retention order: delete runs (references cascade), then unreferenced blobs."""

        async def _operation(conn: Any) -> RunDeletion:
            async with conn.transaction():
                rows = await conn.fetch(_SELECT_EXPIRED_RUNS, RUN_RETENTION_SECONDS, _BATCH_LIMIT)
                if not rows:
                    return RunDeletion(runs=0, artifacts=0)
                owners = [row["owner_id"] for row in rows]
                run_ids = [row["run_id"] for row in rows]
                digest_rows = await conn.fetch(_SELECT_RUN_DIGESTS, owners, run_ids)
                deleted = await conn.fetchval(_DELETE_RUNS, owners, run_ids)
                artifacts = await _delete_unreferenced(conn, *_digest_pairs(digest_rows))
                return RunDeletion(runs=int(deleted), artifacts=artifacts)

        return await self._run_write(_operation)


def _require_replay_match(
    *,
    owner: str,
    key: str | None,
    stored_fingerprint: str,
    fingerprint: str,
    row: Any,
) -> RunCreation:
    if stored_fingerprint != fingerprint:
        raise IdempotencyKeyConflict(
            f"owner {owner} reused idempotency key {key} with different normalized input"
        )
    return RunCreation(run=answer_run_record(row), replayed=True)


def _plan_blob(content: bytes):
    from dlightrag.runtime.blob_chunks import plan_blob

    return plan_blob(content)


__all__ = [
    "ANSWER_RUN_MIGRATIONS",
    "ANSWER_RUN_MIGRATION_SCOPE",
    "ANSWER_RUN_SCHEMA_TABLES",
    "PGAnswerRunStore",
    "answer_run_columns",
    "answer_run_record",
]
