# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL adapter family for the operation-neutral RunRuntime.

The generic run row, claim/lease/event lifecycle, and retention logic live here.
Answer-owned Session, evidence, and resource tables remain narrow projections
linked to the generic row; callers consume owner-specific Protocols rather than
one universal operational-storage interface.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import asyncpg

from dlightrag.adapters.postgres.answer.memory_settings import (
    MEMORY_SETTINGS_DDL,
    MEMORY_SETTINGS_SCHEMA_TABLE,
)
from dlightrag.adapters.postgres.answer.session_repository import (
    PGAgentSessionRepository,
    PGProgressStore,
    write_fetched_resources,
)
from dlightrag.adapters.postgres.answer.workspace import PGWorkspaceStore
from dlightrag.adapters.postgres.connections import PGConnectionPinWriter
from dlightrag.adapters.postgres.core._channels import RUN_CANCEL_CHANNEL
from dlightrag.adapters.postgres.core._migrations import (
    ForeignKeyRequirement,
    IndexRequirement,
    Migration,
    TableRequirement,
    apply_migrations,
    verify_migrations,
)
from dlightrag.adapters.postgres.core._notifications import ChannelWatcher, PGNotificationHub
from dlightrag.adapters.postgres.core._operations import ConnectionPool, PostgresOperationRunner
from dlightrag.adapters.postgres.runtime._child import (
    _CONSUME_CHILD_CONTROLS,
    _CONSUME_PARENT_CONTROLS,
    _INSERT_CONTROL,
    _LOCK_CHILD_SESSION,
    _LOCK_CONTROL_RUN,
    _NEXT_CONTROL_SEQUENCE,
    _SELECT_AGENT_TRANSCRIPT,
    _SELECT_PENDING_CHILD_CONTROLS,
    _SELECT_PENDING_PARENT_CONTROLS,
    PENDING_CONTROL_READ_LIMIT,
    ChildRunStoreMixin,
    cancellation_notify_key,
)
from dlightrag.adapters.postgres.runtime._lease import hold_run_lease
from dlightrag.adapters.postgres.runtime._terminal import (
    SETTLE_TERMINATED_RUN_CHILDREN,
    TerminalStatus,
    finish_fenced_run,
)
from dlightrag.adapters.postgres.runtime.run_blob_store import BlobSizeConflict, write_blob_content
from dlightrag.engine.agent.session.ids import SessionId
from dlightrag.engine.agent.tool_content import decode_tool_content, tool_content_message_fields
from dlightrag.engine.answer.attachment_replay import AttachmentReplaySelection
from dlightrag.engine.answer.execution.connection_binding import RunConnectionBinding
from dlightrag.engine.answer.execution.lineage import ADOPTABLE_LINEAGE_KINDS
from dlightrag.engine.answer.runs.routing import RoutingAcceptance, RoutingRecord
from dlightrag.engine.runtime.contracts import RunKind, RunLane, RunPhase
from dlightrag.engine.runtime.errors import RunSchemaError
from dlightrag.engine.runtime.policy import (
    DEFAULT_RUN_RETENTION_SECONDS,
    MAX_RECLAIMS_WITHOUT_PROGRESS,
    RUN_ABANDONED_ERROR_KIND,
    RUN_LEASE_SECONDS,
)
from dlightrag.engine.runtime.records import (
    CancellationOutcome,
    ClaimedRun,
    DeletedRun,
    IdempotencyKeyConflict,
    LeaseRenewal,
    PendingArtifact,
    PendingArtifactReference,
    PreparedRunEnvelope,
    ReclaimDecision,
    ReclaimState,
    RunAccessScope,
    RunAdmissionLimitExceededError,
    RunArtifactReference,
    RunCreation,
    RunDeletion,
    RunEvent,
    RunExecutionContext,
    RunFetchedResource,
    RunRecord,
    ShutdownOutcome,
    SweepOutcome,
    TerminalOutcome,
    advance_reclaim,
    parse_run_id,
    require_prepared_input_bounds,
)
from dlightrag.engine.runtime.settlements import (
    ArtifactAttachmentUpdate,
    FetchedResourceSettlementUpdate,
)

RUN_MIGRATION_SCOPE = "runs"

# Resource kinds a later Run may adopt from an earlier Run on the same Session.
#: What a later Run of the same Session may adopt by naming a handle. It lives in
#: the engine beside the loader that gates on it, and one declaration drives both
#: this adapter's read and that gate, so a new adoptable kind is added once.
_ADOPTABLE_LINEAGE_KINDS = ADOPTABLE_LINEAGE_KINDS


def _parse_uuid(value: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(str(value))
    except TypeError, ValueError:
        return None


_ABANDONED_ERROR_MESSAGE = "Run exceeded its reclaim-without-progress bound."
_BATCH_LIMIT = 200
_EVENT_PAGE_LIMIT = 500
DEFAULT_QUERY_MAX_NONTERMINAL_RUNS = 30_000
# Validated by the deterministic bounded-control-plane campaign; see
# docs/run-runtime-and-scaling-target.md#captured-local-load-evidence.
DEFAULT_CORPUS_MUTATION_MAX_NONTERMINAL_RUNS = 1_000

# Every index the runs scope owns, declared once. The baseline creates each one; a
# later migration that introduced an index repeats its statement for databases
# initialized before it; readers verify every one by name.
_RUNS_CLAIM_INDEX = IndexRequirement(
    # Claim and sweep scan nonterminal rows oldest-first across every owner.
    "idx_dlightrag_runs_claim",
    "dlightrag_runs",
    "(lane, next_attempt_at, created_at, run_id) WHERE status IN ('queued', 'running')",
)
_RUNS_SUBMISSION_INDEX = IndexRequirement(
    "idx_dlightrag_runs_submission",
    "dlightrag_runs",
    "(run_kind, submitted_by, submission_key)",
    unique=True,
)
# The one uniqueness of a bare run id; the Corpus Mutation window table's foreign
# key to dlightrag_runs (run_id) relies on it (see the baseline below).
_RUNS_GLOBAL_ID_INDEX = IndexRequirement(
    "idx_dlightrag_runs_global_id", "dlightrag_runs", "(run_id)", unique=True
)
_RUNS_RETENTION_INDEX = IndexRequirement(
    "idx_dlightrag_runs_retention",
    "dlightrag_runs",
    "(purge_after) WHERE purge_after IS NOT NULL",
)
_RUNS_CANCEL_PENDING_INDEX = IndexRequirement(
    # Reconnect/notification rescans page only this worker's live cancellations.
    "idx_dlightrag_runs_cancel_pending",
    "dlightrag_runs",
    "(lease_owner, created_at, run_id)"
    " WHERE cancel_requested_at IS NOT NULL AND status = 'running'",
)
_RUNS_MUTATION_FIFO_INDEX = IndexRequirement(
    # Workspace mutation eligibility walks each Workspace's queue oldest-first.
    "idx_dlightrag_runs_mutation_fifo",
    "dlightrag_runs",
    "(owner_id, created_at, run_id)"
    " WHERE lane = 'corpus_mutation' AND status IN ('queued', 'running')",
)
_RUN_EVENTS_TERMINAL_INDEX = IndexRequirement(
    # Exactly one terminal event per run, enforced durably rather than by convention.
    "idx_dlightrag_run_events_terminal",
    "dlightrag_run_events",
    "(owner_id, run_id) WHERE event_type IN ('done', 'error')",
    unique=True,
)
_RUN_ARTIFACTS_DIGEST_INDEX = IndexRequirement(
    # Reverse lookup for ownership-safe blob cleanup and the RESTRICT foreign key.
    "idx_dlightrag_answer_run_artifacts_digest",
    "dlightrag_answer_run_artifacts",
    "(owner_id, digest)",
)
_EVIDENCE_RUN_INDEX = IndexRequirement(
    "idx_dlightrag_answer_evidence_run", "dlightrag_answer_evidence", "(owner_id, run_id)"
)
_RESOURCES_RUN_INDEX = IndexRequirement(
    "idx_dlightrag_answer_resources_run", "dlightrag_answer_resources", "(owner_id, run_id)"
)
_RESOURCES_BLOB_INDEX = IndexRequirement(
    "idx_dlightrag_answer_resources_blob",
    "dlightrag_answer_resources",
    "(owner_id, blob_digest) WHERE blob_digest IS NOT NULL",
)
_ATTACHMENT_OCCURRENCE_INDEX = IndexRequirement(
    # Retained exact Entry occurrences, found without scanning an owner's catalogue.
    "idx_answer_attachment_occurrence",
    "dlightrag_answer_resources",
    "(owner_id, resource_id)"
    " WHERE kind='fetched_blob' AND capabilities->>'resource_kind'='attachment_occurrence'",
)
_CHILD_ROSTER_INDEX = IndexRequirement(
    # Newest-first bounded child-roster keyset pages ride one exact order.
    "idx_answer_child_sessions_roster",
    "dlightrag_answer_child_sessions",
    "(owner_id, run_id, created_at DESC, child_session_id DESC)",
)
_CHILD_OPERATIONS_STATUS_INDEX = IndexRequirement(
    "idx_answer_child_operations_status",
    "dlightrag_answer_child_operations",
    "(owner_id, run_id, child_session_id, status)",
)
_CHILD_CONTROLS_SUBMISSION_INDEX = IndexRequirement(
    "idx_agent_controls_submission",
    "dlightrag_agent_controls",
    "(owner_id, run_id, target_session_id, submission_key)"
    " WHERE target_session_id IS NOT NULL AND submission_key IS NOT NULL",
    unique=True,
)
_PARENT_CONTROLS_SUBMISSION_INDEX = IndexRequirement(
    "idx_agent_parent_controls_submission",
    "dlightrag_agent_controls",
    "(owner_id, run_id, submission_key)"
    " WHERE target_session_id IS NULL AND submission_key IS NOT NULL",
    unique=True,
)
_CHILD_GUIDANCE_PENDING_INDEX = IndexRequirement(
    "idx_child_guidance_pending",
    "dlightrag_answer_child_guidance",
    "(owner_id, run_id, status, expires_at)",
)
_RUN_INDEXES = (
    _RUNS_CLAIM_INDEX,
    _RUNS_SUBMISSION_INDEX,
    _RUNS_GLOBAL_ID_INDEX,
    _RUNS_RETENTION_INDEX,
    _RUNS_CANCEL_PENDING_INDEX,
    _RUNS_MUTATION_FIFO_INDEX,
    _RUN_EVENTS_TERMINAL_INDEX,
    _RUN_ARTIFACTS_DIGEST_INDEX,
    _EVIDENCE_RUN_INDEX,
    _RESOURCES_RUN_INDEX,
    _RESOURCES_BLOB_INDEX,
    _ATTACHMENT_OCCURRENCE_INDEX,
    _CHILD_ROSTER_INDEX,
    _CHILD_OPERATIONS_STATUS_INDEX,
    _CHILD_CONTROLS_SUBMISSION_INDEX,
    _PARENT_CONTROLS_SUBMISSION_INDEX,
    _CHILD_GUIDANCE_PENDING_INDEX,
)

# ─────────────────────────────────────────────────────────────────
# Final clean-break baseline schema
# ─────────────────────────────────────────────────────────────────

# A database that still holds dlightrag_answer_runs (created by releases 2.0.0
# through 2.0.5 and never started by a later release) is not migrated: development
# data is reset instead (docs/postgresql.md), so that table refuses startup with the
# remedy.
_PRE_RUNTIME_ANSWER_SCHEMA = "SELECT to_regclass('dlightrag_answer_runs') IS NOT NULL"
_PRE_RUNTIME_ANSWER_SCHEMA_ERROR = (
    "this database still holds dlightrag_answer_runs (created by DlightRAG 2.0.0-2.0.5 "
    "and never started by a later release), which is not migrated; run a full "
    "development reset (scripts/reset_development.py) and start a writer on the empty "
    "database"
)

_CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS dlightrag_runs (
    owner_id            TEXT        NOT NULL,
    run_id              UUID        NOT NULL,
    run_kind            TEXT        NOT NULL,
    lane                TEXT        NOT NULL,
    submitted_by        TEXT        NOT NULL,
    access_scope_kind   TEXT        NOT NULL,
    submission_key      TEXT        NOT NULL,
    prepared_input_json JSONB,
    accepted_input_json JSONB       NOT NULL DEFAULT '{}'::jsonb,
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
    retention_seconds   BIGINT      NOT NULL,
    purge_after         TIMESTAMPTZ,
    next_attempt_at     TIMESTAMPTZ,
    checkpoint_json     JSONB,
    handoff_started_at  TIMESTAMPTZ,
    superseded_by_run_id UUID,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at          TIMESTAMPTZ,
    finished_at         TIMESTAMPTZ,
    agent_workspace_epoch BIGINT,
    PRIMARY KEY (owner_id, run_id),
    CONSTRAINT dlightrag_runs_kind_check
        CHECK (run_kind IN ('retrieval', 'answer', 'corpus_mutation')),
    CONSTRAINT dlightrag_runs_lane_check
        CHECK (lane IN ('query', 'corpus_mutation')),
    CONSTRAINT dlightrag_runs_scope_check
        CHECK (access_scope_kind IN ('owner', 'workspace')),
    CONSTRAINT dlightrag_runs_status_check
        CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
    CONSTRAINT dlightrag_runs_counter_check
        CHECK (fencing_epoch >= 0 AND next_event_sequence >= 1
               AND durable_progress_version >= 0
               AND last_reclaim_progress_version >= 0
               AND reclaims_without_progress >= 0 AND retention_seconds >= 1),
    CONSTRAINT dlightrag_runs_lease_check
        CHECK ((lease_owner IS NULL) = (lease_expires_at IS NULL)),
    CONSTRAINT dlightrag_runs_terminal_check
        CHECK ((status IN ('succeeded', 'failed', 'cancelled')) = (finished_at IS NOT NULL)),
    CONSTRAINT dlightrag_runs_result_check
        CHECK (status <> 'succeeded' OR result_json IS NOT NULL),
    CONSTRAINT dlightrag_runs_error_check
        CHECK ((status = 'failed') = (error_kind IS NOT NULL)),
    CONSTRAINT dlightrag_runs_prepared_input_check
        CHECK ((status IN ('queued', 'running')) = (prepared_input_json IS NOT NULL)),
    CONSTRAINT dlightrag_runs_workspace_epoch_check
        CHECK (agent_workspace_epoch IS NULL OR agent_workspace_epoch >= 1)
)
"""

_CREATE_EVENTS = """
CREATE TABLE IF NOT EXISTS dlightrag_run_events (
    owner_id       TEXT        NOT NULL,
    run_id         UUID        NOT NULL,
    event_sequence BIGINT      NOT NULL,
    event_type     TEXT        NOT NULL,
    payload        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id, run_id, event_sequence),
    FOREIGN KEY (owner_id, run_id)
        REFERENCES dlightrag_runs (owner_id, run_id) ON DELETE CASCADE,
    CONSTRAINT dlightrag_run_events_sequence_check
        CHECK (event_sequence >= 1)
)
"""

_ENFORCE_RUN_EVENT_CONSTRAINTS = """
CREATE OR REPLACE FUNCTION public.dlightrag_enforce_run_event_constraints()
RETURNS TRIGGER
LANGUAGE plpgsql
SET search_path = pg_catalog, public
AS $$
DECLARE
    parent_status TEXT;
    parent_lease_owner TEXT;
    parent_lease_expires_at TIMESTAMPTZ;
    parent_next_event_sequence BIGINT;
    parent_result JSONB;
    parent_error_kind TEXT;
    parent_error_message TEXT;
BEGIN
    SELECT status, lease_owner, lease_expires_at, next_event_sequence,
           result_json, error_kind, error_message
    INTO parent_status, parent_lease_owner, parent_lease_expires_at,
         parent_next_event_sequence, parent_result, parent_error_kind,
         parent_error_message
    FROM public.dlightrag_runs
    WHERE owner_id = NEW.owner_id AND run_id = NEW.run_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'run event parent does not exist'
            USING ERRCODE = '23503';
    END IF;
    IF NEW.event_sequence >= parent_next_event_sequence THEN
        RAISE EXCEPTION 'run event sequence must precede the parent next sequence'
            USING ERRCODE = '23514';
    END IF;
    IF jsonb_typeof(NEW.payload) IS DISTINCT FROM 'object' THEN
        RAISE EXCEPTION 'run event payload must be a JSON object'
            USING ERRCODE = '23514';
    END IF;

    IF NEW.event_type = 'done' THEN
        IF jsonb_typeof(NEW.payload->'status') IS DISTINCT FROM 'string'
           OR NEW.payload->>'status' IS DISTINCT FROM parent_status THEN
            RAISE EXCEPTION 'done event status does not match its parent'
                USING ERRCODE = '23514';
        END IF;
        IF parent_status = 'succeeded' THEN
            IF NOT NEW.payload ? 'result'
               OR NEW.payload->'result' IS DISTINCT FROM parent_result THEN
                RAISE EXCEPTION 'succeeded run event payload does not match its parent'
                    USING ERRCODE = '23514';
            END IF;
        ELSIF parent_status = 'cancelled' THEN
            IF NEW.payload ? 'result' THEN
                RAISE EXCEPTION 'cancelled run event payload cannot contain a result'
                    USING ERRCODE = '23514';
            END IF;
        ELSE
            RAISE EXCEPTION 'done event requires a succeeded or cancelled parent'
                USING ERRCODE = '23514';
        END IF;
    ELSIF NEW.event_type = 'error' THEN
        IF parent_status IS DISTINCT FROM 'failed' THEN
            RAISE EXCEPTION 'error event requires a failed parent'
                USING ERRCODE = '23514';
        END IF;
        IF jsonb_typeof(NEW.payload->'kind') IS DISTINCT FROM 'string'
           OR jsonb_typeof(NEW.payload->'message') IS DISTINCT FROM 'string'
           OR NEW.payload->>'kind' IS DISTINCT FROM parent_error_kind
           OR NEW.payload->>'message' IS DISTINCT FROM parent_error_message
           OR (NEW.payload ? 'result'
               AND NEW.payload->'result' IS DISTINCT FROM parent_result) THEN
            RAISE EXCEPTION 'failed run event payload does not match its parent'
                USING ERRCODE = '23514';
        END IF;
    ELSIF parent_status IS DISTINCT FROM 'running'
          OR parent_lease_owner IS NULL
          OR parent_lease_expires_at IS NULL
          OR parent_lease_expires_at < NOW() THEN
        RAISE EXCEPTION 'nonterminal event requires a live leased parent'
            USING ERRCODE = '23514';
    END IF;

    RETURN NEW;
END
$$
"""

_CREATE_RUN_EVENT_CONSTRAINT_TRIGGER = """
DROP TRIGGER IF EXISTS trg_dlightrag_run_events_enforce
    ON public.dlightrag_run_events;
CREATE TRIGGER trg_dlightrag_run_events_enforce
    BEFORE INSERT OR UPDATE ON public.dlightrag_run_events
    FOR EACH ROW
    EXECUTE FUNCTION public.dlightrag_enforce_run_event_constraints();
"""

_CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS dlightrag_agent_sessions (
    owner_id             TEXT        NOT NULL,
    session_id           UUID        NOT NULL,
    lease_run_id         UUID        NOT NULL,
    commit_sequence      BIGINT      NOT NULL DEFAULT 0,
    fencing_epoch        BIGINT      NOT NULL,
    last_sequence        BIGINT      NOT NULL DEFAULT 0,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id, session_id),
    CONSTRAINT dlightrag_agent_sessions_commit_sequence_check CHECK (commit_sequence >= 0),
    CONSTRAINT dlightrag_agent_sessions_fencing_check CHECK (fencing_epoch >= 1),
    CONSTRAINT dlightrag_agent_sessions_sequence_check CHECK (last_sequence >= 0)
)
"""

_CREATE_ENTRIES = """
CREATE TABLE IF NOT EXISTS dlightrag_agent_session_entries (
    owner_id       TEXT        NOT NULL,
    session_id     UUID        NOT NULL,
    sequence       BIGINT      NOT NULL,
    entry_id       UUID        NOT NULL,
    parent_entry_id UUID,
    entry_type     TEXT        NOT NULL,
    schema_version INTEGER     NOT NULL,
    timestamp      TIMESTAMPTZ NOT NULL,
    payload_json   JSONB       NOT NULL,
    PRIMARY KEY (owner_id, session_id, sequence),
    UNIQUE (entry_id),
    UNIQUE (owner_id, session_id, entry_id),
    FOREIGN KEY (owner_id, session_id)
        REFERENCES dlightrag_agent_sessions (owner_id, session_id) ON DELETE CASCADE,
    FOREIGN KEY (owner_id, session_id, parent_entry_id)
        REFERENCES dlightrag_agent_session_entries
            (owner_id, session_id, entry_id)
        DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT dlightrag_agent_session_entries_sequence_check CHECK (sequence >= 1),
    CONSTRAINT dlightrag_agent_session_entries_type_check CHECK (entry_type IN (
        'user_message', 'assistant_message', 'tool_result',
        'control_message', 'compaction'
    )),
    CONSTRAINT dlightrag_agent_session_entries_version_check CHECK (schema_version >= 1)
)
"""

_CREATE_SESSION_REGISTERS = """
CREATE TABLE IF NOT EXISTS dlightrag_agent_session_registers (
    owner_id       TEXT        NOT NULL,
    session_id     UUID        NOT NULL,
    register_kind  TEXT        NOT NULL,
    register_key   TEXT        NOT NULL,
    sequence       BIGINT      NOT NULL,
    payload_json   JSONB       NOT NULL,
    PRIMARY KEY (owner_id, session_id, register_kind, register_key),
    FOREIGN KEY (owner_id, session_id)
        REFERENCES dlightrag_agent_sessions (owner_id, session_id) ON DELETE CASCADE,
    CONSTRAINT dlightrag_agent_session_registers_kind_check
        CHECK (register_kind IN (
            'lane_head', 'lane_state', 'operation_meta', 'operation_state',
            'request_snapshot', 'tool_arguments', 'pending_input', 'host_turn_reservation',
            'context_projection', 'session_fault'
        )),
    CONSTRAINT dlightrag_agent_session_registers_sequence_check CHECK (sequence >= 1)
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
        REFERENCES dlightrag_runs (owner_id, run_id) ON DELETE CASCADE,
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
        REFERENCES dlightrag_runs (owner_id, run_id) ON DELETE CASCADE,
    CONSTRAINT dlightrag_answer_evidence_ordinal_check CHECK (result_ordinal >= 0),
    CONSTRAINT dlightrag_answer_evidence_digest_check
        CHECK (content_digest ~ '^[0-9a-f]{64}$' AND locator_digest ~ '^[0-9a-f]{64}$')
)
"""

# Nothing writes kind 'accepted_blob' any more: an accepted upload is its Run's
# current_attachment artifact, which also holds its Blob. The checks still admit
# the kind, because databases created before then keep those rows and those
# checks, and narrowing them would need a migration.
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
    source_locator BYTEA,
    session_id     UUID,
    intent_id      UUID,
    result_ordinal INTEGER,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id, run_id, resource_id),
    FOREIGN KEY (owner_id, run_id)
        REFERENCES dlightrag_runs (owner_id, run_id) ON DELETE CASCADE,
    CONSTRAINT dlightrag_answer_resources_kind_check
        CHECK (kind IN ('accepted_blob', 'evidence', 'fetched_blob', 'committed_spill',
                        'published_artifact')),
    CONSTRAINT dlightrag_answer_resources_blob_link_check
        CHECK ((kind = 'accepted_blob' AND blob_digest IS NOT NULL)
               OR (kind = 'fetched_blob'
                   AND blob_digest IS NOT NULL AND locator_digest IS NOT NULL
                   AND (capabilities->>'resource_kind' IS DISTINCT FROM 'web'
                        OR source_locator IS NOT NULL))
               OR (kind = 'evidence' AND locator_digest IS NOT NULL)
               OR (kind = 'committed_spill'
                   AND blob_digest IS NULL AND locator_digest IS NULL)
               OR (kind = 'published_artifact' AND blob_digest IS NOT NULL))
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
        REFERENCES dlightrag_runs (owner_id, run_id) ON DELETE CASCADE,
    FOREIGN KEY (owner_id, digest)
        REFERENCES dlightrag_blobs (owner_id, digest) ON DELETE RESTRICT,
    CONSTRAINT dlightrag_answer_run_artifacts_kind_check
        CHECK (reference_kind IN
               ('current_attachment', 'history_attachment', 'fetched_resource',
                'published_artifact')),
    CONSTRAINT dlightrag_answer_run_artifacts_ordinal_check
        CHECK (ordinal >= 0)
)
"""

_CREATE_ROUTING = """
CREATE TABLE IF NOT EXISTS dlightrag_answer_run_routing (
    owner_id                 TEXT        NOT NULL,
    run_id                   UUID        NOT NULL,
    requested_mode           TEXT        NOT NULL,
    valid_modes              TEXT[]      NOT NULL,
    resolved_mode            TEXT,
    model_fingerprints       JSONB       NOT NULL DEFAULT '{}'::jsonb,
    context_policy_revision  TEXT        NOT NULL,
    agent_session_id         UUID        NOT NULL,
    agent_lane_id            TEXT        NOT NULL,
    source_lane_id           TEXT,
    fork_point_entry_id      TEXT,
    fork_point_projection_id TEXT,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id, run_id),
    FOREIGN KEY (owner_id, run_id)
        REFERENCES dlightrag_runs (owner_id, run_id) ON DELETE CASCADE,
    CONSTRAINT dlightrag_answer_run_routing_requested_check
        CHECK (requested_mode IN ('auto', 'fast', 'research')),
    CONSTRAINT dlightrag_answer_run_routing_valid_check
        CHECK (COALESCE(array_length(valid_modes, 1), 0) >= 1
               AND valid_modes <@ ARRAY['fast', 'research']::text[]),
    CONSTRAINT dlightrag_answer_run_routing_resolved_check
        CHECK (resolved_mode IS NULL
               OR (resolved_mode IN ('fast', 'research')
                   AND resolved_mode = ANY (valid_modes)))
)
"""

_CREATE_CHILD_SESSIONS = """
CREATE TABLE IF NOT EXISTS dlightrag_answer_child_sessions (
    owner_id           TEXT        NOT NULL,
    run_id             UUID        NOT NULL,
    child_session_id   UUID        NOT NULL,
    parent_session_id  UUID        NOT NULL,
    parent_call_id     TEXT        NOT NULL,
    parent_intent_id   UUID,
    status             TEXT        NOT NULL,
    cancel_requested_at TIMESTAMPTZ,
    summary            TEXT,
    objective          TEXT,
    context_mode       TEXT,
    model_role         TEXT,
    tools_json         JSONB,
    usage_json         JSONB,
    depth              INTEGER     NOT NULL,
    context_snapshot_json JSONB    NOT NULL,
    plan_json          JSONB,
    budget_json        JSONB,
    host_state_json    JSONB       NOT NULL DEFAULT '{}'::jsonb,
    lease_owner        TEXT,
    lease_expires_at   TIMESTAMPTZ,
    fencing_epoch      BIGINT      NOT NULL DEFAULT 0,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id, run_id, child_session_id),
    FOREIGN KEY (owner_id, run_id)
        REFERENCES dlightrag_runs (owner_id, run_id) ON DELETE CASCADE,
    CONSTRAINT dlightrag_answer_child_sessions_status_check
        CHECK (status IN ('running', 'succeeded', 'failed', 'cancelled')),
    CONSTRAINT dlightrag_answer_child_sessions_depth_check CHECK (depth >= 1),
    CONSTRAINT dlightrag_answer_child_sessions_fencing_check CHECK (fencing_epoch >= 0)
)
"""

_CREATE_AGENT_CONTROLS = """
CREATE TABLE IF NOT EXISTS dlightrag_agent_controls (
    owner_id           TEXT        NOT NULL,
    run_id             UUID        NOT NULL,
    control_sequence   BIGINT      NOT NULL,
    kind               TEXT        NOT NULL,
    content            TEXT        NOT NULL,
    target_session_id  UUID,
    target_operation_id UUID,
    origin             TEXT        NOT NULL DEFAULT 'user',
    submission_key     TEXT,
    request_fingerprint TEXT,
    consumed_at        TIMESTAMPTZ,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id, run_id, control_sequence),
    FOREIGN KEY (owner_id, run_id)
        REFERENCES dlightrag_runs (owner_id, run_id) ON DELETE CASCADE,
    CONSTRAINT dlightrag_agent_controls_sequence_check CHECK (control_sequence >= 1),
    CONSTRAINT dlightrag_agent_controls_kind_check CHECK (kind IN ('steer', 'follow_up', 'cancel')),
    CONSTRAINT dlightrag_agent_controls_content_check CHECK (char_length(content) BETWEEN 1 AND 20000),
    CONSTRAINT dlightrag_agent_controls_target_check CHECK (
        (target_session_id IS NULL AND target_operation_id IS NULL)
        OR (target_session_id IS NOT NULL AND target_operation_id IS NOT NULL)
    ),
    CONSTRAINT dlightrag_agent_controls_origin_check CHECK (origin IN ('user', 'parent'))
)
"""

_CREATE_CHILD_OPERATIONS = """
CREATE TABLE IF NOT EXISTS dlightrag_answer_child_operations (
    owner_id           TEXT        NOT NULL,
    run_id             UUID        NOT NULL,
    child_session_id   UUID        NOT NULL,
    operation_sequence BIGINT      NOT NULL,
    operation_id       UUID        NOT NULL,
    idempotency_key    TEXT        NOT NULL,
    request_fingerprint TEXT       NOT NULL,
    content            TEXT        NOT NULL,
    origin             TEXT        NOT NULL,
    status             TEXT        NOT NULL,
    cancellation_origin TEXT,
    summary            TEXT,
    usage_json         JSONB,
    outcome_json       JSONB,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id, run_id, child_session_id, operation_sequence),
    UNIQUE (owner_id, run_id, child_session_id, operation_id),
    UNIQUE (owner_id, run_id, child_session_id, idempotency_key),
    FOREIGN KEY (owner_id, run_id, child_session_id)
        REFERENCES dlightrag_answer_child_sessions (owner_id, run_id, child_session_id)
        ON DELETE CASCADE,
    CONSTRAINT dlightrag_answer_child_operations_sequence_check CHECK (operation_sequence >= 1),
    CONSTRAINT dlightrag_answer_child_operations_status_check
        CHECK (status IN ('running', 'succeeded', 'failed', 'cancelled')),
    CONSTRAINT dlightrag_answer_child_operations_origin_check CHECK (origin IN ('parent', 'user')),
    CONSTRAINT dlightrag_answer_child_operations_cancellation_origin_check CHECK (
        cancellation_origin IS NULL OR cancellation_origin IN ('parent', 'user', 'run')
    ),
    CONSTRAINT dlightrag_answer_child_operations_content_check
        CHECK (char_length(content) BETWEEN 1 AND 20000)
)
"""

_CREATE_CHILD_GUIDANCE = """
CREATE TABLE IF NOT EXISTS dlightrag_answer_child_guidance (
    owner_id           TEXT        NOT NULL,
    run_id             UUID        NOT NULL,
    request_id         UUID        NOT NULL,
    child_session_id   UUID        NOT NULL,
    child_operation_id UUID        NOT NULL,
    parent_session_id  UUID        NOT NULL,
    question           TEXT        NOT NULL,
    status             TEXT        NOT NULL DEFAULT 'pending',
    reply              TEXT,
    reply_origin       TEXT,
    reply_submission_key TEXT,
    reply_fingerprint  TEXT,
    expires_at         TIMESTAMPTZ NOT NULL,
    replied_at         TIMESTAMPTZ,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id, run_id, request_id),
    FOREIGN KEY (owner_id, run_id, child_session_id)
        REFERENCES dlightrag_answer_child_sessions (owner_id, run_id, child_session_id)
        ON DELETE CASCADE,
    CONSTRAINT dlightrag_answer_child_guidance_status_check
        CHECK (status IN ('pending', 'replied', 'expired', 'cancelled')),
    CONSTRAINT dlightrag_answer_child_guidance_question_check
        CHECK (char_length(question) BETWEEN 1 AND 20000),
    CONSTRAINT dlightrag_answer_child_guidance_reply_check
        CHECK (reply IS NULL OR char_length(reply) BETWEEN 1 AND 20000),
    CONSTRAINT dlightrag_answer_child_guidance_reply_origin_check
        CHECK (reply_origin IS NULL OR reply_origin IN ('parent', 'user'))
)
"""


_CREATE_INDEXES = tuple(index.ddl for index in _RUN_INDEXES)

_CREATE_WORKSPACE_INVENTORY = """
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
        REFERENCES dlightrag_runs (owner_id, run_id) ON DELETE CASCADE
)
"""

_CREATE_ARTIFACT_ATTACHMENT_ORDER = """
CREATE SEQUENCE IF NOT EXISTS dlightrag_answer_artifact_attachment_order_seq
"""

_CREATE_ARTIFACT_ATTACHMENTS = """
CREATE TABLE IF NOT EXISTS dlightrag_answer_artifact_attachments (
    owner_id        TEXT        NOT NULL,
    run_id          UUID        NOT NULL,
    relative_path   TEXT        NOT NULL,
    label           TEXT        NOT NULL,
    content_digest  TEXT        NOT NULL,
    size_bytes      BIGINT      NOT NULL,
    presentation    TEXT        NOT NULL,
    session_id      UUID        NOT NULL,
    intent_id       UUID        NOT NULL,
    attachment_order BIGINT     NOT NULL DEFAULT nextval(
        'dlightrag_answer_artifact_attachment_order_seq'
    ),
    attached_at     TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (owner_id, run_id, relative_path),
    FOREIGN KEY (owner_id, run_id)
        REFERENCES dlightrag_runs (owner_id, run_id) ON DELETE CASCADE,
    CONSTRAINT dlightrag_answer_artifact_attachments_digest_check
        CHECK (content_digest ~ '^[0-9a-f]{64}$'),
    CONSTRAINT dlightrag_answer_artifact_attachments_size_check
        CHECK (size_bytes >= 0),
    CONSTRAINT dlightrag_answer_artifact_attachments_presentation_check
        CHECK (presentation IN
            ('image', 'video', 'markdown', 'html', 'pdf', 'text', 'download'))
)
"""

_CREATE_CORPUS_MUTATION_WINDOWS = """
CREATE TABLE IF NOT EXISTS dlightrag_corpus_mutation_windows (
    run_id          UUID        NOT NULL,
    window_number   INTEGER     NOT NULL,
    workspace       TEXT        NOT NULL,
    docs             BIGINT      NOT NULL,
    chunks           BIGINT      NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (run_id, window_number),
    FOREIGN KEY (run_id) REFERENCES dlightrag_runs (run_id) ON DELETE CASCADE,
    CONSTRAINT dlightrag_corpus_mutation_windows_nonnegative
        CHECK (window_number > 0 AND docs >= 0 AND chunks >= 0)
)
"""

_CREATE_COMMITTED_SPILLS = """
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
        REFERENCES dlightrag_runs (owner_id, run_id) ON DELETE CASCADE
)
"""

# Agent Session memory: the notes a Session owns (ADR 0022). It cascades with its
# Session, which is deleted only when no routing row references it, and it holds no
# reference to the Run that wrote a note: memory outlives the Run that wrote it, so
# the writer is attribution rather than a dependency whose cleanup reaches it.
_CREATE_SESSION_NOTES = """
CREATE TABLE IF NOT EXISTS dlightrag_answer_session_notes (
    owner_id           TEXT        NOT NULL,
    session_id         UUID        NOT NULL,
    relative_path      TEXT        NOT NULL,
    size_bytes         BIGINT      NOT NULL,
    content_digest     TEXT        NOT NULL,
    content            BYTEA       NOT NULL,
    revision           BIGINT      NOT NULL,
    written_by_run_id  UUID        NOT NULL,
    updated_at         TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (owner_id, session_id, relative_path),
    FOREIGN KEY (owner_id, session_id)
        REFERENCES dlightrag_agent_sessions (owner_id, session_id) ON DELETE CASCADE,
    CONSTRAINT dlightrag_answer_session_notes_size_check
        CHECK (size_bytes = octet_length(content)),
    CONSTRAINT dlightrag_answer_session_notes_digest_check
        CHECK (length(content_digest) = 64),
    CONSTRAINT dlightrag_answer_session_notes_revision_check CHECK (revision >= 1)
)
"""

# The baseline bakes the current schema directly into CREATE statements, so a fresh
# database is complete once it has run. Later migrations advance databases
# initialized before them, and each is a no-op on a fresh baseline; an integration
# test holds both properties. Nothing here migrates a pre-RunRuntime schema.

RUN_MIGRATIONS = (
    Migration(
        "run_runtime_v1",
        "Create the operation-neutral RunRuntime schema",
        (
            _CREATE_RUNS,
            _CREATE_EVENTS,
            _ENFORCE_RUN_EVENT_CONSTRAINTS,
            _CREATE_RUN_EVENT_CONSTRAINT_TRIGGER,
            _CREATE_SESSIONS,
            _CREATE_ENTRIES,
            _CREATE_SESSION_REGISTERS,
            _CREATE_STAGES,
            _CREATE_EVIDENCE,
            _CREATE_RESOURCES,
            _CREATE_BLOBS,
            _CREATE_BLOB_CHUNKS,
            _CREATE_RUN_ARTIFACTS,
            _CREATE_ROUTING,
            _CREATE_CHILD_SESSIONS,
            _CREATE_AGENT_CONTROLS,
            _CREATE_CHILD_OPERATIONS,
            _CREATE_CHILD_GUIDANCE,
            *_CREATE_INDEXES,
            # Its foreign key to dlightrag_runs (run_id) is backed by the unique global-id
            # index created above. On databases created before the baseline dropped the
            # duplicate UNIQUE (run_id), the key is backed by that constraint's index,
            # dlightrag_runs_run_id_key, instead: repoint the key before dropping it.
            _CREATE_CORPUS_MUTATION_WINDOWS,
            _CREATE_WORKSPACE_INVENTORY,
            _CREATE_ARTIFACT_ATTACHMENT_ORDER,
            _CREATE_ARTIFACT_ATTACHMENTS,
            _CREATE_COMMITTED_SPILLS,
            _CREATE_SESSION_NOTES,
            *MEMORY_SETTINGS_DDL,
        ),
    ),
    Migration(
        "child_roster_index",
        "Index child sessions for bounded newest-first roster pages",
        (_CHILD_ROSTER_INDEX.ddl,),
    ),
    Migration(
        "worker_cancel_pending_index",
        "Index bounded worker-local cancellation rescans",
        (_RUNS_CANCEL_PENDING_INDEX.ddl,),
    ),
    Migration(
        "write_model_published_artifact_kind",
        "Restrict durable output references to the unified publication kind",
        (
            "ALTER TABLE dlightrag_answer_run_artifacts "
            "DROP CONSTRAINT dlightrag_answer_run_artifacts_kind_check",
            "ALTER TABLE dlightrag_answer_run_artifacts "
            "ADD CONSTRAINT dlightrag_answer_run_artifacts_kind_check "
            "CHECK (reference_kind IN "
            "('current_attachment', 'history_attachment', 'fetched_resource', "
            "'published_artifact'))",
        ),
    ),
    Migration(
        "write_model_root_artifact_attachments",
        "Stage root Artifact attachments through settled Agent tool effects",
        (_CREATE_ARTIFACT_ATTACHMENT_ORDER, _CREATE_ARTIFACT_ATTACHMENTS),
    ),
    Migration(
        "write_model_web_resource_catalog",
        "Persist fetched Web locators for process-independent resource recovery",
        (
            "ALTER TABLE dlightrag_answer_resources ADD COLUMN IF NOT EXISTS source_locator BYTEA",
            "ALTER TABLE dlightrag_answer_resources "
            "DROP CONSTRAINT dlightrag_answer_resources_blob_link_check",
            "ALTER TABLE dlightrag_answer_resources "
            "ADD CONSTRAINT dlightrag_answer_resources_blob_link_check "
            "CHECK ((kind = 'accepted_blob' AND blob_digest IS NOT NULL) "
            "OR (kind = 'fetched_blob' AND blob_digest IS NOT NULL "
            "AND locator_digest IS NOT NULL "
            "AND (capabilities->>'resource_kind' IS DISTINCT FROM 'web' "
            "OR source_locator IS NOT NULL)) "
            "OR (kind = 'evidence' AND locator_digest IS NOT NULL) "
            "OR (kind = 'committed_spill' AND blob_digest IS NULL "
            "AND locator_digest IS NULL))",
        ),
    ),
    Migration(
        "corpus_mutation_runtime",
        "Add fenced Corpus Mutation checkpoints and remove the legacy ingest lifecycle",
        (
            "ALTER TABLE dlightrag_runs ADD COLUMN IF NOT EXISTS handoff_started_at TIMESTAMPTZ",
            "ALTER TABLE dlightrag_runs ADD COLUMN IF NOT EXISTS superseded_by_run_id UUID",
            _CREATE_CORPUS_MUTATION_WINDOWS,
            _RUNS_MUTATION_FIFO_INDEX.ddl,
        ),
    ),
    Migration(
        "normalize_run_event_constraints",
        "Enforce event sequence, parent lifecycle, and terminal payload integrity",
        (_ENFORCE_RUN_EVENT_CONSTRAINTS, _CREATE_RUN_EVENT_CONSTRAINT_TRIGGER),
    ),
    Migration(
        "remove_run_active_permit",
        "Remove the obsolete Run occupancy column",
        (
            "ALTER TABLE dlightrag_runs "
            "DROP CONSTRAINT IF EXISTS dlightrag_runs_permit_check, "
            "DROP COLUMN IF EXISTS active_permit",
        ),
    ),
    Migration(
        "interactive_child_async_lifecycle",
        "Persist Child cancellation intent before closing its Agent Operation",
        (
            "ALTER TABLE dlightrag_answer_child_sessions "
            "ADD COLUMN IF NOT EXISTS cancel_requested_at TIMESTAMPTZ",
        ),
    ),
    Migration(
        "interactive_child_controls",
        "Address Child controls and persist same-Session Operations and guidance",
        (
            "ALTER TABLE dlightrag_agent_controls "
            "ADD COLUMN IF NOT EXISTS target_session_id UUID, "
            "ADD COLUMN IF NOT EXISTS target_operation_id UUID, "
            "ADD COLUMN IF NOT EXISTS origin TEXT NOT NULL DEFAULT 'user', "
            "ADD COLUMN IF NOT EXISTS submission_key TEXT, "
            "ADD COLUMN IF NOT EXISTS request_fingerprint TEXT",
            "ALTER TABLE dlightrag_agent_controls "
            "DROP CONSTRAINT IF EXISTS dlightrag_agent_controls_target_check, "
            "ADD CONSTRAINT dlightrag_agent_controls_target_check CHECK ("
            "(target_session_id IS NULL AND target_operation_id IS NULL) OR "
            "(target_session_id IS NOT NULL AND target_operation_id IS NOT NULL))",
            "ALTER TABLE dlightrag_agent_controls "
            "DROP CONSTRAINT IF EXISTS dlightrag_agent_controls_origin_check, "
            "ADD CONSTRAINT dlightrag_agent_controls_origin_check "
            "CHECK (origin IN ('user', 'parent'))",
            _CREATE_CHILD_OPERATIONS,
            _CREATE_CHILD_GUIDANCE,
            _CHILD_OPERATIONS_STATUS_INDEX.ddl,
            _CHILD_CONTROLS_SUBMISSION_INDEX.ddl,
            _PARENT_CONTROLS_SUBMISSION_INDEX.ddl,
            _CHILD_GUIDANCE_PENDING_INDEX.ddl,
        ),
    ),
    Migration(
        "child_cancel_submission_receipts",
        "Retain consumed owner cancellation controls bound to their original Operation",
        (
            "ALTER TABLE dlightrag_agent_controls "
            "DROP CONSTRAINT IF EXISTS dlightrag_agent_controls_kind_check, "
            "ADD CONSTRAINT dlightrag_agent_controls_kind_check "
            "CHECK (kind IN ('steer', 'follow_up', 'cancel'))",
        ),
    ),
    Migration(
        "attachment_occurrence_reference_index",
        "Find retained exact Entry occurrences without scanning an owner's Run catalogue",
        (_ATTACHMENT_OCCURRENCE_INDEX.ddl,),
    ),
    Migration(
        "write_model_fork_points",
        "Record the Lane head and projection each Answer Run settled at",
        (
            "ALTER TABLE dlightrag_answer_run_routing "
            "ADD COLUMN IF NOT EXISTS fork_point_entry_id TEXT",
            "ALTER TABLE dlightrag_answer_run_routing "
            "ADD COLUMN IF NOT EXISTS fork_point_projection_id TEXT",
        ),
    ),
    Migration(
        "answer_session_notes",
        "Give each Agent Session a note plane, so memory outlives the Run that wrote it",
        (_CREATE_SESSION_NOTES,),
    ),
    Migration(
        "published_artifact_resources",
        "Register a published Artifact as a Resource its Session may adopt",
        (
            "ALTER TABLE dlightrag_answer_resources "
            "DROP CONSTRAINT dlightrag_answer_resources_kind_check",
            "ALTER TABLE dlightrag_answer_resources "
            "ADD CONSTRAINT dlightrag_answer_resources_kind_check "
            "CHECK (kind IN ('accepted_blob', 'evidence', 'fetched_blob', 'committed_spill', "
            "'published_artifact'))",
            "ALTER TABLE dlightrag_answer_resources "
            "DROP CONSTRAINT dlightrag_answer_resources_blob_link_check",
            "ALTER TABLE dlightrag_answer_resources "
            "ADD CONSTRAINT dlightrag_answer_resources_blob_link_check "
            "CHECK ((kind = 'accepted_blob' AND blob_digest IS NOT NULL) "
            "OR (kind = 'fetched_blob' "
            "AND blob_digest IS NOT NULL AND locator_digest IS NOT NULL "
            "AND (capabilities->>'resource_kind' IS DISTINCT FROM 'web' "
            "OR source_locator IS NOT NULL)) "
            "OR (kind = 'evidence' AND locator_digest IS NOT NULL) "
            "OR (kind = 'committed_spill' "
            "AND blob_digest IS NULL AND locator_digest IS NULL) "
            "OR (kind = 'published_artifact' AND blob_digest IS NOT NULL))",
        ),
    ),
    Migration(
        "artifact_attachment_video_presentation",
        "Admit a playable video Attachment presentation",
        (
            "ALTER TABLE dlightrag_answer_artifact_attachments "
            "DROP CONSTRAINT dlightrag_answer_artifact_attachments_presentation_check",
            "ALTER TABLE dlightrag_answer_artifact_attachments "
            "ADD CONSTRAINT dlightrag_answer_artifact_attachments_presentation_check "
            "CHECK (presentation IN "
            "('image', 'video', 'markdown', 'html', 'pdf', 'text', 'download'))",
        ),
    ),
)

_RUN_TABLES = (
    TableRequirement(
        name="dlightrag_answer_session_notes",
        columns=(
            "owner_id",
            "session_id",
            "relative_path",
            "size_bytes",
            "content_digest",
            "content",
            "revision",
            "written_by_run_id",
            "updated_at",
        ),
        primary_key=("owner_id", "session_id", "relative_path"),
        foreign_keys=(
            ForeignKeyRequirement(
                columns=("owner_id", "session_id"), references="dlightrag_agent_sessions"
            ),
        ),
        checks=(
            "dlightrag_answer_session_notes_size_check",
            "dlightrag_answer_session_notes_digest_check",
            "dlightrag_answer_session_notes_revision_check",
        ),
    ),
    TableRequirement(
        name="dlightrag_runs",
        columns=(
            "owner_id",
            "run_id",
            "run_kind",
            "lane",
            "submitted_by",
            "access_scope_kind",
            "submission_key",
            "prepared_input_json",
            "accepted_input_json",
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
            "retention_seconds",
            "purge_after",
            "next_attempt_at",
            "checkpoint_json",
            "handoff_started_at",
            "superseded_by_run_id",
            "created_at",
            "updated_at",
            "started_at",
            "finished_at",
            "agent_workspace_epoch",
        ),
        primary_key=("owner_id", "run_id"),
        checks=(
            "dlightrag_runs_kind_check",
            "dlightrag_runs_lane_check",
            "dlightrag_runs_scope_check",
            "dlightrag_runs_status_check",
            "dlightrag_runs_counter_check",
            "dlightrag_runs_lease_check",
            "dlightrag_runs_terminal_check",
            "dlightrag_runs_result_check",
            "dlightrag_runs_error_check",
            "dlightrag_runs_prepared_input_check",
            "dlightrag_runs_workspace_epoch_check",
        ),
    ),
    TableRequirement(
        name="dlightrag_corpus_mutation_windows",
        columns=("run_id", "window_number", "workspace", "docs", "chunks", "created_at"),
        primary_key=("run_id", "window_number"),
        foreign_keys=(ForeignKeyRequirement(columns=("run_id",), references="dlightrag_runs"),),
        checks=("dlightrag_corpus_mutation_windows_nonnegative",),
    ),
    TableRequirement(
        name="dlightrag_run_events",
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
            ForeignKeyRequirement(columns=("owner_id", "run_id"), references="dlightrag_runs"),
        ),
        checks=("dlightrag_run_events_sequence_check",),
        triggers=("trg_dlightrag_run_events_enforce",),
    ),
    TableRequirement(
        name="dlightrag_agent_sessions",
        columns=(
            "owner_id",
            "session_id",
            "lease_run_id",
            "commit_sequence",
            "fencing_epoch",
            "last_sequence",
            "created_at",
            "updated_at",
        ),
        primary_key=("owner_id", "session_id"),
        checks=(
            "dlightrag_agent_sessions_commit_sequence_check",
            "dlightrag_agent_sessions_fencing_check",
            "dlightrag_agent_sessions_sequence_check",
        ),
    ),
    TableRequirement(
        name="dlightrag_agent_session_entries",
        columns=(
            "owner_id",
            "session_id",
            "sequence",
            "entry_id",
            "parent_entry_id",
            "entry_type",
            "schema_version",
            "timestamp",
            "payload_json",
        ),
        primary_key=("owner_id", "session_id", "sequence"),
        unique=(
            ("entry_id",),
            ("owner_id", "session_id", "entry_id"),
        ),
        foreign_keys=(
            ForeignKeyRequirement(
                columns=("owner_id", "session_id"),
                references="dlightrag_agent_sessions",
            ),
            ForeignKeyRequirement(
                columns=("owner_id", "session_id", "parent_entry_id"),
                references="dlightrag_agent_session_entries",
            ),
        ),
        checks=(
            "dlightrag_agent_session_entries_sequence_check",
            "dlightrag_agent_session_entries_type_check",
            "dlightrag_agent_session_entries_version_check",
        ),
    ),
    TableRequirement(
        name="dlightrag_agent_session_registers",
        columns=(
            "owner_id",
            "session_id",
            "register_kind",
            "register_key",
            "sequence",
            "payload_json",
        ),
        primary_key=(
            "owner_id",
            "session_id",
            "register_kind",
            "register_key",
        ),
        foreign_keys=(
            ForeignKeyRequirement(
                columns=("owner_id", "session_id"),
                references="dlightrag_agent_sessions",
            ),
        ),
        checks=(
            "dlightrag_agent_session_registers_kind_check",
            "dlightrag_agent_session_registers_sequence_check",
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
            ForeignKeyRequirement(columns=("owner_id", "run_id"), references="dlightrag_runs"),
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
            ForeignKeyRequirement(columns=("owner_id", "run_id"), references="dlightrag_runs"),
        ),
        checks=(
            "dlightrag_answer_evidence_ordinal_check",
            "dlightrag_answer_evidence_digest_check",
        ),
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
            "source_locator",
            "session_id",
            "intent_id",
            "result_ordinal",
            "created_at",
        ),
        primary_key=("owner_id", "run_id", "resource_id"),
        foreign_keys=(
            ForeignKeyRequirement(columns=("owner_id", "run_id"), references="dlightrag_runs"),
        ),
        checks=(
            "dlightrag_answer_resources_kind_check",
            "dlightrag_answer_resources_blob_link_check",
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
            ForeignKeyRequirement(columns=("owner_id", "run_id"), references="dlightrag_runs"),
            ForeignKeyRequirement(columns=("owner_id", "digest"), references="dlightrag_blobs"),
        ),
        checks=(
            "dlightrag_answer_run_artifacts_kind_check",
            "dlightrag_answer_run_artifacts_ordinal_check",
        ),
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
            ForeignKeyRequirement(columns=("owner_id", "run_id"), references="dlightrag_runs"),
        ),
    ),
    TableRequirement(
        name="dlightrag_answer_artifact_attachments",
        columns=(
            "owner_id",
            "run_id",
            "relative_path",
            "label",
            "content_digest",
            "size_bytes",
            "presentation",
            "session_id",
            "intent_id",
            "attachment_order",
            "attached_at",
        ),
        primary_key=("owner_id", "run_id", "relative_path"),
        foreign_keys=(
            ForeignKeyRequirement(columns=("owner_id", "run_id"), references="dlightrag_runs"),
        ),
        checks=(
            "dlightrag_answer_artifact_attachments_digest_check",
            "dlightrag_answer_artifact_attachments_size_check",
            "dlightrag_answer_artifact_attachments_presentation_check",
        ),
    ),
    TableRequirement(
        name="dlightrag_answer_run_routing",
        columns=(
            "owner_id",
            "run_id",
            "requested_mode",
            "valid_modes",
            "resolved_mode",
            "model_fingerprints",
            "context_policy_revision",
            "agent_session_id",
            "agent_lane_id",
            "source_lane_id",
            "fork_point_entry_id",
            "fork_point_projection_id",
            "created_at",
            "updated_at",
        ),
        primary_key=("owner_id", "run_id"),
        foreign_keys=(
            ForeignKeyRequirement(columns=("owner_id", "run_id"), references="dlightrag_runs"),
        ),
        checks=(
            "dlightrag_answer_run_routing_requested_check",
            "dlightrag_answer_run_routing_valid_check",
            "dlightrag_answer_run_routing_resolved_check",
        ),
    ),
    TableRequirement(
        name="dlightrag_answer_child_sessions",
        columns=(
            "owner_id",
            "run_id",
            "child_session_id",
            "parent_session_id",
            "parent_call_id",
            "parent_intent_id",
            "status",
            "cancel_requested_at",
            "summary",
            "objective",
            "context_mode",
            "model_role",
            "tools_json",
            "usage_json",
            "depth",
            "context_snapshot_json",
            "plan_json",
            "budget_json",
            "host_state_json",
            "lease_owner",
            "lease_expires_at",
            "fencing_epoch",
            "created_at",
            "updated_at",
        ),
        primary_key=("owner_id", "run_id", "child_session_id"),
        foreign_keys=(
            ForeignKeyRequirement(columns=("owner_id", "run_id"), references="dlightrag_runs"),
        ),
        checks=(
            "dlightrag_answer_child_sessions_status_check",
            "dlightrag_answer_child_sessions_depth_check",
            "dlightrag_answer_child_sessions_fencing_check",
        ),
    ),
    TableRequirement(
        name="dlightrag_agent_controls",
        columns=(
            "owner_id",
            "run_id",
            "control_sequence",
            "kind",
            "content",
            "target_session_id",
            "target_operation_id",
            "origin",
            "submission_key",
            "request_fingerprint",
            "consumed_at",
            "created_at",
        ),
        primary_key=("owner_id", "run_id", "control_sequence"),
        foreign_keys=(
            ForeignKeyRequirement(columns=("owner_id", "run_id"), references="dlightrag_runs"),
        ),
        checks=(
            "dlightrag_agent_controls_sequence_check",
            "dlightrag_agent_controls_kind_check",
            "dlightrag_agent_controls_content_check",
            "dlightrag_agent_controls_target_check",
            "dlightrag_agent_controls_origin_check",
        ),
    ),
    TableRequirement(
        name="dlightrag_answer_child_operations",
        columns=(
            "owner_id",
            "run_id",
            "child_session_id",
            "operation_sequence",
            "operation_id",
            "idempotency_key",
            "request_fingerprint",
            "content",
            "origin",
            "status",
            "cancellation_origin",
            "summary",
            "usage_json",
            "outcome_json",
            "created_at",
            "updated_at",
        ),
        primary_key=("owner_id", "run_id", "child_session_id", "operation_sequence"),
        unique=(
            ("owner_id", "run_id", "child_session_id", "operation_id"),
            ("owner_id", "run_id", "child_session_id", "idempotency_key"),
        ),
        foreign_keys=(
            ForeignKeyRequirement(
                columns=("owner_id", "run_id", "child_session_id"),
                references="dlightrag_answer_child_sessions",
            ),
        ),
        checks=(
            "dlightrag_answer_child_operations_sequence_check",
            "dlightrag_answer_child_operations_status_check",
            "dlightrag_answer_child_operations_origin_check",
            "dlightrag_answer_child_operations_cancellation_origin_check",
            "dlightrag_answer_child_operations_content_check",
        ),
    ),
    TableRequirement(
        name="dlightrag_answer_child_guidance",
        columns=(
            "owner_id",
            "run_id",
            "request_id",
            "child_session_id",
            "child_operation_id",
            "parent_session_id",
            "question",
            "status",
            "reply",
            "reply_origin",
            "reply_submission_key",
            "reply_fingerprint",
            "expires_at",
            "replied_at",
            "created_at",
            "updated_at",
        ),
        primary_key=("owner_id", "run_id", "request_id"),
        foreign_keys=(
            ForeignKeyRequirement(
                columns=("owner_id", "run_id", "child_session_id"),
                references="dlightrag_answer_child_sessions",
            ),
        ),
        checks=(
            "dlightrag_answer_child_guidance_status_check",
            "dlightrag_answer_child_guidance_question_check",
            "dlightrag_answer_child_guidance_reply_check",
            "dlightrag_answer_child_guidance_reply_origin_check",
        ),
    ),
    MEMORY_SETTINGS_SCHEMA_TABLE,
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
            ForeignKeyRequirement(columns=("owner_id", "run_id"), references="dlightrag_runs"),
        ),
    ),
)

# What a reader verifies: every table above, plus every index the scope declares.
RUN_SCHEMA_TABLES = tuple(table.with_indexes(_RUN_INDEXES) for table in _RUN_TABLES)

#: ``(expression, output name)`` for every column :func:`run_record` reads.
_RUN_COLUMN_SPECS: tuple[tuple[str, str], ...] = (
    ("owner_id", "owner_id"),
    ("run_id::text", "run_id"),
    ("run_kind", "run_kind"),
    ("lane", "lane"),
    ("submitted_by", "submitted_by"),
    ("access_scope_kind", "access_scope_kind"),
    ("submission_key", "submission_key"),
    ("request_fingerprint", "request_fingerprint"),
    ("prepared_input_json", "prepared_input"),
    ("accepted_input_json", "accepted_input"),
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
    ("purge_after", "purge_after"),
    ("next_attempt_at", "next_attempt_at"),
    ("checkpoint_json", "checkpoint_json"),
    ("handoff_started_at", "handoff_started_at"),
    ("superseded_by_run_id::text", "superseded_by_run_id"),
    ("agent_workspace_epoch", "agent_workspace_epoch"),
)


def run_columns(alias: str = "") -> str:
    """Project one run row's columns, optionally through a join alias."""
    prefix = f"{alias}." if alias else ""
    return ",\n".join(f"{prefix}{expression} AS {name}" for expression, name in _RUN_COLUMN_SPECS)


_RUN_COLUMNS = run_columns()
_LIST_RUNS = f"""
SELECT {_RUN_COLUMNS}
FROM dlightrag_runs
WHERE owner_id = $1 ORDER BY created_at, run_id LIMIT $2
"""  # noqa: S608 - interpolates only the trusted _RUN_COLUMNS constant
_LIST_RUNS_AFTER = f"""
SELECT {_RUN_COLUMNS}
FROM dlightrag_runs
WHERE owner_id = $1 AND (created_at, run_id) > (
 SELECT created_at, run_id FROM dlightrag_runs
 WHERE owner_id = $1 AND run_id = $2)
ORDER BY created_at, run_id LIMIT $3
"""  # noqa: S608 - interpolates only the trusted _RUN_COLUMNS constant

_ACTIVE_REQUIREMENTS_FRONTIER = """
SELECT created_at, run_id
FROM dlightrag_runs
WHERE run_kind IN ('answer', 'retrieval')
  AND status IN ('queued', 'running')
  AND cancel_requested_at IS NULL
  AND NOT (status = 'running' AND lease_expires_at < NOW()
           AND reclaims_without_progress >= $1)
ORDER BY created_at DESC, run_id DESC
LIMIT 1
"""
_ACTIVE_REQUIREMENTS_FIRST_PAGE = """
SELECT created_at, run_id, run_kind, prepared_input_json
FROM dlightrag_runs
WHERE run_kind IN ('answer', 'retrieval')
  AND status IN ('queued', 'running')
  AND cancel_requested_at IS NULL
  AND NOT (status = 'running' AND lease_expires_at < NOW()
           AND reclaims_without_progress >= $1)
  AND (created_at, run_id) <= ($2::timestamptz, $3::uuid)
ORDER BY created_at, run_id
LIMIT $4
"""
_ACTIVE_REQUIREMENTS_AFTER = """
SELECT created_at, run_id, run_kind, prepared_input_json
FROM dlightrag_runs
WHERE run_kind IN ('answer', 'retrieval')
  AND status IN ('queued', 'running')
  AND cancel_requested_at IS NULL
  AND NOT (status = 'running' AND lease_expires_at < NOW()
           AND reclaims_without_progress >= $1)
  AND (created_at, run_id) <= ($2::timestamptz, $3::uuid)
  AND (created_at, run_id) > ($4::timestamptz, $5::uuid)
ORDER BY created_at, run_id
LIMIT $6
"""

_CANCEL_PENDING_FRONTIER = """
SELECT created_at, run_id
FROM dlightrag_runs
WHERE cancel_requested_at IS NOT NULL
  AND status = 'running'
  AND lease_owner = $1
  AND lease_expires_at > NOW()
ORDER BY created_at DESC, run_id DESC
LIMIT 1
"""
_CANCEL_PENDING_FIRST_PAGE = """
SELECT owner_id, run_id, created_at
FROM dlightrag_runs
WHERE cancel_requested_at IS NOT NULL
  AND status = 'running'
  AND lease_owner = $1
  AND lease_expires_at > NOW()
  AND (created_at, run_id) <= ($2::timestamptz, $3::uuid)
ORDER BY created_at, run_id
LIMIT $4
"""
_CANCEL_PENDING_AFTER = """
SELECT owner_id, run_id, created_at
FROM dlightrag_runs
WHERE cancel_requested_at IS NOT NULL
  AND status = 'running'
  AND lease_owner = $1
  AND lease_expires_at > NOW()
  AND (created_at, run_id) <= ($2::timestamptz, $3::uuid)
  AND (created_at, run_id) > ($4::timestamptz, $5::uuid)
ORDER BY created_at, run_id
LIMIT $6
"""

_INSERT_RUN = f"""
INSERT INTO dlightrag_runs (
    owner_id, run_id, run_kind, lane, submitted_by, access_scope_kind,
    submission_key, prepared_input_json, accepted_input_json,
    request_fingerprint, retention_seconds
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9::jsonb, $10, $11)
ON CONFLICT (run_kind, submitted_by, submission_key) DO NOTHING
RETURNING {_RUN_COLUMNS}
"""  # noqa: S608 - interpolates only the trusted _RUN_COLUMNS constant

_SELECT_RUN_BY_KEY = f"""
SELECT {_RUN_COLUMNS}
FROM dlightrag_runs
WHERE run_kind = $1 AND submitted_by = $2 AND submission_key = $3
"""  # noqa: S608 - interpolates only the trusted _RUN_COLUMNS constant

_COUNT_NONTERMINAL_LANE = """
SELECT COUNT(*) FROM dlightrag_runs
WHERE lane = $1 AND status IN ('queued', 'running')
"""

_SELECT_RUN = f"""
SELECT {_RUN_COLUMNS}
FROM dlightrag_runs
WHERE owner_id = $1 AND run_id = $2
"""  # noqa: S608 - interpolates only the trusted _RUN_COLUMNS constant

_SELECT_RUN_FOR_UPDATE = f"""
SELECT {_RUN_COLUMNS}
FROM dlightrag_runs
WHERE owner_id = $1 AND run_id = $2
FOR UPDATE
"""  # noqa: S608 - interpolates only the trusted _RUN_COLUMNS constant

_SELECT_RUN_GLOBAL = f"""
SELECT {_RUN_COLUMNS}
FROM dlightrag_runs
WHERE run_id = $1
"""  # noqa: S608 - interpolates only the trusted _RUN_COLUMNS constant

_RECORD_CORPUS_WINDOW = """
WITH inserted AS (
    INSERT INTO dlightrag_corpus_mutation_windows (
        run_id, window_number, workspace, docs, chunks
    ) VALUES ($1, $2, $3, $4, $5)
    ON CONFLICT (run_id, window_number) DO NOTHING
    RETURNING 1
), updated AS (
    UPDATE dlightrag_workspace_meta
    SET ingested_docs_total = ingested_docs_total + $4,
        ingested_chunks_total = ingested_chunks_total + $5,
        promotion_state = CASE
            WHEN promotion_state = 'none' AND storage_tier = 'shared'
             AND (($6::bigint IS NOT NULL AND ingested_docs_total + $4 >= $6)
               OR ($7::bigint IS NOT NULL AND ingested_chunks_total + $5 >= $7))
            THEN 'pending' ELSE promotion_state END,
        updated_at = NOW()
    WHERE workspace = $3 AND EXISTS (SELECT 1 FROM inserted)
    RETURNING promotion_state
), promoted AS (
    INSERT INTO dlightrag_promotion_jobs (workspace, state)
    SELECT $3, 'pending' FROM updated WHERE promotion_state = 'pending'
    ON CONFLICT DO NOTHING
)
SELECT EXISTS (SELECT 1 FROM inserted)
"""

_SELECT_EVENTS = """
SELECT event_sequence, event_type, payload, created_at
FROM dlightrag_run_events
WHERE owner_id = $1 AND run_id = $2 AND event_sequence > $3
ORDER BY event_sequence
LIMIT $4
"""

_SELECT_CLAIM_CANDIDATE = """
SELECT r.owner_id, r.run_id
FROM dlightrag_runs r
WHERE r.run_kind = ANY($2::text[])
  AND r.lane = ANY($3::text[])
  AND r.cancel_requested_at IS NULL
  AND (r.next_attempt_at IS NULL OR r.next_attempt_at <= NOW())
  AND (
      r.status = 'queued'
      OR (r.status = 'running' AND r.lease_expires_at < NOW()
          AND r.reclaims_without_progress < $1)
  )
  AND (
      r.lane <> 'corpus_mutation'
      OR NOT EXISTS (
          SELECT 1 FROM dlightrag_runs earlier
          WHERE earlier.lane = 'corpus_mutation'
            AND earlier.owner_id = r.owner_id
            AND earlier.status IN ('queued', 'running')
            AND (earlier.created_at, earlier.run_id) < (r.created_at, r.run_id)
      )
  )
ORDER BY r.created_at, r.run_id
LIMIT 1
FOR UPDATE OF r SKIP LOCKED
"""

_LOCK_AGENT_SESSION_IF_PRESENT = """
SELECT 1
FROM dlightrag_agent_sessions
WHERE owner_id = $1 AND session_id = $2
FOR UPDATE
"""

_INSERT_ROUTING = """
INSERT INTO dlightrag_answer_run_routing (
    owner_id, run_id, requested_mode, valid_modes, resolved_mode,
    model_fingerprints, context_policy_revision,
    agent_session_id, agent_lane_id, source_lane_id
)
VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, $9, $10)
"""

_SELECT_ROUTING = """
SELECT requested_mode, valid_modes, resolved_mode,
       agent_session_id::text, agent_lane_id, source_lane_id,
       fork_point_entry_id, fork_point_projection_id
FROM dlightrag_answer_run_routing
WHERE owner_id = $1 AND run_id = $2
"""

_RECORD_FORK_POINT = """
UPDATE dlightrag_answer_run_routing AS rt
SET fork_point_entry_id = $5,
    fork_point_projection_id = $6,
    updated_at = NOW()
FROM dlightrag_runs AS r
WHERE rt.owner_id = r.owner_id AND rt.run_id = r.run_id
  AND rt.owner_id = $1 AND rt.run_id = $2
  AND r.lease_owner = $3 AND r.fencing_epoch = $4
  AND r.status = 'running' AND r.lease_expires_at > NOW()
RETURNING TRUE
"""

_RESOLVE_ROUTING = """
UPDATE dlightrag_answer_run_routing AS rt
SET resolved_mode = $5,
    updated_at = NOW()
FROM dlightrag_runs AS r
WHERE rt.owner_id = r.owner_id AND rt.run_id = r.run_id
  AND rt.owner_id = $1 AND rt.run_id = $2
  AND r.lease_owner = $3 AND r.fencing_epoch = $4
  AND r.status = 'running' AND r.lease_expires_at > NOW()
  AND (rt.resolved_mode IS NULL OR rt.resolved_mode = $5)
RETURNING rt.resolved_mode
"""

_CLAIM_RUN = f"""
UPDATE dlightrag_runs
SET status = 'running',
    lease_owner = $3,
    lease_expires_at = NOW() + ($4 * INTERVAL '1 second'),
    fencing_epoch = fencing_epoch + 1,
    reclaims_without_progress = $5,
    last_reclaim_progress_version = $6,
    started_at = COALESCE(started_at, NOW()),
    next_attempt_at = NULL,
    updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2
  AND status IN ('queued', 'running')
RETURNING {_RUN_COLUMNS}
"""  # noqa: S608 - interpolates only the trusted _RUN_COLUMNS constant

_HEARTBEAT = """
UPDATE dlightrag_runs
SET lease_expires_at = NOW() + ($5 * INTERVAL '1 second'),
    updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2
  AND lease_owner = $3 AND fencing_epoch = $4
  AND status = 'running' AND lease_expires_at > NOW()
RETURNING (cancel_requested_at IS NOT NULL) AS cancel_requested
"""

_WRITE_CHECKPOINT = """
UPDATE dlightrag_runs
SET checkpoint_json = $5::jsonb,
    phase = COALESCE($6::text, phase),
    durable_progress_version = durable_progress_version + 1,
    lease_expires_at = NOW() + ($7 * INTERVAL '1 second'),
    updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2
  AND lease_owner = $3 AND fencing_epoch = $4
  AND status = 'running' AND lease_expires_at > NOW()
RETURNING 1
"""

_START_HANDOFF = """
UPDATE dlightrag_runs
SET handoff_started_at = COALESCE(handoff_started_at, NOW()),
    checkpoint_json = $5::jsonb,
    durable_progress_version = durable_progress_version + 1,
    lease_expires_at = NOW() + ($6 * INTERVAL '1 second'),
    updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2
  AND lease_owner = $3 AND fencing_epoch = $4
  AND status = 'running' AND lease_expires_at > NOW()
  AND (handoff_started_at IS NOT NULL OR cancel_requested_at IS NULL)
RETURNING 1
"""

_APPEND_EVENT = """
WITH bumped AS (
    UPDATE dlightrag_runs
    SET next_event_sequence = next_event_sequence + 1,
        phase = COALESCE($5::text, phase),
        lease_expires_at = NOW() + ($8 * INTERVAL '1 second'),
        updated_at = NOW()
    WHERE owner_id = $1 AND run_id = $2
      AND lease_owner = $3 AND fencing_epoch = $4
      AND status = 'running' AND lease_expires_at > NOW()
    RETURNING next_event_sequence - 1 AS event_sequence
), inserted AS (
    INSERT INTO dlightrag_run_events (
        owner_id, run_id, event_sequence, event_type, payload
    )
    SELECT $1, $2, event_sequence, $6::text, $7::jsonb FROM bumped
    RETURNING event_sequence
)
SELECT event_sequence FROM inserted
"""

_FINALIZE_UNLEASED = f"""
WITH bumped AS (
    UPDATE dlightrag_runs AS r
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
        purge_after = NOW() + make_interval(secs => retention_seconds::double precision),
        updated_at = NOW(),
        next_event_sequence = r.next_event_sequence + 1
    WHERE (r.owner_id, r.run_id) IN (SELECT * FROM unnest($1::text[], $2::uuid[]))
      AND r.status IN ('queued', 'running')
      AND (r.lease_expires_at IS NULL OR r.lease_expires_at < NOW())
    RETURNING r.owner_id, r.run_id, r.next_event_sequence - 1 AS event_sequence
), {SETTLE_TERMINATED_RUN_CHILDREN}, inserted AS (
    INSERT INTO dlightrag_run_events (
        owner_id, run_id, event_sequence, event_type, payload
    )
    SELECT owner_id, run_id, event_sequence, $6::text, $7::jsonb FROM bumped
    RETURNING event_sequence
)
SELECT count(*)::int FROM inserted
"""  # noqa: S608 - interpolates only the trusted SETTLE_TERMINATED_RUN_CHILDREN constant

_SUPERSEDE_WAITING_MUTATION = """
WITH updated AS (
    UPDATE dlightrag_runs
    SET status = 'failed', phase = NULL,
        error_kind = 'repair_superseded',
        error_message = 'Waiting mutation was superseded by an authorized Corpus Reset.',
        result_json = jsonb_build_object(
            'action', COALESCE(accepted_input_json->>'action', 'unknown'),
            'superseded_by_run_id', $3::uuid,
            'repair_reason', 'Superseded by an authorized full Corpus Reset.'
        ),
        prepared_input_json = NULL,
        superseded_by_run_id = $3::uuid,
        lease_owner = NULL, lease_expires_at = NULL,
        finished_at = NOW(),
        purge_after = NOW() + make_interval(secs => retention_seconds::double precision),
        updated_at = NOW(), next_event_sequence = next_event_sequence + 1
    WHERE owner_id = $1 AND run_id = $2
      AND run_kind = 'corpus_mutation' AND status = 'running'
      AND phase = 'waiting_for_repair'
      AND lease_owner IS NULL AND lease_expires_at IS NULL
    RETURNING owner_id, run_id, next_event_sequence - 1 AS event_sequence,
              result_json
), inserted AS (
    INSERT INTO dlightrag_run_events (
        owner_id, run_id, event_sequence, event_type, payload
    )
    SELECT owner_id, run_id, event_sequence, 'error',
           jsonb_build_object(
               'kind', 'repair_superseded',
               'message', 'Waiting mutation was superseded by an authorized Corpus Reset.',
               'result', result_json
           )
    FROM updated
    RETURNING 1
)
SELECT count(*)::int FROM inserted
"""

_REQUEST_CANCELLATION = """
WITH updated AS (
    UPDATE dlightrag_runs
    SET cancel_requested_at = COALESCE(cancel_requested_at, NOW()),
        updated_at = NOW()
    WHERE owner_id = $1 AND run_id = $2
      AND status = 'running'
      AND handoff_started_at IS NULL
    RETURNING 1
), notified AS (
    SELECT pg_notify($3, $4) FROM updated
)
SELECT count(*)::int FROM notified
"""

_REQUEUE_RUN = """
UPDATE dlightrag_runs
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

_DEFER_RUN = """
UPDATE dlightrag_runs
SET status = 'queued', phase = 'deferred', checkpoint_json = $5::jsonb,
    next_attempt_at = $6, lease_owner = NULL, lease_expires_at = NULL,
    updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2 AND lease_owner = $3
  AND fencing_epoch = $4 AND status = 'running' AND lease_expires_at > NOW()
RETURNING 1
"""

# A Run that is not settled has no state it ended at, and the two columns are written
# by whichever attempt does settle it. Clearing them with the requeue keeps that true
# structurally: a later attempt whose own write fails inherits a refusal, not a head
# some abandoned attempt left behind.
_CLEAR_FORK_POINT = """
UPDATE dlightrag_answer_run_routing
SET fork_point_entry_id = NULL, fork_point_projection_id = NULL, updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2
"""

_RESUME_REPAIR = """
UPDATE dlightrag_runs
SET status = 'queued', phase = 'repair_resumed', next_attempt_at = NOW(),
    checkpoint_json = jsonb_set(
        COALESCE(checkpoint_json, '{}'::jsonb),
        '{repair_resume_confirmed}',
        'true'::jsonb,
        TRUE
    ),
    updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2 AND phase = 'waiting_for_repair'
RETURNING 1
"""

_WAIT_FOR_REPAIR = """
UPDATE dlightrag_runs
SET phase = 'waiting_for_repair', checkpoint_json = $5::jsonb,
    next_attempt_at = NULL, lease_owner = NULL, lease_expires_at = NULL,
    updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2 AND lease_owner = $3
  AND fencing_epoch = $4 AND status = 'running' AND lease_expires_at > NOW()
RETURNING 1
"""

_SELECT_CANCEL_PENDING = """
SELECT owner_id, run_id
FROM dlightrag_runs
WHERE cancel_requested_at IS NOT NULL
  AND status IN ('queued', 'running')
  AND (lease_expires_at IS NULL OR lease_expires_at < NOW())
ORDER BY updated_at
LIMIT $1
FOR UPDATE SKIP LOCKED
"""

_INSERT_RESOURCE = """
INSERT INTO dlightrag_answer_resources (
    owner_id, run_id, resource_id, kind, safe_name, media_type, capabilities,
    ordinal, blob_digest, locator_digest, source_locator,
    session_id, intent_id, result_ordinal
)
VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10, $11, $12, $13, $14)
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

_SELECT_RUN_ARTIFACTS = """
SELECT resource_id, reference_kind, ordinal, digest, filename, mime_type,
       transform_locator, created_at
FROM dlightrag_answer_run_artifacts
WHERE owner_id = $1 AND run_id = $2
ORDER BY reference_kind, ordinal
"""

_SELECT_LINEAGE_RESOURCE = """
SELECT run_id, resource_id, ordinal, blob_digest, safe_name, media_type, source_locator, capabilities
FROM dlightrag_answer_resources
WHERE owner_id = $1 AND session_id = $2 AND blob_digest IS NOT NULL
  AND (
      (resource_id = $3
       AND (kind, capabilities->>'resource_kind') IN (
           SELECT * FROM unnest($5::text[], $6::text[])
       ))
      OR (source_locator = $4::bytea AND kind = 'fetched_blob'
          AND capabilities->>'resource_kind' = ANY(ARRAY['conversion_snapshot', 'conversion_asset']))
  )
ORDER BY (resource_id = $3) DESC, created_at DESC, run_id DESC, resource_id
FOR SHARE
"""

_SELECT_RUN_FETCHED_RESOURCES = """
SELECT resource_id, ordinal, blob_digest, safe_name, media_type, source_locator, capabilities
FROM dlightrag_answer_resources
WHERE owner_id = $1 AND run_id = $2 AND kind = 'fetched_blob'
  AND capabilities->>'resource_kind' IN (
      'web', 'tool_attachment', 'conversion_snapshot', 'conversion_asset', 'lineage_adoption'
  )
  AND ordinal IS NOT NULL AND source_locator IS NOT NULL
ORDER BY ordinal, resource_id
"""

_SELECT_RUN_RESOURCE_BY_ID = """
SELECT resource_id, ordinal, blob_digest, safe_name, media_type, source_locator, capabilities
FROM dlightrag_answer_resources
WHERE owner_id = $1 AND run_id = $2 AND resource_id = $3
"""

_SELECT_ARTIFACT_ATTACHMENTS = """
SELECT relative_path, label, content_digest, size_bytes, presentation,
       session_id::text, intent_id::text
FROM dlightrag_answer_artifact_attachments
WHERE owner_id = $1 AND run_id = $2
ORDER BY attachment_order
"""

_SELECT_RUN_DIGESTS = """
SELECT owner_id, digest
FROM dlightrag_answer_run_artifacts
WHERE (owner_id, run_id) IN (SELECT * FROM unnest($1::text[], $2::uuid[]))
UNION
SELECT owner_id, blob_digest AS digest
FROM dlightrag_answer_resources
WHERE (owner_id, run_id) IN (SELECT * FROM unnest($1::text[], $2::uuid[]))
  AND blob_digest IS NOT NULL
"""

_DELETE_RUNS = """
DELETE FROM dlightrag_runs
WHERE (owner_id, run_id) IN (SELECT * FROM unnest($1::text[], $2::uuid[]))
RETURNING owner_id, run_id, run_kind
"""

_SELECT_RUN_AGENT_SESSIONS = """
SELECT owner_id, agent_session_id AS session_id
FROM dlightrag_answer_run_routing
WHERE (owner_id, run_id) IN (SELECT * FROM unnest($1::text[], $2::uuid[]))
UNION
SELECT owner_id, child_session_id AS session_id
FROM dlightrag_answer_child_sessions
WHERE (owner_id, run_id) IN (SELECT * FROM unnest($1::text[], $2::uuid[]))
"""

_LOCK_AGENT_SESSION_CANDIDATES = """
SELECT owner_id, session_id
FROM dlightrag_agent_sessions
WHERE (owner_id, session_id) IN (
    SELECT * FROM unnest($1::text[], $2::uuid[])
)
ORDER BY owner_id, session_id
FOR UPDATE
"""

_DELETE_UNREFERENCED_AGENT_SESSIONS = """
DELETE FROM dlightrag_agent_sessions AS sessions
WHERE (sessions.owner_id, sessions.session_id) IN (
    SELECT * FROM unnest($1::text[], $2::uuid[])
)
AND NOT EXISTS (
    SELECT 1 FROM dlightrag_answer_run_routing AS routing
    WHERE routing.owner_id = sessions.owner_id
      AND routing.agent_session_id = sessions.session_id
)
RETURNING 1
"""

# Retention order: references first, then blob chunks/metadata only when no
# run/resource reference remains. Resources cascade with their run; orphan
# reference rows are gone with the run row too.
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
FROM dlightrag_runs AS runs
WHERE runs.status IN ('succeeded', 'failed', 'cancelled')
  AND runs.purge_after <= NOW()
ORDER BY runs.purge_after
LIMIT $1
FOR UPDATE OF runs SKIP LOCKED
"""

_SELECT_TRIMMABLE_RUNS = """
SELECT owner_id, run_id
FROM dlightrag_runs
WHERE status IN ('succeeded', 'failed', 'cancelled')
  AND purge_after <= NOW()
  AND events_trimmed_at IS NULL
ORDER BY purge_after
LIMIT $1
FOR UPDATE SKIP LOCKED
"""

_DELETE_EVENTS_FOR_RUNS = """
DELETE FROM dlightrag_run_events
WHERE (owner_id, run_id) IN (SELECT * FROM unnest($1::text[], $2::uuid[]))
"""

_MARK_EVENTS_TRIMMED = """
UPDATE dlightrag_runs
SET events_trimmed_at = NOW(),
    updated_at = NOW()
WHERE (owner_id, run_id) IN (SELECT * FROM unnest($1::text[], $2::uuid[]))
"""


def _deleted_runs(rows: Sequence[Any]) -> tuple[DeletedRun, ...]:
    """Project RETURNING identities into the storage-neutral deletion record."""
    return tuple(
        DeletedRun(
            owner_id=str(row["owner_id"]),
            run_id=str(row["run_id"]),
            run_kind=cast(RunKind, str(row["run_kind"])),
        )
        for row in rows
    )


def _digest_pairs(rows: Sequence[Any]) -> tuple[list[str], list[str]]:
    """Split reference rows into parallel owner and digest lists."""
    owners: list[str] = []
    digests: list[str] = []
    for row in rows:
        owners.append(str(row["owner_id"]))
        digests.append(str(row["digest"]))
    return owners, digests


async def _delete_unreferenced_agent_sessions(conn: Any, rows: Sequence[Any]) -> int:
    candidates = {(str(row["owner_id"]), uuid.UUID(str(row["session_id"]))) for row in rows}
    if not candidates:
        return 0
    ordered = sorted(candidates, key=lambda item: (item[0], str(item[1])))
    owners = [owner for owner, _session_id in ordered]
    session_ids = [session_id for _owner, session_id in ordered]
    await conn.fetch(_LOCK_AGENT_SESSION_CANDIDATES, owners, session_ids)
    deleted = await conn.fetch(_DELETE_UNREFERENCED_AGENT_SESSIONS, owners, session_ids)
    return len(deleted)


async def _try_delete_unreferenced(
    conn: Any, owners: Sequence[str], digests: Sequence[str]
) -> int | None:
    """Delete one savepointed blob batch, or None when a concurrent reference wins."""
    try:
        async with conn.transaction():
            deleted = await conn.fetchval(_DELETE_UNREFERENCED_BLOBS, owners, digests)
    except asyncpg.RestrictViolationError:
        return None
    except asyncpg.PostgresError:
        raise
    return int(deleted or 0)


async def _delete_unreferenced(conn: Any, owners: Sequence[str], digests: Sequence[str]) -> int:
    """Delete blobs no run or resource still references, yielding to concurrent links.

    Each delete runs inside its own savepoint. A RESTRICT raised by a reference
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


def _require_owner(owner_id: str) -> str:
    owner = str(owner_id).strip()
    if not owner:
        raise ValueError("owner_id cannot be empty")
    return owner


@dataclass(frozen=True, slots=True)
class _AcceptedRun:
    """What validation derived from one envelope before a connection is taken."""

    owner: str
    run_uuid: uuid.UUID
    prepared_json: str
    accepted_json: str
    superseded_uuid: uuid.UUID | None


def _validate_acceptance(
    envelope: PreparedRunEnvelope,
    run_id: str,
    *,
    carries_answer_projections: bool,
    references: Sequence[PendingArtifactReference],
) -> _AcceptedRun:
    if envelope.run_kind in {"answer", "retrieval"} and envelope.lane != "query":
        raise ValueError("Answer and Retrieval runs execute on the query lane")
    if envelope.run_kind == "corpus_mutation" and envelope.lane != "corpus_mutation":
        raise ValueError("Corpus Mutation runs execute on the corpus_mutation lane")
    if envelope.run_kind == "corpus_mutation" and envelope.access_scope.kind != "workspace":
        raise ValueError("Corpus Mutation runs require workspace access scope")
    if envelope.supersedes_run_id is not None and envelope.run_kind != "corpus_mutation":
        raise ValueError("only Corpus Mutation runs may supersede a waiting mutation")
    superseded_uuid = (
        parse_run_id(envelope.supersedes_run_id) if envelope.supersedes_run_id is not None else None
    )
    if envelope.supersedes_run_id is not None and superseded_uuid is None:
        raise ValueError("supersedes_run_id is invalid")
    if envelope.run_kind != "answer" and carries_answer_projections:
        raise ValueError("non-Answer runs cannot carry Answer-owned projections")
    if any(reference.reference_kind == "fetched_resource" for reference in references):
        # A fetched resource is worker-fenced run state, never accepted input.
        raise ValueError("fetched_resource references cannot be run creation inputs")
    owner, run_uuid, prepared_json, accepted_json = _validate_envelope(envelope, run_id)
    return _AcceptedRun(
        owner=owner,
        run_uuid=run_uuid,
        prepared_json=prepared_json,
        accepted_json=accepted_json,
        superseded_uuid=superseded_uuid,
    )


async def _replay(
    conn: Any, run_kind: str, submitted_by: str, key: str, fingerprint: str
) -> RunCreation | None:
    """Return the run this submitter's key already accepted; changed input conflicts."""
    row = await conn.fetchrow(_SELECT_RUN_BY_KEY, run_kind, submitted_by, key)
    if row is None:
        return None
    if str(row["request_fingerprint"]) != fingerprint:
        raise IdempotencyKeyConflict(
            f"owner {submitted_by} reused idempotency key {key} with different normalized input"
        )
    return RunCreation(run=run_record(row), replayed=True)


async def _replay_in(conn: Any, envelope: PreparedRunEnvelope) -> RunCreation | None:
    return await _replay(
        conn,
        envelope.run_kind,
        envelope.submitted_by,
        envelope.submission_key,
        envelope.request_fingerprint,
    )


def _validate_envelope(
    envelope: PreparedRunEnvelope, run_id: str
) -> tuple[str, uuid.UUID, str, str]:
    submitter = _require_owner(envelope.submitted_by)
    owner = _require_owner(envelope.access_scope.scope_id)
    if envelope.access_scope.kind == "owner" and owner != submitter:
        raise ValueError("owner-scoped runs must be submitted by their owner")
    if not envelope.submission_key:
        raise ValueError("submission_key must be non-empty")
    if not envelope.request_fingerprint:
        raise ValueError("request_fingerprint must be non-empty")
    if envelope.retention_seconds < 1:
        raise ValueError("retention_seconds must be positive")
    run_uuid = parse_run_id(run_id)
    if run_uuid is None:
        raise ValueError("run_id must be a canonical UUID")
    require_prepared_input_bounds(envelope.payload)
    payload = json.dumps(dict(envelope.payload), ensure_ascii=False, sort_keys=True)
    accepted_input = json.dumps(dict(envelope.accepted_input), ensure_ascii=False, sort_keys=True)
    return owner, run_uuid, payload, accepted_input


def _json_object(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        loaded = json.loads(value)
        return dict(loaded) if isinstance(loaded, dict) else {}
    return dict(value) if isinstance(value, Mapping) else {}


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _optional_int(row: Any, name: str) -> int | None:
    try:
        value = row[name]
    except KeyError, TypeError:
        return None
    return int(value) if value is not None else None


def run_record(row: Any) -> RunRecord:
    """Project one stored run row into the storage-neutral Runtime record."""
    prepared = row["prepared_input"]
    return RunRecord(
        run_id=str(row["run_id"]),
        run_kind=cast(RunKind, str(row["run_kind"])),
        lane=cast(RunLane, str(row["lane"])),
        submitted_by=str(row["submitted_by"]),
        access_scope=RunAccessScope(
            kind=cast(Literal["owner", "workspace"], str(row["access_scope_kind"])),
            scope_id=str(row["owner_id"]),
        ),
        submission_key=str(row["submission_key"]),
        request_fingerprint=str(row["request_fingerprint"]),
        prepared_input=_json_object(prepared) if prepared is not None else None,
        accepted_input=_json_object(row["accepted_input"]),
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
        purge_after=row["purge_after"],
        next_attempt_at=row["next_attempt_at"],
        checkpoint=(
            _json_object(row["checkpoint_json"]) if row["checkpoint_json"] is not None else None
        ),
        handoff_started_at=row["handoff_started_at"],
        superseded_by_run_id=(
            str(row["superseded_by_run_id"]) if row["superseded_by_run_id"] is not None else None
        ),
        agent_workspace_epoch=_optional_int(row, "agent_workspace_epoch"),
    )


def _event_record(row: Any) -> RunEvent:
    return RunEvent(
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


class PGRunStore(ChildRunStoreMixin, PostgresOperationRunner):
    """Generic durable lifecycle plus Answer-owned PostgreSQL projections."""

    def __init__(
        self,
        *,
        pool: ConnectionPool | None = None,
        notifications: PGNotificationHub | None = None,
        retention_seconds: int = DEFAULT_RUN_RETENTION_SECONDS,
        query_max_nonterminal_runs: int = DEFAULT_QUERY_MAX_NONTERMINAL_RUNS,
        corpus_mutation_max_nonterminal_runs: int = DEFAULT_CORPUS_MUTATION_MAX_NONTERMINAL_RUNS,
        promotion_doc_threshold: int | None = None,
        promotion_chunk_threshold: int | None = None,
    ) -> None:
        super().__init__(pool=pool, notifications=notifications)
        self._retention_seconds = retention_seconds
        self._query_max_nonterminal_runs = max(1, int(query_max_nonterminal_runs))
        self._corpus_mutation_max_nonterminal_runs = max(
            1, int(corpus_mutation_max_nonterminal_runs)
        )
        self._promotion_doc_threshold = promotion_doc_threshold
        self._promotion_chunk_threshold = promotion_chunk_threshold
        self._initialized = False

    async def _run_read[T](self, operation: Callable[[Any], Awaitable[T]]) -> T:
        return await self._run(operation)

    async def _run_write[T](self, operation: Callable[[Any], Awaitable[T]]) -> T:
        return await self._run_once(operation)

    async def initialize(self, *, validate_only: bool = False) -> None:
        """Create the RunRuntime and Answer projection schema, or validate a reader."""
        if self._initialized:
            return

        async def _operation(conn: Any) -> None:
            if await conn.fetchval(_PRE_RUNTIME_ANSWER_SCHEMA):
                raise RunSchemaError(_PRE_RUNTIME_ANSWER_SCHEMA_ERROR)
            if validate_only:
                await verify_migrations(
                    conn,
                    scope=RUN_MIGRATION_SCOPE,
                    migrations=RUN_MIGRATIONS,
                    tables=RUN_SCHEMA_TABLES,
                    schema_error=RunSchemaError,
                )
                return
            await apply_migrations(
                conn,
                scope=RUN_MIGRATION_SCOPE,
                migrations=RUN_MIGRATIONS,
                schema_error=RunSchemaError,
            )

        await self._run(_operation)
        self._initialized = True

    # -- acceptance ---------------------------------------------------
    async def replay_run(
        self,
        *,
        owner_id: str,
        idempotency_key: str,
        idempotency_fingerprint: str,
        run_kind: RunKind,
    ) -> RunCreation | None:
        """Replay a matching operation-scoped key before preparation happens."""
        submitter = _require_owner(owner_id)

        async def _operation(conn: Any) -> RunCreation | None:
            return await _replay(
                conn, run_kind, submitter, idempotency_key, idempotency_fingerprint
            )

        return await self._run_read(_operation)

    async def accept_run(
        self,
        *,
        envelope: PreparedRunEnvelope,
        run_id: str,
        artifacts: Sequence[PendingArtifact] = (),
        references: Sequence[PendingArtifactReference] = (),
        routing: RoutingAcceptance | None = None,
        connection_bindings: tuple[RunConnectionBinding, ...] = (),
    ) -> RunCreation:
        """Atomically accept one operation input and its generic run row."""
        accepted = _validate_acceptance(
            envelope,
            run_id,
            carries_answer_projections=bool(
                artifacts or references or routing or connection_bindings
            ),
            references=references,
        )

        async def _operation(conn: Any) -> RunCreation:
            async with conn.transaction():
                return await self._accept_in(
                    conn,
                    envelope,
                    accepted,
                    artifacts=artifacts,
                    references=references,
                    routing=routing,
                    connection_bindings=connection_bindings,
                )

        return await self._run_write(_operation)

    async def accept_run_in(
        self,
        conn: Any,
        *,
        envelope: PreparedRunEnvelope,
        run_id: str,
        artifacts: Sequence[PendingArtifact] = (),
        references: Sequence[PendingArtifactReference] = (),
        routing: RoutingAcceptance | None = None,
        connection_bindings: tuple[RunConnectionBinding, ...] = (),
    ) -> RunCreation:
        """Accept or replay one Web Answer run inside a transaction the caller already owns.

        This is the composition seam another durable table uses to link its own
        row to the accepted run atomically. It performs no transaction control
        of its own, so the caller's commit is what makes the run and its link
        durable together. ``envelope`` carries the bounded accepted execution input.
        """
        if envelope.run_kind != "answer" or envelope.lane != "query":
            raise ValueError("Web Answer acceptance requires answer kind on the query lane")
        if envelope.access_scope.kind != "owner":
            raise ValueError("Web Answer runs require owner access scope")
        if envelope.supersedes_run_id is not None:
            raise ValueError("Web Answer runs cannot supersede another run")
        accepted = _validate_acceptance(
            envelope,
            run_id,
            carries_answer_projections=bool(
                artifacts or references or routing or connection_bindings
            ),
            references=references,
        )
        return await self._accept_in(
            conn,
            envelope,
            accepted,
            artifacts=artifacts,
            references=references,
            routing=routing,
            connection_bindings=connection_bindings,
        )

    async def _accept_in(
        self,
        conn: Any,
        envelope: PreparedRunEnvelope,
        accepted: _AcceptedRun,
        *,
        artifacts: Sequence[PendingArtifact],
        references: Sequence[PendingArtifactReference],
        routing: RoutingAcceptance | None,
        connection_bindings: tuple[RunConnectionBinding, ...],
    ) -> RunCreation:
        """Replay or insert one validated run inside the caller's transaction.

        The one acceptance body both entry points share: an idempotent replay is
        answered before and after the lane's acceptance lock, admission is
        counted under that lock, and a key conflict the insert still meets is
        answered exactly as a replay is.
        """
        owner, run_uuid = accepted.owner, accepted.run_uuid
        replayed = await _replay_in(conn, envelope)
        if replayed is not None:
            return replayed
        await conn.execute(
            "SELECT pg_advisory_xact_lock(hashtext($1))",
            f"dlightrag:run-accept:{envelope.lane}",
        )
        replayed = await _replay_in(conn, envelope)
        if replayed is not None:
            return replayed
        if accepted.superseded_uuid is not None:
            superseded = await conn.fetchval(
                _SUPERSEDE_WAITING_MUTATION,
                owner,
                accepted.superseded_uuid,
                run_uuid,
            )
            if int(superseded or 0) != 1:
                raise ValueError("supersedes_run_id is not this Workspace's waiting mutation")
        nonterminal = int(await conn.fetchval(_COUNT_NONTERMINAL_LANE, envelope.lane) or 0)
        max_nonterminal = (
            self._query_max_nonterminal_runs
            if envelope.lane == "query"
            else self._corpus_mutation_max_nonterminal_runs
        )
        if nonterminal >= max_nonterminal:
            raise RunAdmissionLimitExceededError(
                "Deployment-wide nonterminal admission limit reached"
            )
        if envelope.run_kind == "answer":
            await PGConnectionPinWriter.validate_in(
                conn, owner_id=owner, payload=envelope.payload, bindings=connection_bindings
            )
        await self._write_blobs(conn, owner, artifacts)
        row = await conn.fetchrow(
            _INSERT_RUN,
            owner,
            run_uuid,
            envelope.run_kind,
            envelope.lane,
            envelope.submitted_by,
            envelope.access_scope.kind,
            envelope.submission_key,
            accepted.prepared_json,
            accepted.accepted_json,
            envelope.request_fingerprint,
            envelope.retention_seconds,
        )
        if row is None:
            replayed = await _replay_in(conn, envelope)
            if replayed is None:
                raise RuntimeError("run insert reported a vanished conflict")
            return replayed
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
        if envelope.run_kind == "answer":
            await PGConnectionPinWriter.insert_in(
                conn, owner_id=owner, run_id=run_uuid, bindings=connection_bindings
            )
            await self._insert_routing(
                conn, owner, run_uuid, routing, prepared_input=envelope.payload
            )
        return RunCreation(run=run_record(row), replayed=False)

    async def record_corpus_window(
        self,
        *,
        run_id: str,
        workspace: str,
        window_number: int,
        docs: int,
        chunks: int,
    ) -> bool:
        """Idempotently account one successful mutation window for promotion."""
        run_uuid = parse_run_id(run_id)
        if run_uuid is None or window_number < 1 or docs < 0 or chunks < 0:
            raise ValueError("invalid corpus mutation window")

        async def _operation(conn: Any) -> bool:
            async with conn.transaction():
                return bool(
                    await conn.fetchval(
                        _RECORD_CORPUS_WINDOW,
                        run_uuid,
                        window_number,
                        _require_owner(workspace),
                        docs,
                        chunks,
                        self._promotion_doc_threshold,
                        self._promotion_chunk_threshold,
                    )
                )

        return await self._run_write(_operation)

    async def _write_publications(
        self, conn: Any, owner: str, run_uuid: uuid.UUID, publications: Sequence[Any]
    ) -> None:
        planned = tuple((item, PendingArtifact(content=item.content)) for item in publications)
        # A product's conversion view, written as the rows a read of it settles, so a
        # later Run of the Session adopts bytes and view. A product with no Session
        # stamp registers no Resource, so its view is not written either.
        views = tuple(
            update for item, _blob in planned if item.session_id is not None for update in item.view
        )
        # Every Blob this commit names, products and views alike, is written in one
        # canonical order, so two commits that share bytes never wait on each other in
        # a cycle; the view rows below then find their Blobs already written.
        await self._write_blobs(
            conn,
            owner,
            (
                *(blob for _item, blob in planned),
                *(
                    PendingArtifact(content=b"".join(update.complete_blob.chunks))
                    for update in views
                ),
            ),
        )
        for index, (item, blob) in enumerate(planned):
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
            if item.session_id is None:
                # A publication with no Agent Session stamp cannot be adopted by any
                # later Run, so it stays a Run-owned product and registers no Resource.
                continue
            await conn.execute(
                _INSERT_RESOURCE,
                owner,
                run_uuid,
                item.resource_id,
                "published_artifact",
                item.filename,
                item.mime_type,
                json.dumps(
                    {
                        "resource_kind": "published_artifact",
                        "artifact_path": item.relative_path,
                        "presentation": item.presentation,
                        "label": item.label,
                    },
                    ensure_ascii=False,
                ),
                index,
                blob.digest,
                None,
                None,
                uuid.UUID(item.session_id),
                None,
                None,
            )
        if views:
            await write_fetched_resources(conn, owner_id=owner, run_id=run_uuid, updates=views)

    async def _write_blobs(self, conn: Any, owner: str, blobs: Sequence[PendingArtifact]) -> None:
        """Acquire new blob identities in one canonical order per transaction."""
        unique = {blob.digest: blob for blob in blobs}
        for digest in sorted(unique):
            await self._write_blob(conn, owner, unique[digest])

    async def _write_blob(self, conn: Any, owner: str, blob: PendingArtifact) -> None:
        try:
            await write_blob_content(
                conn,
                owner_id=owner,
                digest=blob.digest,
                content=blob.content,
            )
        except BlobSizeConflict as exc:
            raise ValueError("blob digest collision with a different byte size") from exc

    async def _insert_routing(
        self,
        conn: Any,
        owner: str,
        run_uuid: uuid.UUID,
        routing: RoutingAcceptance | None,
        *,
        prepared_input: Mapping[str, Any],
    ) -> None:
        record = routing or RoutingAcceptance.fallback(prepared_input)
        session_uuid = uuid.UUID(record.agent_session_id)
        await conn.fetchval(_LOCK_AGENT_SESSION_IF_PRESENT, owner, session_uuid)
        await conn.execute(
            _INSERT_ROUTING,
            owner,
            run_uuid,
            record.requested_mode,
            list(record.valid_modes),
            record.resolved_mode,
            json.dumps(dict(record.model_fingerprints), ensure_ascii=False),
            record.context_policy_revision,
            uuid.UUID(record.agent_session_id),
            record.agent_lane_id,
            record.source_lane_id,
        )

    async def load_routing(self, *, owner_id: str, run_id: str) -> RoutingRecord | None:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return None

        async def _operation(conn: Any) -> RoutingRecord | None:
            row = await conn.fetchrow(_SELECT_ROUTING, owner, run_uuid)
            if row is None:
                return None
            valid = tuple(str(item) for item in (row["valid_modes"] or ()))
            return RoutingRecord(
                requested_mode=str(row["requested_mode"]),
                valid_modes=valid,
                resolved_mode=row["resolved_mode"],
                agent_session_id=str(row["agent_session_id"]),
                agent_lane_id=str(row["agent_lane_id"]),
                source_lane_id=(str(row["source_lane_id"]) if row["source_lane_id"] else None),
                fork_point_entry_id=(
                    str(row["fork_point_entry_id"]) if row["fork_point_entry_id"] else None
                ),
                fork_point_projection_id=(
                    str(row["fork_point_projection_id"])
                    if row["fork_point_projection_id"]
                    else None
                ),
            )

        return await self._run_read(_operation)

    async def record_fork_point(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        entry_id: str | None,
        projection_id: str | None,
    ) -> bool:
        """Record the state this Run settled at, under its live claim.

        A Run that is reclaim-eligible settles again from durable authority, so the
        last worker holding the claim owns the answer and overwrites an earlier
        attempt's row: only `status = 'running'` plus the matching lease/epoch may
        write, and the terminal transition closes the window behind it. Answers
        whether the write landed: a Run whose lane has no head yet records two
        NULLs, which is a recorded refusal-free state rather than a lost claim.
        """
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            raise ValueError("run_id must be a canonical UUID")

        async def _operation(conn: Any) -> bool:
            return bool(
                await conn.fetchval(
                    _RECORD_FORK_POINT,
                    owner,
                    run_uuid,
                    worker_id,
                    fencing_epoch,
                    entry_id,
                    projection_id,
                )
            )

        return await self._run_write(_operation)

    async def resolve(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        resolved_mode: str,
    ) -> str | None:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            raise ValueError("run_id must be a canonical UUID")

        async def _operation(conn: Any) -> str | None:
            value = await conn.fetchval(
                _RESOLVE_ROUTING,
                owner,
                run_uuid,
                worker_id,
                fencing_epoch,
                resolved_mode,
            )
            return str(value) if value is not None else None

        return await self._run_write(_operation)

    async def load_agent_transcript(
        self,
        *,
        owner_id: str,
        run_id: str,
        session_id: str,
        limit: int,
    ) -> tuple[dict[str, Any], ...]:
        """Project parent or owned child Session ancestry without exposing storage rows."""
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        session_uuid = parse_run_id(session_id)
        if run_uuid is None or session_uuid is None:
            return ()

        async def _operation(conn: Any) -> tuple[dict[str, Any], ...]:
            rows = await conn.fetch(
                _SELECT_AGENT_TRANSCRIPT,
                owner,
                run_uuid,
                session_uuid,
                max(1, min(int(limit), 100)),
            )
            messages: list[dict[str, Any]] = []
            for row in reversed(rows):
                payload = _json_object(row["payload_json"])
                entry_type = str(row["entry_type"])
                if entry_type in {"user_message", "control_message"}:
                    messages.append({"role": "user", "content": payload.get("content")})
                elif entry_type == "assistant_message":
                    messages.append(
                        {
                            "role": "assistant",
                            "content": payload.get("content") or "",
                            "tool_calls": list(payload.get("tool_calls") or ()),
                        }
                    )
                elif entry_type == "tool_result":
                    outcome = str(payload.get("outcome") or "failed")
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": str(payload.get("call_id") or ""),
                            "name": str(payload.get("tool_name") or ""),
                            **tool_content_message_fields(
                                decode_tool_content(payload.get("content"))
                            ),
                            "is_error": outcome != "succeeded",
                        }
                    )
            return tuple(messages)

        return await self._run_read(_operation)

    async def enqueue_agent_control(
        self,
        *,
        owner_id: str,
        run_id: str,
        kind: str,
        content: str,
    ) -> dict[str, Any] | None:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        text = content.strip()
        if run_uuid is None or kind not in {"steer", "follow_up"} or not text:
            return None

        async def _operation(conn: Any) -> dict[str, Any] | None:
            async with conn.transaction():
                run = await conn.fetchrow(_LOCK_CONTROL_RUN, owner, run_uuid)
                if run is None or str(run["status"]) not in {"queued", "running"}:
                    return None
                resolved = str(run["resolved_mode"] or "")
                requested = str(run["requested_mode"] or "")
                if resolved != "research" and not (not resolved and requested == "research"):
                    return None
                sequence = int(await conn.fetchval(_NEXT_CONTROL_SEQUENCE, owner, run_uuid) or 1)
                await conn.execute(
                    _INSERT_CONTROL,
                    owner,
                    run_uuid,
                    sequence,
                    kind,
                    text,
                    None,
                    None,
                    "user",
                    None,
                    None,
                )
                return {
                    "run_id": run_id,
                    "control_sequence": sequence,
                    "kind": kind,
                    "content": text,
                }

        return await self._run_write(_operation)

    async def load_pending_agent_controls(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        target_session_id: str | None = None,
        target_operation_id: str | None = None,
        child_fencing_epoch: int | None = None,
    ) -> tuple[dict[str, Any], ...] | None:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        target_uuid = parse_run_id(target_session_id) if target_session_id is not None else None
        operation_uuid = (
            parse_run_id(target_operation_id) if target_operation_id is not None else None
        )
        if run_uuid is None or (target_uuid is None) != (operation_uuid is None):
            return None

        async def _operation(conn: Any) -> tuple[dict[str, Any], ...] | None:
            async with conn.transaction():
                if not await hold_run_lease(conn, owner, run_uuid, worker_id, fencing_epoch):
                    return None
                if target_uuid is None:
                    rows = await conn.fetch(
                        _SELECT_PENDING_PARENT_CONTROLS,
                        owner,
                        run_uuid,
                        PENDING_CONTROL_READ_LIMIT,
                    )
                else:
                    child = await conn.fetchrow(_LOCK_CHILD_SESSION, owner, run_uuid, target_uuid)
                    if (
                        child is None
                        or child_fencing_epoch is None
                        or str(child["lease_owner"] or "") != worker_id
                        or int(child["fencing_epoch"]) != child_fencing_epoch
                        or str(child["operation_id"] or "") != str(operation_uuid)
                    ):
                        return None
                    rows = await conn.fetch(
                        _SELECT_PENDING_CHILD_CONTROLS,
                        owner,
                        run_uuid,
                        target_uuid,
                        operation_uuid,
                        PENDING_CONTROL_READ_LIMIT,
                    )
                return tuple(
                    {
                        "control_sequence": int(row["control_sequence"]),
                        "kind": str(row["kind"]),
                        "content": str(row["content"]),
                        "origin": str(row["origin"]),
                        "created_at": row["created_at"],
                    }
                    for row in rows
                )

        return await self._run_write(_operation)

    async def acknowledge_agent_controls(
        self,
        *,
        owner_id: str,
        run_id: str,
        control_sequences: Sequence[int],
        worker_id: str,
        fencing_epoch: int,
        target_session_id: str | None = None,
        target_operation_id: str | None = None,
        child_fencing_epoch: int | None = None,
    ) -> bool:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        target_uuid = parse_run_id(target_session_id) if target_session_id is not None else None
        operation_uuid = (
            parse_run_id(target_operation_id) if target_operation_id is not None else None
        )
        if run_uuid is None or (target_uuid is None) != (operation_uuid is None):
            return False

        async def _operation(conn: Any) -> bool:
            async with conn.transaction():
                if not await hold_run_lease(conn, owner, run_uuid, worker_id, fencing_epoch):
                    return False
                values = [int(value) for value in control_sequences]
                if target_uuid is None:
                    if values:
                        await conn.execute(_CONSUME_PARENT_CONTROLS, owner, run_uuid, values)
                    return True
                child = await conn.fetchrow(_LOCK_CHILD_SESSION, owner, run_uuid, target_uuid)
                if (
                    child is None
                    or child_fencing_epoch is None
                    or str(child["lease_owner"] or "") != worker_id
                    or int(child["fencing_epoch"]) != child_fencing_epoch
                    or str(child["operation_id"] or "") != str(operation_uuid)
                ):
                    return False
                if values:
                    await conn.execute(
                        _CONSUME_CHILD_CONTROLS,
                        owner,
                        run_uuid,
                        target_uuid,
                        operation_uuid,
                        values,
                    )
                return True

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
        session_rows = await conn.fetch(_SELECT_RUN_AGENT_SESSIONS, owners, run_uuids)
        deleted_rows = await conn.fetch(_DELETE_RUNS, owners, run_uuids)
        await _delete_unreferenced_agent_sessions(conn, session_rows)
        artifacts = await _delete_unreferenced(conn, *_digest_pairs(pairs))
        deleted = _deleted_runs(deleted_rows)
        return RunDeletion(runs=len(deleted), artifacts=artifacts, deleted=deleted)

    async def iter_active_run_requirements(
        self,
        *,
        page_size: int = _BATCH_LIMIT,
    ) -> AsyncIterator[Mapping[str, Any]]:
        """Stream active-run compatibility facts in bounded keyset pages."""
        cap = max(1, min(int(page_size), _BATCH_LIMIT))

        async def _frontier(conn: Any) -> Any:
            return await conn.fetchrow(
                _ACTIVE_REQUIREMENTS_FRONTIER,
                MAX_RECLAIMS_WITHOUT_PROGRESS,
            )

        upper = await self._run_read(_frontier)
        if upper is None:
            return
        upper_position = (upper["created_at"], upper["run_id"])
        position: tuple[Any, Any] | None = None
        while True:

            async def _page(
                conn: Any,
                after: tuple[Any, Any] | None = position,
            ) -> list[Any]:
                if after is None:
                    return await conn.fetch(
                        _ACTIVE_REQUIREMENTS_FIRST_PAGE,
                        MAX_RECLAIMS_WITHOUT_PROGRESS,
                        *upper_position,
                        cap,
                    )
                return await conn.fetch(
                    _ACTIVE_REQUIREMENTS_AFTER,
                    MAX_RECLAIMS_WITHOUT_PROGRESS,
                    *upper_position,
                    *after,
                    cap,
                )

            rows = await self._run_read(_page)
            if not rows:
                return
            for row in rows:
                yield {
                    "run_kind": str(row["run_kind"]),
                    "prepared_input": _json_value(row["prepared_input_json"]),
                }
            if len(rows) < cap:
                return
            next_position = (rows[-1]["created_at"], rows[-1]["run_id"])
            if position is not None and next_position <= position:
                raise RuntimeError("active-run requirement cursor did not advance")
            position = next_position

    # -- reads --------------------------------------------------------
    async def get_run(self, *, owner_id: str, run_id: str) -> RunRecord | None:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return None

        async def _operation(conn: Any) -> RunRecord | None:
            row = await conn.fetchrow(_SELECT_RUN, owner, run_uuid)
            return run_record(row) if row is not None else None

        return await self._run_read(_operation)

    async def get_run_global(self, *, run_id: str) -> RunRecord | None:
        """Read by globally unique id for a later fail-closed authorization check."""
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return None

        async def _operation(conn: Any) -> RunRecord | None:
            row = await conn.fetchrow(_SELECT_RUN_GLOBAL, run_uuid)
            return run_record(row) if row is not None else None

        return await self._run_read(_operation)

    async def list_runs(
        self, *, owner_id: str, after_run_id: str | None = None, limit: int = 50
    ) -> tuple[RunRecord, ...]:
        owner = _require_owner(owner_id)
        cap = max(1, min(int(limit), 100))
        after = parse_run_id(after_run_id) if after_run_id else None

        async def _operation(conn: Any) -> tuple[RunRecord, ...]:
            if after is None:
                rows = await conn.fetch(_LIST_RUNS, owner, cap)
            else:
                rows = await conn.fetch(_LIST_RUNS_AFTER, owner, after, cap)
            return tuple(run_record(row) for row in rows)

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

    async def lineage_resource_rows(
        self, *, owner_id: str, session_id: str, resource_id: str
    ) -> tuple[RunFetchedResource, ...]:
        """Read one earlier Run Resource this Session may adopt, with its view.

        The Session stamp decides admission: a row another owner or another Agent
        Session registered is not returned at all, so the caller cannot adopt it by
        naming its handle.
        """
        owner = _require_owner(owner_id)
        session_uuid = _parse_uuid(session_id)
        if session_uuid is None or not resource_id:
            return ()

        async def _operation(conn: Any) -> tuple[RunFetchedResource, ...]:
            rows = await conn.fetch(
                _SELECT_LINEAGE_RESOURCE,
                owner,
                session_uuid,
                resource_id,
                resource_id.encode("utf-8"),
                [table_kind for _capability, table_kind in _ADOPTABLE_LINEAGE_KINDS],
                [capability for capability, _table_kind in _ADOPTABLE_LINEAGE_KINDS],
            )
            return tuple(
                RunFetchedResource(
                    resource_id=str(row["resource_id"]),
                    ordinal=int(row["ordinal"] or 0),
                    digest=str(row["blob_digest"]),
                    filename=str(row["safe_name"] or row["resource_id"]),
                    mime_type=str(row["media_type"] or "application/octet-stream"),
                    source_locator=bytes(row["source_locator"] or b""),
                    # The origin Run is the row's own run_id; surface it the way
                    # occurrence rows record theirs, so one loader reads both.
                    capabilities={
                        **_json_object(row["capabilities"]),
                        "origin_run_id": str(row["run_id"]),
                    },
                )
                for row in rows
            )

        return await self._run(_operation)

    async def read_run_resource_row(
        self, *, owner_id: str, run_id: str, resource_id: str
    ) -> RunFetchedResource | None:
        """Read one registered Row of a Run by the id it recorded.

        The read surface accepts every kind a run registered, including an adopted
        entry attachment whose locator is a digest rather than a URL, so this
        lookup never filters by kind; the URL-bearing catalog keeps its own query.
        """
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None or not resource_id:
            return None

        async def _operation(conn: Any) -> RunFetchedResource | None:
            row = await conn.fetchrow(_SELECT_RUN_RESOURCE_BY_ID, owner, run_uuid, resource_id)
            if row is None:
                return None
            return RunFetchedResource(
                resource_id=str(row["resource_id"]),
                ordinal=int(row["ordinal"] or 0),
                digest=str(row["blob_digest"]),
                filename=str(row["safe_name"] or row["resource_id"]),
                mime_type=str(row["media_type"] or "application/octet-stream"),
                source_locator=bytes(row["source_locator"] or b""),
                capabilities=_json_object(row["capabilities"]),
            )

        return await self._run_read(_operation)

    async def list_fetched_resources(
        self, *, owner_id: str, run_id: str
    ) -> tuple[RunFetchedResource, ...]:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return ()

        async def _operation(conn: Any) -> tuple[RunFetchedResource, ...]:
            rows = await conn.fetch(_SELECT_RUN_FETCHED_RESOURCES, owner, run_uuid)
            return tuple(
                RunFetchedResource(
                    resource_id=str(row["resource_id"]),
                    ordinal=int(row["ordinal"]),
                    digest=str(row["blob_digest"]),
                    filename=str(row["safe_name"]),
                    mime_type=str(row["media_type"]),
                    source_locator=bytes(row["source_locator"]),
                    capabilities=_json_object(row["capabilities"]),
                )
                for row in rows
            )

        return await self._run_read(_operation)

    async def load_child_attachment_occurrences(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        child_session_id: str,
        context_snapshot: dict[str, Any],
    ) -> tuple[RunFetchedResource, ...]:
        from dlightrag.adapters.postgres.answer.attachment_replay import (
            load_child_attachment_occurrences,
        )

        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        child_uuid = parse_run_id(child_session_id)
        if run_uuid is None or child_uuid is None:
            raise ValueError("invalid Child attachment replay identity")

        async def operation(conn: Any) -> tuple[RunFetchedResource, ...]:
            return await load_child_attachment_occurrences(
                conn,
                owner_id=owner,
                run_id=run_uuid,
                worker_id=worker_id,
                fencing_epoch=fencing_epoch,
                child_session_id=child_uuid,
                context_snapshot=context_snapshot,
            )

        return await self._run_read(operation)

    async def retain_attachment_occurrences(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        selection: AttachmentReplaySelection,
    ) -> tuple[RunFetchedResource, ...]:
        from dlightrag.adapters.postgres.answer.attachment_replay import (
            retain_attachment_occurrences,
        )

        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            raise ValueError("invalid attachment replay Run")

        async def operation(conn: Any) -> tuple[RunFetchedResource, ...]:
            return await retain_attachment_occurrences(
                conn,
                owner_id=owner,
                run_id=run_uuid,
                worker_id=worker_id,
                fencing_epoch=fencing_epoch,
                selection=selection,
            )

        return await self._run_write(operation)

    async def record_lineage_adoption(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        resources: tuple[FetchedResourceSettlementUpdate, ...],
    ) -> None:
        """Record one adopted Resource as this Run's own rows, fenced by its lease."""
        from dlightrag.adapters.postgres.answer.session_repository import record_lineage_adoption

        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            raise ValueError("invalid lineage adoption Run")

        async def operation(conn: Any) -> None:
            await record_lineage_adoption(
                conn,
                owner_id=owner,
                run_id=run_uuid,
                worker_id=worker_id,
                fencing_epoch=fencing_epoch,
                resources=resources,
            )

        await self._run_write(operation)

    async def list_artifact_attachments(
        self, *, owner_id: str, run_id: str
    ) -> tuple[ArtifactAttachmentUpdate, ...]:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return ()

        async def _operation(conn: Any) -> tuple[ArtifactAttachmentUpdate, ...]:
            rows = await conn.fetch(_SELECT_ARTIFACT_ATTACHMENTS, owner, run_uuid)
            return tuple(
                ArtifactAttachmentUpdate(
                    relative_path=str(row["relative_path"]),
                    label=str(row["label"]),
                    content_digest=str(row["content_digest"]),
                    size_bytes=int(row["size_bytes"]),
                    presentation=str(row["presentation"]),
                    session_id=str(row["session_id"]),
                    intent_id=str(row["intent_id"]),
                )
                for row in rows
            )

        return await self._run_read(_operation)

    async def read_event_page(
        self, *, owner_id: str, run_id: str, after_sequence: int = 0
    ) -> tuple[RunEvent, ...]:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return ()

        async def _operation(conn: Any) -> tuple[RunEvent, ...]:
            rows = await conn.fetch(
                _SELECT_EVENTS,
                owner,
                run_uuid,
                max(0, int(after_sequence)),
                _EVENT_PAGE_LIMIT,
            )
            return tuple(_event_record(row) for row in rows)

        return await self._run_read(_operation)

    # -- cancellation -------------------------------------------------
    async def iter_cancel_pending(
        self,
        *,
        worker_id: str,
        page_size: int = _BATCH_LIMIT,
    ) -> AsyncIterator[tuple[str, str]]:
        """Stream this worker's live cancel-pending leases in bounded pages."""
        cap = max(1, min(int(page_size), _BATCH_LIMIT))

        async def _frontier(conn: Any) -> Any:
            return await conn.fetchrow(_CANCEL_PENDING_FRONTIER, worker_id)

        upper = await self._run_read(_frontier)
        if upper is None:
            return
        upper_position = (upper["created_at"], upper["run_id"])
        position: tuple[Any, Any] | None = None
        while True:

            async def _page(
                conn: Any,
                after: tuple[Any, Any] | None = position,
            ) -> list[Any]:
                if after is None:
                    return await conn.fetch(
                        _CANCEL_PENDING_FIRST_PAGE,
                        worker_id,
                        *upper_position,
                        cap,
                    )
                return await conn.fetch(
                    _CANCEL_PENDING_AFTER,
                    worker_id,
                    *upper_position,
                    *after,
                    cap,
                )

            rows = await self._run_read(_page)
            if not rows:
                return
            for row in rows:
                yield str(row["owner_id"]), str(row["run_id"])
            if len(rows) < cap:
                return
            next_position = (rows[-1]["created_at"], rows[-1]["run_id"])
            if position is not None and next_position <= position:
                raise RuntimeError("cancel-pending cursor did not advance")
            position = next_position

    def build_cancellation_listener(
        self,
        *,
        worker_id: str,
        on_cancel: Callable[[str, str], Awaitable[None]],
    ) -> ChannelWatcher:
        """Signal this worker's cancel-pending Runs whenever the cancel channel wakes.

        A wake digest names a Run but never authorizes: every wake rescans this
        worker's live cancel-pending leases and signals each one. The listener is
        ``ready`` once a rescan after the channel went live, and every signal it
        sent, succeeded.
        """

        async def rescan() -> None:
            async for owner_id, run_id in self.iter_cancel_pending(worker_id=worker_id):
                await on_cancel(owner_id, run_id)

        return ChannelWatcher(
            self._notification_hub,
            RUN_CANCEL_CHANNEL,
            rescan,
            name="Run cancellation rescan",
        )

    async def request_cancellation(self, *, owner_id: str, run_id: str) -> CancellationOutcome:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return CancellationOutcome(outcome="unknown", run=None)

        async def _operation(conn: Any) -> CancellationOutcome:
            async with conn.transaction():
                row = await conn.fetchrow(_SELECT_RUN_FOR_UPDATE, owner, run_uuid)
                if row is None:
                    return CancellationOutcome(outcome="unknown", run=None)
                run = run_record(row)
                if run.terminal:
                    return CancellationOutcome(outcome="already_terminal", run=run)
                if run.handoff_started_at is not None:
                    return CancellationOutcome(outcome="rejected", run=run)
                if run.status == "queued":
                    finalized = await self._finalize_cancelled_queued(conn, owner, run_uuid)
                    if not finalized:
                        raise RuntimeError("locked queued run could not be cancelled")
                    return CancellationOutcome(
                        outcome="cancelled",
                        run=run_record(await conn.fetchrow(_SELECT_RUN, owner, run_uuid)),
                    )
                updated = await conn.fetchval(
                    _REQUEST_CANCELLATION,
                    owner,
                    run_uuid,
                    RUN_CANCEL_CHANNEL,
                    cancellation_notify_key(owner_id=owner, run_id=str(run_uuid)),
                )
                if int(updated or 0) != 1:
                    current = run_record(await conn.fetchrow(_SELECT_RUN, owner, run_uuid))
                    return CancellationOutcome(outcome="rejected", run=current)
                return CancellationOutcome(
                    outcome="pending",
                    run=run_record(await conn.fetchrow(_SELECT_RUN, owner, run_uuid)),
                )

        return await self._run_write(_operation)

    async def _finalize_cancelled_queued(self, conn: Any, owner: str, run_uuid: uuid.UUID) -> bool:
        finalized = await conn.fetchval(
            _FINALIZE_UNLEASED,
            [owner],
            [run_uuid],
            "cancelled",
            None,
            None,
            "done",
            json.dumps({"status": "cancelled"}),
        )
        return int(finalized or 0) == 1

    # -- claim and writes ---------------------------------------------
    async def claim_next(
        self,
        *,
        worker_id: str,
        run_kinds: Sequence[RunKind] = ("answer",),
        lanes: Sequence[RunLane] = ("query",),
    ) -> ClaimedRun | None:
        """Claim the oldest eligible registered kind without double-claiming."""
        worker = str(worker_id).strip()
        if not worker:
            raise ValueError("worker_id cannot be empty")
        requested_lanes = tuple(dict.fromkeys(lanes))
        if len(requested_lanes) != 1:
            raise ValueError("claim_next requires exactly one execution lane")
        lane = requested_lanes[0]

        async def _operation(conn: Any) -> ClaimedRun | None:
            while True:
                candidate = await conn.fetchrow(
                    _SELECT_CLAIM_CANDIDATE,
                    MAX_RECLAIMS_WITHOUT_PROGRESS,
                    list(run_kinds),
                    [lane],
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
                        json.dumps(
                            {
                                "kind": RUN_ABANDONED_ERROR_KIND,
                                "message": _ABANDONED_ERROR_MESSAGE,
                            }
                        ),
                    )
                    continue
                row = await conn.fetchrow(
                    _CLAIM_RUN,
                    candidate["owner_id"],
                    candidate["run_id"],
                    worker,
                    RUN_LEASE_SECONDS,
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
        run = run_record(row)
        owner = run.owner_id
        run_uuid = parse_run_id(run.run_id)
        if run_uuid is None:
            raise RuntimeError("claimed run id is not a canonical UUID")
        if run.run_kind != "answer":
            return ClaimedRun(
                run=run,
                execution=RunExecutionContext(
                    owner_id=owner,
                    run_id=run.run_id,
                    worker_id=worker,
                    lease_owner=worker,
                    fencing_epoch=run.fencing_epoch,
                ),
            )
        prepared = run.prepared_input or run.accepted_input or {}
        raw_session_id = prepared.get("agent_session_id")
        if not raw_session_id:
            raise RuntimeError("claimed Answer run has no canonical Agent Session mapping")
        primary_session_id = SessionId(str(raw_session_id))
        execution = RunExecutionContext(
            owner_id=owner,
            run_id=run.run_id,
            worker_id=worker,
            lease_owner=worker,
            fencing_epoch=run.fencing_epoch,
            session_repository=PGAgentSessionRepository(
                pool=self._operation_pool,
                owner_id=owner,
                run_id=run_uuid,
                worker_id=worker,
                lease_owner=worker,
                fencing_epoch=run.fencing_epoch,
                primary_session_id=primary_session_id,
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
                RUN_LEASE_SECONDS,
            )
            if row is None:
                return LeaseRenewal(renewed=False, cancel_requested=False)
            return LeaseRenewal(renewed=True, cancel_requested=bool(row["cancel_requested"]))

        return await self._run_write(_operation)

    async def write_checkpoint(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        checkpoint: Mapping[str, object],
        phase: RunPhase | None = None,
    ) -> bool:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return False

        async def _operation(conn: Any) -> bool:
            value = await conn.fetchval(
                _WRITE_CHECKPOINT,
                owner,
                run_uuid,
                worker_id,
                fencing_epoch,
                json.dumps(dict(checkpoint), ensure_ascii=False),
                phase,
                RUN_LEASE_SECONDS,
            )
            return value is not None

        return await self._run_write(_operation)

    async def start_handoff(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        checkpoint: Mapping[str, object],
    ) -> bool:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return False

        async def _operation(conn: Any) -> bool:
            value = await conn.fetchval(
                _START_HANDOFF,
                owner,
                run_uuid,
                worker_id,
                fencing_epoch,
                json.dumps(dict(checkpoint), ensure_ascii=False),
                RUN_LEASE_SECONDS,
            )
            return value is not None

        return await self._run_write(_operation)

    async def record_phase(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        phase: RunPhase,
    ) -> int | None:
        return await self.append_event(
            owner_id=owner_id,
            run_id=run_id,
            worker_id=worker_id,
            fencing_epoch=fencing_epoch,
            phase=phase,
            event_type="progress",
            payload={"phase": phase},
        )

    async def append_event(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        phase: RunPhase | None,
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
                RUN_LEASE_SECONDS,
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
        result: Mapping[str, object] | None = None,
    ) -> TerminalOutcome:
        return await self._finish_run(
            owner_id=owner_id,
            run_id=run_id,
            worker_id=worker_id,
            fencing_epoch=fencing_epoch,
            status="failed",
            stop_reason=None,
            result=result,
            error_kind=error_kind,
            error_message=error_message,
            event_type="error",
            payload={"kind": error_kind, "message": error_message},
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
        status: TerminalStatus,
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
                outcome = await finish_fenced_run(
                    conn,
                    owner_id=owner,
                    run_id=run_uuid,
                    lease_owner=worker_id,
                    fencing_epoch=fencing_epoch,
                    status=status,
                    stop_reason=stop_reason,
                    result=result,
                    error_kind=error_kind,
                    error_message=error_message,
                    event_type=event_type,
                    payload=payload,
                    withhold_on_cancel=withhold_on_cancel,
                )
                # Preserve publication and spill cleanup ownership for the
                # requested transition; a cancellation that beat it owns only
                # its terminal row and event.
                if outcome.committed and outcome.status == status:
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
                return outcome

        return await self._run_write(_operation)

    async def defer(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        checkpoint: Mapping[str, object],
        next_attempt_at: Any,
    ) -> bool:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return False

        async def _operation(conn: Any) -> bool:
            async with conn.transaction():
                value = await conn.fetchval(
                    _DEFER_RUN,
                    owner,
                    run_uuid,
                    worker_id,
                    fencing_epoch,
                    json.dumps(dict(checkpoint), ensure_ascii=False),
                    next_attempt_at,
                )
                if value is None:
                    return False
                await conn.execute(_CLEAR_FORK_POINT, owner, run_uuid)
                return True

        return await self._run_write(_operation)

    async def resume_repair(self, *, owner_id: str, run_id: str) -> bool:
        """Explicitly make one waiting mutation claimable again without a new Run."""
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return False

        async def _operation(conn: Any) -> bool:
            value = await conn.fetchval(_RESUME_REPAIR, owner, run_uuid)
            return value is not None

        return await self._run_write(_operation)

    async def wait_for_repair(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        checkpoint: Mapping[str, object],
    ) -> bool:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return False

        async def _operation(conn: Any) -> bool:
            value = await conn.fetchval(
                _WAIT_FOR_REPAIR,
                owner,
                run_uuid,
                worker_id,
                fencing_epoch,
                json.dumps(dict(checkpoint), ensure_ascii=False),
            )
            return value is not None

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
                    await conn.execute(_CLEAR_FORK_POINT, owner, run_uuid)
                    return "requeued"
                current = await conn.fetchrow(_SELECT_RUN, owner, run_uuid)
                if current is not None and current["cancel_requested_at"] is not None:
                    outcome = await finish_fenced_run(
                        conn,
                        owner_id=owner,
                        run_id=run_uuid,
                        lease_owner=worker_id,
                        fencing_epoch=fencing_epoch,
                        status="cancelled",
                        stop_reason=None,
                        result=None,
                        error_kind=None,
                        error_message=None,
                        event_type="done",
                        payload={"status": "cancelled"},
                        withhold_on_cancel=False,
                        cancel_requested=True,
                    )
                    return "cancelled" if outcome.committed else "lease_lost"
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
                        json.dumps({"status": "cancelled"}),
                    )
                poisoned = await conn.fetch(
                    "SELECT owner_id, run_id FROM dlightrag_runs"
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
                        json.dumps(
                            {
                                "kind": RUN_ABANDONED_ERROR_KIND,
                                "message": _ABANDONED_ERROR_MESSAGE,
                            }
                        ),
                    )
                return SweepOutcome(cancelled=int(cancelled), abandoned=int(abandoned))

        return await self._run_write(_operation)

    async def trim_expired_event_logs(self) -> int:
        async def _operation(conn: Any) -> int:
            async with conn.transaction():
                rows = await conn.fetch(_SELECT_TRIMMABLE_RUNS, _BATCH_LIMIT)
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
                rows = await conn.fetch(_SELECT_EXPIRED_RUNS, _BATCH_LIMIT)
                if not rows:
                    return RunDeletion(runs=0, artifacts=0)
                owners = [row["owner_id"] for row in rows]
                run_ids = [row["run_id"] for row in rows]
                digest_rows = await conn.fetch(_SELECT_RUN_DIGESTS, owners, run_ids)
                session_rows = await conn.fetch(_SELECT_RUN_AGENT_SESSIONS, owners, run_ids)
                deleted_rows = await conn.fetch(_DELETE_RUNS, owners, run_ids)
                await _delete_unreferenced_agent_sessions(conn, session_rows)
                artifacts = await _delete_unreferenced(conn, *_digest_pairs(digest_rows))
                deleted = _deleted_runs(deleted_rows)
                return RunDeletion(runs=len(deleted), artifacts=artifacts, deleted=deleted)

        return await self._run_write(_operation)


__all__ = [
    "RUN_MIGRATIONS",
    "RUN_MIGRATION_SCOPE",
    "RUN_SCHEMA_TABLES",
    "PGRunStore",
    "run_columns",
    "run_record",
]
