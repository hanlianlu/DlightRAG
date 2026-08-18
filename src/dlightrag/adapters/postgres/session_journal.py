# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Claim-bound PostgreSQL journal, progress, evidence, resource, and blob stores.

Every bound store embeds owner, run, worker, lease owner, and fencing epoch at
claim time; its public methods carry no fencing parameters (M3 claim-bound
execution stores). Each mutating transaction starts by locking the run row
under the live lease/epoch predicate, so a stale or lost lease changes zero
rows (M3 transactional invariant 1).

Settlements are one transaction each: host updates, ordered result entries,
projection, settlement columns, session version, and durable run progress
commit atomically (M3-D26/D27, transactional invariant 2).
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any

from dlightrag_agent.session.effects import EffectSettlement, JsonValue
from dlightrag_agent.session.entries import EffectIntentEntry, SessionEntry
from dlightrag_agent.session.ids import IntentId, ProjectionId, SessionId, StageIntentId
from dlightrag_agent.session.projection import ContextProjection
from dlightrag_agent.session.store import (
    AgentSessionSnapshot,
    AppendCommit,
    EffectAlreadySettled,
    EffectCommit,
    EffectMissing,
    EvidenceConflict,
    LeaseLost,
    SessionCommit,
    SessionProgressClass,
    SettleCommit,
    VersionConflict,
)

from dlightrag.adapters.postgres._operations import ConnectionPool
from dlightrag.adapters.postgres._pool import pg_pool
from dlightrag.runtime.progress import (
    StageCommit,
    StageCommitResult,
    StageConflict,
    StageEvidenceConflict,
    StageLeaseLost,
    StageProgressConflict,
    StageRecord,
)
from dlightrag.runtime.settlements import (
    CommittedSpillUpdate,
    CompleteBlobDescriptor,
    EvidenceSettlementUpdate,
    FetchedResourceSettlementUpdate,
    M3HostUpdate,
    OpaqueEvidenceResourceWrite,
    OpaqueEvidenceWrite,
    OpaqueFetchedResourceWrite,
    WorkspaceInventoryUpdate,
)

_LEASE_PREDICATE = """
SELECT 1
FROM dlightrag_answer_runs
WHERE owner_id = $1 AND run_id = $2
  AND lease_owner = $3 AND fencing_epoch = $4
  AND status = 'running' AND lease_expires_at > NOW()
FOR UPDATE
"""

_LOCK_SESSION = """
SELECT version, active_projection_id
FROM dlightrag_agent_sessions
WHERE owner_id = $1 AND run_id = $2 AND session_id = $3
FOR UPDATE
"""

_CREATE_SESSION = """
INSERT INTO dlightrag_agent_sessions (
    owner_id, run_id, session_id, version, active_projection_id
)
VALUES ($1, $2, $3, 0, NULL)
ON CONFLICT (owner_id, run_id, session_id) DO NOTHING
"""

_INSERT_ENTRY = """
INSERT INTO dlightrag_agent_session_entries (
    owner_id, run_id, session_id, sequence, entry_id, entry_type,
    schema_version, timestamp, payload_json
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb)
"""

_INSERT_PROJECTION = """
INSERT INTO dlightrag_agent_context_projections (
    owner_id, run_id, session_id, projection_id, first_retained_sequence,
    covered_through_sequence, summary, token_anchors, schema_version
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9)
ON CONFLICT (owner_id, run_id, session_id, projection_id) DO NOTHING
"""

_INSERT_INTENT_ROW = """
INSERT INTO dlightrag_agent_effects (
    owner_id, run_id, session_id, intent_id, tool_name, replay_policy,
    contract_version, input_schema_digest, canonical_input, source_call_id
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10)
ON CONFLICT (owner_id, run_id, session_id, intent_id) DO NOTHING
"""

_ADVANCE_SESSION = """
UPDATE dlightrag_agent_sessions
SET version = version + 1,
    last_sequence = last_sequence + $4,
    active_projection_id = COALESCE($5, active_projection_id),
    updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2 AND session_id = $3
RETURNING version
"""

_ADVANCE_PROGRESS = """
UPDATE dlightrag_answer_runs
SET durable_progress_version = durable_progress_version + 1,
    updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2
"""

_SELECT_SESSION_SNAPSHOT = """
SELECT version, active_projection_id
FROM dlightrag_agent_sessions
WHERE owner_id = $1 AND run_id = $2 AND session_id = $3
"""

_SELECT_ENTRIES = """
SELECT sequence, entry_id::text, entry_type, schema_version, timestamp, payload_json
FROM dlightrag_agent_session_entries
WHERE owner_id = $1 AND run_id = $2 AND session_id = $3
ORDER BY sequence
"""

_SELECT_PROJECTION = """
SELECT projection_id::text, first_retained_sequence, covered_through_sequence,
       summary, token_anchors, schema_version
FROM dlightrag_agent_context_projections
WHERE owner_id = $1 AND run_id = $2 AND session_id = $3 AND projection_id = $4
"""

_SELECT_INTENT_FOR_UPDATE = """
SELECT intent_id::text, outcome, contract_version
FROM dlightrag_agent_effects
WHERE owner_id = $1 AND run_id = $2 AND session_id = $3 AND intent_id = $4
FOR UPDATE
"""

_SETTLE_INTENT = """
UPDATE dlightrag_agent_effects
SET outcome = $5::text,
    result_entry_sequence = $6,
    result_digest = $7,
    host_update_digest = $8,
    settled_at = NOW()
WHERE owner_id = $1 AND run_id = $2 AND session_id = $3 AND intent_id = $4
"""

_INSERT_EVIDENCE = """
INSERT INTO dlightrag_answer_evidence (
    owner_id, run_id, session_id, intent_id, result_ordinal,
    content_digest, locator_digest, content, locator
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
ON CONFLICT (owner_id, run_id, session_id, intent_id, result_ordinal) DO NOTHING
"""

_SELECT_EVIDENCE_DIGESTS = """
SELECT content_digest, locator_digest
FROM dlightrag_answer_evidence
WHERE owner_id = $1 AND run_id = $2 AND session_id = $3
  AND intent_id = $4 AND result_ordinal = $5
"""

_INSERT_BLOB_METADATA = """
INSERT INTO dlightrag_blobs (owner_id, digest, byte_size)
VALUES ($1, $2, $3)
ON CONFLICT (owner_id, digest) DO NOTHING
"""

_SELECT_BLOB_SIZE = """
SELECT byte_size FROM dlightrag_blobs WHERE owner_id = $1 AND digest = $2
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

_SELECT_RESOURCE_DIGESTS = """
SELECT kind, blob_digest, locator_digest
FROM dlightrag_answer_resources
WHERE owner_id = $1 AND run_id = $2 AND resource_id = $3
"""


class _EvidenceIdentityConflict(Exception):
    """Rolls back the current settlement transaction as an EvidenceConflict."""


def _uuid(value: Any) -> Any:
    """Coerce a canonical id value to a PostgreSQL UUID parameter."""
    return uuid.UUID(str(value))


class PGJournalStore:
    """One claim-bound AgentSessionStore over PostgreSQL."""

    def __init__(
        self,
        *,
        pool: ConnectionPool | None,
        owner_id: str,
        run_id: Any,
        worker_id: str,
        lease_owner: str,
        fencing_epoch: int,
    ) -> None:
        self._pool = pool
        self._owner_id = owner_id
        self._run_id = run_id
        self._worker_id = worker_id
        self._lease_owner = lease_owner
        self._fencing_epoch = fencing_epoch

    @asynccontextmanager
    async def _connection(self) -> AsyncIterator[Any]:
        pool = self._pool if self._pool is not None else await pg_pool.get()
        async with pool.acquire() as conn:
            yield conn

    async def load(self, session_id: SessionId) -> AgentSessionSnapshot:
        async with self._connection() as conn:
            session_row = await conn.fetchrow(
                _SELECT_SESSION_SNAPSHOT, self._owner_id, self._run_id, _uuid(session_id.value)
            )
            if session_row is None:
                return AgentSessionSnapshot(
                    session_id=session_id, version=0, entries=(), active_projection=None
                )
            entries = await self._load_entries(conn, session_id)
            projection = await self._load_projection(
                conn, session_id, session_row["active_projection_id"]
            )
            return AgentSessionSnapshot(
                session_id=session_id,
                version=int(session_row["version"]),
                entries=tuple(entries),
                active_projection=projection,
            )

    async def _load_entries(self, conn: Any, session_id: SessionId) -> list[SessionEntry]:
        rows = await conn.fetch(
            _SELECT_ENTRIES, self._owner_id, self._run_id, _uuid(session_id.value)
        )
        entries: list[SessionEntry] = []
        for row in rows:
            entries.append(
                _decode_entry(
                    row,
                    owner_id=self._owner_id,
                    run_id=str(self._run_id),
                    session_id=session_id,
                )
            )
        return entries

    async def _load_projection(
        self, conn: Any, session_id: SessionId, projection_id: Any
    ) -> ContextProjection | None:
        if projection_id is None:
            return None
        row = await conn.fetchrow(
            _SELECT_PROJECTION,
            self._owner_id,
            self._run_id,
            _uuid(session_id.value),
            projection_id,
        )
        if row is None:
            return None
        return ContextProjection(
            projection_id=ProjectionId(str(projection_id)),
            first_retained_sequence=int(row["first_retained_sequence"]),
            covered_through_sequence=int(row["covered_through_sequence"]),
            summary=row["summary"],
            token_anchors=tuple(
                _token_anchor(anchor) for anchor in (_json_payload(row["token_anchors"]) or [])
            ),
            schema_version=int(row["schema_version"]),
        )

    async def append(
        self,
        *,
        session_id: SessionId,
        expected_version: int,
        entries: Sequence[SessionEntry],
        projection: ContextProjection | None = None,
    ) -> AppendCommit:
        if not entries:
            raise ValueError("a session transaction requires at least one entry")
        async with self._connection() as conn:
            async with conn.transaction():
                if await self._hold_lease(conn) is None:
                    return LeaseLost()
                await conn.execute(
                    _CREATE_SESSION, self._owner_id, self._run_id, _uuid(session_id.value)
                )
                session_row = await conn.fetchrow(
                    _LOCK_SESSION, self._owner_id, self._run_id, _uuid(session_id.value)
                )
                version = int(session_row["version"])
                if version != expected_version:
                    return VersionConflict(
                        expected_version=expected_version, current_version=version
                    )
                last_sequence = await conn.fetchval(
                    "SELECT last_sequence FROM dlightrag_agent_sessions"
                    " WHERE owner_id = $1 AND run_id = $2 AND session_id = $3",
                    self._owner_id,
                    self._run_id,
                    _uuid(session_id.value),
                )
                sequences = tuple(
                    range(int(last_sequence) + 1, int(last_sequence) + len(entries) + 1)
                )
                for entry, sequence in zip(entries, sequences, strict=True):
                    await conn.execute(
                        _INSERT_ENTRY,
                        self._owner_id,
                        self._run_id,
                        _uuid(session_id.value),
                        sequence,
                        _uuid(entry.entry_id.value),
                        entry.entry_type,
                        entry.schema_version,
                        entry.timestamp,
                        json.dumps(entry.canonical_payload(), ensure_ascii=False),
                    )
                    if isinstance(entry, EffectIntentEntry):
                        await conn.execute(
                            _INSERT_INTENT_ROW,
                            self._owner_id,
                            self._run_id,
                            _uuid(session_id.value),
                            _uuid(entry.intent.intent_id.value),
                            entry.intent.tool_name,
                            entry.intent.replay_policy,
                            entry.intent.contract_version,
                            entry.intent.input_schema_digest,
                            entry.intent.canonical_input,
                            entry.intent.source_call_id,
                        )
                if projection is not None:
                    await self._insert_projection(conn, session_id, projection)
                await conn.execute(
                    _ADVANCE_SESSION,
                    self._owner_id,
                    self._run_id,
                    _uuid(session_id.value),
                    len(entries),
                    _uuid(projection.projection_id.value) if projection else None,
                )
                await conn.execute(_ADVANCE_PROGRESS, self._owner_id, self._run_id)
                return SessionCommit(version=version + 1, appended_sequences=sequences)

    async def settle_effect(
        self,
        *,
        session_id: SessionId,
        expected_version: int,
        intent_id: IntentId,
        settlement: EffectSettlement[M3HostUpdate],
        entries: Sequence[SessionEntry],
        projection: ContextProjection | None = None,
        progress: SessionProgressClass = "live",
    ) -> SettleCommit:
        if not entries:
            raise ValueError("a settlement requires at least one result entry")
        async with self._connection() as conn:
            async with conn.transaction():
                try:
                    return await self._settle_locked(
                        conn,
                        session_id=session_id,
                        expected_version=expected_version,
                        intent_id=intent_id,
                        settlement=settlement,
                        entries=entries,
                        projection=projection,
                        progress=progress,
                    )
                except _EvidenceIdentityConflict:
                    return EvidenceConflict()

    async def _settle_locked(
        self,
        conn: Any,
        *,
        session_id: SessionId,
        expected_version: int,
        intent_id: IntentId,
        settlement: EffectSettlement[M3HostUpdate],
        entries: Sequence[SessionEntry],
        projection: ContextProjection | None,
        progress: SessionProgressClass,
    ) -> SettleCommit:
        if await self._hold_lease(conn) is None:
            return LeaseLost()
        session_row = await conn.fetchrow(
            _LOCK_SESSION, self._owner_id, self._run_id, _uuid(session_id.value)
        )
        if session_row is None:
            return EffectMissing(intent_id=intent_id)
        version = int(session_row["version"])
        if version != expected_version:
            return VersionConflict(expected_version=expected_version, current_version=version)
        intent_row = await conn.fetchrow(
            _SELECT_INTENT_FOR_UPDATE,
            self._owner_id,
            self._run_id,
            _uuid(session_id.value),
            _uuid(intent_id.value),
        )
        if intent_row is None:
            return EffectMissing(intent_id=intent_id)
        if intent_row["outcome"] is not None:
            return EffectAlreadySettled(intent_id=intent_id)

        host_update_digest = await self._write_host_update(
            conn, session_id, intent_id, settlement.host_update
        )

        last_sequence = await conn.fetchval(
            "SELECT last_sequence FROM dlightrag_agent_sessions"
            " WHERE owner_id = $1 AND run_id = $2 AND session_id = $3",
            self._owner_id,
            self._run_id,
            _uuid(session_id.value),
        )
        sequences = tuple(range(int(last_sequence) + 1, int(last_sequence) + len(entries) + 1))
        for entry, sequence in zip(entries, sequences, strict=True):
            await conn.execute(
                _INSERT_ENTRY,
                self._owner_id,
                self._run_id,
                _uuid(session_id.value),
                sequence,
                _uuid(entry.entry_id.value),
                entry.entry_type,
                entry.schema_version,
                entry.timestamp,
                json.dumps(entry.canonical_payload(), ensure_ascii=False),
            )
            if isinstance(entry, EffectIntentEntry):
                await conn.execute(
                    _INSERT_INTENT_ROW,
                    self._owner_id,
                    self._run_id,
                    _uuid(session_id.value),
                    _uuid(entry.intent.intent_id.value),
                    entry.intent.tool_name,
                    entry.intent.replay_policy,
                    entry.intent.contract_version,
                    entry.intent.input_schema_digest,
                    entry.intent.canonical_input,
                    entry.intent.source_call_id,
                )
        if projection is not None:
            await self._insert_projection(conn, session_id, projection)

        await conn.execute(
            _SETTLE_INTENT,
            self._owner_id,
            self._run_id,
            _uuid(session_id.value),
            _uuid(intent_id.value),
            settlement.outcome,
            sequences[0],
            json.dumps(settlement.result.content, ensure_ascii=False),
            host_update_digest,
        )
        await conn.execute(
            _ADVANCE_SESSION,
            self._owner_id,
            self._run_id,
            _uuid(session_id.value),
            len(entries),
            _uuid(projection.projection_id.value) if projection else None,
        )
        if progress == "live":
            await conn.execute(_ADVANCE_PROGRESS, self._owner_id, self._run_id)
        return EffectCommit(
            version=version + 1,
            appended_sequences=sequences,
            intent_id=intent_id,
            outcome=settlement.outcome,
        )

    async def _hold_lease(self, conn: Any) -> Any:
        return await conn.fetchval(
            _LEASE_PREDICATE,
            self._owner_id,
            self._run_id,
            self._lease_owner,
            self._fencing_epoch,
        )

    async def load_evidence(self, session_id: SessionId) -> list[OpaqueEvidenceWrite]:
        """Return this run's durable evidence writes, oldest first (adapter read)."""
        async with self._connection() as conn:
            rows = await conn.fetch(
                "SELECT session_id::text, intent_id::text, result_ordinal,"
                " content_digest, locator_digest, content, locator"
                " FROM dlightrag_answer_evidence"
                " WHERE owner_id = $1 AND run_id = $2 AND session_id = $3"
                " ORDER BY created_at, result_ordinal",
                self._owner_id,
                self._run_id,
                _uuid(session_id.value),
            )
        return [
            OpaqueEvidenceWrite(
                session_id=str(row["session_id"]),
                intent_id=str(row["intent_id"]),
                result_ordinal=int(row["result_ordinal"]),
                content_digest=str(row["content_digest"]),
                locator_digest=str(row["locator_digest"]),
                content=bytes(row["content"]),
                locator=bytes(row["locator"]),
            )
            for row in rows
        ]

    async def _insert_projection(
        self, conn: Any, session_id: SessionId, projection: ContextProjection
    ) -> None:
        await conn.execute(
            _INSERT_PROJECTION,
            self._owner_id,
            self._run_id,
            _uuid(session_id.value),
            _uuid(projection.projection_id.value),
            projection.first_retained_sequence,
            projection.covered_through_sequence,
            projection.summary,
            json.dumps(
                [
                    {
                        "through_sequence": anchor.through_sequence,
                        "measured_input_tokens": anchor.measured_input_tokens,
                        "measured_output_tokens": anchor.measured_output_tokens,
                    }
                    for anchor in projection.token_anchors
                ],
                ensure_ascii=False,
            ),
            projection.schema_version,
        )

    async def _write_host_update(
        self,
        conn: Any,
        session_id: SessionId,
        intent_id: IntentId,
        update: M3HostUpdate,
    ) -> str:
        if isinstance(update, EvidenceSettlementUpdate):
            for write in update.evidence:
                await self._write_evidence(conn, write)
            for resource in update.resources:
                await self._write_evidence_resource(conn, resource)
            return _host_update_digest(update)
        if isinstance(update, FetchedResourceSettlementUpdate):
            await self._write_complete_blob(conn, update.complete_blob)
            await self._write_fetched_resource(conn, update.resource)
            for write in update.evidence:
                await self._write_evidence(conn, write)
            return _host_update_digest(update)
        raise ValueError(f"unknown host update variant: {type(update).__name__}")

    async def _write_evidence(self, conn: Any, write: OpaqueEvidenceWrite) -> None:
        await conn.execute(
            _INSERT_EVIDENCE,
            self._owner_id,
            self._run_id,
            write.session_id,
            write.intent_id,
            write.result_ordinal,
            write.content_digest,
            write.locator_digest,
            write.content,
            write.locator,
        )
        row = await conn.fetchrow(
            _SELECT_EVIDENCE_DIGESTS,
            self._owner_id,
            self._run_id,
            write.session_id,
            write.intent_id,
            write.result_ordinal,
        )
        if (
            row is None
            or row["content_digest"] != write.content_digest
            or row["locator_digest"] != write.locator_digest
        ):
            raise _EvidenceIdentityConflict()

    async def _write_evidence_resource(self, conn: Any, write: OpaqueEvidenceResourceWrite) -> None:
        await conn.execute(
            _INSERT_RESOURCE,
            self._owner_id,
            self._run_id,
            write.resource_id,
            "evidence",
            write.safe_name,
            write.media_type,
            json.dumps(write.capabilities, ensure_ascii=False),
            None,
            None,
            write.locator_digest,
            write.session_id,
            write.intent_id,
            write.result_ordinal,
        )
        row = await conn.fetchrow(
            _SELECT_RESOURCE_DIGESTS, self._owner_id, self._run_id, write.resource_id
        )
        if (
            row is None
            or row["kind"] != "evidence"
            or row["locator_digest"] != write.locator_digest
        ):
            raise _EvidenceIdentityConflict()

    async def _write_complete_blob(self, conn: Any, blob: CompleteBlobDescriptor) -> None:
        await conn.execute(_INSERT_BLOB_METADATA, self._owner_id, blob.digest, blob.total_bytes)
        existing = await conn.fetchval(_SELECT_BLOB_SIZE, self._owner_id, blob.digest)
        if existing != blob.total_bytes:
            raise _EvidenceIdentityConflict()
        for index, chunk in enumerate(blob.chunks):
            await conn.execute(_INSERT_BLOB_CHUNK, self._owner_id, blob.digest, index, chunk)

    async def _write_fetched_resource(self, conn: Any, write: OpaqueFetchedResourceWrite) -> None:
        await conn.execute(
            _INSERT_RESOURCE,
            self._owner_id,
            self._run_id,
            write.resource_id,
            "fetched_blob",
            write.safe_name,
            write.media_type,
            json.dumps(write.capabilities, ensure_ascii=False),
            None,
            write.blob_digest,
            write.source_locator_digest,
            write.session_id,
            write.intent_id,
            None,
        )
        row = await conn.fetchrow(
            _SELECT_RESOURCE_DIGESTS, self._owner_id, self._run_id, write.resource_id
        )
        if (
            row is None
            or row["kind"] != "fetched_blob"
            or row["blob_digest"] != write.blob_digest
            or row["locator_digest"] != write.source_locator_digest
        ):
            raise _EvidenceIdentityConflict()


_FINISH_TERMINAL = """
WITH bumped AS (
    UPDATE dlightrag_answer_runs
    SET status = 'succeeded',
        result_json = $5::jsonb,
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
    RETURNING next_event_sequence - 1 AS event_sequence
), inserted AS (
    INSERT INTO dlightrag_answer_run_events (
        owner_id, run_id, event_sequence, event_type, payload
    )
    SELECT $1, $2, event_sequence, 'done',
           jsonb_build_object('status', 'succeeded', 'result', $5::jsonb)
    FROM bumped
    RETURNING event_sequence
)
SELECT event_sequence FROM inserted
"""


class PGProgressStore:
    """One claim-bound RunProgressStore over PostgreSQL."""

    def __init__(
        self,
        *,
        pool: ConnectionPool | None,
        owner_id: str,
        run_id: Any,
        worker_id: str,
        lease_owner: str,
        fencing_epoch: int,
    ) -> None:
        self._pool = pool
        self._owner_id = owner_id
        self._run_id = run_id
        self._worker_id = worker_id
        self._lease_owner = lease_owner
        self._fencing_epoch = fencing_epoch

    @asynccontextmanager
    async def _connection(self) -> AsyncIterator[Any]:
        pool = self._pool if self._pool is not None else await pg_pool.get()
        async with pool.acquire() as conn:
            yield conn

    async def load_stage(self, stage_intent_id: StageIntentId) -> StageRecord | None:
        async with self._connection() as conn:
            row = await conn.fetchrow(
                "SELECT stage_intent_id::text, stage_name, progress_version, state,"
                " state_digest, settled_at::text"
                " FROM dlightrag_answer_run_stages"
                " WHERE owner_id = $1 AND run_id = $2 AND stage_intent_id = $3",
                self._owner_id,
                self._run_id,
                _uuid(stage_intent_id.value),
            )
            if row is None:
                return None
            return StageRecord(
                stage_intent_id=stage_intent_id,
                stage_name=str(row["stage_name"]),
                progress_version=int(row["progress_version"]),
                state=_json_payload(row["state"]),
                state_digest=str(row["state_digest"]),
                evidence_count=0,
                settled_at=row["settled_at"],
            )

    async def settle_terminal(
        self,
        *,
        expected_progress_version: int,
        stage_intent_id: StageIntentId,
        state: JsonValue,
        result: Mapping[str, Any],
    ) -> StageCommitResult:
        """Settle the final Fast stage and the run's succeeded terminal in ONE
        transaction: stage record, result, unique done event, lease release,
        and one Durable Progress increment (M3-D22).
        """
        async with self._connection() as conn:
            async with conn.transaction():
                if (
                    await conn.fetchval(
                        _LEASE_PREDICATE,
                        self._owner_id,
                        self._run_id,
                        self._lease_owner,
                        self._fencing_epoch,
                    )
                    is None
                ):
                    return StageLeaseLost()
                progress = await conn.fetchval(
                    "SELECT durable_progress_version FROM dlightrag_answer_runs"
                    " WHERE owner_id = $1 AND run_id = $2 FOR UPDATE",
                    self._owner_id,
                    self._run_id,
                )
                if progress is None:
                    return StageLeaseLost()
                if int(progress) != expected_progress_version:
                    return StageProgressConflict(
                        expected_progress_version=expected_progress_version,
                        current_progress_version=int(progress),
                    )
                existing = await conn.fetchrow(
                    "SELECT state_digest FROM dlightrag_answer_run_stages"
                    " WHERE owner_id = $1 AND run_id = $2 AND stage_intent_id = $3"
                    " FOR UPDATE",
                    self._owner_id,
                    self._run_id,
                    _uuid(stage_intent_id.value),
                )
                state_json = json.dumps(state, ensure_ascii=False, sort_keys=True)
                state_digest = _sha256(state_json)
                if existing is not None and existing["state_digest"] != state_digest:
                    return StageConflict(stage_intent_id=stage_intent_id)
                if existing is None:
                    await conn.execute(
                        "INSERT INTO dlightrag_answer_run_stages ("
                        " owner_id, run_id, stage_intent_id, stage_name,"
                        " progress_version, state, state_digest)"
                        " VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)",
                        self._owner_id,
                        self._run_id,
                        _uuid(stage_intent_id.value),
                        "final_generation",
                        expected_progress_version,
                        state_json,
                        state_digest,
                    )
                sequence = await conn.fetchval(
                    _FINISH_TERMINAL,
                    self._owner_id,
                    self._run_id,
                    self._lease_owner,
                    self._fencing_epoch,
                    json.dumps(dict(result), ensure_ascii=False),
                )
                if sequence is None:
                    return StageLeaseLost()
                await conn.execute(
                    "UPDATE dlightrag_answer_runs SET"
                    " durable_progress_version = durable_progress_version + 1"
                    " WHERE owner_id = $1 AND run_id = $2",
                    self._owner_id,
                    self._run_id,
                )
                return StageCommit(
                    progress_version=expected_progress_version + 1,
                    stage_intent_id=stage_intent_id,
                    evidence_count=0,
                )

    async def settle_stage(
        self,
        *,
        expected_progress_version: int,
        stage_intent_id: StageIntentId,
        stage_name: str,
        state: JsonValue,
        evidence: Sequence[Any],
    ) -> StageCommitResult:
        async with self._connection() as conn:
            async with conn.transaction():
                if (
                    await conn.fetchval(
                        _LEASE_PREDICATE,
                        self._owner_id,
                        self._run_id,
                        self._lease_owner,
                        self._fencing_epoch,
                    )
                    is None
                ):
                    return StageLeaseLost()
                progress = await conn.fetchval(
                    "SELECT durable_progress_version FROM dlightrag_answer_runs"
                    " WHERE owner_id = $1 AND run_id = $2 FOR UPDATE",
                    self._owner_id,
                    self._run_id,
                )
                if progress is None:
                    return StageLeaseLost()
                if int(progress) != expected_progress_version:
                    return StageProgressConflict(
                        expected_progress_version=expected_progress_version,
                        current_progress_version=int(progress),
                    )
                existing = await conn.fetchrow(
                    "SELECT state_digest FROM dlightrag_answer_run_stages"
                    " WHERE owner_id = $1 AND run_id = $2 AND stage_intent_id = $3"
                    " FOR UPDATE",
                    self._owner_id,
                    self._run_id,
                    _uuid(stage_intent_id.value),
                )
                state_json = json.dumps(state, ensure_ascii=False, sort_keys=True)
                state_digest = _sha256(state_json)
                if existing is not None:
                    if existing["state_digest"] != state_digest:
                        return StageConflict(stage_intent_id=stage_intent_id)
                    return StageCommit(
                        progress_version=expected_progress_version,
                        stage_intent_id=stage_intent_id,
                        evidence_count=0,
                    )
                try:
                    evidence_count = 0
                    for write in evidence:
                        evidence_count += 1
                        # Fast evidence uses the synthetic UUIDv5 fast namespace.
                        await conn.execute(
                            _INSERT_EVIDENCE,
                            self._owner_id,
                            self._run_id,
                            write.session_id,
                            write.intent_id,
                            write.result_ordinal,
                            write.content_digest,
                            write.locator_digest,
                            write.content,
                            write.locator,
                        )
                        row = await conn.fetchrow(
                            _SELECT_EVIDENCE_DIGESTS,
                            self._owner_id,
                            self._run_id,
                            write.session_id,
                            write.intent_id,
                            write.result_ordinal,
                        )
                        if (
                            row is None
                            or row["content_digest"] != write.content_digest
                            or row["locator_digest"] != write.locator_digest
                        ):
                            return StageEvidenceConflict()
                    await conn.execute(
                        "INSERT INTO dlightrag_answer_run_stages ("
                        " owner_id, run_id, stage_intent_id, stage_name,"
                        " progress_version, state, state_digest)"
                        " VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)",
                        self._owner_id,
                        self._run_id,
                        _uuid(stage_intent_id.value),
                        stage_name,
                        expected_progress_version,
                        state_json,
                        state_digest,
                    )
                    await conn.execute(_ADVANCE_PROGRESS, self._owner_id, self._run_id)
                    return StageCommit(
                        progress_version=expected_progress_version + 1,
                        stage_intent_id=stage_intent_id,
                        evidence_count=evidence_count,
                    )
                except _EvidenceIdentityConflict:
                    return StageEvidenceConflict()


def _sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _host_update_digest(update: M3HostUpdate) -> str:
    payload: dict[str, Any]
    if isinstance(update, EvidenceSettlementUpdate):
        payload = {
            "kind": "evidence",
            "evidence": [
                {
                    "session_id": w.session_id,
                    "intent_id": w.intent_id,
                    "result_ordinal": w.result_ordinal,
                    "content_digest": w.content_digest,
                    "locator_digest": w.locator_digest,
                }
                for w in update.evidence
            ],
            "resources": [
                {
                    "resource_id": r.resource_id,
                    "locator_digest": r.locator_digest,
                }
                for r in update.resources
            ],
        }
    elif isinstance(update, FetchedResourceSettlementUpdate):
        payload = {
            "kind": "fetched_resource",
            "resource_id": update.resource.resource_id,
            "blob_digest": update.resource.blob_digest,
            "source_locator_digest": update.resource.source_locator_digest,
            "evidence": [
                {
                    "session_id": w.session_id,
                    "intent_id": w.intent_id,
                    "result_ordinal": w.result_ordinal,
                    "content_digest": w.content_digest,
                    "locator_digest": w.locator_digest,
                }
                for w in update.evidence
            ],
        }
    elif isinstance(update, CommittedSpillUpdate):
        payload = {
            "kind": "committed_spill",
            "resource_id": update.resource_id,
            "content_digest": update.content_digest,
            "size_bytes": update.size_bytes,
        }
    elif isinstance(update, WorkspaceInventoryUpdate):
        payload = {
            "kind": "workspace_inventory",
            "replace_all": update.replace_all,
            "upserts": [record.relative_path for record in update.upserts],
            "deletes": list(update.deletes),
        }
    else:
        raise ValueError(f"unknown host update variant: {type(update).__name__}")
    return _sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _json_payload(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        loaded = json.loads(value)
        return dict(loaded) if isinstance(loaded, dict) else {}
    return dict(value) if isinstance(value, Mapping) else {}


def _token_anchor(payload: Any) -> Any:
    from dlightrag_agent.session.projection import TokenAnchor

    return TokenAnchor(
        through_sequence=int(payload["through_sequence"]),
        measured_input_tokens=int(payload["measured_input_tokens"]),
        measured_output_tokens=int(payload["measured_output_tokens"]),
    )


def _decode_entry(row: Any, *, owner_id: str, run_id: str, session_id: SessionId) -> SessionEntry:
    from dlightrag_agent.session.entries import ENTRY_TYPE_TO_CLASS, decode_entry_payload
    from dlightrag_agent.session.ids import EntryId

    entry_class = ENTRY_TYPE_TO_CLASS.get(str(row["entry_type"]))
    if entry_class is None:
        raise ValueError(f"unknown journal entry type in storage: {row['entry_type']}")
    payload = _json_payload(row["payload_json"])
    return decode_entry_payload(
        entry_type=str(row["entry_type"]),
        entry_id=EntryId(str(row["entry_id"])),
        session_id=session_id,
        sequence=int(row["sequence"]),
        timestamp=row["timestamp"],
        payload=payload,
    )


__all__ = [
    "PGJournalStore",
    "PGProgressStore",
]
