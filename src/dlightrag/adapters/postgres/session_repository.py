# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Claim-bound PostgreSQL Session, progress, evidence, resource, and blob repositories.

Every bound store embeds owner, run, worker, lease owner, and fencing epoch at
claim time; its public methods carry no fencing parameters. Each mutating
transaction starts by locking the run row under the live lease/epoch predicate,
so a stale or lost lease changes zero rows.

Runtime transitions atomically commit HostDelta, ordered Entries, exact typed
registers, Session commit sequence, and durable run progress.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any

from dlightrag.adapters.postgres._operations import ConnectionPool
from dlightrag.adapters.postgres._pool import pg_pool
from dlightrag.engine.agent.session.effects import JsonValue
from dlightrag.engine.agent.session.entries import (
    SessionEntry,
)
from dlightrag.engine.agent.session.ids import (
    EntryId,
    IntentId,
    LaneId,
    SessionId,
    StageIntentId,
)
from dlightrag.engine.agent.session.registers import (
    DeleteRegister,
    LaneHead,
    LaneState,
    RegisterRecord,
    RegisterRef,
    SetRegister,
    decode_register,
)
from dlightrag.engine.agent.session.repository import (
    AgentSessionSnapshot,
)
from dlightrag.engine.agent.session.transactions import (
    RegisterConflict,
    SessionTransaction,
    TransactionCommit,
    TransactionLeaseLost,
    TransactionOutcome,
)
from dlightrag.engine.runtime.progress import (
    StageCommit,
    StageCommitResult,
    StageConflict,
    StageEvidenceConflict,
    StageLeaseLost,
    StageProgressConflict,
    StageRecord,
)
from dlightrag.engine.runtime.settlements import (
    CompleteBlobDescriptor,
    EffectHostUpdate,
    MemoryOperationSettlement,
    OpaqueEvidenceResourceWrite,
    OpaqueEvidenceWrite,
    OpaqueFetchedResourceWrite,
)

_LEASE_PREDICATE = """
SELECT 1
FROM dlightrag_answer_runs
WHERE owner_id = $1 AND run_id = $2
  AND lease_owner = $3 AND fencing_epoch = $4
  AND status = 'running' AND lease_expires_at > NOW()
FOR UPDATE
"""

_CHILD_LEASE_PREDICATE = """
SELECT 1
FROM dlightrag_answer_child_sessions
WHERE owner_id = $1 AND run_id = $2 AND child_session_id = $3
  AND lease_owner = $4 AND fencing_epoch = $5
  AND status = 'running' AND lease_expires_at > NOW()
FOR UPDATE
"""

_LOCK_SESSION = """
SELECT lease_run_id::text, commit_sequence, fencing_epoch, last_sequence
FROM dlightrag_agent_sessions
WHERE owner_id = $1 AND session_id = $2
FOR UPDATE
"""

_CREATE_SESSION = """
INSERT INTO dlightrag_agent_sessions (
    owner_id, session_id, lease_run_id, commit_sequence, fencing_epoch
)
VALUES ($1, $2, $3, 0, $4)
ON CONFLICT (owner_id, session_id) DO NOTHING
"""

_ACTIVE_SESSION_RUN = """
SELECT 1
FROM dlightrag_answer_runs
WHERE owner_id = $1 AND run_id = $2
  AND status = 'running' AND lease_expires_at > NOW()
"""

_INSERT_ENTRY = """
INSERT INTO dlightrag_agent_session_entries (
    owner_id, session_id, sequence, entry_id, parent_entry_id,
    entry_type, schema_version, timestamp, payload_json
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb)
"""

_ADVANCE_SESSION = """
UPDATE dlightrag_agent_sessions
SET commit_sequence = commit_sequence + 1,
    last_sequence = last_sequence + $3,
    updated_at = NOW()
WHERE owner_id = $1 AND session_id = $2
RETURNING commit_sequence
"""

_ADVANCE_PROGRESS = """
UPDATE dlightrag_answer_runs
SET durable_progress_version = durable_progress_version + 1,
    updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2
"""

_INSERT_MEMORY_OPERATION_EVENT = """
WITH bumped AS (
    UPDATE dlightrag_answer_runs
    SET next_event_sequence = next_event_sequence + 1,
        updated_at = NOW()
    WHERE owner_id = $1 AND run_id = $2
      AND lease_owner = $3 AND fencing_epoch = $4
      AND status = 'running' AND lease_expires_at > NOW()
    RETURNING next_event_sequence - 1 AS event_sequence
)
INSERT INTO dlightrag_answer_run_events (
    owner_id, run_id, event_sequence, event_type, payload
)
SELECT $1, $2, event_sequence, 'memory_operation_settled', $5::jsonb
FROM bumped
"""

_SELECT_SESSION_SNAPSHOT = """
SELECT commit_sequence
FROM dlightrag_agent_sessions
WHERE owner_id = $1 AND session_id = $2
"""

_SELECT_ENTRIES = """
SELECT sequence, entry_id::text, parent_entry_id::text, entry_type,
       schema_version, timestamp, payload_json
FROM dlightrag_agent_session_entries
WHERE owner_id = $1 AND session_id = $2
ORDER BY sequence
"""

_SELECT_REGISTERS = """
SELECT register_kind, register_key, sequence, payload_json
FROM dlightrag_agent_session_registers
WHERE owner_id = $1 AND session_id = $2
ORDER BY register_kind, register_key
"""

_SELECT_REGISTER_FOR_UPDATE = """
SELECT sequence, payload_json
FROM dlightrag_agent_session_registers
WHERE owner_id = $1 AND session_id = $2
  AND register_kind = $3 AND register_key = $4
FOR UPDATE
"""

_SET_REGISTER = """
INSERT INTO dlightrag_agent_session_registers (
    owner_id, session_id, register_kind, register_key, sequence, payload_json
)
VALUES ($1, $2, $3, $4, $5, $6::jsonb)
ON CONFLICT (owner_id, session_id, register_kind, register_key)
DO UPDATE SET sequence = EXCLUDED.sequence, payload_json = EXCLUDED.payload_json
"""

_DELETE_REGISTER = """
DELETE FROM dlightrag_agent_session_registers
WHERE owner_id = $1 AND session_id = $2
  AND register_kind = $3 AND register_key = $4
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


class PGAgentSessionRepository:
    """One claim-bound Agent Session Repository over PostgreSQL."""

    def __init__(
        self,
        *,
        pool: ConnectionPool | None,
        owner_id: str,
        run_id: Any,
        worker_id: str,
        lease_owner: str,
        fencing_epoch: int,
        primary_session_id: SessionId | None = None,
        child_session_id: SessionId | None = None,
    ) -> None:
        self._pool = pool
        self._owner_id = owner_id
        self._run_id = run_id
        self._worker_id = worker_id
        self._lease_owner = lease_owner
        self._fencing_epoch = fencing_epoch
        self._primary_session_id = primary_session_id
        self._child_session_id = child_session_id

    def for_child(
        self, child_session_id: SessionId, *, fencing_epoch: int
    ) -> PGAgentSessionRepository:
        """Bind the same HostDelta owner to one independently leased Child Session."""
        return PGAgentSessionRepository(
            pool=self._pool,
            owner_id=self._owner_id,
            run_id=self._run_id,
            worker_id=self._worker_id,
            lease_owner=self._lease_owner,
            fencing_epoch=fencing_epoch,
            primary_session_id=self._primary_session_id,
            child_session_id=child_session_id,
        )

    @asynccontextmanager
    async def _connection(self) -> AsyncIterator[Any]:
        pool = self._pool if self._pool is not None else await pg_pool.get()
        async with pool.acquire() as conn:
            yield conn

    async def load(self, session_id: SessionId) -> AgentSessionSnapshot:
        async with self._connection() as conn:
            session_row = await conn.fetchrow(
                _SELECT_SESSION_SNAPSHOT, self._owner_id, _uuid(session_id.value)
            )
            if session_row is None:
                return AgentSessionSnapshot(
                    session_id=session_id,
                    commit_sequence=0,
                    entries=(),
                    registers=(),
                )
            entries = await self._load_entries(conn, session_id)
            registers = await self._load_registers(conn, session_id)
            return AgentSessionSnapshot(
                session_id=session_id,
                commit_sequence=int(session_row["commit_sequence"]),
                entries=tuple(entries),
                registers=tuple(registers),
            )

    async def _load_entries(self, conn: Any, session_id: SessionId) -> list[SessionEntry]:
        rows = await conn.fetch(_SELECT_ENTRIES, self._owner_id, _uuid(session_id.value))
        entries: list[SessionEntry] = []
        for row in rows:
            entries.append(_decode_entry(row, session_id=session_id))
        return entries

    async def _load_registers(self, conn: Any, session_id: SessionId) -> list[RegisterRecord]:
        rows = await conn.fetch(
            _SELECT_REGISTERS,
            self._owner_id,
            _uuid(session_id.value),
        )
        records: list[RegisterRecord] = []
        for row in rows:
            ref = RegisterRef(
                kind=str(row["register_kind"]),  # type: ignore[arg-type]
                key=str(row["register_key"]),
            )
            value = decode_register(
                kind=ref.kind,
                payload=_json_payload(row["payload_json"]),
            )
            if value.ref != ref:
                raise ValueError("Agent Session register payload identity is corrupt")
            records.append(RegisterRecord(value=value, sequence=int(row["sequence"])))
        return records

    async def transact(
        self,
        *,
        session_id: SessionId,
        fencing_epoch: int,
        transaction: SessionTransaction[EffectHostUpdate],
    ) -> TransactionOutcome:
        """Apply one exact-register-CAS Session transaction."""
        if fencing_epoch != self._fencing_epoch:
            return TransactionLeaseLost()
        if self._child_session_id is not None and session_id != self._child_session_id:
            return TransactionLeaseLost()
        if (
            self._child_session_id is None
            and self._primary_session_id is not None
            and session_id != self._primary_session_id
        ):
            return TransactionLeaseLost()
        async with self._connection() as conn:
            async with conn.transaction():
                if await self._hold_lease(conn) is None:
                    return TransactionLeaseLost()
                await conn.execute(
                    _CREATE_SESSION,
                    self._owner_id,
                    _uuid(session_id.value),
                    self._run_id,
                    fencing_epoch,
                )
                session_row = await conn.fetchrow(
                    _LOCK_SESSION, self._owner_id, _uuid(session_id.value)
                )
                stored_epoch = int(session_row["fencing_epoch"])
                stored_run_id = str(session_row["lease_run_id"])
                current_run_id = str(self._run_id)
                if stored_run_id != current_run_id:
                    if (
                        await conn.fetchval(
                            _ACTIVE_SESSION_RUN,
                            self._owner_id,
                            _uuid(stored_run_id),
                        )
                        is not None
                    ):
                        return TransactionLeaseLost()
                    await conn.execute(
                        "UPDATE dlightrag_agent_sessions"
                        " SET lease_run_id = $3, fencing_epoch = $4"
                        " WHERE owner_id = $1 AND session_id = $2",
                        self._owner_id,
                        _uuid(session_id.value),
                        self._run_id,
                        fencing_epoch,
                    )
                elif stored_epoch > fencing_epoch:
                    return TransactionLeaseLost()
                elif stored_epoch < fencing_epoch:
                    await conn.execute(
                        "UPDATE dlightrag_agent_sessions SET fencing_epoch = $3"
                        " WHERE owner_id = $1 AND session_id = $2",
                        self._owner_id,
                        _uuid(session_id.value),
                        fencing_epoch,
                    )
                current_sequence = int(session_row["commit_sequence"])
                for expectation in transaction.expectations:
                    row = await conn.fetchrow(
                        _SELECT_REGISTER_FOR_UPDATE,
                        self._owner_id,
                        _uuid(session_id.value),
                        expectation.ref.kind,
                        expectation.ref.key,
                    )
                    actual = int(row["sequence"]) if row is not None else None
                    if actual != expectation.sequence:
                        return RegisterConflict(
                            ref=expectation.ref,
                            expected_sequence=expectation.sequence,
                            current_sequence=actual,
                        )
                await self._validate_transaction_entries(conn, session_id, transaction.entries)
                await self._validate_transaction_registers(conn, session_id, transaction)
                next_sequence = current_sequence + 1
                last_entry_sequence = int(session_row["last_sequence"])
                entry_sequences = tuple(
                    range(
                        last_entry_sequence + 1,
                        last_entry_sequence + 1 + len(transaction.entries),
                    )
                )
                if transaction.host_delta is not None:
                    settlement = transaction.host_delta
                    await self._write_host_update(
                        conn,
                        session_id,
                        settlement.intent_id,
                        settlement.value,
                    )
                    if settlement.value.memory_operation is not None:
                        await conn.execute(
                            _INSERT_MEMORY_OPERATION_EVENT,
                            self._owner_id,
                            self._run_id,
                            self._lease_owner,
                            self._fencing_epoch,
                            json.dumps(
                                _memory_event_payload(
                                    session_id,
                                    settlement.intent_id,
                                    settlement.value.memory_operation,
                                ),
                                ensure_ascii=False,
                            ),
                        )
                for entry, sequence in zip(transaction.entries, entry_sequences, strict=True):
                    await self._insert_entry(conn, session_id, entry, sequence)
                register_sequences: list[tuple[RegisterRef, int]] = []
                for write in transaction.register_writes:
                    if isinstance(write, SetRegister):
                        await conn.execute(
                            _SET_REGISTER,
                            self._owner_id,
                            _uuid(session_id.value),
                            write.ref.kind,
                            write.ref.key,
                            next_sequence,
                            json.dumps(write.value.canonical_payload(), ensure_ascii=False),
                        )
                    elif isinstance(write, DeleteRegister):
                        await conn.execute(
                            _DELETE_REGISTER,
                            self._owner_id,
                            _uuid(session_id.value),
                            write.ref.kind,
                            write.ref.key,
                        )
                    register_sequences.append((write.ref, next_sequence))
                await conn.execute(
                    _ADVANCE_SESSION,
                    self._owner_id,
                    _uuid(session_id.value),
                    len(transaction.entries),
                )
                if transaction.advances_durable_progress:
                    await conn.execute(_ADVANCE_PROGRESS, self._owner_id, self._run_id)
                return TransactionCommit(
                    commit_sequence=next_sequence,
                    appended_sequences=entry_sequences,
                    register_sequences=tuple(register_sequences),
                )

    async def _validate_transaction_registers(
        self,
        conn: Any,
        session_id: SessionId,
        transaction: SessionTransaction[EffectHostUpdate],
    ) -> None:
        rows = await conn.fetch(
            "SELECT register_kind, register_key, payload_json"
            " FROM dlightrag_agent_session_registers"
            " WHERE owner_id = $1 AND session_id = $2",
            self._owner_id,
            _uuid(session_id.value),
        )
        values = {}
        for row in rows:
            ref = RegisterRef(
                kind=str(row["register_kind"]),  # type: ignore[arg-type]
                key=str(row["register_key"]),
            )
            values[ref] = decode_register(
                kind=ref.kind,
                payload=_json_payload(row["payload_json"]),
            )
        for write in transaction.register_writes:
            if isinstance(write, SetRegister):
                values[write.ref] = write.value
            elif isinstance(write, DeleteRegister):
                values.pop(write.ref, None)
        refs = set(values)
        heads = {ref.key for ref in refs if ref.kind == "lane_head"}
        states = {ref.key for ref in refs if ref.kind == "lane_state"}
        if heads != states or LaneId.main().value not in heads:
            raise ValueError("Session registers require complete main and Lane pairs")
        if transaction.entries:
            advanced_lanes = {
                write.value.lane_id
                for write in transaction.register_writes
                if isinstance(write, SetRegister)
                and isinstance(write.value, LaneHead)
                and write.value.entry_id == transaction.entries[-1].entry_id
            }
            for advanced_lane_id in advanced_lanes:
                state = values.get(LaneState(advanced_lane_id).ref)
                if isinstance(state, LaneState) and state.archived:
                    raise ValueError("an archived Lane is not writable")

    async def _validate_transaction_entries(
        self,
        conn: Any,
        session_id: SessionId,
        entries: Sequence[SessionEntry],
    ) -> None:
        rows = await conn.fetch(
            "SELECT entry_id::text, parent_entry_id::text"
            " FROM dlightrag_agent_session_entries"
            " WHERE owner_id = $1 AND session_id = $2",
            self._owner_id,
            _uuid(session_id.value),
        )
        known = {EntryId(str(row["entry_id"])) for row in rows}
        roots = sum(row["parent_entry_id"] is None for row in rows)
        for entry in entries:
            if entry.session_id != session_id:
                raise ValueError("transaction Entry belongs to another Session")
            if entry.entry_id in known:
                raise ValueError("transaction Entry identity already exists")
            if entry.parent_entry_id is None:
                if known or roots:
                    raise ValueError("only the first Session Entry can be a root")
                roots += 1
            elif entry.parent_entry_id not in known:
                raise ValueError("transaction Entry parent is missing")
            known.add(entry.entry_id)

    async def _insert_entry(
        self,
        conn: Any,
        session_id: SessionId,
        entry: SessionEntry,
        sequence: int,
    ) -> None:
        await conn.execute(
            _INSERT_ENTRY,
            self._owner_id,
            _uuid(session_id.value),
            sequence,
            _uuid(entry.entry_id.value),
            (_uuid(entry.parent_entry_id.value) if entry.parent_entry_id is not None else None),
            entry.entry_type,
            entry.schema_version,
            entry.timestamp,
            json.dumps(entry.canonical_payload(), ensure_ascii=False),
        )

    async def _hold_lease(self, conn: Any) -> Any:
        if self._child_session_id is not None:
            return await conn.fetchval(
                _CHILD_LEASE_PREDICATE,
                self._owner_id,
                self._run_id,
                _uuid(self._child_session_id.value),
                self._lease_owner,
                self._fencing_epoch,
            )
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

    async def _write_host_update(
        self,
        conn: Any,
        session_id: SessionId,
        intent_id: IntentId,
        update: EffectHostUpdate,
    ) -> str:
        for write in update.evidence:
            await self._write_evidence(conn, write)
        for resource in update.resources:
            await self._write_evidence_resource(conn, resource)
        for fetched in update.fetched:
            await self._write_complete_blob(conn, fetched.complete_blob)
            await self._write_fetched_resource(conn, fetched.resource)
            for write in fetched.evidence:
                await self._write_evidence(conn, write)

        if update.committed_outputs:
            from dlightrag.adapters.postgres.workspace import _upsert_spill
            from dlightrag.engine.runtime.workspace import CommittedSpillRecord

            for output in update.committed_outputs:
                await _upsert_spill(
                    conn,
                    self._owner_id,
                    self._run_id,
                    CommittedSpillRecord(
                        resource_id=output.resource_id,
                        content_digest=output.content_digest,
                        size_bytes=output.size_bytes,
                        session_id=output.session_id,
                        intent_id=output.intent_id,
                    ),
                )

        inventory = update.workspace_inventory
        if inventory is not None:
            if inventory.replace_all:
                await conn.execute(
                    "DELETE FROM dlightrag_answer_workspace_inventory"
                    " WHERE owner_id = $1 AND run_id = $2",
                    self._owner_id,
                    self._run_id,
                )
            else:
                for path in inventory.deletes:
                    await conn.execute(
                        "DELETE FROM dlightrag_answer_workspace_inventory"
                        " WHERE owner_id = $1 AND run_id = $2 AND relative_path = $3",
                        self._owner_id,
                        self._run_id,
                        path,
                    )
            for record in inventory.upserts:
                await conn.execute(
                    "INSERT INTO dlightrag_answer_workspace_inventory ("
                    " owner_id, run_id, relative_path, entry_type, mode, size_bytes, content_digest)"
                    " VALUES ($1, $2, $3, $4, $5, $6, $7)"
                    " ON CONFLICT (owner_id, run_id, relative_path) DO UPDATE SET"
                    " entry_type = EXCLUDED.entry_type, mode = EXCLUDED.mode,"
                    " size_bytes = EXCLUDED.size_bytes, content_digest = EXCLUDED.content_digest",
                    self._owner_id,
                    self._run_id,
                    record.relative_path,
                    record.entry_type,
                    record.mode,
                    record.size_bytes,
                    record.content_digest,
                )
        return _host_update_digest(update)

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
        and one Durable Progress increment.
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


def _memory_event_payload(
    session_id: SessionId,
    intent_id: IntentId,
    operation: MemoryOperationSettlement,
) -> dict[str, Any]:
    return {
        "body": operation.body,
        "change_id": operation.change_id,
        "intent_id": intent_id.value,
        "kind": operation.kind,
        "memory_ids": list(operation.memory_ids),
        "operation": operation.operation,
        "outcome": operation.outcome,
        "session_id": session_id.value,
        "supersedes_id": operation.supersedes_id,
        "target_change_id": operation.target_change_id,
    }


def _host_update_digest(update: EffectHostUpdate) -> str:
    payload = {
        "evidence": [
            {
                "session_id": write.session_id,
                "intent_id": write.intent_id,
                "result_ordinal": write.result_ordinal,
                "content_digest": write.content_digest,
                "locator_digest": write.locator_digest,
            }
            for write in update.evidence
        ],
        "resources": [
            {
                "resource_id": resource.resource_id,
                "locator_digest": resource.locator_digest,
            }
            for resource in update.resources
        ],
        "fetched": [
            {
                "resource_id": item.resource.resource_id,
                "blob_digest": item.resource.blob_digest,
                "source_locator_digest": item.resource.source_locator_digest,
            }
            for item in update.fetched
        ],
        "committed_outputs": [
            {
                "resource_id": output.resource_id,
                "content_digest": output.content_digest,
                "size_bytes": output.size_bytes,
            }
            for output in update.committed_outputs
        ],
        "workspace_inventory": (
            None
            if update.workspace_inventory is None
            else {
                "replace_all": update.workspace_inventory.replace_all,
                "upserts": [record.relative_path for record in update.workspace_inventory.upserts],
                "deletes": list(update.workspace_inventory.deletes),
            }
        ),
        "memory_operation": (
            None
            if update.memory_operation is None
            else {
                "operation": update.memory_operation.operation,
                "outcome": update.memory_operation.outcome,
                "change_id": update.memory_operation.change_id,
                "memory_ids": list(update.memory_operation.memory_ids),
                "kind": update.memory_operation.kind,
                "body": update.memory_operation.body,
                "supersedes_id": update.memory_operation.supersedes_id,
                "target_change_id": update.memory_operation.target_change_id,
            }
        ),
    }
    return _sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _json_payload(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        loaded = json.loads(value)
        return dict(loaded) if isinstance(loaded, dict) else {}
    return dict(value) if isinstance(value, Mapping) else {}


def _decode_entry(row: Any, *, session_id: SessionId) -> SessionEntry:
    from dlightrag.engine.agent.session.entries import ENTRY_TYPE_TO_CLASS, decode_entry_payload
    from dlightrag.engine.agent.session.ids import EntryId

    entry_class = ENTRY_TYPE_TO_CLASS.get(str(row["entry_type"]))
    if entry_class is None:
        raise ValueError(f"unknown Session Entry type in storage: {row['entry_type']}")
    payload = _json_payload(row["payload_json"])
    return decode_entry_payload(
        entry_type=str(row["entry_type"]),
        entry_id=EntryId(str(row["entry_id"])),
        session_id=session_id,
        sequence=int(row["sequence"]),
        timestamp=row["timestamp"],
        payload=payload,
        parent_entry_id=(
            EntryId(str(row["parent_entry_id"])) if row["parent_entry_id"] is not None else None
        ),
    )


__all__ = [
    "PGAgentSessionRepository",
    "PGProgressStore",
]
