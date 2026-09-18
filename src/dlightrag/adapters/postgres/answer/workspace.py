# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Claim-bound PostgreSQL workspace epoch, inventory, and spill digests."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

from dlightrag.adapters.postgres.core._operations import ConnectionPool
from dlightrag.adapters.postgres.core._pool import pg_pool
from dlightrag.engine.runtime.settlements import InventoryPathRecord
from dlightrag.engine.runtime.workspace import (
    DEFAULT_SESSION_NOTES_LIMITS,
    SESSION_NOTES_BUDGET_REFUSED,
    SESSION_NOTES_LEASE_LOST,
    CommittedSpillRecord,
    HandoffCommit,
    HandoffConflict,
    HandoffLeaseLost,
    HandoffResult,
    InventoryReplaceResult,
    SessionNoteRecord,
    SessionNotesLimits,
    SessionNotesPromotion,
    _validate_spill_page_limit,
    note_digest,
    select_promotable_session_notes,
    validate_note_path,
)

#: Serializes same-Session promotions. The row locks below cover the notes that
#: exist, not the notes that do not, so two Runs of one Session could each admit a note
#: against a plane neither had seen the other's: the Session row is the one lock they
#: share. A Session with no row yet (a bind-time migration into a brand-new Session)
#: has nothing to serialize against, and its write is refused by the note table's own
#: foreign key instead.
_LOCK_SESSION_FOR_NOTES = """
SELECT 1
FROM dlightrag_agent_sessions
WHERE owner_id = $1 AND session_id = $2::text::uuid
FOR UPDATE
"""

_LEASE = """
SELECT 1
FROM dlightrag_runs
WHERE owner_id = $1 AND run_id = $2
  AND lease_owner = $3 AND fencing_epoch = $4
  AND status = 'running' AND lease_expires_at > NOW()
FOR UPDATE
"""


class PGWorkspaceStore:
    """Fenced workspace metadata for one claimed run."""

    def __init__(
        self,
        *,
        pool: ConnectionPool | None = None,
        owner_id: str,
        run_id: uuid.UUID,
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

    async def handoff_epoch(
        self,
        *,
        expected_epoch: int | None,
        destination_epoch: int,
        inventory: Sequence[InventoryPathRecord],
    ) -> HandoffResult:
        if destination_epoch < 1:
            raise ValueError("destination epoch must be positive")
        async with self._connection() as conn:
            async with conn.transaction():
                if (
                    await conn.fetchval(
                        _LEASE, self._owner_id, self._run_id, self._lease_owner, self._fencing_epoch
                    )
                    is None
                ):
                    return HandoffLeaseLost()
                current = await conn.fetchval(
                    "SELECT agent_workspace_epoch FROM dlightrag_runs"
                    " WHERE owner_id = $1 AND run_id = $2 FOR UPDATE",
                    self._owner_id,
                    self._run_id,
                )
                current_epoch = int(current) if current is not None else None
                if current_epoch != expected_epoch:
                    return HandoffConflict(
                        expected_epoch=expected_epoch, current_epoch=current_epoch
                    )
                updated = await conn.fetchval(
                    "UPDATE dlightrag_runs SET agent_workspace_epoch = $3, updated_at = NOW()"
                    " WHERE owner_id = $1 AND run_id = $2"
                    " AND agent_workspace_epoch IS NOT DISTINCT FROM $4"
                    " RETURNING agent_workspace_epoch",
                    self._owner_id,
                    self._run_id,
                    destination_epoch,
                    expected_epoch,
                )
                if updated is None:
                    return HandoffLeaseLost()
                await self._replace_inventory_locked(conn, inventory)
                return HandoffCommit(workspace_epoch=int(updated))

    async def load_inventory(self) -> tuple[InventoryPathRecord, ...]:
        return await load_run_inventory(
            owner_id=self._owner_id, run_id=self._run_id, pool=self._pool
        )

    async def replace_inventory(
        self, records: Sequence[InventoryPathRecord]
    ) -> InventoryReplaceResult:
        async with self._connection() as conn:
            async with conn.transaction():
                if (
                    await conn.fetchval(
                        _LEASE, self._owner_id, self._run_id, self._lease_owner, self._fencing_epoch
                    )
                    is None
                ):
                    return "lease_lost"
                await self._replace_inventory_locked(conn, records)
                return "committed"

    async def load_session_notes(self, *, session_id: str) -> tuple[SessionNoteRecord, ...]:
        """Read one Session's notes without needing a live claim.

        Memory belongs to the Session, so the read is not fenced by this Run's lease:
        a recovered attempt reads the same plane a fresh one would, and a Session whose
        plane is unreadable is the caller's degradation to record, not an error here.
        """
        return await load_session_notes(
            owner_id=self._owner_id, session_id=session_id, pool=self._pool
        )

    async def promote_session_notes(
        self,
        *,
        session_id: str,
        upserts: Sequence[SessionNoteRecord],
        deletes: Sequence[str] = (),
        limits: SessionNotesLimits = DEFAULT_SESSION_NOTES_LIMITS,
    ) -> SessionNotesPromotion:
        """Promote notes under this Run's lease, refusing what the plane cannot hold.

        The plane's budget is the Session's — and configurable — so admission is decided
        inside the same transaction that reads it: a note that does not fit is refused
        for that note alone, and nothing older is evicted to make room.
        """
        refused_paths = tuple(
            note.relative_path for note in upserts if not validate_note_path(note.relative_path)
        )
        admissible = tuple(note for note in upserts if validate_note_path(note.relative_path))
        async with self._connection() as conn:
            async with conn.transaction():
                if (
                    await conn.fetchval(
                        _LEASE, self._owner_id, self._run_id, self._lease_owner, self._fencing_epoch
                    )
                    is None
                ):
                    return SessionNotesPromotion(degraded_reason=SESSION_NOTES_LEASE_LOST)
                await conn.fetchval(_LOCK_SESSION_FOR_NOTES, self._owner_id, session_id)
                rows = await conn.fetch(
                    "SELECT relative_path, size_bytes FROM dlightrag_answer_session_notes"
                    " WHERE owner_id = $1 AND session_id = $2::uuid FOR UPDATE",
                    self._owner_id,
                    session_id,
                )
                existing = {str(row["relative_path"]): int(row["size_bytes"]) for row in rows}
                admitted, over_budget = select_promotable_session_notes(
                    existing=existing,
                    upserts=admissible,
                    deletes=deletes,
                    limits=limits,
                )
                deleted = 0
                for path in deletes:
                    if path not in existing:
                        continue
                    await conn.execute(
                        "DELETE FROM dlightrag_answer_session_notes"
                        " WHERE owner_id = $1 AND session_id = $2::uuid AND relative_path = $3",
                        self._owner_id,
                        session_id,
                        path,
                    )
                    deleted += 1
                for note in admitted:
                    await conn.execute(
                        "INSERT INTO dlightrag_answer_session_notes ("
                        " owner_id, session_id, relative_path, size_bytes, content_digest,"
                        " content, revision, written_by_run_id, updated_at)"
                        " VALUES ($1, $2::uuid, $3, $4, $5, $6, 1, $7, NOW())"
                        " ON CONFLICT (owner_id, session_id, relative_path) DO UPDATE SET"
                        " size_bytes = EXCLUDED.size_bytes,"
                        " content_digest = EXCLUDED.content_digest,"
                        " content = EXCLUDED.content,"
                        " revision = dlightrag_answer_session_notes.revision + 1,"
                        " written_by_run_id = EXCLUDED.written_by_run_id,"
                        " updated_at = NOW()",
                        self._owner_id,
                        session_id,
                        note.relative_path,
                        len(note.content),
                        note_digest(note.content),
                        note.content,
                        self._run_id,
                    )
        refused = (*refused_paths, *over_budget)
        return SessionNotesPromotion(
            promoted=len(admitted),
            deleted=deleted,
            refused_paths=refused,
            degraded_reason=(SESSION_NOTES_BUDGET_REFUSED if refused else None),
        )

    async def register_spill(self, spill: CommittedSpillRecord) -> InventoryReplaceResult:
        async with self._connection() as conn:
            async with conn.transaction():
                if (
                    await conn.fetchval(
                        _LEASE, self._owner_id, self._run_id, self._lease_owner, self._fencing_epoch
                    )
                    is None
                ):
                    return "lease_lost"
                await _upsert_spill(conn, self._owner_id, self._run_id, spill)
                return "committed"

    async def load_spills_page(
        self, *, after_resource_id: str | None, limit: int
    ) -> tuple[CommittedSpillRecord, ...]:
        _validate_spill_page_limit(limit)
        async with self._connection() as conn:
            if after_resource_id is None:
                rows = await conn.fetch(
                    "SELECT resource_id, content_digest, size_bytes, session_id::text,"
                    " intent_id::text FROM dlightrag_answer_committed_spills"
                    " WHERE owner_id = $1 AND run_id = $2"
                    " ORDER BY resource_id LIMIT $3",
                    self._owner_id,
                    self._run_id,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    "SELECT resource_id, content_digest, size_bytes, session_id::text,"
                    " intent_id::text FROM dlightrag_answer_committed_spills"
                    " WHERE owner_id = $1 AND run_id = $2 AND resource_id > $3"
                    " ORDER BY resource_id LIMIT $4",
                    self._owner_id,
                    self._run_id,
                    after_resource_id,
                    limit,
                )
        return tuple(
            CommittedSpillRecord(
                resource_id=str(row["resource_id"]),
                content_digest=str(row["content_digest"]),
                size_bytes=int(row["size_bytes"]),
                session_id=str(row["session_id"]),
                intent_id=str(row["intent_id"]),
            )
            for row in rows
        )

    async def load_recent_spills(self, *, limit: int) -> tuple[CommittedSpillRecord, ...]:
        _validate_spill_page_limit(limit)
        async with self._connection() as conn:
            rows = await conn.fetch(
                "SELECT resource_id, content_digest, size_bytes, session_id::text,"
                " intent_id::text FROM dlightrag_answer_committed_spills"
                " WHERE owner_id = $1 AND run_id = $2"
                " ORDER BY intent_id DESC, resource_id DESC LIMIT $3",
                self._owner_id,
                self._run_id,
                limit,
            )
        return tuple(
            CommittedSpillRecord(
                resource_id=str(row["resource_id"]),
                content_digest=str(row["content_digest"]),
                size_bytes=int(row["size_bytes"]),
                session_id=str(row["session_id"]),
                intent_id=str(row["intent_id"]),
            )
            for row in rows
        )

    async def clear_spills(self) -> InventoryReplaceResult:
        async with self._connection() as conn:
            async with conn.transaction():
                if (
                    await conn.fetchval(
                        _LEASE, self._owner_id, self._run_id, self._lease_owner, self._fencing_epoch
                    )
                    is None
                ):
                    return "lease_lost"
                await conn.execute(
                    "DELETE FROM dlightrag_answer_committed_spills"
                    " WHERE owner_id = $1 AND run_id = $2",
                    self._owner_id,
                    self._run_id,
                )
                await conn.execute(
                    "DELETE FROM dlightrag_answer_resources"
                    " WHERE owner_id = $1 AND run_id = $2 AND kind = 'committed_spill'",
                    self._owner_id,
                    self._run_id,
                )
                return "committed"

    async def _replace_inventory_locked(
        self, conn: Any, records: Sequence[InventoryPathRecord]
    ) -> None:
        await conn.execute(
            "DELETE FROM dlightrag_answer_workspace_inventory WHERE owner_id = $1 AND run_id = $2",
            self._owner_id,
            self._run_id,
        )
        for record in records:
            await conn.execute(
                "INSERT INTO dlightrag_answer_workspace_inventory ("
                " owner_id, run_id, relative_path, entry_type, mode, size_bytes, content_digest)"
                " VALUES ($1, $2, $3, $4, $5, $6, $7)",
                self._owner_id,
                self._run_id,
                record.relative_path,
                record.entry_type,
                record.mode,
                record.size_bytes,
                record.content_digest,
            )


async def load_session_notes(
    *,
    owner_id: str,
    session_id: str,
    pool: ConnectionPool | None = None,
) -> tuple[SessionNoteRecord, ...]:
    """Read one Agent Session's notes, by path, without a live claim."""
    connection_pool = pool if pool is not None else await pg_pool.get()
    async with connection_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT relative_path, content FROM dlightrag_answer_session_notes"
            " WHERE owner_id = $1 AND session_id = $2::uuid"
            " ORDER BY relative_path",
            owner_id,
            session_id,
        )
    return tuple(
        SessionNoteRecord(relative_path=str(row["relative_path"]), content=bytes(row["content"]))
        for row in rows
    )


async def load_run_inventory(
    *,
    owner_id: str,
    run_id: uuid.UUID,
    pool: ConnectionPool | None = None,
) -> tuple[InventoryPathRecord, ...]:
    """Read one Run's Workspace Inventory without a live claim.

    The one last carry reads a terminal parent Run's registered notes. That Run
    holds no lease, and the fenced store's write methods would refuse it. The read
    is the same query; it does not need the claim.
    """
    connection_pool = pool if pool is not None else await pg_pool.get()
    async with connection_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT relative_path, entry_type, mode, size_bytes, content_digest"
            " FROM dlightrag_answer_workspace_inventory"
            " WHERE owner_id = $1 AND run_id = $2"
            " ORDER BY relative_path",
            owner_id,
            run_id,
        )
    return tuple(_inventory_row(row) for row in rows)


def _inventory_row(row: Any) -> InventoryPathRecord:
    return InventoryPathRecord(
        relative_path=str(row["relative_path"]),
        entry_type=str(row["entry_type"]),
        mode=int(row["mode"]) if row["mode"] is not None else None,
        size_bytes=int(row["size_bytes"]),
        content_digest=str(row["content_digest"]) if row["content_digest"] is not None else None,
    )


async def _upsert_spill(
    conn: Any, owner_id: str, run_id: uuid.UUID, spill: CommittedSpillRecord
) -> None:
    await conn.execute(
        "INSERT INTO dlightrag_answer_committed_spills ("
        " owner_id, run_id, resource_id, content_digest, size_bytes, session_id, intent_id)"
        " VALUES ($1, $2, $3, $4, $5, $6::uuid, $7::uuid)"
        " ON CONFLICT (owner_id, run_id, resource_id) DO UPDATE SET"
        " content_digest = EXCLUDED.content_digest, size_bytes = EXCLUDED.size_bytes",
        owner_id,
        run_id,
        spill.resource_id,
        spill.content_digest,
        spill.size_bytes,
        spill.session_id,
        spill.intent_id,
    )
    await conn.execute(
        "INSERT INTO dlightrag_answer_resources ("
        " owner_id, run_id, resource_id, kind, safe_name, media_type, capabilities,"
        " session_id, intent_id)"
        " VALUES ($1, $2, $3, 'committed_spill', $3, 'text/plain', '{}'::jsonb, $4::uuid, $5::uuid)"
        " ON CONFLICT (owner_id, run_id, resource_id) DO NOTHING",
        owner_id,
        run_id,
        spill.resource_id,
        spill.session_id,
        spill.intent_id,
    )


__all__ = ["PGWorkspaceStore", "load_run_inventory", "load_session_notes"]
