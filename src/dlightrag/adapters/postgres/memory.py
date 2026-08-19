# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Owner-scoped Memory Record schema and PostgreSQL store."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from dlightrag.adapters.postgres._migrations import TableRequirement
from dlightrag.adapters.postgres._operations import ConnectionPool, PostgresOperationRunner
from dlightrag.answer.memory import MemoryProvenance, MemoryRecord, select_auto_recall

_CREATE_MEMORY_RECORDS = """
CREATE TABLE IF NOT EXISTS dlightrag_answer_memory_records (
    owner_id       TEXT             NOT NULL,
    memory_id      UUID             NOT NULL,
    kind           TEXT             NOT NULL,
    body           TEXT             NOT NULL,
    confidence     DOUBLE PRECISION NOT NULL,
    run_id         UUID             NOT NULL,
    session_id     UUID             NOT NULL,
    status         TEXT             NOT NULL,
    supersedes_id  UUID,
    created_at     TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    updated_at     TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id, memory_id),
    CONSTRAINT dlightrag_answer_memory_records_kind_check
        CHECK (kind IN ('preference', 'fact')),
    CONSTRAINT dlightrag_answer_memory_records_status_check
        CHECK (status IN ('active', 'superseded')),
    CONSTRAINT dlightrag_answer_memory_records_body_check
        CHECK (char_length(body) BETWEEN 1 AND 500),
    CONSTRAINT dlightrag_answer_memory_records_confidence_check
        CHECK (confidence > 0 AND confidence <= 1)
)
"""

_CREATE_MEMORY_WRITE_LOG = """
CREATE TABLE IF NOT EXISTS dlightrag_answer_memory_write_log (
    owner_id    TEXT        NOT NULL,
    written_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

_CREATE_MEMORY_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_dlightrag_answer_memory_records_recall "
    "ON dlightrag_answer_memory_records (owner_id, status, updated_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_dlightrag_answer_memory_records_purge "
    "ON dlightrag_answer_memory_records (status, updated_at) "
    "WHERE status = 'superseded'",
    "CREATE INDEX IF NOT EXISTS idx_dlightrag_answer_memory_write_log_owner "
    "ON dlightrag_answer_memory_write_log (owner_id, written_at)",
)

_ALIGN_STATUS_CHECK = (
    "ALTER TABLE dlightrag_answer_memory_records "
    "DROP CONSTRAINT IF EXISTS dlightrag_answer_memory_records_status_check",
    "ALTER TABLE dlightrag_answer_memory_records "
    "ADD CONSTRAINT dlightrag_answer_memory_records_status_check "
    "CHECK (status IN ('active', 'superseded'))",
)

MEMORY_DDL = (
    _CREATE_MEMORY_RECORDS,
    _CREATE_MEMORY_WRITE_LOG,
    *_CREATE_MEMORY_INDEXES,
    *_ALIGN_STATUS_CHECK,
)

MEMORY_SCHEMA_TABLE = TableRequirement(
    name="dlightrag_answer_memory_records",
    columns=(
        "owner_id",
        "memory_id",
        "kind",
        "body",
        "confidence",
        "run_id",
        "session_id",
        "status",
        "supersedes_id",
        "created_at",
        "updated_at",
    ),
    primary_key=("owner_id", "memory_id"),
    checks=(
        "dlightrag_answer_memory_records_kind_check",
        "dlightrag_answer_memory_records_status_check",
        "dlightrag_answer_memory_records_body_check",
        "dlightrag_answer_memory_records_confidence_check",
    ),
    indexes=(
        "idx_dlightrag_answer_memory_records_recall",
        "idx_dlightrag_answer_memory_records_purge",
    ),
)

MEMORY_WRITE_LOG_TABLE = TableRequirement(
    name="dlightrag_answer_memory_write_log",
    columns=("owner_id", "written_at"),
    indexes=("idx_dlightrag_answer_memory_write_log_owner",),
)

_COUNT_ACTIVE = """
SELECT COUNT(*)::int FROM dlightrag_answer_memory_records
WHERE owner_id = $1 AND status = 'active'
"""

_COUNT_WRITES = """
SELECT COUNT(*)::int FROM dlightrag_answer_memory_write_log
WHERE owner_id = $1 AND written_at >= $2
"""

_INSERT = """
INSERT INTO dlightrag_answer_memory_records (
    owner_id, memory_id, kind, body, confidence, run_id, session_id,
    status, supersedes_id, created_at, updated_at
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
"""

_LOG_WRITE = """
INSERT INTO dlightrag_answer_memory_write_log (owner_id, written_at)
VALUES ($1, NOW())
"""

_MARK_SUPERSEDED = """
UPDATE dlightrag_answer_memory_records
SET status = 'superseded', updated_at = NOW()
WHERE owner_id = $1 AND memory_id = $2 AND status = 'active'
"""

_DELETE = """
DELETE FROM dlightrag_answer_memory_records
WHERE owner_id = $1 AND memory_id = $2
"""

_DELETE_BODY = """
DELETE FROM dlightrag_answer_memory_records
WHERE owner_id = $1 AND body = $2
"""

_SELECT_ONE = """
SELECT owner_id, memory_id, kind, body, confidence, run_id, session_id,
       status, supersedes_id, created_at, updated_at
FROM dlightrag_answer_memory_records
WHERE owner_id = $1 AND memory_id = $2
"""

_SELECT_ACTIVE = """
SELECT owner_id, memory_id, kind, body, confidence, run_id, session_id,
       status, supersedes_id, created_at, updated_at
FROM dlightrag_answer_memory_records
WHERE owner_id = $1 AND status = 'active'
ORDER BY updated_at DESC
"""

_PURGE = """
DELETE FROM dlightrag_answer_memory_records
WHERE status = 'superseded' AND updated_at < $1
"""

_PRUNE_LOG = """
DELETE FROM dlightrag_answer_memory_write_log
WHERE written_at < $1
"""


def _uuid(value: str, *, label: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be a canonical UUID") from exc


def _row(row: Any) -> MemoryRecord:
    return MemoryRecord(
        owner_id=str(row["owner_id"]),
        memory_id=str(row["memory_id"]),
        kind=str(row["kind"]),  # type: ignore[arg-type]
        body=str(row["body"]),
        confidence=float(row["confidence"]),
        provenance=MemoryProvenance(
            run_id=str(row["run_id"]),
            session_id=str(row["session_id"]),
        ),
        status=str(row["status"]),  # type: ignore[arg-type]
        supersedes_id=str(row["supersedes_id"]) if row["supersedes_id"] is not None else None,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class PGAnswerMemoryStore(PostgresOperationRunner):
    """Owner-isolated Memory Records in PostgreSQL."""

    def __init__(self, *, pool: ConnectionPool | None = None) -> None:
        super().__init__(pool=pool)

    async def initialize(self) -> None:
        async def _operation(conn: Any) -> None:
            for statement in MEMORY_DDL:
                await conn.execute(statement)

        await self._run(_operation)

    async def count_active(self, *, owner_id: str) -> int:
        async def _operation(conn: Any) -> int:
            return int(await conn.fetchval(_COUNT_ACTIVE, owner_id) or 0)

        return await self._run(_operation)

    async def count_writes_since(self, *, owner_id: str, since: datetime) -> int:
        async def _operation(conn: Any) -> int:
            return int(await conn.fetchval(_COUNT_WRITES, owner_id, since) or 0)

        return await self._run(_operation)

    async def insert(self, record: MemoryRecord) -> None:
        params = _insert_params(record)

        async def _operation(conn: Any) -> None:
            async with conn.transaction():
                await conn.execute(_INSERT, *params)
                await conn.execute(_LOG_WRITE, record.owner_id)

        await self._run_once(_operation)

    async def supersede(self, *, owner_id: str, old_id: str, new: MemoryRecord) -> None:
        if new.owner_id != owner_id:
            raise ValueError("supersede cannot change owner")
        old_uuid = _uuid(old_id, label="memory_id")
        params = _insert_params(new)

        async def _operation(conn: Any) -> None:
            async with conn.transaction():
                tag = await conn.execute(_MARK_SUPERSEDED, owner_id, old_uuid)
                if str(tag).endswith(" 0"):
                    raise KeyError(old_id)
                await conn.execute(_INSERT, *params)
                await conn.execute(_LOG_WRITE, owner_id)

        await self._run_once(_operation)

    async def forget(self, *, owner_id: str, memory_id: str) -> bool:
        memory_uuid = _uuid(memory_id, label="memory_id")

        async def _operation(conn: Any) -> bool:
            async with conn.transaction():
                tag = await conn.execute(_DELETE, owner_id, memory_uuid)
                removed = not str(tag).endswith(" 0")
                if removed:
                    await conn.execute(_LOG_WRITE, owner_id)
                return removed

        return await self._run_once(_operation)

    async def forget_matching(self, *, owner_id: str, body: str) -> int:
        async def _operation(conn: Any) -> int:
            async with conn.transaction():
                result = await conn.execute(_DELETE_BODY, owner_id, body.strip())
                deleted = int(str(result).rsplit(" ", 1)[-1])
                if deleted:
                    await conn.execute(_LOG_WRITE, owner_id)
                return deleted

        return await self._run_once(_operation)

    async def get(self, *, owner_id: str, memory_id: str) -> MemoryRecord | None:
        memory_uuid = _uuid(memory_id, label="memory_id")

        async def _operation(conn: Any) -> MemoryRecord | None:
            row = await conn.fetchrow(_SELECT_ONE, owner_id, memory_uuid)
            return None if row is None else _row(row)

        return await self._run(_operation)

    async def list_active(self, *, owner_id: str) -> tuple[MemoryRecord, ...]:
        async def _operation(conn: Any) -> tuple[MemoryRecord, ...]:
            rows = await conn.fetch(_SELECT_ACTIVE, owner_id)
            return tuple(_row(row) for row in rows)

        return await self._run(_operation)

    async def list_for_recall(self, *, owner_id: str) -> tuple[MemoryRecord, ...]:
        return select_auto_recall(await self.list_active(owner_id=owner_id))

    async def purge_superseded(self, *, older_than: datetime) -> int:
        async def _operation(conn: Any) -> int:
            result = await conn.execute(_PURGE, older_than)
            return int(str(result).rsplit(" ", 1)[-1])

        return await self._run_once(_operation)

    async def prune_write_log(self, *, older_than: datetime) -> int:
        async def _operation(conn: Any) -> int:
            result = await conn.execute(_PRUNE_LOG, older_than)
            return int(str(result).rsplit(" ", 1)[-1])

        return await self._run_once(_operation)


def _insert_params(record: MemoryRecord) -> tuple[Any, ...]:
    return (
        record.owner_id,
        _uuid(record.memory_id, label="memory_id"),
        record.kind,
        record.body,
        record.confidence,
        _uuid(record.provenance.run_id, label="run_id"),
        _uuid(record.provenance.session_id, label="session_id"),
        record.status,
        _uuid(record.supersedes_id, label="supersedes_id") if record.supersedes_id else None,
        record.created_at,
        record.updated_at,
    )


__all__ = [
    "MEMORY_DDL",
    "MEMORY_SCHEMA_TABLE",
    "MEMORY_WRITE_LOG_TABLE",
    "PGAnswerMemoryStore",
]
