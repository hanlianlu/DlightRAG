# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL adapter for the Memory facade (P3/P4 substrate).

Owns its own schema, namespace, and migration registry: the package never
shares DlightRAG's answer-run migrations or tables. The three recall legs are
implemented here behind the neutral ports:

- exact:  ``normalized_body`` btree equality (Python-side NFKC normalization)
- sparse: pg_textsearch BM25 with the corpus-tuned k1/b and both textsearch
  configs (``simple`` + ``public.jiebacfg``), merged by best score into one
  ranking so a record never double-counts in fusion
- dense:  optional ``halfvec`` column + HNSW index when a TextEmbedder is bound

Dense is opt-in: with the NullEmbedder the adapter runs exact + sparse only.
A changed embedder fingerprint leaves old rows out of the dense leg (exact and
sparse still reach them); automatic re-embedding is a P4 decision.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, Protocol

from dlightrag_memory._storage.pg_bm25 import (
    build_bm25_sql,
    ensure_bm25_indexes,
    extension_bootstrap_sql,
    text_configs_available,
)
from dlightrag_memory.errors import MemoryWriteRejectedError
from dlightrag_memory.models import (
    MemoryOperation,
    MemoryOperationReceipt,
    MemoryProvenance,
    MemoryRecord,
)
from dlightrag_memory.normalize import normalized_body
from dlightrag_memory.ports import (
    NullEmbedder,
    SearchCandidate,
    TextEmbedder,
    Vector,
)
from dlightrag_memory.recall import recall_recency
from dlightrag_memory.store import (
    OperationGuard,
    operation_change_id,
    operation_fingerprint,
    operation_receipt,
    operation_record_id,
)


class PGConnection(Protocol):
    """The duck-typed asyncpg surface the PostgreSQL adapter consumes."""

    async def fetch(self, query: str, *args: Any) -> list[Any]: ...

    async def fetchrow(self, query: str, *args: Any) -> Any: ...

    async def fetchval(self, query: str, *args: Any) -> Any: ...

    async def execute(self, query: str, *args: Any) -> Any: ...

    def transaction(self) -> Any: ...


class PGPool(Protocol):
    """A pool whose ``acquire()`` yields one connection context.

    Duck-typed: both a bound asyncpg pool and a lazy pool holder satisfy it.
    """

    def acquire(self) -> Any: ...


_RECORDS_TABLE = """
CREATE TABLE IF NOT EXISTS dlightrag_memory_records (
    owner_id             TEXT             NOT NULL,
    memory_id            UUID             NOT NULL,
    kind                 TEXT             NOT NULL,
    body                 TEXT             NOT NULL,
    normalized_body      TEXT             NOT NULL,
    origin_kind          TEXT             NOT NULL,
    origin_id            TEXT             NOT NULL,
    run_id               TEXT,
    session_id           TEXT,
    status               TEXT             NOT NULL,
    supersedes_id        UUID,
    embedding_fingerprint TEXT,
    created_at           TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    updated_at           TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id, memory_id),
    CONSTRAINT dlightrag_memory_records_kind_check
        CHECK (kind IN ('preference', 'fact')),
    CONSTRAINT dlightrag_memory_records_origin_check
        CHECK (origin_kind IN ('answer_run', 'management', 'mcp', 'undo')),
    CONSTRAINT dlightrag_memory_records_status_check
        CHECK (status IN ('active', 'superseded', 'forgotten')),
    CONSTRAINT dlightrag_memory_records_body_check
        CHECK (char_length(body) BETWEEN 1 AND 500)
)
"""

_OPERATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS dlightrag_memory_operations (
    owner_id            TEXT        NOT NULL,
    change_id           UUID        NOT NULL,
    idempotency_key     TEXT        NOT NULL,
    request_fingerprint TEXT        NOT NULL,
    operation           TEXT        NOT NULL,
    outcome             TEXT        NOT NULL,
    mutation_scope      TEXT,
    receipt             JSONB       NOT NULL,
    before_records      JSONB       NOT NULL DEFAULT '[]'::jsonb,
    undone_by           UUID,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id, change_id),
    UNIQUE (owner_id, idempotency_key),
    CONSTRAINT dlightrag_memory_operations_action_check
        CHECK (operation IN ('remember', 'forget', 'undo')),
    CONSTRAINT dlightrag_memory_operations_outcome_check
        CHECK (outcome IN ('changed', 'unchanged', 'conflict'))
)
"""

_RECORD_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_dlightrag_memory_records_recall "
    "ON dlightrag_memory_records (owner_id, status, updated_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_dlightrag_memory_records_list "
    "ON dlightrag_memory_records (owner_id, status, updated_at DESC, memory_id DESC)",
    "CREATE INDEX IF NOT EXISTS idx_dlightrag_memory_records_exact "
    "ON dlightrag_memory_records (owner_id, normalized_body)",
    "CREATE INDEX IF NOT EXISTS idx_dlightrag_memory_records_purge "
    "ON dlightrag_memory_records (status, updated_at) "
    "WHERE status = 'superseded'",
)

_RECORD_COLUMNS = """
owner_id, memory_id, kind, body, normalized_body, origin_kind, origin_id, run_id,
session_id, status, supersedes_id, embedding_fingerprint, created_at, updated_at
"""

_OPERATION_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_dlightrag_memory_operations_scope "
    "ON dlightrag_memory_operations (owner_id, mutation_scope, outcome)",
    "CREATE INDEX IF NOT EXISTS idx_dlightrag_memory_operations_retention "
    "ON dlightrag_memory_operations (created_at)",
)


def _vector_text(vector: Vector) -> str:
    return "[" + ",".join(f"{float(value):g}" for value in vector) + "]"


class PostgresMemoryStore:
    """Owner-isolated Memory Records in one package-owned PostgreSQL schema."""

    def __init__(
        self,
        pool: PGPool | None = None,
        *,
        dsn: str | None = None,
        pool_factory: Callable[[], Awaitable[Any]] | None = None,
        embedder: TextEmbedder = NullEmbedder(),
    ) -> None:
        if pool is None and not dsn and pool_factory is None:
            raise ValueError("PostgresMemoryStore needs a pool, a dsn, or a pool factory")
        self._dsn = dsn
        self._pool = pool
        self._pool_factory = pool_factory
        self._owned_pool: Any = None
        self._embedder = embedder
        self._dense = not isinstance(embedder, NullEmbedder)
        self._bm25_indexes: tuple[str, ...] = ()
        self._operation_pool = pool  # test hook: backdate rows directly
        self._initialized = False

    async def aclose(self) -> None:
        """Close the adapter-owned pool; safe to call more than once."""
        owned = self._owned_pool
        self._owned_pool = None
        if owned is not None:
            await owned.close()

    async def _acquire_context(self) -> Any:
        """Return one pool acquire context, creating an owned pool when needed."""
        if self._pool is None:
            if self._pool_factory is not None:
                self._pool = await self._pool_factory()
            else:
                import asyncpg

                if self._owned_pool is None:
                    self._owned_pool = await asyncpg.create_pool(self._dsn)
                self._pool = self._owned_pool
        return self._pool.acquire()

    async def initialize(self) -> None:
        async def operation(conn: PGConnection) -> None:
            await conn.execute(_RECORDS_TABLE)
            await conn.execute(_OPERATIONS_TABLE)
            for statement in (*_RECORD_INDEXES, *_OPERATION_INDEXES):
                await conn.execute(statement)
            for statement in extension_bootstrap_sql():
                await conn.execute(statement)
            available = await text_configs_available(conn)
            self._bm25_indexes = await ensure_bm25_indexes(conn, available=available)
            if self._dense:
                dim = int(self._embedder.dim)
                if dim < 1:
                    raise ValueError("embedder dim must be positive for the dense leg")
                await conn.execute(_embedding_column_sql(dim))
                await conn.execute(_embedding_index_sql())

        acquire = await self._acquire_context()
        async with acquire as conn:
            await operation(conn)
        await self.verify()

    async def verify(self) -> None:
        """Validate the writer's schema and load search-index facts, no DDL."""

        async def operation(conn: PGConnection) -> None:
            rows = await conn.fetch(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_name = ANY($1::text[])",
                ["dlightrag_memory_records", "dlightrag_memory_operations"],
            )
            found = {str(row["table_name"]) for row in rows}
            missing = {"dlightrag_memory_records", "dlightrag_memory_operations"} - found
            if missing:
                raise RuntimeError(
                    f"Memory schema is missing {', '.join(sorted(missing))}; "
                    "initialize it on the writer first"
                )
            columns = await conn.fetch(
                "SELECT table_name, column_name FROM information_schema.columns "
                "WHERE table_name = ANY($1::text[])",
                ["dlightrag_memory_records", "dlightrag_memory_operations"],
            )
            by_table: dict[str, set[str]] = {}
            for row in columns:
                by_table.setdefault(str(row["table_name"]), set()).add(str(row["column_name"]))
            if "confidence" in by_table.get("dlightrag_memory_records", set()):
                raise RuntimeError("Memory schema still contains the removed confidence column")
            required = {
                "dlightrag_memory_records": {"origin_kind", "origin_id", "normalized_body"},
                "dlightrag_memory_operations": {
                    "change_id",
                    "request_fingerprint",
                    "receipt",
                    "before_records",
                    "undone_by",
                },
            }
            for table, names in required.items():
                missing_columns = names - by_table.get(table, set())
                if missing_columns:
                    raise RuntimeError(f"{table} is missing {', '.join(sorted(missing_columns))}")
            self._bm25_indexes = await ensure_bm25_indexes(conn, verify_only=True)

        acquire = await self._acquire_context()
        async with acquire as conn:
            await operation(conn)
        self._initialized = True

    def _embedder_fingerprint(self) -> str:
        """One canonical embedding-space identity for TEXT persistence."""
        return self._embedder.embedding_fingerprint

    async def apply_operation(
        self,
        operation: MemoryOperation,
        *,
        guard: OperationGuard | None = None,
    ) -> MemoryOperationReceipt:
        """Atomically settle one operation, its receipt, journal, and mutation cap."""
        change_id = operation_change_id(operation)
        fingerprint = operation_fingerprint(operation)
        embedding = (
            await self._embedding(operation.body.strip())
            if operation.action == "remember" and self._dense
            else None
        )

        async def settle(conn: PGConnection) -> MemoryOperationReceipt:
            async with conn.transaction():
                await conn.fetchval(_LOCK_OWNER, operation.owner_id)
                if guard is not None:
                    await guard(conn)
                replay = await conn.fetchrow(
                    _SELECT_OPERATION, operation.owner_id, _uuid(change_id, label="change_id")
                )
                if replay is not None:
                    if str(replay["request_fingerprint"]) != fingerprint:
                        raise MemoryWriteRejectedError(
                            "Memory idempotency key was reused with different input."
                        )
                    return _receipt_row(replay)

                if operation.action == "remember":
                    receipt, before = await self._settle_remember(
                        conn,
                        operation=operation,
                        change_id=change_id,
                        embedding=embedding,
                    )
                elif operation.action == "forget":
                    receipt, before = await self._settle_forget(
                        conn, operation=operation, change_id=change_id
                    )
                else:
                    receipt, before = await self._settle_undo(
                        conn, operation=operation, change_id=change_id
                    )

                if receipt.changed and operation.mutation_scope is not None:
                    used = int(
                        await conn.fetchval(
                            _COUNT_SCOPE_MUTATIONS,
                            operation.owner_id,
                            operation.mutation_scope,
                        )
                        or 0
                    )
                    if used >= int(operation.mutation_limit or 0):
                        raise MemoryWriteRejectedError(
                            "This Answer Run reached its Memory mutation limit."
                        )

                await conn.execute(
                    _INSERT_OPERATION,
                    operation.owner_id,
                    _uuid(change_id, label="change_id"),
                    operation.idempotency_key,
                    fingerprint,
                    receipt.action,
                    receipt.outcome,
                    operation.mutation_scope,
                    json.dumps(_receipt_json(receipt), ensure_ascii=False),
                    json.dumps([_record_json(record) for record in before], ensure_ascii=False),
                    receipt.created_at,
                )
                if receipt.action == "undo" and receipt.changed and receipt.target_change_id:
                    await conn.execute(
                        _MARK_OPERATION_UNDONE,
                        operation.owner_id,
                        _uuid(receipt.target_change_id, label="target_change_id"),
                        _uuid(change_id, label="change_id"),
                    )
                return receipt

        return await self._write(settle)

    async def _settle_remember(
        self,
        conn: PGConnection,
        *,
        operation: MemoryOperation,
        change_id: str,
        embedding: Vector | None,
    ) -> tuple[MemoryOperationReceipt, tuple[MemoryRecord, ...]]:
        now = datetime.now(UTC)
        body = operation.body.strip()
        duplicate_row = await conn.fetchrow(
            _SELECT_ACTIVE_NORMALIZED_FOR_UPDATE,
            operation.owner_id,
            normalized_body(body),
        )
        if duplicate_row is not None:
            duplicate = _row(duplicate_row)
            outcome = (
                "conflict"
                if operation.supersedes_id and duplicate.memory_id != operation.supersedes_id
                else "unchanged"
            )
            return (
                operation_receipt(
                    operation,
                    change_id,
                    outcome,
                    memory_ids=(duplicate.memory_id,),
                    kind=duplicate.kind,
                    body=duplicate.body,
                    now=now,
                ),
                (),
            )

        before: tuple[MemoryRecord, ...] = ()
        if operation.supersedes_id:
            old_row = await conn.fetchrow(
                _SELECT_ONE_FOR_UPDATE,
                operation.owner_id,
                _uuid(operation.supersedes_id, label="supersedes_id"),
            )
            if old_row is None or str(old_row["status"]) != "active":
                return (
                    operation_receipt(operation, change_id, "conflict", body=body, now=now),
                    (),
                )
            old = _row(old_row)
            before = (old,)
            await conn.execute(
                _MARK_SUPERSEDED,
                operation.owner_id,
                _uuid(operation.supersedes_id, label="supersedes_id"),
            )

        memory_id = operation_record_id(operation.owner_id, change_id)
        record = MemoryRecord(
            owner_id=operation.owner_id,
            memory_id=memory_id,
            kind=operation.kind or "fact",
            body=body,
            provenance=operation.provenance,
            status="active",
            supersedes_id=operation.supersedes_id,
            created_at=now,
            updated_at=now,
        )
        await _insert_record(self, conn, record=record, embedding=embedding)
        return (
            operation_receipt(
                operation,
                change_id,
                "changed",
                memory_ids=(memory_id,),
                kind=record.kind,
                body=record.body,
                now=now,
            ),
            before,
        )

    async def _settle_forget(
        self,
        conn: PGConnection,
        *,
        operation: MemoryOperation,
        change_id: str,
    ) -> tuple[MemoryOperationReceipt, tuple[MemoryRecord, ...]]:
        now = datetime.now(UTC)
        if operation.memory_id:
            rows = await conn.fetch(
                _SELECT_ACTIVE_ID_FOR_UPDATE,
                operation.owner_id,
                _uuid(operation.memory_id, label="memory_id"),
            )
        else:
            rows = await conn.fetch(
                _SELECT_ACTIVE_NORMALIZED_ALL_FOR_UPDATE,
                operation.owner_id,
                normalized_body(operation.body),
            )
        matches = tuple(_row(row) for row in rows)
        if not matches:
            return (operation_receipt(operation, change_id, "unchanged", now=now), ())
        ids = [_uuid(record.memory_id, label="memory_id") for record in matches]
        await conn.execute(_MARK_FORGOTTEN_IDS, operation.owner_id, ids)
        first = matches[0]
        return (
            operation_receipt(
                operation,
                change_id,
                "changed",
                memory_ids=tuple(record.memory_id for record in matches),
                kind=first.kind,
                body=first.body,
                now=now,
            ),
            matches,
        )

    async def _settle_undo(
        self,
        conn: PGConnection,
        *,
        operation: MemoryOperation,
        change_id: str,
    ) -> tuple[MemoryOperationReceipt, tuple[MemoryRecord, ...]]:
        now = datetime.now(UTC)
        target_id = operation.target_change_id or ""
        target = await conn.fetchrow(
            _SELECT_OPERATION_FOR_UPDATE,
            operation.owner_id,
            _uuid(target_id, label="target_change_id"),
        )
        if target is None or target["undone_by"] is not None:
            return (
                operation_receipt(
                    operation,
                    change_id,
                    "conflict",
                    target_change_id=target_id,
                    now=now,
                ),
                (),
            )
        target_receipt = _receipt_row(target)
        before = _records_json(target["before_records"])
        if not target_receipt.changed or target_receipt.action == "undo":
            return (
                operation_receipt(
                    operation,
                    change_id,
                    "conflict",
                    target_change_id=target_id,
                    now=now,
                ),
                (),
            )

        if target_receipt.action == "remember":
            current_id = target_receipt.memory_id or ""
            current_row = await conn.fetchrow(
                _SELECT_ONE_FOR_UPDATE,
                operation.owner_id,
                _uuid(current_id, label="memory_id"),
            )
            if current_row is None or str(current_row["status"]) != "active":
                return (
                    operation_receipt(
                        operation,
                        change_id,
                        "conflict",
                        target_change_id=target_id,
                        now=now,
                    ),
                    (),
                )
            current = _row(current_row)
            if target_receipt.supersedes_id and before:
                await conn.execute(
                    _MARK_SUPERSEDED,
                    operation.owner_id,
                    _uuid(current_id, label="memory_id"),
                )
                restored_id = operation_record_id(operation.owner_id, change_id)
                restored = replace(
                    before[0],
                    memory_id=restored_id,
                    provenance=operation.provenance,
                    status="active",
                    supersedes_id=current_id,
                    created_at=now,
                    updated_at=now,
                )
                await _insert_record(self, conn, record=restored, embedding=None)
                return (
                    operation_receipt(
                        operation,
                        change_id,
                        "changed",
                        memory_ids=(restored_id,),
                        kind=restored.kind,
                        body=restored.body,
                        supersedes_id=current_id,
                        target_change_id=target_id,
                        now=now,
                    ),
                    (current,),
                )
            await conn.execute(
                _MARK_FORGOTTEN_IDS,
                operation.owner_id,
                [_uuid(current_id, label="memory_id")],
            )
            return (
                operation_receipt(
                    operation,
                    change_id,
                    "changed",
                    memory_ids=(current_id,),
                    kind=current.kind,
                    body=current.body,
                    target_change_id=target_id,
                    now=now,
                ),
                (current,),
            )

        # Forget target: preflight the full batch set-wise before mutating
        # anything. The persisted journal must be well formed (nonempty, every
        # before record belongs to this owner, memory ids are unique and, in
        # order, exactly the target receipt's memory_ids), every target id
        # must still exist for this owner as a forgotten row, and no active
        # record outside the batch may share a normalized body (siblings
        # inside the batch compensate the exact prior forget and are never
        # conflicts).
        if (
            not before
            or any(old.owner_id != operation.owner_id for old in before)
            or len({old.memory_id for old in before}) != len(before)
            or tuple(old.memory_id for old in before) != target_receipt.memory_ids
        ):
            return (
                operation_receipt(
                    operation,
                    change_id,
                    "conflict",
                    target_change_id=target_id,
                    now=now,
                ),
                (),
            )
        target_ids = [_uuid(old.memory_id, label="memory_id") for old in before]
        current_rows = await conn.fetch(_SELECT_IDS_FOR_UPDATE, operation.owner_id, target_ids)
        current_by_id = {str(row["memory_id"]): row for row in current_rows}
        if any(
            current_by_id.get(old.memory_id) is None
            or str(current_by_id[old.memory_id]["status"]) != "forgotten"
            for old in before
        ):
            return (
                operation_receipt(
                    operation,
                    change_id,
                    "conflict",
                    target_change_id=target_id,
                    now=now,
                ),
                (),
            )
        bodies = sorted({normalized_body(old.body) for old in before})
        if await conn.fetchval(
            _SELECT_ACTIVE_NORMALIZED_CONFLICT_EXISTS, operation.owner_id, bodies
        ):
            return (
                operation_receipt(
                    operation,
                    change_id,
                    "conflict",
                    target_change_id=target_id,
                    now=now,
                ),
                (),
            )

        restored_records: list[MemoryRecord] = []
        for index, old in enumerate(before):
            restored_records.append(
                replace(
                    old,
                    memory_id=operation_record_id(operation.owner_id, change_id, index=index),
                    provenance=operation.provenance,
                    status="active",
                    supersedes_id=old.memory_id,
                    created_at=now,
                    updated_at=now,
                )
            )
        if restored_records:
            restored_rows = await conn.fetch(
                _INSERT_RESTORED_BATCH,
                operation.owner_id,
                operation.provenance.origin_kind,
                operation.provenance.origin_id,
                self._embedder_fingerprint() if self._dense else None,
                now,
                _restore_batch_json(restored_records),
            )
            restored_ids = {str(row["memory_id"]) for row in restored_rows}
            if len(restored_rows) != len(restored_records) or restored_ids != {
                record.memory_id for record in restored_records
            }:
                raise ValueError("memory id already exists with different content")
        first = restored_records[0] if restored_records else None
        return (
            operation_receipt(
                operation,
                change_id,
                "changed",
                memory_ids=tuple(record.memory_id for record in restored_records),
                kind=None if first is None else first.kind,
                body="" if first is None else first.body,
                target_change_id=target_id,
                now=now,
            ),
            before,
        )

    async def clear_owner(
        self,
        *,
        owner_id: str,
        guard: OperationGuard | None = None,
    ) -> int:
        async def operation(conn: PGConnection) -> int:
            async with conn.transaction():
                await conn.fetchval(_LOCK_OWNER, owner_id)
                if guard is not None:
                    await guard(conn)
                await conn.execute(_CLEAR_OPERATIONS, owner_id)
                records = await conn.execute(_CLEAR_RECORDS, owner_id)
                return _command_count(records)

        return await self._write(operation)

    async def count_active(self, *, owner_id: str) -> int:
        async def operation(conn: PGConnection) -> int:
            return int(await conn.fetchval(_COUNT_ACTIVE, owner_id) or 0)

        return await self._read(operation)

    async def insert(self, record: MemoryRecord) -> None:
        embedding = await self._embedding(record.body) if self._dense else None

        async def operation(conn: PGConnection) -> None:
            async with conn.transaction():
                if embedding is None:
                    await conn.execute(_INSERT, *_insert_params(self, record=record))
                else:
                    await conn.execute(
                        _INSERT_WITH_EMBEDDING,
                        *_insert_params(self, record=record),
                        _vector_text(embedding),
                    )
                current = await conn.fetchrow(
                    _SELECT_ONE, record.owner_id, _uuid(record.memory_id, label="memory_id")
                )
                if current is None or _row(current) != record:
                    raise ValueError("memory id already exists with different content")

        await self._write(operation)

    async def supersede(self, *, owner_id: str, old_id: str, new: MemoryRecord) -> None:
        if new.owner_id != owner_id:
            raise ValueError("supersede cannot change owner")
        embedding = await self._embedding(new.body) if self._dense else None

        async def operation(conn: PGConnection) -> None:
            async with conn.transaction():
                tag = await conn.execute(
                    _MARK_SUPERSEDED, owner_id, _uuid(old_id, label="memory_id")
                )
                if str(tag).endswith(" 0"):
                    raise KeyError(old_id)
                if embedding is None:
                    await conn.execute(_INSERT, *_insert_params(self, record=new))
                else:
                    await conn.execute(
                        _INSERT_WITH_EMBEDDING,
                        *_insert_params(self, record=new),
                        _vector_text(embedding),
                    )

        await self._write(operation)

    async def forget(self, *, owner_id: str, memory_id: str) -> bool:
        async def operation(conn: PGConnection) -> bool:
            async with conn.transaction():
                tag = await conn.execute(_DELETE, owner_id, _uuid(memory_id, label="memory_id"))
                return not str(tag).endswith(" 0")

        return await self._write(operation)

    async def forget_matching(self, *, owner_id: str, body: str) -> int:
        async def operation(conn: PGConnection) -> int:
            async with conn.transaction():
                result = await conn.execute(_DELETE_BODY, owner_id, body.strip())
                return int(str(result).rsplit(" ", 1)[-1])

        return await self._write(operation)

    async def forget_all(self, *, owner_id: str) -> int:
        async def operation(conn: PGConnection) -> int:
            async with conn.transaction():
                result = await conn.execute(_DELETE_ALL, owner_id)
                return int(str(result).rsplit(" ", 1)[-1])

        return await self._write(operation)

    async def get(self, *, owner_id: str, memory_id: str) -> MemoryRecord | None:
        async def operation(conn: PGConnection) -> MemoryRecord | None:
            row = await conn.fetchrow(_SELECT_ONE, owner_id, _uuid(memory_id, label="memory_id"))
            return None if row is None else _row(row)

        return await self._read(operation)

    async def list_active_page(
        self,
        *,
        owner_id: str,
        after: tuple[datetime, str] | None = None,
        limit: int = 50,
    ) -> tuple[tuple[MemoryRecord, ...], tuple[datetime, str] | None]:
        """Keyset page over (updated_at DESC, memory_id) for owner browse."""
        cap = max(1, min(int(limit), 100))

        async def operation(
            conn: PGConnection,
        ) -> tuple[tuple[MemoryRecord, ...], tuple[datetime, str] | None]:
            if after is None:
                rows = await conn.fetch(_SELECT_ACTIVE_PAGE, owner_id, cap + 1)
            else:
                rows = await conn.fetch(
                    _SELECT_ACTIVE_PAGE_AFTER, owner_id, after[0], after[1], cap + 1
                )
            page = tuple(_row(row) for row in rows[:cap])
            if len(rows) <= cap:
                return page, None
            last = _row(rows[cap - 1])
            return page, (recall_recency(last), last.memory_id)

        return await self._read(operation)

    async def purge_superseded(self, *, older_than: datetime) -> int:
        async def operation(conn: PGConnection) -> int:
            async with conn.transaction():
                operations = await conn.execute(_PURGE_OPERATIONS, older_than)
                records = await conn.execute(_PURGE_RECORDS, older_than)
                return _command_count(operations) + _command_count(records)

        return await self._write(operation)

    async def search_candidates(
        self, *, owner_id: str, query: str, limit: int
    ) -> tuple[SearchCandidate, ...]:
        """Return leg-tagged candidates in per-leg rank order, no cross-leg merge.

        The sparse leg merges both BM25 configs by best score into ONE ranking
        (a record matching both configs must not double-count in RRF); the
        façade fuses exact/sparse/dense with RRF.
        """
        cap = max(1, min(int(limit), 100))
        key = normalized_body(query)

        async def operation(conn: PGConnection) -> tuple[SearchCandidate, ...]:
            candidates: list[SearchCandidate] = []
            exact_rows = await conn.fetch(_SEARCH_EXACT, owner_id, key, cap)
            candidates.extend(
                SearchCandidate(record=_row(row), leg="exact", score=2.0) for row in exact_rows
            )
            sparse_by_id: dict[str, SearchCandidate] = {}
            for bm25_index in self._bm25_indexes:
                rows = await conn.fetch(
                    build_bm25_sql(index_name=bm25_index, limit=cap), query, owner_id
                )
                for row in rows:
                    record = _row(row)
                    score = float(row["score"])
                    existing = sparse_by_id.get(record.memory_id)
                    if existing is None or score > existing.score:
                        sparse_by_id[record.memory_id] = SearchCandidate(
                            record=record, leg="sparse", score=score
                        )
            candidates.extend(
                sorted(
                    sparse_by_id.values(),
                    key=lambda candidate: candidate.score,
                    reverse=True,
                )[:cap]
            )
            if self._dense:
                vector = await self._query_embedding(query)
                dense_rows = await conn.fetch(
                    _SEARCH_DENSE,
                    owner_id,
                    self._embedder_fingerprint(),
                    _vector_text(vector),
                    cap,
                )
                candidates.extend(
                    SearchCandidate(record=_row(row), leg="dense", score=float(row["score"]))
                    for row in dense_rows
                )
            return tuple(candidates)

        return await self._read(operation)

    async def _embedding(self, text: str) -> Vector:
        (vector,) = await self._embedder.embed_documents((text,))
        return vector

    async def _query_embedding(self, text: str) -> Vector:
        """Embed one query with the port's query context (asymmetric-aware)."""
        return await self._embedder.embed_query(text)

    async def _write(self, operation: Any) -> Any:
        async with await self._acquire_context() as conn:
            return await operation(conn)

    async def _read(self, operation: Any) -> Any:
        async with await self._acquire_context() as conn:
            return await operation(conn)


async def _insert_record(
    store: PostgresMemoryStore,
    conn: PGConnection,
    *,
    record: MemoryRecord,
    embedding: Vector | None,
) -> None:
    if embedding is None:
        await conn.execute(_INSERT, *_insert_params(store, record=record))
    else:
        await conn.execute(
            _INSERT_WITH_EMBEDDING,
            *_insert_params(store, record=record),
            _vector_text(embedding),
        )


def _receipt_json(receipt: MemoryOperationReceipt) -> dict[str, Any]:
    return {
        "action": receipt.action,
        "body": receipt.body,
        "change_id": receipt.change_id,
        "created_at": None if receipt.created_at is None else receipt.created_at.isoformat(),
        "kind": receipt.kind,
        "memory_ids": list(receipt.memory_ids),
        "mutation_scope": receipt.mutation_scope,
        "outcome": receipt.outcome,
        "provenance": _provenance_json(receipt.provenance),
        "supersedes_id": receipt.supersedes_id,
        "target_change_id": receipt.target_change_id,
    }


def _receipt_row(row: Any) -> MemoryOperationReceipt:
    value = _json_object(row["receipt"])
    return MemoryOperationReceipt(
        change_id=str(value["change_id"]),
        action=str(value["action"]),  # type: ignore[arg-type]
        outcome=str(value["outcome"]),  # type: ignore[arg-type]
        memory_ids=tuple(str(item) for item in value.get("memory_ids", [])),
        provenance=_provenance_from_json(value["provenance"]),
        kind=str(value["kind"]) if value.get("kind") is not None else None,  # type: ignore[arg-type]
        body=str(value.get("body") or ""),
        supersedes_id=(
            str(value["supersedes_id"]) if value.get("supersedes_id") is not None else None
        ),
        target_change_id=(
            str(value["target_change_id"]) if value.get("target_change_id") is not None else None
        ),
        mutation_scope=(
            str(value["mutation_scope"]) if value.get("mutation_scope") is not None else None
        ),
        created_at=_datetime_value(value.get("created_at")),
    )


def _record_json(record: MemoryRecord) -> dict[str, Any]:
    return {
        "body": record.body,
        "created_at": None if record.created_at is None else record.created_at.isoformat(),
        "kind": record.kind,
        "memory_id": record.memory_id,
        "owner_id": record.owner_id,
        "provenance": _provenance_json(record.provenance),
        "status": record.status,
        "supersedes_id": record.supersedes_id,
        "updated_at": None if record.updated_at is None else record.updated_at.isoformat(),
    }


def _records_json(value: Any) -> tuple[MemoryRecord, ...]:
    rows = _json_array(value)
    return tuple(
        MemoryRecord(
            owner_id=str(row["owner_id"]),
            memory_id=str(row["memory_id"]),
            kind=str(row["kind"]),  # type: ignore[arg-type]
            body=str(row["body"]),
            provenance=_provenance_from_json(row["provenance"]),
            status=str(row["status"]),  # type: ignore[arg-type]
            supersedes_id=(
                str(row["supersedes_id"]) if row.get("supersedes_id") is not None else None
            ),
            created_at=_datetime_value(row.get("created_at")),
            updated_at=_datetime_value(row.get("updated_at")),
        )
        for row in rows
    )


def _provenance_json(provenance: MemoryProvenance) -> dict[str, Any]:
    return {
        "origin_kind": provenance.origin_kind,
        "origin_id": provenance.origin_id,
        "run_id": provenance.run_id,
        "session_id": provenance.session_id,
    }


def _provenance_from_json(value: Any) -> MemoryProvenance:
    row = _json_object(value)
    return MemoryProvenance(
        origin_kind=str(row["origin_kind"]),  # type: ignore[arg-type]
        origin_id=str(row["origin_id"]),
        run_id=str(row["run_id"]) if row.get("run_id") is not None else None,
        session_id=str(row["session_id"]) if row.get("session_id") is not None else None,
    )


def _json_object(value: Any) -> dict[str, Any]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, dict):
        raise ValueError("Memory operation receipt is not an object")
    return parsed


def _json_array(value: Any) -> list[dict[str, Any]]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
        raise ValueError("Memory operation before-records is not an array")
    return parsed


def _datetime_value(value: Any) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _command_count(tag: Any) -> int:
    return int(str(tag).rsplit(" ", 1)[-1])


def _insert_params(store: PostgresMemoryStore, *, record: MemoryRecord) -> tuple[Any, ...]:
    return (
        record.owner_id,
        _uuid(record.memory_id, label="memory_id"),
        record.kind,
        record.body,
        normalized_body(record.body),
        record.provenance.origin_kind,
        record.provenance.origin_id,
        record.provenance.run_id,
        record.provenance.session_id,
        record.status,
        _uuid(record.supersedes_id, label="supersedes_id") if record.supersedes_id else None,
        store._embedder_fingerprint() if store._dense else None,  # noqa: SLF001
        record.created_at,
        record.updated_at,
    )


def _restore_batch_json(records: list[MemoryRecord]) -> str:
    """Encode one undo restoration batch as a single JSONB recordset parameter."""
    return json.dumps(
        [
            {
                "body": record.body,
                "kind": record.kind,
                "memory_id": record.memory_id,
                "normalized_body": normalized_body(record.body),
                "run_id": record.provenance.run_id,
                "session_id": record.provenance.session_id,
                "supersedes_id": record.supersedes_id,
            }
            for record in records
        ],
        ensure_ascii=False,
    )


def _embedding_column_sql(dim: int) -> str:  # noqa: S608 - dim is a validated int
    return f"ALTER TABLE dlightrag_memory_records ADD COLUMN IF NOT EXISTS embedding halfvec({dim})"


def _embedding_index_sql() -> str:
    return (
        "CREATE INDEX IF NOT EXISTS idx_dlightrag_memory_records_dense "
        "ON dlightrag_memory_records USING hnsw (embedding halfvec_cosine_ops)"
    )


_INSERT = """
INSERT INTO dlightrag_memory_records (
    owner_id, memory_id, kind, body, normalized_body, origin_kind, origin_id, run_id,
    session_id, status, supersedes_id, embedding_fingerprint, created_at, updated_at
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
ON CONFLICT (owner_id, memory_id) DO NOTHING
"""

_INSERT_WITH_EMBEDDING = """
INSERT INTO dlightrag_memory_records (
    owner_id, memory_id, kind, body, normalized_body, origin_kind, origin_id, run_id,
    session_id, status, supersedes_id, embedding_fingerprint, embedding, created_at, updated_at
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $15::halfvec, $13, $14)
ON CONFLICT (owner_id, memory_id) DO NOTHING
"""

_LOCK_OWNER = "SELECT pg_advisory_xact_lock(hashtext($1))"

_SELECT_OPERATION = """
SELECT request_fingerprint, receipt, before_records, undone_by
FROM dlightrag_memory_operations
WHERE owner_id = $1 AND change_id = $2
"""

_SELECT_OPERATION_FOR_UPDATE = _SELECT_OPERATION + " FOR UPDATE"

_INSERT_OPERATION = """
INSERT INTO dlightrag_memory_operations (
    owner_id, change_id, idempotency_key, request_fingerprint, operation, outcome,
    mutation_scope, receipt, before_records, created_at
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9::jsonb, $10)
"""

_COUNT_SCOPE_MUTATIONS = """
SELECT COUNT(*)
FROM dlightrag_memory_operations
WHERE owner_id = $1 AND mutation_scope = $2 AND outcome = 'changed'
"""

_MARK_OPERATION_UNDONE = """
UPDATE dlightrag_memory_operations
SET undone_by = $3
WHERE owner_id = $1 AND change_id = $2 AND undone_by IS NULL
"""

_SELECT_ACTIVE_NORMALIZED_FOR_UPDATE = f"""
SELECT {_RECORD_COLUMNS}
FROM dlightrag_memory_records
WHERE owner_id = $1 AND status = 'active' AND normalized_body = $2
ORDER BY updated_at DESC
LIMIT 1
FOR UPDATE
"""  # noqa: S608

_SELECT_ACTIVE_NORMALIZED_ALL_FOR_UPDATE = f"""
SELECT {_RECORD_COLUMNS}
FROM dlightrag_memory_records
WHERE owner_id = $1 AND status = 'active' AND normalized_body = $2
ORDER BY updated_at DESC
FOR UPDATE
"""  # noqa: S608

_SELECT_ACTIVE_ID_FOR_UPDATE = f"""
SELECT {_RECORD_COLUMNS}
FROM dlightrag_memory_records
WHERE owner_id = $1 AND memory_id = $2 AND status = 'active'
FOR UPDATE
"""  # noqa: S608

_SELECT_ONE_FOR_UPDATE = f"""
SELECT {_RECORD_COLUMNS}
FROM dlightrag_memory_records
WHERE owner_id = $1 AND memory_id = $2
FOR UPDATE
"""  # noqa: S608

_SELECT_IDS_FOR_UPDATE = f"""
SELECT {_RECORD_COLUMNS}
FROM dlightrag_memory_records
WHERE owner_id = $1 AND memory_id = ANY($2::uuid[])
FOR UPDATE
"""  # noqa: S608

_SELECT_ACTIVE_NORMALIZED_CONFLICT_EXISTS = """
SELECT EXISTS (
    SELECT 1
    FROM dlightrag_memory_records
    WHERE owner_id = $1 AND status = 'active' AND normalized_body = ANY($2::text[])
)
"""

_INSERT_RESTORED_BATCH = """
INSERT INTO dlightrag_memory_records (
    owner_id, memory_id, kind, body, normalized_body, origin_kind, origin_id, run_id,
    session_id, status, supersedes_id, embedding_fingerprint, created_at, updated_at
)
SELECT
    $1,
    (record->>'memory_id')::uuid,
    record->>'kind',
    record->>'body',
    record->>'normalized_body',
    $2,
    $3,
    record->>'run_id',
    record->>'session_id',
    'active',
    NULLIF(record->>'supersedes_id', '')::uuid,
    $4,
    $5,
    $5
FROM jsonb_array_elements($6::jsonb) AS record
ON CONFLICT (owner_id, memory_id) DO NOTHING
RETURNING memory_id
"""

_MARK_FORGOTTEN_IDS = """
UPDATE dlightrag_memory_records
SET status = 'forgotten', updated_at = NOW()
WHERE owner_id = $1 AND memory_id = ANY($2::uuid[]) AND status = 'active'
"""

_COUNT_ACTIVE = """
SELECT COUNT(*) FROM dlightrag_memory_records
WHERE owner_id = $1 AND status = 'active'
"""

_CLEAR_OPERATIONS = "DELETE FROM dlightrag_memory_operations WHERE owner_id = $1"
_CLEAR_RECORDS = "DELETE FROM dlightrag_memory_records WHERE owner_id = $1"

_MARK_SUPERSEDED = """
UPDATE dlightrag_memory_records
SET status = 'superseded', updated_at = NOW()
WHERE owner_id = $1 AND memory_id = $2 AND status = 'active'
"""

_DELETE = """
UPDATE dlightrag_memory_records
SET status = 'forgotten', updated_at = NOW()
WHERE owner_id = $1 AND memory_id = $2 AND status != 'forgotten'
"""

_DELETE_BODY = """
UPDATE dlightrag_memory_records
SET status = 'forgotten', updated_at = NOW()
WHERE owner_id = $1 AND body = $2 AND status != 'forgotten'
"""

_DELETE_ALL = """
UPDATE dlightrag_memory_records
SET status = 'forgotten', updated_at = NOW()
WHERE owner_id = $1 AND status != 'forgotten'
"""

_SELECT_ONE = f"""
SELECT {_RECORD_COLUMNS}
FROM dlightrag_memory_records
WHERE owner_id = $1 AND memory_id = $2
"""  # noqa: S608 - interpolates only the trusted _RECORD_COLUMNS constant

_SELECT_ACTIVE_PAGE = f"""
SELECT {_RECORD_COLUMNS}
FROM dlightrag_memory_records
WHERE owner_id = $1 AND status = 'active'
ORDER BY updated_at DESC, memory_id DESC
LIMIT $2
"""  # noqa: S608 - interpolates only the trusted _RECORD_COLUMNS constant

_SELECT_ACTIVE_PAGE_AFTER = f"""
SELECT {_RECORD_COLUMNS}
FROM dlightrag_memory_records
WHERE owner_id = $1 AND status = 'active'
  AND (updated_at, memory_id) < ($2, $3)
ORDER BY updated_at DESC, memory_id DESC
LIMIT $4
"""  # noqa: S608 - interpolates only the trusted _RECORD_COLUMNS constant

_PURGE_RECORDS = """
DELETE FROM dlightrag_memory_records
WHERE status != 'active' AND updated_at < $1
"""

_PURGE_OPERATIONS = """
DELETE FROM dlightrag_memory_operations
WHERE created_at < $1
"""

_SEARCH_EXACT = f"""
SELECT {_RECORD_COLUMNS}
FROM dlightrag_memory_records
WHERE owner_id = $1 AND status = 'active' AND normalized_body = $2
ORDER BY updated_at DESC
LIMIT $3
"""  # noqa: S608 - interpolates only the trusted _RECORD_COLUMNS constant

_SEARCH_DENSE = f"""
SELECT {_RECORD_COLUMNS}, 1 - (embedding <=> $3::halfvec) AS score
FROM dlightrag_memory_records
WHERE owner_id = $1 AND status = 'active'
  AND embedding_fingerprint = $2
  AND embedding IS NOT NULL
ORDER BY embedding <=> $3::halfvec
LIMIT $4
"""  # noqa: S608 - interpolates only the trusted _RECORD_COLUMNS constant


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
        provenance=MemoryProvenance(
            origin_kind=str(row["origin_kind"]),  # type: ignore[arg-type]
            origin_id=str(row["origin_id"]),
            run_id=str(row["run_id"]) if row["run_id"] is not None else None,
            session_id=str(row["session_id"]) if row["session_id"] is not None else None,
        ),
        status=str(row["status"]),  # type: ignore[arg-type]
        supersedes_id=str(row["supersedes_id"]) if row["supersedes_id"] is not None else None,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


__all__ = ["PostgresMemoryStore"]
