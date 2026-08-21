# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL adapter for the Memory facade (P3/P4 substrate).

Owns its own schema, namespace, and migration registry: the package never
shares DlightRAG's answer-run migrations or tables. The three recall legs are
implemented here behind the neutral ports:

- exact:  ``normalized_body`` btree equality (Python-side NFKC normalization)
- sparse: pg_textsearch BM25 with the corpus-tuned k1/b and both textsearch
  configs (``simple`` + ``public.jiebacfg``), merged by best score
- dense:  optional ``halfvec`` column + HNSW index when a TextEmbedder is bound

Dense is opt-in: with the NullEmbedder the adapter runs exact + sparse only.
A changed embedder fingerprint leaves old rows out of the dense leg (exact and
sparse still reach them); automatic re-embedding is a P4 decision.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from dlightrag_memory._storage.pg_bm25 import (
    build_bm25_sql,
    ensure_bm25_indexes,
    extension_bootstrap_sql,
    text_configs_available,
)
from dlightrag_memory.models import MemoryProvenance, MemoryRecord
from dlightrag_memory.normalize import normalized_body
from dlightrag_memory.ports import (
    Migration,
    NullEmbedder,
    PGConnection,
    PGPool,
    SearchCandidate,
    TextEmbedder,
    Vector,
)

_MIGRATION_TABLE = """
CREATE TABLE IF NOT EXISTS dlightrag_memory_migrations (
    migration_id TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

_RECORDS_TABLE = """
CREATE TABLE IF NOT EXISTS dlightrag_memory_records (
    owner_id             TEXT             NOT NULL,
    memory_id            UUID             NOT NULL,
    kind                 TEXT             NOT NULL,
    body                 TEXT             NOT NULL,
    normalized_body      TEXT             NOT NULL,
    confidence           DOUBLE PRECISION NOT NULL,
    run_id               TEXT             NOT NULL,
    session_id           TEXT             NOT NULL DEFAULT '',
    status               TEXT             NOT NULL,
    supersedes_id        UUID,
    embedding_fingerprint TEXT,
    created_at           TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    updated_at           TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id, memory_id),
    CONSTRAINT dlightrag_memory_records_kind_check
        CHECK (kind IN ('preference', 'fact')),
    CONSTRAINT dlightrag_memory_records_status_check
        CHECK (status IN ('active', 'superseded')),
    CONSTRAINT dlightrag_memory_records_body_check
        CHECK (char_length(body) BETWEEN 1 AND 500),
    CONSTRAINT dlightrag_memory_records_confidence_check
        CHECK (confidence > 0 AND confidence <= 1)
)
"""

_RECORD_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_dlightrag_memory_records_recall "
    "ON dlightrag_memory_records (owner_id, status, updated_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_dlightrag_memory_records_exact "
    "ON dlightrag_memory_records (owner_id, normalized_body)",
    "CREATE INDEX IF NOT EXISTS idx_dlightrag_memory_records_purge "
    "ON dlightrag_memory_records (status, updated_at) "
    "WHERE status = 'superseded'",
)

_RECORD_COLUMNS = """
owner_id, memory_id, kind, body, normalized_body, confidence, run_id, session_id,
status, supersedes_id, embedding_fingerprint, created_at, updated_at
"""


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
        migrations: Sequence[Migration] = (),
    ) -> None:
        if pool is None and not dsn and pool_factory is None:
            raise ValueError("PostgresMemoryStore needs a pool, a dsn, or a pool factory")
        self._dsn = dsn
        self._pool = pool
        self._pool_factory = pool_factory
        self._owned_pool: Any = None
        self._embedder = embedder
        self._migrations = tuple(migrations)
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
            await conn.execute(_MIGRATION_TABLE)
            await conn.execute(_RECORDS_TABLE)
            for statement in _RECORD_INDEXES:
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
            applied = {
                row["migration_id"]
                for row in await conn.fetch("SELECT migration_id FROM dlightrag_memory_migrations")
            }
            for migration in self._migrations:
                if migration.id in applied:
                    continue
                async with conn.transaction():
                    await migration.apply(conn)
                    await conn.execute(
                        "INSERT INTO dlightrag_memory_migrations (migration_id) VALUES ($1)",
                        migration.id,
                    )

        acquire = await self._acquire_context()
        async with acquire as conn:
            await operation(conn)
        self._initialized = True

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

    async def get(self, *, owner_id: str, memory_id: str) -> MemoryRecord | None:
        async def operation(conn: PGConnection) -> MemoryRecord | None:
            row = await conn.fetchrow(_SELECT_ONE, owner_id, _uuid(memory_id, label="memory_id"))
            return None if row is None else _row(row)

        return await self._read(operation)

    async def list_active(self, *, owner_id: str) -> tuple[MemoryRecord, ...]:
        async def operation(conn: PGConnection) -> tuple[MemoryRecord, ...]:
            rows = await conn.fetch(_SELECT_ACTIVE, owner_id)
            return tuple(_row(row) for row in rows)

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
            return page, (_cursor_time(last), last.memory_id)

        return await self._read(operation)

    async def purge_superseded(self, *, older_than: datetime) -> int:
        async def operation(conn: PGConnection) -> int:
            result = await conn.execute(_PURGE, older_than)
            return int(str(result).rsplit(" ", 1)[-1])

        return await self._write(operation)

    async def search_candidates(
        self, *, owner_id: str, query: str, limit: int
    ) -> tuple[SearchCandidate, ...]:
        """Generate candidates from the exact, sparse, and dense legs."""
        cap = max(1, min(int(limit), 100))
        key = normalized_body(query)

        async def operation(conn: PGConnection) -> tuple[SearchCandidate, ...]:
            candidates: list[SearchCandidate] = []
            exact_rows = await conn.fetch(_SEARCH_EXACT, owner_id, key, cap)
            candidates.extend(
                SearchCandidate(record=_row(row), leg="exact", score=2.0) for row in exact_rows
            )
            for bm25_index in self._bm25_indexes:
                rows = await conn.fetch(
                    build_bm25_sql(index_name=bm25_index, limit=cap), query, owner_id
                )
                for row in rows:
                    record = _row(row)
                    score = float(row["score"])
                    existing = next(
                        (c for c in candidates if c.record.memory_id == record.memory_id), None
                    )
                    if existing is not None:
                        if score > existing.score:
                            existing.score = score
                            existing.leg = "sparse"
                        continue
                    candidates.append(SearchCandidate(record=record, leg="sparse", score=score))
            if self._dense:
                vector = await self._embedding(query)
                dense_rows = await conn.fetch(
                    _SEARCH_DENSE,
                    owner_id,
                    self._embedder.fingerprint,
                    _vector_text(vector),
                    cap,
                )
                for row in dense_rows:
                    record = _row(row)
                    score = float(row["score"])
                    existing = next(
                        (c for c in candidates if c.record.memory_id == record.memory_id), None
                    )
                    if existing is not None:
                        if score > existing.score:
                            existing.score = score
                            existing.leg = "dense"
                        continue
                    candidates.append(SearchCandidate(record=record, leg="dense", score=score))
            candidates.sort(key=lambda candidate: candidate.score, reverse=True)
            return tuple(candidates[:cap])

        return await self._read(operation)

    async def _embedding(self, text: str) -> Vector:
        (vector,) = await self._embedder.embed_documents((text,))
        return vector

    async def _write(self, operation: Any) -> Any:
        async with await self._acquire_context() as conn:
            return await operation(conn)

    async def _read(self, operation: Any) -> Any:
        async with await self._acquire_context() as conn:
            return await operation(conn)


def _insert_params(store: PostgresMemoryStore, *, record: MemoryRecord) -> tuple[Any, ...]:
    return (
        record.owner_id,
        _uuid(record.memory_id, label="memory_id"),
        record.kind,
        record.body,
        normalized_body(record.body),
        record.confidence,
        record.provenance.run_id,
        record.provenance.session_id or "",
        record.status,
        _uuid(record.supersedes_id, label="supersedes_id") if record.supersedes_id else None,
        store._embedder.fingerprint if store._dense else None,  # noqa: SLF001
        record.created_at,
        record.updated_at,
    )


def _embedding_column_sql(dim: int) -> str:
    return f"ALTER TABLE dlightrag_memory_records ADD COLUMN IF NOT EXISTS embedding halfvec({dim})"


def _embedding_index_sql() -> str:
    return (
        "CREATE INDEX IF NOT EXISTS idx_dlightrag_memory_records_dense "
        "ON dlightrag_memory_records USING hnsw (embedding halfvec_cosine_ops)"
    )


_INSERT = """
INSERT INTO dlightrag_memory_records (
    owner_id, memory_id, kind, body, normalized_body, confidence, run_id, session_id,
    status, supersedes_id, embedding_fingerprint, created_at, updated_at
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
"""

_INSERT_WITH_EMBEDDING = """
INSERT INTO dlightrag_memory_records (
    owner_id, memory_id, kind, body, normalized_body, confidence, run_id, session_id,
    status, supersedes_id, embedding_fingerprint, embedding, created_at, updated_at
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $14::halfvec, $12, $13)
"""

_MARK_SUPERSEDED = """
UPDATE dlightrag_memory_records
SET status = 'superseded', updated_at = NOW()
WHERE owner_id = $1 AND memory_id = $2 AND status = 'active'
"""

_DELETE = """
DELETE FROM dlightrag_memory_records
WHERE owner_id = $1 AND memory_id = $2
"""

_DELETE_BODY = """
DELETE FROM dlightrag_memory_records
WHERE owner_id = $1 AND body = $2
"""

_SELECT_ONE = f"""
SELECT {_RECORD_COLUMNS}
FROM dlightrag_memory_records
WHERE owner_id = $1 AND memory_id = $2
"""  # noqa: S608 - interpolates only the trusted _RECORD_COLUMNS constant

_SELECT_ACTIVE = f"""
SELECT {_RECORD_COLUMNS}
FROM dlightrag_memory_records
WHERE owner_id = $1 AND status = 'active'
ORDER BY updated_at DESC
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

_PURGE = """
DELETE FROM dlightrag_memory_records
WHERE status = 'superseded' AND updated_at < $1
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


def _cursor_time(record: MemoryRecord) -> datetime:
    if record.updated_at is not None:
        return record.updated_at
    if record.created_at is not None:
        return record.created_at
    return datetime.min.replace(tzinfo=UTC)


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


class SqlMigration(Migration):
    """One named SQL migration for the package-owned migration registry."""

    def __init__(self, migration_id: str, statements: Sequence[str]) -> None:
        self.id = migration_id
        self._statements = tuple(statements)

    async def apply(self, conn: PGConnection) -> None:
        for statement in self._statements:
            await conn.execute(statement)


__all__ = ["PostgresMemoryStore", "SqlMigration"]
