# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Owner Profile Memory activation and invalidation epoch in PostgreSQL."""

from __future__ import annotations

from typing import Any

from dlightrag.adapters.postgres._migrations import TableRequirement
from dlightrag.adapters.postgres._operations import ConnectionPool, PostgresOperationRunner
from dlightrag.engine.answer.memory import MemoryCapability

_CREATE_MEMORY_SETTINGS = """
CREATE TABLE IF NOT EXISTS dlightrag_answer_memory_settings (
    owner_id   TEXT        NOT NULL,
    enabled    BOOLEAN     NOT NULL DEFAULT TRUE,
    epoch      BIGINT      NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id)
)
"""

MEMORY_SETTINGS_DDL = (_CREATE_MEMORY_SETTINGS,)

MEMORY_SETTINGS_SCHEMA_TABLE = TableRequirement(
    name="dlightrag_answer_memory_settings",
    columns=("owner_id", "enabled", "epoch", "updated_at"),
    primary_key=("owner_id",),
)

_GET_SETTINGS = """
SELECT enabled, epoch
FROM dlightrag_answer_memory_settings
WHERE owner_id = $1
"""

_SET_ENABLED = """
INSERT INTO dlightrag_answer_memory_settings (owner_id, enabled, epoch, updated_at)
VALUES ($1, $2, CASE WHEN $2 THEN 0 ELSE 1 END, NOW())
ON CONFLICT (owner_id) DO UPDATE
SET enabled = EXCLUDED.enabled,
    epoch = dlightrag_answer_memory_settings.epoch
        + CASE WHEN dlightrag_answer_memory_settings.enabled AND NOT EXCLUDED.enabled
               THEN 1 ELSE 0 END,
    updated_at = NOW()
RETURNING enabled, epoch
"""

_LOCK_OWNER = "SELECT pg_advisory_xact_lock(hashtext($1))"

_BUMP_EPOCH = """
INSERT INTO dlightrag_answer_memory_settings (owner_id, enabled, epoch, updated_at)
VALUES ($1, TRUE, 1, NOW())
ON CONFLICT (owner_id) DO UPDATE
SET epoch = dlightrag_answer_memory_settings.epoch + 1,
    updated_at = NOW()
RETURNING enabled, epoch
"""


class PGMemorySettingsStore(PostgresOperationRunner):
    """Durable hard capability gate for one owner."""

    def __init__(self, *, pool: ConnectionPool | None = None) -> None:
        super().__init__(pool=pool)

    async def state(self, *, owner_id: str) -> MemoryCapability:
        async def operation(conn: Any) -> MemoryCapability:
            row = await conn.fetchrow(_GET_SETTINGS, owner_id)
            if row is None:
                return MemoryCapability(enabled=True, epoch=0)
            return MemoryCapability(enabled=bool(row["enabled"]), epoch=int(row["epoch"]))

        return await self._run(operation)

    async def state_in_settlement(
        self, *, owner_id: str, settlement: object | None
    ) -> MemoryCapability:
        if settlement is None:
            return await self.state(owner_id=owner_id)
        row = await settlement.fetchrow(_GET_SETTINGS, owner_id)  # type: ignore[attr-defined]
        if row is None:
            return MemoryCapability(enabled=True, epoch=0)
        return MemoryCapability(enabled=bool(row["enabled"]), epoch=int(row["epoch"]))

    async def set_enabled(self, *, owner_id: str, enabled: bool) -> MemoryCapability:
        async def operation(conn: Any) -> MemoryCapability:
            async with conn.transaction():
                await conn.fetchval(_LOCK_OWNER, owner_id)
                row = await conn.fetchrow(_SET_ENABLED, owner_id, enabled)
                return MemoryCapability(enabled=bool(row["enabled"]), epoch=int(row["epoch"]))

        return await self._run_once(operation)

    async def bump_epoch(self, *, owner_id: str) -> MemoryCapability:
        async def operation(conn: Any) -> MemoryCapability:
            async with conn.transaction():
                await conn.fetchval(_LOCK_OWNER, owner_id)
                row = await conn.fetchrow(_BUMP_EPOCH, owner_id)
                return MemoryCapability(enabled=bool(row["enabled"]), epoch=int(row["epoch"]))

        return await self._run_once(operation)


__all__ = [
    "MEMORY_SETTINGS_DDL",
    "MEMORY_SETTINGS_SCHEMA_TABLE",
    "PGMemorySettingsStore",
]
