# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Owner-scoped Memory enablement settings in PostgreSQL.

Root product state: the Memory package owns records and recall; whether an
owner's memory is enabled for answer injection is product policy. ``clear`` is
an operation, not a stored flag — it tombstones the owner's records and
leaves ``enabled`` untouched.
"""

from __future__ import annotations

from typing import Any

from dlightrag.adapters.postgres._migrations import TableRequirement
from dlightrag.adapters.postgres._operations import ConnectionPool, PostgresOperationRunner

_CREATE_MEMORY_SETTINGS = """
CREATE TABLE IF NOT EXISTS dlightrag_answer_memory_settings (
    owner_id   TEXT        NOT NULL,
    enabled    BOOLEAN     NOT NULL DEFAULT TRUE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id)
)
"""

MEMORY_SETTINGS_DDL = (_CREATE_MEMORY_SETTINGS,)

MEMORY_SETTINGS_SCHEMA_TABLE = TableRequirement(
    name="dlightrag_answer_memory_settings",
    columns=("owner_id", "enabled", "updated_at"),
    primary_key=("owner_id",),
)

_GET_SETTINGS = """
SELECT enabled
FROM dlightrag_answer_memory_settings
WHERE owner_id = $1
"""

_UPSERT_SETTINGS = """
INSERT INTO dlightrag_answer_memory_settings (owner_id, enabled, updated_at)
VALUES ($1, $2, NOW())
ON CONFLICT (owner_id) DO UPDATE
SET enabled = EXCLUDED.enabled, updated_at = NOW()
"""


class PGMemorySettingsStore(PostgresOperationRunner):
    """Owner-scoped Memory enablement flags."""

    def __init__(self, *, pool: ConnectionPool | None = None) -> None:
        super().__init__(pool=pool)

    async def enabled(self, *, owner_id: str) -> bool:
        """Memory is enabled unless the owner explicitly disabled it."""

        async def operation(conn: Any) -> bool:
            row = await conn.fetchrow(_GET_SETTINGS, owner_id)
            return True if row is None else bool(row["enabled"])

        return await self._run(operation)

    async def set_enabled(self, *, owner_id: str, enabled: bool) -> None:
        async def operation(conn: Any) -> None:
            await conn.execute(_UPSERT_SETTINGS, owner_id, enabled)

        await self._run_once(operation)


__all__ = [
    "MEMORY_SETTINGS_DDL",
    "MEMORY_SETTINGS_SCHEMA_TABLE",
    "PGMemorySettingsStore",
]
