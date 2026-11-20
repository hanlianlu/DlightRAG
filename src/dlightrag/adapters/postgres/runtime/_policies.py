# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Runtime Policy persistence mixed into PGRunStore."""

import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from dlightrag.application.policies.service import StoredPolicy

_SELECT_POLICY = """
SELECT value::text AS value, updated_at, updated_by
FROM dlightrag_runtime_policies
WHERE key = $1
"""

# The document is replaced whole and the last write stands, whoever wrote the one before.
_UPSERT_POLICY = """
INSERT INTO dlightrag_runtime_policies (key, value, updated_by)
VALUES ($1, $2::jsonb, $3)
ON CONFLICT (key) DO UPDATE
SET value = EXCLUDED.value,
    updated_at = NOW(),
    updated_by = EXCLUDED.updated_by
RETURNING value::text AS value, updated_at, updated_by
"""


def _stored_policy(row: Any) -> StoredPolicy:
    return StoredPolicy(
        value=json.loads(row["value"]),
        updated_at=row["updated_at"],
        updated_by=str(row["updated_by"]),
    )


class PolicyRunStoreMixin:
    """Runtime Policy operations on PGRunStore.

    A policy is one row read by its key whenever a request needs it. Nothing is cached and no
    process is told of a change, so every process reads what was last written.
    """

    _run_read: Callable[..., Awaitable[Any]]
    _run_write: Callable[..., Awaitable[Any]]

    async def read_runtime_policy(self, key: str) -> StoredPolicy | None:
        """The document last written under ``key``, or None when none ever was."""

        async def _operation(conn: Any) -> StoredPolicy | None:
            row = await conn.fetchrow(_SELECT_POLICY, key)
            return None if row is None else _stored_policy(row)

        return await self._run_read(_operation)

    async def write_runtime_policy(
        self, key: str, value: Mapping[str, Any], actor: str
    ) -> StoredPolicy:
        """Replace the document under ``key`` whole, naming who wrote it."""
        document = json.dumps(value, ensure_ascii=False)

        async def _operation(conn: Any) -> StoredPolicy:
            return _stored_policy(await conn.fetchrow(_UPSERT_POLICY, key, document, actor))

        return await self._run_write(_operation)
