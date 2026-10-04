# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Each owner's Agent Accounts in PostgreSQL, their passwords sealed under the key ring (ADR 0034)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from dlightrag.adapters.postgres.core._migrations import TableRequirement
from dlightrag.adapters.postgres.core._operations import ConnectionPool, PostgresOperationRunner
from dlightrag.engine.answer.agent_browser import StoredAgentAccount

_CREATE_AGENT_ACCOUNTS = """
CREATE TABLE IF NOT EXISTS dlightrag_agent_accounts (
    owner_id           TEXT        NOT NULL,
    site               TEXT        NOT NULL,
    account_id         TEXT        NOT NULL,
    email              TEXT,
    username           TEXT,
    key_id             TEXT        NOT NULL,
    encrypted_envelope TEXT        NOT NULL,
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id, site),
    CONSTRAINT dlightrag_agent_accounts_identity_check
        CHECK (email IS NOT NULL OR username IS NOT NULL)
)
"""

AGENT_ACCOUNTS_DDL = (_CREATE_AGENT_ACCOUNTS,)

AGENT_ACCOUNTS_SCHEMA_TABLE = TableRequirement(
    name="dlightrag_agent_accounts",
    columns=(
        "owner_id",
        "site",
        "account_id",
        "email",
        "username",
        "key_id",
        "encrypted_envelope",
        "updated_at",
    ),
    primary_key=("owner_id", "site"),
    checks=("dlightrag_agent_accounts_identity_check",),
)

_SELECT_ACCOUNT = """
SELECT owner_id, site, account_id, email, username, key_id, encrypted_envelope
FROM dlightrag_agent_accounts
WHERE owner_id = $1 AND site = $2
"""

# The account id and the envelope are one fact: the envelope is bound to the id, so a reset
# that keeps the id replaces the envelope and a first registration inserts both.
_UPSERT_ACCOUNT = """
INSERT INTO dlightrag_agent_accounts
    (owner_id, site, account_id, email, username, key_id, encrypted_envelope)
VALUES ($1, $2, $3, $4, $5, $6, $7)
ON CONFLICT (owner_id, site) DO UPDATE
SET account_id = EXCLUDED.account_id,
    email = EXCLUDED.email,
    username = EXCLUDED.username,
    key_id = EXCLUDED.key_id,
    encrypted_envelope = EXCLUDED.encrypted_envelope,
    updated_at = NOW()
"""

_SELECT_SEALED_UNDER = """
SELECT owner_id, site, account_id, email, username, key_id, encrypted_envelope
FROM dlightrag_agent_accounts
WHERE key_id = ANY($1::text[])
ORDER BY owner_id, site
LIMIT $2
"""

# A re-seal changes no account, so the row's time stays what the owner last did to it, and it
# loses to anything that replaced the envelope after it was read.
_RESEAL_ACCOUNT = """
UPDATE dlightrag_agent_accounts
SET key_id = $5, encrypted_envelope = $6
WHERE owner_id = $1 AND site = $2 AND account_id = $3 AND encrypted_envelope = $4
RETURNING 1
"""


def _stored(row: Any) -> StoredAgentAccount:
    return StoredAgentAccount(
        owner_id=row["owner_id"],
        site=row["site"],
        account_id=row["account_id"],
        email=row["email"],
        username=row["username"],
        key_id=row["key_id"],
        envelope=row["encrypted_envelope"],
    )


class PGAgentAccountStore(PostgresOperationRunner):
    """Each owner's Agent Accounts, one row per site."""

    def __init__(self, *, pool: ConnectionPool | None = None) -> None:
        super().__init__(pool=pool)

    async def account(self, *, owner_id: str, site: str) -> StoredAgentAccount | None:
        async def operation(conn: Any) -> StoredAgentAccount | None:
            row = await conn.fetchrow(_SELECT_ACCOUNT, owner_id, site)
            return None if row is None else _stored(row)

        return await self._run(operation)

    async def save(self, account: StoredAgentAccount) -> None:
        async def operation(conn: Any) -> None:
            await conn.execute(
                _UPSERT_ACCOUNT,
                account.owner_id,
                account.site,
                account.account_id,
                account.email,
                account.username,
                account.key_id,
                account.envelope,
            )

        await self._run(operation)

    async def sealed_under(
        self, *, key_ids: Sequence[str], limit: int
    ) -> tuple[StoredAgentAccount, ...]:
        async def operation(conn: Any) -> tuple[StoredAgentAccount, ...]:
            rows = await conn.fetch(_SELECT_SEALED_UNDER, list(key_ids), limit)
            return tuple(_stored(row) for row in rows)

        return await self._run(operation)

    async def reseal(self, account: StoredAgentAccount, *, key_id: str, envelope: str) -> bool:
        async def operation(conn: Any) -> bool:
            changed = await conn.fetchval(
                _RESEAL_ACCOUNT,
                account.owner_id,
                account.site,
                account.account_id,
                account.envelope,
                key_id,
                envelope,
            )
            return bool(changed)

        return await self._run(operation)


__all__ = [
    "AGENT_ACCOUNTS_DDL",
    "AGENT_ACCOUNTS_SCHEMA_TABLE",
    "PGAgentAccountStore",
]
