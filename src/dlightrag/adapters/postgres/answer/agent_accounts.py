# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Each owner's Agent Accounts in PostgreSQL, their passwords sealed under the key ring, and the
owner's switch for new sign-ups (ADR 0034)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from dlightrag.adapters.postgres.core._migrations import TableRequirement
from dlightrag.adapters.postgres.core._operations import ConnectionPool, PostgresOperationRunner
from dlightrag.engine.answer.agent_browser import AgentAccountSummary, StoredAgentAccount

# ``updated_at`` is the last time the owner's credentials changed, ``created_at`` the first
# registration, which a reset keeps, and ``last_used_at`` the last login that filled them.
_CREATE_AGENT_ACCOUNTS = """
CREATE TABLE IF NOT EXISTS dlightrag_agent_accounts (
    owner_id           TEXT        NOT NULL,
    site               TEXT        NOT NULL,
    account_id         TEXT        NOT NULL,
    email              TEXT,
    username           TEXT,
    key_id             TEXT        NOT NULL,
    encrypted_envelope TEXT        NOT NULL,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_used_at       TIMESTAMPTZ,
    PRIMARY KEY (owner_id, site),
    CONSTRAINT dlightrag_agent_accounts_identity_check
        CHECK (email IS NOT NULL OR username IS NOT NULL)
)
"""

# Each owner's switch for new sign-ups. A row exists only once the owner has chosen, and an owner
# with none has them on, so the deployment's allowance is the only thing that has to be said.
_CREATE_AGENT_ACCOUNT_SETTINGS = """
CREATE TABLE IF NOT EXISTS dlightrag_agent_account_settings (
    owner_id             TEXT        NOT NULL,
    sign_ups_enabled     BOOLEAN     NOT NULL DEFAULT TRUE,
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id)
)
"""

AGENT_ACCOUNTS_DDL = (_CREATE_AGENT_ACCOUNTS,)
AGENT_ACCOUNT_SETTINGS_DDL = (_CREATE_AGENT_ACCOUNT_SETTINGS,)

# What advances a database whose accounts table was made before it kept these times. A
# registration time it never recorded is taken to be the last time the account's credentials
# changed, which is the registration of an account no one has reset. Only a row that the new
# column gave the time of this migration, later than anything the row was written at, is touched,
# so a fresh baseline, whose table is empty, and any row written since are left as they are.
AGENT_ACCOUNT_ACTIVITY_AND_SIGN_UPS_DDL = (
    "ALTER TABLE dlightrag_agent_accounts "
    "ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()",
    "ALTER TABLE dlightrag_agent_accounts ADD COLUMN IF NOT EXISTS last_used_at TIMESTAMPTZ",
    "UPDATE dlightrag_agent_accounts SET created_at = updated_at WHERE created_at > updated_at",
    *AGENT_ACCOUNT_SETTINGS_DDL,
)

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
        "created_at",
        "updated_at",
        "last_used_at",
    ),
    primary_key=("owner_id", "site"),
    checks=("dlightrag_agent_accounts_identity_check",),
)

AGENT_ACCOUNT_SETTINGS_SCHEMA_TABLE = TableRequirement(
    name="dlightrag_agent_account_settings",
    columns=("owner_id", "sign_ups_enabled", "updated_at"),
    primary_key=("owner_id",),
)

_SELECT_ACCOUNT = """
SELECT owner_id, site, account_id, email, username, key_id, encrypted_envelope
FROM dlightrag_agent_accounts
WHERE owner_id = $1 AND site = $2
"""

_SELECT_SUMMARIES = """
SELECT site, email, username, created_at, last_used_at
FROM dlightrag_agent_accounts
WHERE owner_id = $1
ORDER BY site
"""

# The account id and the envelope are one fact: the envelope is bound to the id, so a reset
# that keeps the id replaces the envelope and a first registration inserts both. A reset is
# the same account, so it keeps the time of its registration and of its last login.
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

_DELETE_ACCOUNT = """
DELETE FROM dlightrag_agent_accounts
WHERE owner_id = $1 AND site = $2
RETURNING 1
"""

# A login that read an account which was replaced or removed meanwhile touches nothing: the
# account id is the account the login filled.
_MARK_USED = """
UPDATE dlightrag_agent_accounts
SET last_used_at = NOW()
WHERE owner_id = $1 AND site = $2 AND account_id = $3
"""

_GET_SIGN_UPS = """
SELECT sign_ups_enabled
FROM dlightrag_agent_account_settings
WHERE owner_id = $1
"""

_SET_SIGN_UPS = """
INSERT INTO dlightrag_agent_account_settings (owner_id, sign_ups_enabled)
VALUES ($1, $2)
ON CONFLICT (owner_id) DO UPDATE
SET sign_ups_enabled = EXCLUDED.sign_ups_enabled,
    updated_at = NOW()
RETURNING sign_ups_enabled
"""

_SELECT_SEALED_UNDER = """
SELECT owner_id, site, account_id, email, username, key_id, encrypted_envelope
FROM dlightrag_agent_accounts
WHERE key_id = ANY($1::text[]) AND (owner_id, site) > ($2::text, $3::text)
ORDER BY owner_id, site
LIMIT $4
"""

# A re-seal changes no account, so the row's times stay what the owner last did to it, and it
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

    async def summaries(self, *, owner_id: str) -> tuple[AgentAccountSummary, ...]:
        async def operation(conn: Any) -> tuple[AgentAccountSummary, ...]:
            rows = await conn.fetch(_SELECT_SUMMARIES, owner_id)
            return tuple(
                AgentAccountSummary(
                    site=row["site"],
                    email=row["email"],
                    username=row["username"],
                    created_at=row["created_at"],
                    last_used_at=row["last_used_at"],
                )
                for row in rows
            )

        return await self._run(operation)

    async def delete(self, *, owner_id: str, site: str) -> bool:
        async def operation(conn: Any) -> bool:
            return bool(await conn.fetchval(_DELETE_ACCOUNT, owner_id, site))

        # Whether there was an account to remove is the answer, so a retry must not give it.
        return await self._run_once(operation)

    async def mark_used(self, account: StoredAgentAccount) -> None:
        async def operation(conn: Any) -> None:
            await conn.execute(_MARK_USED, account.owner_id, account.site, account.account_id)

        await self._run(operation)

    async def sealed_under(
        self, *, key_ids: Sequence[str], after: tuple[str, str] = ("", ""), limit: int
    ) -> tuple[StoredAgentAccount, ...]:
        async def operation(conn: Any) -> tuple[StoredAgentAccount, ...]:
            rows = await conn.fetch(_SELECT_SEALED_UNDER, list(key_ids), *after, limit)
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


class PGAgentAccountSettingsStore(PostgresOperationRunner):
    """Whether each owner lets the Agent sign up for new accounts."""

    def __init__(self, *, pool: ConnectionPool | None = None) -> None:
        super().__init__(pool=pool)

    async def sign_ups_enabled(self, *, owner_id: str) -> bool:
        async def operation(conn: Any) -> bool:
            enabled = await conn.fetchval(_GET_SIGN_UPS, owner_id)
            return True if enabled is None else bool(enabled)

        return await self._run(operation)

    async def set_sign_ups(self, *, owner_id: str, enabled: bool) -> bool:
        async def operation(conn: Any) -> bool:
            return bool(await conn.fetchval(_SET_SIGN_UPS, owner_id, enabled))

        return await self._run(operation)


__all__ = [
    "AGENT_ACCOUNTS_DDL",
    "AGENT_ACCOUNTS_SCHEMA_TABLE",
    "AGENT_ACCOUNT_ACTIVITY_AND_SIGN_UPS_DDL",
    "AGENT_ACCOUNT_SETTINGS_DDL",
    "AGENT_ACCOUNT_SETTINGS_SCHEMA_TABLE",
    "PGAgentAccountSettingsStore",
    "PGAgentAccountStore",
]
