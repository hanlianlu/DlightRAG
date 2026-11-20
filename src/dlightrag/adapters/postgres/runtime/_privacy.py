# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Conversation Contribution persistence mixed into PGRunStore."""

from collections.abc import Awaitable, Callable
from typing import Any

from dlightrag.application.privacy import PrivacySettings

_SELECT_PRIVACY = """
SELECT contribute, contribute_since
FROM dlightrag_privacy_settings
WHERE owner_id = $1
"""

# The moment is decided by this one statement and by nothing before it: the clock when the
# permission turns on after being off or absent, the stored moment when it stays on, none when it
# turns off. Writers that meet at the owner's row go one after the other, the later one reading
# what the earlier one left, so none decides on a state that has already changed and a second
# turn-on never moves the moment. The clock is `clock_timestamp()`, read when the statement writes
# and not `NOW()`, the instant its transaction began: a turn-on that waited behind a turn-off is
# dated after that turn-off committed, never before it, so it cannot cover a conversation begun
# under the permission just withdrawn. `updated_at` reads the same clock, so it is never earlier
# than the moment. A Run's and a Session's `created_at` are `NOW()` of the transaction that wrote
# them, so the moment can only come later than they would have it, and later covers less.
_UPSERT_PRIVACY = """
INSERT INTO dlightrag_privacy_settings AS settings
    (owner_id, contribute, contribute_since, updated_at)
VALUES ($1, $2::boolean, CASE WHEN $2::boolean THEN clock_timestamp() END, clock_timestamp())
ON CONFLICT (owner_id) DO UPDATE
SET contribute = EXCLUDED.contribute,
    contribute_since = CASE
        WHEN NOT EXCLUDED.contribute THEN NULL
        WHEN settings.contribute THEN settings.contribute_since
        ELSE clock_timestamp()
    END,
    updated_at = clock_timestamp()
RETURNING contribute, contribute_since
"""


def _settings(row: Any) -> PrivacySettings:
    return PrivacySettings(contribute=bool(row["contribute"]), since=row["contribute_since"])


class PrivacyRunStoreMixin:
    """Conversation Contribution operations on PGRunStore.

    An owner who has chosen has one row, and one who has not has none, which a read answers as not
    contributing. A write is one statement made once and never replayed, because a replay that
    reached the database after a later turn-off would give back a permission the owner had
    withdrawn. A read changes nothing, so it may be retried.
    """

    _run_read: Callable[..., Awaitable[Any]]
    _run_write: Callable[..., Awaitable[Any]]

    async def read_privacy_settings(self, *, owner_id: str) -> PrivacySettings:
        """The owner's setting; an owner with no row is not contributing."""

        async def _operation(conn: Any) -> PrivacySettings:
            row = await conn.fetchrow(_SELECT_PRIVACY, owner_id)
            return PrivacySettings(contribute=False, since=None) if row is None else _settings(row)

        return await self._run_read(_operation)

    async def write_privacy_settings(self, *, owner_id: str, contribute: bool) -> PrivacySettings:
        """Keep the owner's choice and answer the row as it is kept."""

        async def _operation(conn: Any) -> PrivacySettings:
            return _settings(await conn.fetchrow(_UPSERT_PRIVACY, owner_id, contribute))

        return await self._run_write(_operation)
