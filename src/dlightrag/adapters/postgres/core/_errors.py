# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Private PostgreSQL exception classification."""

import asyncio
from collections.abc import Awaitable

import asyncpg

from dlightrag.engine.agent.session.repository import UnrepresentablePayloadError

_UNAVAILABLE_EXCEPTIONS = (
    asyncio.TimeoutError,
    TimeoutError,
    ConnectionError,
    OSError,
    asyncpg.exceptions.TooManyConnectionsError,
    asyncpg.exceptions.CannotConnectNowError,
    asyncpg.exceptions.AdminShutdownError,
    asyncpg.exceptions.CrashShutdownError,
    asyncpg.exceptions.PostgresConnectionError,
    asyncpg.exceptions.ConnectionDoesNotExistError,
    asyncpg.exceptions.ConnectionFailureError,
    asyncpg.exceptions.InterfaceError,
)

# PostgreSQL cannot carry U+0000 in a text value, so a payload that reaches the
# server with one is refused rather than stored: 22P05 is
# untranslatable_character (a \u0000 escape inside json/jsonb) and 22021 is
# character_not_in_repertoire (a raw NUL byte inside text). Both classify the
# value, not the connection: re-sending the same value cannot succeed.
_UNREPRESENTABLE_SQLSTATES = frozenset({"22P05", "22021"})


def is_postgres_unavailable(exc: BaseException) -> bool:
    """Return whether ``exc`` means the PostgreSQL session is unavailable."""
    return isinstance(exc, _UNAVAILABLE_EXCEPTIONS)


def is_unrepresentable_payload(exc: BaseException) -> bool:
    """Return whether ``exc`` is PostgreSQL refusing a value it cannot store."""
    return isinstance(exc, asyncpg.exceptions.PostgresError) and (
        getattr(exc, "sqlstate", None) in _UNREPRESENTABLE_SQLSTATES
    )


async def guard_payload[T](statement: Awaitable[T], *, surface: str) -> T:
    """Await one statement, naming a durable payload PostgreSQL refused.

    ``surface`` names the record being written for operator diagnosis. The
    message stays bounded and content-free, because a payload that cannot be
    stored must not be echoed while reporting its own refusal.
    """
    try:
        return await statement
    except asyncpg.exceptions.PostgresError as exc:
        if not is_unrepresentable_payload(exc):
            raise
        raise UnrepresentablePayloadError(
            f"{surface} holds a character PostgreSQL cannot store."
        ) from exc


__all__ = ["guard_payload", "is_postgres_unavailable", "is_unrepresentable_payload"]
