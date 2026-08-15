# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Private PostgreSQL exception classification."""

import asyncio

import asyncpg

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


def is_postgres_unavailable(exc: BaseException) -> bool:
    """Return whether ``exc`` means the PostgreSQL session is unavailable."""
    return isinstance(exc, _UNAVAILABLE_EXCEPTIONS)


__all__ = ["is_postgres_unavailable"]
