# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Owned-database teardown must not surface an unrelated privilege error as a test failure."""

import asyncpg
import pytest

from tests.integration.run_runtime_pg_harness import drop_owned_database


class FlakyAdmin:
    """A connection whose first `failures` drop attempts hit a privileged backend."""

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.attempts = 0
        self.statements: list[str] = []

    async def execute(self, query: str) -> str:
        self.statements.append(query)
        self.attempts += 1
        if self.attempts <= self.failures:
            raise asyncpg.exceptions.InsufficientPrivilegeError(
                "permission denied to terminate process"
            )
        return "DROP DATABASE"

    async def fetch(self, query: str, *args: object) -> list[object]:
        self.statements.append(query)
        return [("1234", "idle", 'LISTEN "fixture"')]


@pytest.mark.asyncio
async def test_drop_retries_a_transient_privileged_backend():
    """A backend the role may not terminate is transient: retry, then drop."""
    admin = FlakyAdmin(failures=1)
    await drop_owned_database(admin, "dlightrag_demo")
    assert admin.attempts == 2, "the drop must be retried once the privileged backend detaches"
    assert admin.statements[-1] == 'DROP DATABASE IF EXISTS "dlightrag_demo" WITH (FORCE)'


@pytest.mark.asyncio
async def test_drop_reports_the_obstacle_instead_of_a_bare_permission_error():
    """When the backend never detaches, say what was attached instead of leaking a privilege error."""
    admin = FlakyAdmin(failures=99)
    with pytest.raises(RuntimeError, match="still attached"):
        await drop_owned_database(admin, "dlightrag_demo")
