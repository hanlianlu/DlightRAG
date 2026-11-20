# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Runtime Policies on PostgreSQL: what the store keeps, that every process reads what was last
written, and that a database made before the table gains it.

A policy is one row read by its key whenever a request needs it, so these tests hold the store to
what makes that correct across several API processes at once: no process remembers a document,
two writers never leave a mixture of two, and the table keeps only what is a document at all.

Every test runs in a throwaway database created and dropped per test; skipped without PostgreSQL.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from dlightrag.adapters.postgres.runtime.run_store import (
    PGRunStore,
)
from dlightrag.application.access.principal import UserContext
from dlightrag.application.policies.service import (
    ANSWER_FEEDBACK,
    RuntimePolicies,
)
from tests.support.pg import (
    PG_CONN_KWARGS,
    drop_database,
    skip_without_postgres,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_TABLE = "dlightrag_runtime_policies"


@pytest.fixture
async def pool() -> AsyncIterator[Any]:
    await skip_without_postgres()
    database = f"dlightrag_policies_{uuid.uuid4().hex[:12]}"
    admin = await asyncpg.connect(**PG_CONN_KWARGS)
    try:
        await admin.execute(f'CREATE DATABASE "{database}"')
    finally:
        await admin.close()
    created = await asyncpg.create_pool(
        **{**PG_CONN_KWARGS, "database": database}, min_size=1, max_size=10
    )
    try:
        yield created
    finally:
        await created.close()
        await drop_database(database)


@pytest.fixture
async def store(pool: Any) -> PGRunStore:
    created = PGRunStore(pool=pool)
    await created.initialize()
    return created


def _off_for(person: str) -> dict[str, Any]:
    return {"default": True, "rules": [{"claim": "email", "value": person, "then": False}]}


def _signed_in(email: str) -> UserContext:
    return UserContext(user_id=email, auth_mode="jwt", claims={"email": email})


async def _rows(pool: Any) -> int:
    async with pool.acquire() as conn:
        return int(await conn.fetchval(f"SELECT count(*) FROM {_TABLE}"))


async def test_a_policy_nobody_wrote_has_no_row_and_reads_as_its_code_default(
    store: PGRunStore, pool: Any
) -> None:
    state = await RuntimePolicies(store=store).read(ANSWER_FEEDBACK)

    assert await store.read_runtime_policy(ANSWER_FEEDBACK) is None
    assert await _rows(pool) == 0
    assert state.document.model_dump(mode="json") == {"default": True, "rules": []}
    assert (state.updated_at, state.updated_by) == (None, None)


async def test_a_write_replaces_the_document_whole_and_stamps_who_wrote_it_and_when(
    store: PGRunStore, pool: Any
) -> None:
    policies = RuntimePolicies(store=store)
    # A person whose identifier is not ASCII is kept and read back as it was written.
    first = await policies.write(ANSWER_FEEDBACK, _off_for("张伟@example.com"), "alice")

    second = await policies.write(ANSWER_FEEDBACK, {"default": False}, "bob")

    assert first.document.model_dump(mode="json") == _off_for("张伟@example.com")
    assert first.updated_at is not None and second.updated_at is not None
    assert (first.updated_by, second.updated_by) == ("alice", "bob")
    assert second.updated_at > first.updated_at
    stored = await store.read_runtime_policy(ANSWER_FEEDBACK)
    assert stored is not None
    assert stored.value == {"default": False, "rules": []}, "nothing of the first document stays"
    assert (stored.updated_by, stored.updated_at) == ("bob", second.updated_at)
    assert await _rows(pool) == 1


async def test_a_change_one_process_makes_is_read_by_another_at_its_next_request(
    store: PGRunStore, pool: Any
) -> None:
    """Nothing is cached, so two processes sharing the database never disagree for long."""
    writer = RuntimePolicies(store=store)
    reader = RuntimePolicies(store=PGRunStore(pool=pool))
    intern = _signed_in("intern@example.com")
    assert await reader.value_for(ANSWER_FEEDBACK, intern) is True

    await writer.write(ANSWER_FEEDBACK, _off_for("intern@example.com"), "admin")
    assert await reader.value_for(ANSWER_FEEDBACK, intern) is False

    await writer.write(ANSWER_FEEDBACK, {"default": True}, "admin")
    assert await reader.value_for(ANSWER_FEEDBACK, intern) is True


async def test_writers_at_once_leave_one_whole_document_and_none_of_them_fails(
    store: PGRunStore, pool: Any
) -> None:
    """Two administrators saving the same new policy at once both succeed; the later write wins."""
    documents = [_off_for(f"person-{number}@example.com") for number in range(8)]

    states = await asyncio.gather(
        *(
            RuntimePolicies(store=PGRunStore(pool=pool)).write(ANSWER_FEEDBACK, document, "admin")
            for document in documents
        )
    )

    assert len(states) == len(documents)
    stored = await store.read_runtime_policy(ANSWER_FEEDBACK)
    assert stored is not None and stored.value in documents, "one document as written, never a mix"
    assert await _rows(pool) == 1


async def test_the_table_keeps_only_documents_under_a_key_of_a_sensible_length(
    store: PGRunStore, pool: Any
) -> None:
    """The application decides what a policy holds; the table is the last word on its shape."""
    insert = f"INSERT INTO {_TABLE} (key, value, updated_by) VALUES ($1, $2::jsonb, 'admin')"
    cannot_be = [
        ("key_check", "", "{}"),
        ("key_check", "k" * 65, "{}"),
        ("value_check", ANSWER_FEEDBACK, "[]"),
        ("value_check", ANSWER_FEEDBACK, "true"),
        ("value_check", ANSWER_FEEDBACK, "null"),
    ]

    async with pool.acquire() as conn:
        for constraint, key, value in cannot_be:
            with pytest.raises(asyncpg.CheckViolationError, match=f"{_TABLE}_{constraint}"):
                await conn.execute(insert, key, value)
        await conn.execute(insert, "k" * 64, '{"default": true}')

    assert await _rows(pool) == 1
