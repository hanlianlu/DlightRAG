# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""What one Run's writers left in an owner's Profile Memory, over the real PostgreSQL store.

The Agent's own memory tools and the Session Memory Pass both write under a Run. The read lists
the records of that Run that are still active, each with the change behind it, so a person can
see them under the answer and undo one. Which writer made a record is the record's own
provenance; whether it added, corrected or merged is read from the journal row behind it.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest
from dlightrag_memory import Memory, MemoryOriginKind, MemoryProvenance
from dlightrag_memory.postgres import PostgresMemoryStore

from tests.support.pg import PG_CONN_KWARGS, drop_database, skip_without_postgres

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

OWNER = "owner-1"
RUN = "run-1"


@pytest.fixture
async def pool() -> AsyncIterator[Any]:
    await skip_without_postgres()
    database = f"dlightrag_runchg_{uuid.uuid4().hex[:12]}"
    admin = await asyncpg.connect(**PG_CONN_KWARGS)
    try:
        await admin.execute(f'CREATE DATABASE "{database}"')
    finally:
        await admin.close()
    created = await asyncpg.create_pool(
        **{**PG_CONN_KWARGS, "database": database}, min_size=1, max_size=4
    )
    try:
        yield created
    finally:
        await created.close()
        await drop_database(database)


@pytest.fixture
async def memory(pool: Any) -> Memory:
    store = PostgresMemoryStore(pool=pool)
    await store.initialize()
    return Memory(store)


def _by(origin: MemoryOriginKind, run: str | None = RUN) -> MemoryProvenance:
    return MemoryProvenance(
        origin_kind=origin, origin_id=run or "origin", run_id=run, session_id="session-1"
    )


async def _remember(
    memory: Memory,
    body: str,
    key: str,
    *,
    kind: Any = "fact",
    origin: MemoryOriginKind = "answer_run",
    run: str | None = RUN,
    owner: str = OWNER,
    supersedes: str | None = None,
) -> Any:
    receipt = await memory.remember(
        owner_id=owner,
        kind=kind,
        body=body,
        provenance=_by(origin, run),
        idempotency_key=key,
        supersedes_id=supersedes,
    )
    assert receipt.outcome == "changed"
    return receipt


async def test_a_run_lists_what_the_agent_and_the_pass_each_left_in_the_order_they_wrote(
    memory: Memory,
) -> None:
    agent = await _remember(memory, "Prefers tables first.", "a", kind="preference")
    learned = await _remember(memory, "Lives in Munich.", "b", origin="session_pass")

    changes = await memory.run_changes(owner_id=OWNER, run_id=RUN)

    assert [(c.record.body, c.record.provenance.origin_kind, c.action) for c in changes] == [
        ("Prefers tables first.", "answer_run", "add"),
        ("Lives in Munich.", "session_pass", "add"),
    ]
    assert [c.change_id for c in changes] == [agent.change_id, learned.change_id]
    assert [c.record.memory_id for c in changes] == [agent.memory_id, learned.memory_id]
    assert [c.record.kind for c in changes] == ["preference", "fact"]


async def test_a_correction_is_an_update_and_a_merge_is_a_fuse(memory: Memory) -> None:
    older = await _remember(memory, "Lives in Berlin.", "old", run="earlier-run")
    corrected = await _remember(
        memory, "Lives in Munich.", "fix", supersedes=older.memory_id, origin="session_pass"
    )
    first = await _remember(memory, "Likes short answers.", "s1", kind="preference")
    second = await _remember(memory, "Dislikes long answers.", "s2", kind="preference")
    merged = await memory.fuse(
        owner_id=OWNER,
        memory_ids=[first.memory_id, second.memory_id],
        body="Wants short answers.",
        provenance=_by("session_pass"),
        idempotency_key="merge",
    )

    changes = await memory.run_changes(owner_id=OWNER, run_id=RUN)

    # The merged sentence is what is left; the two it replaced are not changes still standing.
    assert [(c.record.body, c.action, c.change_id) for c in changes] == [
        ("Lives in Munich.", "update", corrected.change_id),
        ("Wants short answers.", "fuse", merged.change_id),
    ]


async def test_what_was_forgotten_or_corrected_since_is_no_longer_listed(memory: Memory) -> None:
    kept = await _remember(memory, "Keeps.", "k")
    forgotten = await _remember(memory, "Forgotten later.", "f")
    replaced = await _remember(memory, "Replaced later.", "r")
    await memory.forget(
        owner_id=OWNER,
        memory_id=forgotten.memory_id,
        provenance=_by("management", None),
        idempotency_key="forget",
    )
    await _remember(
        memory, "Corrected by someone else.", "c", supersedes=replaced.memory_id, run="other-run"
    )

    changes = await memory.run_changes(owner_id=OWNER, run_id=RUN)

    assert [c.record.memory_id for c in changes] == [kept.memory_id]


async def test_an_undone_change_leaves_the_list_and_what_the_undo_restores_is_not_a_change(
    memory: Memory,
) -> None:
    older = await _remember(memory, "Lives in Berlin.", "old", run="earlier-run")
    added = await _remember(memory, "Prefers tea.", "added")
    updated = await _remember(memory, "Lives in Munich.", "upd", supersedes=older.memory_id)
    first = await _remember(memory, "Likes short answers.", "s1")
    second = await _remember(memory, "Dislikes long answers.", "s2")
    merged = await memory.fuse(
        owner_id=OWNER,
        memory_ids=[first.memory_id, second.memory_id],
        body="Wants short answers.",
        provenance=_by("answer_run"),
        idempotency_key="merge",
    )
    assert len(await memory.run_changes(owner_id=OWNER, run_id=RUN)) == 3

    # An add is taken back by forgetting the record, an update and a fuse by their change.
    forgot = await memory.forget(
        owner_id=OWNER,
        memory_id=added.memory_id,
        provenance=_by("management", None),
        idempotency_key="forget-added",
    )
    undid_update = await memory.undo(
        owner_id=OWNER,
        change_id=updated.change_id,
        provenance=_by("undo", None),
        idempotency_key="undo-update",
    )
    undid_fuse = await memory.undo(
        owner_id=OWNER,
        change_id=merged.change_id,
        provenance=_by("undo", None),
        idempotency_key="undo-fuse",
    )

    assert [r.outcome for r in (forgot, undid_update, undid_fuse)] == ["changed"] * 3
    assert await memory.run_changes(owner_id=OWNER, run_id=RUN) == ()
    # The restored records exist and stand, but they belong to nobody's Run.
    bodies = {record.body for record in (await memory.browse(owner_id=OWNER))[0]}
    assert {"Lives in Berlin.", "Likes short answers.", "Dislikes long answers."} <= bodies


async def test_a_record_an_agent_undo_restored_is_not_offered_for_another_undo(
    memory: Memory,
) -> None:
    """An undo of an undo is refused, so the record an undo leaves is not listed as a change."""
    earlier = await _remember(memory, "Prefers tea.", "added", run="earlier-run")
    forgot = await memory.forget(
        owner_id=OWNER,
        memory_id=earlier.memory_id,
        provenance=_by("answer_run"),
        idempotency_key="forget",
    )
    restored = await memory.undo(
        owner_id=OWNER,
        change_id=forgot.change_id,
        provenance=_by("answer_run"),
        idempotency_key="undo",
    )
    assert restored.outcome == "changed"

    # The restored record carries this Run's id through the undo's provenance, yet it is a restoring.
    (tea,) = [r for r in (await memory.browse(owner_id=OWNER))[0] if r.body == "Prefers tea."]
    assert tea.provenance.run_id == RUN
    assert await memory.run_changes(owner_id=OWNER, run_id=RUN) == ()


async def test_one_owner_never_sees_another_owners_changes_under_the_same_run_id(
    memory: Memory,
) -> None:
    mine = await _remember(memory, "Mine.", "m")
    await _remember(memory, "Theirs.", "t", owner="owner-2")

    assert [c.record.memory_id for c in await memory.run_changes(owner_id=OWNER, run_id=RUN)] == [
        mine.memory_id
    ]
    assert [c.record.body for c in await memory.run_changes(owner_id="owner-2", run_id=RUN)] == [
        "Theirs."
    ]
    assert await memory.run_changes(owner_id="owner-3", run_id=RUN) == ()


async def test_a_run_that_wrote_nothing_and_a_record_no_run_wrote_are_empty(
    memory: Memory,
) -> None:
    await _remember(memory, "Owner typed it.", "m", origin="management", run=None)
    await _remember(memory, "Another run's.", "o", run="other-run")

    assert await memory.run_changes(owner_id=OWNER, run_id=RUN) == ()
    assert await memory.run_changes(owner_id=OWNER, run_id="never-ran") == ()


async def test_the_read_is_served_by_an_index_on_the_owner_and_run(pool: Any) -> None:
    store = PostgresMemoryStore(pool=pool)
    await store.initialize()
    async with pool.acquire() as conn:
        definition = await conn.fetchval(
            "SELECT indexdef FROM pg_indexes WHERE indexname = $1",
            "idx_dlightrag_memory_records_run",
        )

    assert definition is not None
    assert "(owner_id, run_id)" in definition
    assert "run_id IS NOT NULL" in definition
