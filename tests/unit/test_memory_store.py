# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""In-memory Profile Memory operation, receipt, and lifecycle semantics."""

from datetime import UTC, datetime

import pytest
from dlightrag_memory import Memory, MemoryProvenance, MemoryRecord
from dlightrag_memory.store import InMemoryMemoryStore

from dlightrag.answer.errors import MemoryWriteRejectedError


def _provenance(run_id: str = "run-1") -> MemoryProvenance:
    return MemoryProvenance(
        origin_kind="answer_run",
        origin_id=run_id,
        run_id=run_id,
        session_id="session-1",
    )


def _record(
    *, owner: str = "alpha", memory_id: str = "m1", body: str = "No email."
) -> MemoryRecord:
    now = datetime.now(UTC)
    return MemoryRecord(
        owner_id=owner,
        memory_id=memory_id,
        kind="preference",
        body=body,
        provenance=_provenance(),
        created_at=now,
        updated_at=now,
    )


async def _remember(
    memory: Memory,
    *,
    key: str,
    body: str = "No email.",
    kind: str = "preference",
    supersedes_id: str | None = None,
    scope: str | None = None,
    limit: int | None = None,
):
    return await memory.remember(
        owner_id="alpha",
        kind=kind,  # type: ignore[arg-type]
        body=body,
        provenance=_provenance(),
        idempotency_key=key,
        supersedes_id=supersedes_id,
        mutation_scope=scope,
        mutation_limit=limit,
    )


async def test_owners_cannot_read_each_other() -> None:
    store = InMemoryMemoryStore()
    await store.insert(_record(owner="alpha", memory_id="m1"))
    await store.insert(_record(owner="beta", memory_id="m1", body="Other."))
    assert [row.body for row in await store.list_active(owner_id="alpha")] == ["No email."]
    assert [row.body for row in await store.list_active(owner_id="beta")] == ["Other."]


async def test_operation_replay_returns_the_original_receipt() -> None:
    memory = Memory(InMemoryMemoryStore())
    first = await _remember(memory, key="call-1")
    replay = await _remember(memory, key="call-1")

    assert first.outcome == "changed"
    assert replay == first
    assert len(await memory.list_active(owner_id="alpha")) == 1


async def test_owner_guard_rejects_before_journal_or_record_settlement() -> None:
    memory = Memory(InMemoryMemoryStore())

    async def reject(_settlement: object | None) -> None:
        raise MemoryWriteRejectedError("capability changed")

    with pytest.raises(MemoryWriteRejectedError, match="capability changed"):
        await memory.remember(
            owner_id="alpha",
            kind="fact",
            body="Lives in Gothenburg.",
            provenance=_provenance(),
            idempotency_key="call-1",
            guard=reject,
        )

    assert await memory.list_active(owner_id="alpha") == ()
    settled = await _remember(memory, key="call-1", body="Lives in Gothenburg.", kind="fact")
    assert settled.outcome == "changed"


async def test_reusing_an_idempotency_key_with_different_input_rejects() -> None:
    memory = Memory(InMemoryMemoryStore())
    await _remember(memory, key="call-1")

    with pytest.raises(MemoryWriteRejectedError, match="different input"):
        await _remember(memory, key="call-1", body="Use chat.")


async def test_semantic_duplicate_is_unchanged_without_another_record() -> None:
    memory = Memory(InMemoryMemoryStore())
    first = await _remember(memory, key="call-1", body="Use Chinese.")
    duplicate = await _remember(memory, key="call-2", body="  use chinese.  ")

    assert first.outcome == "changed"
    assert duplicate.outcome == "unchanged"
    assert duplicate.memory_id == first.memory_id
    assert await memory.count_active(owner_id="alpha") == 1


async def test_supersede_and_compensating_undo_preserve_history() -> None:
    store = InMemoryMemoryStore()
    memory = Memory(store)
    old = await _remember(memory, key="call-1", body="Lives in Beijing.", kind="fact")
    replacement = await _remember(
        memory,
        key="call-2",
        body="Lives in Gothenburg.",
        kind="fact",
        supersedes_id=old.memory_id,
    )

    assert replacement.outcome == "changed"
    assert [row.body for row in await memory.list_active(owner_id="alpha")] == [
        "Lives in Gothenburg."
    ]

    undone = await memory.undo(
        owner_id="alpha",
        change_id=replacement.change_id,
        provenance=MemoryProvenance(origin_kind="undo", origin_id="undo-1"),
        idempotency_key="undo-1",
    )

    assert undone.outcome == "changed"
    assert [row.body for row in await memory.list_active(owner_id="alpha")] == ["Lives in Beijing."]
    current = await store.get(owner_id="alpha", memory_id=replacement.memory_id or "")
    assert current is not None and current.status == "superseded"


async def test_forget_and_undo_restore_as_a_new_active_record() -> None:
    memory = Memory(InMemoryMemoryStore())
    remembered = await _remember(memory, key="call-1", body="Prefers concise answers.")
    forgotten = await memory.forget(
        owner_id="alpha",
        memory_id=remembered.memory_id,
        provenance=_provenance(),
        idempotency_key="forget-1",
    )
    assert forgotten.outcome == "changed"
    assert await memory.list_active(owner_id="alpha") == ()

    undone = await memory.undo(
        owner_id="alpha",
        change_id=forgotten.change_id,
        provenance=MemoryProvenance(origin_kind="undo", origin_id="undo-1"),
        idempotency_key="undo-1",
    )
    assert undone.outcome == "changed"
    (restored,) = await memory.list_active(owner_id="alpha")
    assert restored.body == "Prefers concise answers."
    assert restored.memory_id != remembered.memory_id


async def test_stale_or_repeated_undo_conflicts() -> None:
    memory = Memory(InMemoryMemoryStore())
    remembered = await _remember(memory, key="call-1")
    first = await memory.undo(
        owner_id="alpha",
        change_id=remembered.change_id,
        provenance=MemoryProvenance(origin_kind="undo", origin_id="undo-1"),
        idempotency_key="undo-1",
    )
    second = await memory.undo(
        owner_id="alpha",
        change_id=remembered.change_id,
        provenance=MemoryProvenance(origin_kind="undo", origin_id="undo-2"),
        idempotency_key="undo-2",
    )
    assert first.outcome == "changed"
    assert second.outcome == "conflict"


async def test_per_run_cap_counts_only_changed_mutations() -> None:
    memory = Memory(InMemoryMemoryStore())
    first = await _remember(memory, key="call-1", body="One.", scope="run-1", limit=2)
    duplicate = await _remember(memory, key="call-2", body="one.", scope="run-1", limit=2)
    await _remember(memory, key="call-3", body="Two.", scope="run-1", limit=2)

    assert first.changed
    assert duplicate.outcome == "unchanged"
    with pytest.raises(MemoryWriteRejectedError, match="mutation limit"):
        await _remember(memory, key="call-4", body="Three.", scope="run-1", limit=2)
    assert await memory.count_active(owner_id="alpha") == 2


async def test_clear_physically_removes_records_and_operation_replay() -> None:
    memory = Memory(InMemoryMemoryStore())
    first = await _remember(memory, key="call-1")
    assert await memory.clear(owner_id="alpha") == 1
    assert await memory.list_active(owner_id="alpha") == ()

    replay_after_clear = await _remember(memory, key="call-1")
    assert replay_after_clear.changed
    assert replay_after_clear.change_id == first.change_id
    assert await memory.count_active(owner_id="alpha") == 1


async def test_recall_searches_with_query() -> None:
    memory = Memory(InMemoryMemoryStore())
    await _remember(memory, key="call-1", body="No email.")
    await _remember(memory, key="call-2", body="Unrelated fact about trains.", kind="fact")

    result = await memory.recall(owner_id="alpha", query="email", top_k=5)

    assert result.strategy == "query_search"
    assert [record.body for record in result.records] == ["No email."]
    assert result.content_chars == len("No email.")
