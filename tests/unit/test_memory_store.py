# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""In-memory Memory Record store isolation and checklist commit."""

from datetime import UTC, datetime, timedelta

import pytest

from dlightrag.answer.errors import MemoryUnavailableError, MemoryWriteRejectedError
from dlightrag.answer.memory import (
    MEMORY_SUPERSEDE_RETENTION_DAYS,
    MemoryProvenance,
    MemoryRecord,
    MemoryWrite,
)
from dlightrag.answer.memory_store import InMemoryAnswerMemoryStore, commit_memory_write


def _record(
    *, owner: str = "alpha", memory_id: str = "m1", body: str = "No email."
) -> MemoryRecord:
    now = datetime.now(UTC)
    return MemoryRecord(
        owner_id=owner,
        memory_id=memory_id,
        kind="preference",
        body=body,
        confidence=0.8,
        provenance=MemoryProvenance(run_id="run-1", session_id="sess-1"),
        created_at=now,
        updated_at=now,
    )


def _remember(**overrides: object) -> MemoryWrite:
    payload: dict[str, object] = {
        "owner_id": "alpha",
        "auth_mode": "jwt",
        "kind": "preference",
        "body": "No email.",
        "confidence": 0.8,
        "provenance": MemoryProvenance(run_id="run-1", session_id="sess-1"),
    }
    payload.update(overrides)
    return MemoryWrite(**payload)  # type: ignore[arg-type]


async def test_owners_cannot_read_each_other() -> None:
    store = InMemoryAnswerMemoryStore()
    await store.insert(_record(owner="alpha", memory_id="m1"))
    await store.insert(_record(owner="beta", memory_id="m1", body="Other."))
    alpha = await store.list_active(owner_id="alpha")
    beta = await store.list_active(owner_id="beta")
    assert [row.body for row in alpha] == ["No email."]
    assert [row.body for row in beta] == ["Other."]


async def test_supersede_hides_old_and_forget_hard_deletes() -> None:
    store = InMemoryAnswerMemoryStore()
    await store.insert(_record(memory_id="old"))
    replacement = _record(memory_id="new", body="Use chat only.")
    await store.supersede(owner_id="alpha", old_id="old", new=replacement)
    active = await store.list_active(owner_id="alpha")
    assert [row.memory_id for row in active] == ["new"]
    superseded = await store.get(owner_id="alpha", memory_id="old")
    assert superseded is not None
    assert superseded.status == "superseded"
    assert await store.forget(owner_id="alpha", memory_id="new") is True
    assert await store.get(owner_id="alpha", memory_id="new") is None


async def test_commit_remember_and_reject_none_owner() -> None:
    store = InMemoryAnswerMemoryStore()
    written = await commit_memory_write(store, _remember())
    assert written is not None
    assert written.body == "No email."
    with pytest.raises(MemoryUnavailableError):
        await commit_memory_write(store, _remember(auth_mode="none"))


async def test_supersede_missing_id_is_a_public_reject() -> None:
    store = InMemoryAnswerMemoryStore()
    with pytest.raises(MemoryWriteRejectedError, match="No matching memory to replace"):
        await commit_memory_write(
            store,
            MemoryWrite(
                owner_id="alpha",
                auth_mode="jwt",
                kind="fact",
                body="Replacement.",
                confidence=1.0,
                provenance=MemoryProvenance(run_id="r", session_id="s"),
                supersedes_id="missing",
            ),
        )


async def test_service_purge_expired_uses_retention_cutoff() -> None:
    from dlightrag.services.memory import MemoryService

    store = InMemoryAnswerMemoryStore()
    await store.insert(_record(memory_id="old"))
    await store.supersede(owner_id="alpha", old_id="old", new=_record(memory_id="new"))
    stale = await store.get(owner_id="alpha", memory_id="old")
    assert stale is not None
    store._rows[("alpha", "old")] = MemoryRecord(
        owner_id=stale.owner_id,
        memory_id=stale.memory_id,
        kind=stale.kind,
        body=stale.body,
        confidence=stale.confidence,
        provenance=stale.provenance,
        status="superseded",
        created_at=stale.created_at,
        updated_at=datetime.now(UTC) - timedelta(days=MEMORY_SUPERSEDE_RETENTION_DAYS + 10),
    )
    removed = await MemoryService(store).purge_expired()
    assert removed == 1


async def test_purge_only_old_superseded_rows() -> None:
    store = InMemoryAnswerMemoryStore()
    await store.insert(_record(memory_id="old"))
    await store.supersede(owner_id="alpha", old_id="old", new=_record(memory_id="new"))
    stale = await store.get(owner_id="alpha", memory_id="old")
    assert stale is not None
    store._rows[("alpha", "old")] = MemoryRecord(
        owner_id=stale.owner_id,
        memory_id=stale.memory_id,
        kind=stale.kind,
        body=stale.body,
        confidence=stale.confidence,
        provenance=stale.provenance,
        status="superseded",
        created_at=stale.created_at,
        updated_at=datetime.now(UTC) - timedelta(days=MEMORY_SUPERSEDE_RETENTION_DAYS + 10),
    )
    assert MEMORY_SUPERSEDE_RETENTION_DAYS == 365
    removed = await store.purge_superseded(
        older_than=datetime.now(UTC) - timedelta(days=MEMORY_SUPERSEDE_RETENTION_DAYS)
    )
    assert removed == 1
    assert await store.get(owner_id="alpha", memory_id="old") is None
    assert await store.get(owner_id="alpha", memory_id="new") is not None
