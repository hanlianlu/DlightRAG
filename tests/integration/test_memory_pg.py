# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL Profile Memory records, journal, atomic receipts, undo, and recall."""

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest
from dlightrag_memory import (
    Memory,
    MemoryKind,
    MemoryOperation,
    MemoryOperationReceipt,
    MemoryProvenance,
    MemoryRecord,
)
from dlightrag_memory._storage.pg_bm25 import INDEX_NAME
from dlightrag_memory.errors import MemoryWriteRejectedError
from dlightrag_memory.mcp_server import _forget as mcp_forget
from dlightrag_memory.mcp_server import _recall as mcp_recall
from dlightrag_memory.mcp_server import _remember as mcp_remember
from dlightrag_memory.mcp_server import _undo as mcp_undo
from dlightrag_memory.normalize import normalized_body
from dlightrag_memory.policy import RECALL_CHAR_BUDGET, RECALL_TOP_K
from dlightrag_memory.ports import NullEmbedder, TextEmbedder
from dlightrag_memory.postgres import PostgresMemoryStore
from dlightrag_memory.store import operation_change_id, operation_record_id

from dlightrag.adapters.postgres.answer.memory_settings import (
    MEMORY_SETTINGS_DDL,
    PGMemorySettingsStore,
)
from dlightrag.application.memory import (
    MemoryDisabledError,
    MemoryListPageRequest,
    MemoryService,
)
from dlightrag.engine.agent.tools import ToolResult
from dlightrag.engine.answer.tools.memory import (
    ForgetInput,
    MemoryHost,
    RecallInput,
    RememberInput,
    forget_tool,
    recall_memory_tool,
    remember_tool,
)
from tests.support.pg import PG_CONN_KWARGS, drop_database, skip_without_postgres
from tests.tool_helpers import recording_tool_runtime, tool_runtime

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_PG: dict[str, Any] = PG_CONN_KWARGS


@asynccontextmanager
async def _scratch_store(
    embedder: TextEmbedder = NullEmbedder(),
) -> AsyncIterator[PostgresMemoryStore]:
    await skip_without_postgres()
    db_name = f"dlightrag_mem_{uuid.uuid4().hex[:12]}"
    admin = await asyncpg.connect(**_PG)
    try:
        await admin.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin.close()
    pool = await asyncpg.create_pool(**{**_PG, "database": db_name}, min_size=1, max_size=4)
    try:
        if not isinstance(embedder, NullEmbedder):
            # The host schema owns pgvector; a scratch database has to add it.
            async with pool.acquire() as conn:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        created = PostgresMemoryStore(pool=pool, embedder=embedder)
        await created.initialize()
        try:
            yield created
        finally:
            await created.aclose()
    finally:
        await pool.close()
        await drop_database(db_name)


@pytest.fixture
async def store() -> AsyncIterator[PostgresMemoryStore]:
    async with _scratch_store() as created:
        yield created


class _TopicEmbedder:
    """Deterministic vectors with one axis per topic word, recording document calls."""

    dim = 3
    _TOPICS = ("tea", "coffee", "train")

    def __init__(self, relevance_floor: float | None = 0.5) -> None:
        self.document_calls: list[tuple[str, ...]] = []
        self._relevance_floor = relevance_floor

    @property
    def embedding_fingerprint(self) -> str:
        return "test:topic@local"

    @property
    def relevance_floor(self) -> float | None:
        return self._relevance_floor

    def _vector(self, text: str) -> list[float]:
        lowered = text.lower()
        return [1.0 if topic in lowered else 0.01 for topic in self._TOPICS]

    async def embed_documents(self, texts: Sequence[str]) -> Sequence[list[float]]:
        self.document_calls.append(tuple(texts))
        return [self._vector(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@pytest.fixture
async def dense_store() -> AsyncIterator[tuple[PostgresMemoryStore, _TopicEmbedder]]:
    embedder = _TopicEmbedder()
    async with _scratch_store(embedder) as created:
        yield created, embedder


def _provenance(run: str = "run-1") -> MemoryProvenance:
    return MemoryProvenance(
        origin_kind="answer_run", origin_id=run, run_id=run, session_id="session-1"
    )


def _record(
    *,
    owner: str = "alpha",
    body: str = "No email.",
    memory_id: str | None = None,
    kind: MemoryKind = "preference",
) -> MemoryRecord:
    now = datetime.now(UTC)
    return MemoryRecord(
        owner_id=owner,
        memory_id=memory_id or str(uuid.uuid4()),
        kind=kind,
        body=body,
        provenance=_provenance(),
        created_at=now,
        updated_at=now,
    )


async def _plant(store: PostgresMemoryStore, record: MemoryRecord) -> None:
    """Write one row directly, for a state no operation produces on demand."""
    async with store._pool.acquire() as conn:  # type: ignore[union-attr]
        await conn.execute(
            "INSERT INTO dlightrag_memory_records (owner_id, memory_id, kind, body, "
            "normalized_body, origin_kind, origin_id, run_id, session_id, status, "
            "created_at, updated_at) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 'active', "
            "$10, $11)",
            record.owner_id,
            uuid.UUID(record.memory_id),
            record.kind,
            record.body,
            normalized_body(record.body),
            record.provenance.origin_kind,
            record.provenance.origin_id,
            record.provenance.run_id,
            record.provenance.session_id,
            record.created_at,
            record.updated_at,
        )


async def _remember(
    memory: Memory, body: str, *, kind: MemoryKind = "fact", owner_id: str = "alpha"
) -> MemoryOperationReceipt:
    return await memory.remember(
        owner_id=owner_id,
        kind=kind,
        body=body,
        provenance=_provenance(),
        idempotency_key=f"{kind}:{uuid.uuid5(uuid.NAMESPACE_URL, body)}",
    )


async def _active(memory: Memory, *, owner_id: str = "alpha") -> tuple[MemoryRecord, ...]:
    records, _ = await memory.browse(owner_id=owner_id, limit=100)
    return records


async def _store_active(
    store: PostgresMemoryStore, *, owner_id: str = "alpha"
) -> tuple[MemoryRecord, ...]:
    records, _ = await store.list_active_page(owner_id=owner_id, limit=100)
    return records


async def test_pg_owners_are_isolated(store: PostgresMemoryStore) -> None:
    memory = Memory(store)
    await _remember(memory, "Alpha only.", owner_id="alpha")
    await _remember(memory, "Beta only.", owner_id="beta")
    assert [row.body for row in await _store_active(store, owner_id="alpha")] == ["Alpha only."]
    assert [row.body for row in await _store_active(store, owner_id="beta")] == ["Beta only."]


async def test_pg_initialization_rejects_legacy_confidence_schema(
    store: PostgresMemoryStore,
) -> None:
    pool = store._pool
    assert pool is not None
    async with pool.acquire() as conn:
        await conn.execute(
            "ALTER TABLE dlightrag_memory_records "
            "ADD COLUMN confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0"
        )
    with pytest.raises(RuntimeError, match="removed confidence"):
        await store.initialize()


async def test_pg_operation_replay_duplicate_cap_and_schema(store: PostgresMemoryStore) -> None:
    memory = Memory(store)
    first = await memory.remember(
        owner_id="alpha",
        kind="preference",
        body="Use Chinese.",
        provenance=_provenance(),
        idempotency_key="call-1",
        mutation_scope="run-1",
        mutation_limit=1,
    )
    replay = await memory.remember(
        owner_id="alpha",
        kind="preference",
        body="Use Chinese.",
        provenance=_provenance(),
        idempotency_key="call-1",
        mutation_scope="run-1",
        mutation_limit=1,
    )
    duplicate = await memory.remember(
        owner_id="alpha",
        kind="preference",
        body="  use chinese. ",
        provenance=_provenance(),
        idempotency_key="call-2",
        mutation_scope="run-1",
        mutation_limit=1,
    )
    assert replay == first
    assert duplicate.outcome == "unchanged"
    assert duplicate.memory_id == first.memory_id
    with pytest.raises(MemoryWriteRejectedError, match="mutation limit"):
        await memory.remember(
            owner_id="alpha",
            kind="fact",
            body="Lives in Gothenburg.",
            provenance=_provenance(),
            idempotency_key="call-3",
            mutation_scope="run-1",
            mutation_limit=1,
        )
    async with store._pool.acquire() as conn:  # type: ignore[union-attr]
        assert (
            await conn.fetchval(
                "SELECT COUNT(*) FROM dlightrag_memory_operations WHERE owner_id = 'alpha'"
            )
            == 2
        )
        assert not await conn.fetchval(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'dlightrag_memory_records' AND column_name = 'confidence'"
        )


async def test_pg_reusing_an_idempotency_key_with_different_input_rejects(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)
    await memory.remember(
        owner_id="alpha",
        kind="preference",
        body="No email.",
        provenance=_provenance(),
        idempotency_key="call-1",
    )

    with pytest.raises(MemoryWriteRejectedError, match="different input"):
        await memory.remember(
            owner_id="alpha",
            kind="preference",
            body="Use chat.",
            provenance=_provenance(),
            idempotency_key="call-1",
        )
    assert [row.body for row in await _active(memory)] == ["No email."]


async def test_pg_guard_rejection_settles_neither_journal_nor_record(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)

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

    assert await _active(memory) == ()
    async with store._pool.acquire() as conn:  # type: ignore[union-attr]
        assert await conn.fetchval("SELECT COUNT(*) FROM dlightrag_memory_operations") == 0
    settled = await memory.remember(
        owner_id="alpha",
        kind="fact",
        body="Lives in Gothenburg.",
        provenance=_provenance(),
        idempotency_key="call-1",
    )
    assert settled.outcome == "changed"


async def test_pg_mutation_cap_counts_only_changed_operations(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)

    async def remember(key: str, body: str):
        return await memory.remember(
            owner_id="alpha",
            kind="preference",
            body=body,
            provenance=_provenance(),
            idempotency_key=key,
            mutation_scope="run-1",
            mutation_limit=2,
        )

    assert (await remember("call-1", "One.")).outcome == "changed"
    assert (await remember("call-2", "one.")).outcome == "unchanged"
    assert (await remember("call-3", "Two.")).outcome == "changed"
    with pytest.raises(MemoryWriteRejectedError, match="mutation limit"):
        await remember("call-4", "Three.")
    assert await memory.count_active(owner_id="alpha") == 2


async def _service(store: PostgresMemoryStore) -> MemoryService:
    """The product gate over this store, with its settings table beside it."""
    pool = store._pool
    assert pool is not None
    async with pool.acquire() as conn:
        for statement in MEMORY_SETTINGS_DDL:
            await conn.execute(statement)
    return MemoryService(
        store,
        settings_store=PGMemorySettingsStore(pool=pool),
        memory_list_cursor_secret=b"memory-pg-list-test",
    )


async def test_pg_owner_lock_rechecks_deactivation_before_mutation_commit(
    store: PostgresMemoryStore,
) -> None:
    service = await _service(store)
    pool = store._pool
    assert pool is not None

    async with pool.acquire() as conn, conn.transaction():
        await conn.fetchval("SELECT pg_advisory_xact_lock(hashtext($1))", "alpha")
        await conn.execute(
            "INSERT INTO dlightrag_answer_memory_settings "
            "(owner_id, enabled, epoch) VALUES ($1, FALSE, 1)",
            "alpha",
        )
        pending = asyncio.create_task(
            service.remember(
                owner_id="alpha",
                auth_mode="jwt",
                kind="fact",
                body="Stable.",
                provenance=_provenance(),
                idempotency_key="call-after-disable",
            )
        )
        await asyncio.sleep(0.05)
        assert not pending.done()

    with pytest.raises(MemoryDisabledError):
        await pending
    assert await store.count_active(owner_id="alpha") == 0


def _management() -> MemoryProvenance:
    return MemoryProvenance(origin_kind="management", origin_id="request-1")


async def test_pg_disabled_owner_keeps_only_the_settings_control_plane(
    store: PostgresMemoryStore,
) -> None:
    service = await _service(store)
    owner = {"owner_id": "alpha", "auth_mode": "jwt"}
    disabled = await service.set_enabled(**owner, enabled=False)

    assert disabled.enabled is False
    assert disabled.active_count is None
    assert disabled.epoch == 1
    assert (await service.settings(**owner)).active_count is None
    with pytest.raises(MemoryDisabledError):
        await service.list_active_page(owner_id="alpha", auth_mode="jwt")
    with pytest.raises(MemoryDisabledError):
        await service.remember(
            **owner,
            kind="fact",
            body="Stable.",
            provenance=_management(),
            idempotency_key="request-1",
        )
    with pytest.raises(MemoryDisabledError):
        await service.clear(**owner)
    assert await store.count_active(owner_id="alpha") == 0


async def test_pg_deactivation_and_clear_invalidate_run_epochs(
    store: PostgresMemoryStore,
) -> None:
    service = await _service(store)
    owner = {"owner_id": "alpha", "auth_mode": "jwt"}
    initial = await service.settings(**owner)
    assert initial.enabled and initial.epoch == 0
    assert await service.capability_current(owner_id="alpha", epoch=0)

    assert (await service.set_enabled(**owner, enabled=False)).epoch == 1
    assert (await service.set_enabled(**owner, enabled=True)).epoch == 1
    assert not await service.capability_current(owner_id="alpha", epoch=0)
    assert await service.capability_current(owner_id="alpha", epoch=1)

    await service.remember(
        **owner,
        kind="preference",
        body="Use Chinese.",
        provenance=_management(),
        idempotency_key="request-1",
    )
    assert (await service.settings(**owner)).active_count == 1
    # The public count reports Profile Memory records, not journal rows.
    assert await service.clear(**owner) == 1
    cleared = await service.settings(**owner)
    assert (cleared.enabled, cleared.epoch, cleared.active_count) == (True, 2, 0)
    assert not await service.capability_current(owner_id="alpha", epoch=1)


async def _page_through(service: MemoryService, *, limit: int, max_pages: int) -> list[list[str]]:
    """Every page's bodies, failing instead of looping when a cursor never ends."""
    pages: list[list[str]] = []
    request = MemoryListPageRequest(limit=limit)
    for _ in range(max_pages):
        page = await service.list_active_page(owner_id="alpha", auth_mode="jwt", page=request)
        pages.append([record.body for record in page.records])
        if page.next_cursor is None:
            return pages
        request = MemoryListPageRequest(limit=limit, cursor=page.next_cursor)
    pytest.fail(f"paging did not end within {max_pages} pages")


# Six records fill the last page exactly: it must end the listing, not promise an
# empty page after it.
@pytest.mark.parametrize(("count", "page_sizes"), [(7, [3, 3, 1]), (6, [3, 3])])
async def test_pg_service_pages_active_memories_with_continuation(
    store: PostgresMemoryStore, count: int, page_sizes: list[int]
) -> None:
    service = await _service(store)
    for index in range(count):
        await service.remember(
            owner_id="alpha",
            auth_mode="jwt",
            kind="preference",
            body=f"Memory {index}.",
            provenance=_management(),
            idempotency_key=f"key-{index}",
        )

    pages = await _page_through(service, limit=3, max_pages=len(page_sizes) + 1)

    assert [len(page) for page in pages] == page_sizes
    assert sorted(body for page in pages for body in page) == [
        f"Memory {index}." for index in range(count)
    ]


async def test_pg_supersede_forget_and_compensating_undo(store: PostgresMemoryStore) -> None:
    memory = Memory(store)
    old = await memory.remember(
        owner_id="alpha",
        kind="fact",
        body="Lives in Beijing.",
        provenance=_provenance(),
        idempotency_key="call-1",
    )
    replacement = await memory.remember(
        owner_id="alpha",
        kind="fact",
        body="Lives in Gothenburg.",
        provenance=_provenance(),
        idempotency_key="call-2",
        supersedes_id=old.memory_id,
    )
    assert [row.body for row in await _active(memory)] == ["Lives in Gothenburg."]
    undone = await memory.undo(
        owner_id="alpha",
        change_id=replacement.change_id,
        provenance=MemoryProvenance(origin_kind="undo", origin_id="undo-1"),
        idempotency_key="undo-1",
    )
    assert undone.outcome == "changed"
    assert [row.body for row in await _active(memory)] == ["Lives in Beijing."]
    current = await store.get(owner_id="alpha", memory_id=replacement.memory_id or "")
    assert current is not None and current.status == "superseded"
    # The undo restores the original as a new row that supersedes the replacement.
    back = await store.get(owner_id="alpha", memory_id=undone.memory_id or "")
    assert back is not None
    assert back.memory_id != old.memory_id
    assert (back.body, back.status, back.supersedes_id) == (
        "Lives in Beijing.",
        "active",
        replacement.memory_id,
    )

    forgotten = await memory.forget(
        owner_id="alpha",
        memory_id=undone.memory_id,
        provenance=_provenance(),
        idempotency_key="forget-1",
    )
    restored = await memory.undo(
        owner_id="alpha",
        change_id=forgotten.change_id,
        provenance=MemoryProvenance(origin_kind="undo", origin_id="undo-2"),
        idempotency_key="undo-2",
    )
    assert restored.outcome == "changed"
    # A restoration is a new active record; the forgotten one keeps its history.
    assert restored.memory_id != undone.memory_id
    assert [row.body for row in await _active(memory)] == ["Lives in Beijing."]


async def test_pg_repeated_undo_of_a_remember_conflicts(store: PostgresMemoryStore) -> None:
    memory = Memory(store)
    remembered = await memory.remember(
        owner_id="alpha",
        kind="preference",
        body="No email.",
        provenance=_provenance(),
        idempotency_key="call-1",
    )
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
    assert await _active(memory) == ()


async def test_pg_clear_physically_erases_records_and_operations(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)
    first = await memory.remember(
        owner_id="alpha",
        kind="fact",
        body="Stable.",
        provenance=_provenance(),
        idempotency_key="call-1",
    )
    assert await memory.clear(owner_id="alpha") == 1
    assert await _active(memory) == ()

    # The journal went with the records, so the same key settles anew.
    again = await memory.remember(
        owner_id="alpha",
        kind="fact",
        body="Stable.",
        provenance=_provenance(),
        idempotency_key="call-1",
    )
    assert again.changed
    assert again.change_id == first.change_id
    assert await memory.count_active(owner_id="alpha") == 1


async def test_pg_purge_expired_non_active_rows(store: PostgresMemoryStore) -> None:
    memory = Memory(store)
    old = await _remember(memory, "Stale.")
    assert old.memory_id is not None
    await memory.remember(
        owner_id="alpha",
        kind="fact",
        body="Fresh.",
        provenance=_provenance(),
        idempotency_key="fresh",
        supersedes_id=old.memory_id,
    )
    async with store._pool.acquire() as conn:  # type: ignore[union-attr]
        await conn.execute(
            "UPDATE dlightrag_memory_records "
            "SET updated_at = NOW() - INTERVAL '400 days' WHERE memory_id = $1",
            uuid.UUID(old.memory_id),
        )
    removed = await store.purge_superseded(older_than=datetime.now(UTC) - timedelta(days=365))
    assert removed == 1
    assert await store.get(owner_id="alpha", memory_id=old.memory_id) is None


async def test_pg_fact_search_finds_only_the_owner(store: PostgresMemoryStore) -> None:
    memory = Memory(store)
    await _remember(memory, "No email.", owner_id="alpha")
    await _remember(memory, "No email.", owner_id="beta")
    await _remember(memory, "Deploy at midnight.", owner_id="alpha")

    candidates = await store.search_facts(owner_id="alpha", query="email", limit=10)

    assert {candidate.record.body for candidate in candidates} == {"No email."}
    assert all(candidate.record.owner_id == "alpha" for candidate in candidates)


def _bodies(records: Sequence[MemoryRecord]) -> list[str]:
    return [record.body for record in records]


async def test_pg_recall_keeps_preferences_standing_and_facts_relevant(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)
    await _remember(memory, "Answer in Chinese.", kind="preference")
    await _remember(memory, "Works as a quantitative trader.")
    await _remember(memory, "Keeps a corgi named Doudou.")

    result = await memory.recall(
        owner_id="alpha", query="How should a quantitative fund size positions?"
    )

    assert _bodies(result.preferences) == ["Answer in Chinese."]
    assert _bodies(result.facts) == ["Works as a quantitative trader."]
    assert result.content_chars == sum(len(record.body) for record in result.records)


async def test_pg_recall_needs_a_shared_content_word_not_a_stopword(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)
    await _remember(memory, "The office is in Berlin.")

    result = await memory.recall(owner_id="alpha", query="What is the capital of Australia?")

    assert result.facts == ()


async def test_pg_recall_needs_a_chinese_content_word_not_a_space_or_a_function_word(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)
    await _remember(memory, "周末 在 上海 跑步")

    unrelated = await memory.recall(owner_id="alpha", query="量化 交易 策略")
    function_words = await memory.recall(owner_id="alpha", query="你 在 做 什么")
    related = await memory.recall(owner_id="alpha", query="上海 天气 怎么样")

    # pg_jieba indexes the spaces as terms, and 在 is a function word.
    assert unrelated.facts == ()
    assert function_words.facts == ()
    assert _bodies(related.facts) == ["周末 在 上海 跑步"]


async def test_pg_recall_finds_a_fact_restated_word_for_word(store: PostgresMemoryStore) -> None:
    memory = Memory(store)
    # Every word is a stopword, so only the exact leg can find it.
    await _remember(memory, "It is what it is.")

    result = await memory.recall(owner_id="alpha", query="it is what it is.")

    assert _bodies(result.facts) == ["It is what it is."]


@pytest.mark.parametrize(("floor", "recalled"), [(0.5, ["Runs a small coffeehouse."]), (None, [])])
async def test_pg_recall_trusts_dense_similarity_only_at_the_floor(
    floor: float | None, recalled: list[str]
) -> None:
    async with _scratch_store(_TopicEmbedder(relevance_floor=floor)) as store:
        memory = Memory(store)
        for key, body in (("cafe", "Runs a small coffeehouse."), ("train", "Commutes by train.")):
            await memory.remember(
                owner_id="alpha",
                kind="fact",
                body=body,
                provenance=_provenance(),
                idempotency_key=key,
            )

        # No word in common with either fact: only the embedding can relate them.
        result = await memory.recall(owner_id="alpha", query="Where can I buy good coffee?")

    assert _bodies(result.facts) == recalled


class _SlowTopicEmbedder(_TopicEmbedder):
    """An embedding endpoint queued behind other work when a query arrives."""

    async def embed_query(self, text: str) -> list[float]:
        await asyncio.sleep(0.2)
        return await super().embed_query(text)


async def test_pg_recall_matches_facts_by_words_when_the_query_embedding_is_slow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dlightrag_memory._storage.pg._QUERY_EMBEDDING_DEADLINE_SECONDS", 0.05)
    async with _scratch_store(_SlowTopicEmbedder()) as store:
        memory = Memory(store)
        await memory.remember(
            owner_id="alpha",
            kind="fact",
            body="Milestone 3 shipped.",
            provenance=_provenance(),
            idempotency_key="milestone",
        )

        result = await memory.recall(owner_id="alpha", query="Milestone 3 shipped.")

    assert _bodies(result.facts) == ["Milestone 3 shipped."]


async def test_pg_recall_caps_each_section_and_keeps_the_newest_preferences(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)
    for index in range(15):
        await _remember(memory, f"Preference {index}.", kind="preference")
        await _remember(memory, f"Milestone {index} shipped.")

    # Fresh statistics must not move the sparse leg off the owner's rows.
    async with store._pool.acquire() as conn:  # type: ignore[union-attr]
        await conn.execute("ANALYZE dlightrag_memory_records")

    result = await memory.recall(owner_id="alpha", query="milestone shipped")

    assert _bodies(result.preferences) == [f"Preference {index}." for index in range(5, 15)]
    assert len(result.facts) == RECALL_TOP_K


async def test_pg_recall_gives_preferences_the_character_budget_first(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)
    for index in range(9):
        await _remember(memory, f"{index} " + "p" * 448, kind="preference")
    await _remember(memory, "Milestone shipped. " + "f" * 431)

    result = await memory.recall(owner_id="alpha", query="milestone")

    # Eight 450-character preferences take 3,600 of the 4,000; neither the
    # oldest preference nor the matching fact fits in what is left.
    assert [record.body[0] for record in result.preferences] == [str(m) for m in range(1, 9)]
    assert result.facts == ()
    assert result.content_chars <= RECALL_CHAR_BUDGET


@asynccontextmanager
async def _managed_server_store(
    *, preinstalled: tuple[str, ...] = ()
) -> AsyncIterator[tuple[PostgresMemoryStore, str]]:
    """A store whose owner role is no superuser, as on a managed server."""
    await skip_without_postgres()
    role = db_name = f"dlightrag_mem_{uuid.uuid4().hex[:12]}"
    password = uuid.uuid4().hex
    admin = await asyncpg.connect(**_PG)
    try:
        await admin.execute(f"CREATE ROLE {role} LOGIN NOSUPERUSER PASSWORD '{password}'")
        await admin.execute(f'CREATE DATABASE "{db_name}" OWNER {role}')
    finally:
        await admin.close()
    try:
        if preinstalled:
            operator = await asyncpg.connect(**{**_PG, "database": db_name})
            try:
                for extension in preinstalled:
                    await operator.execute(f"CREATE EXTENSION {extension}")
            finally:
                await operator.close()
        pool = await asyncpg.create_pool(
            **{**_PG, "user": role, "password": password, "database": db_name},
            min_size=1,
            max_size=2,
        )
        try:
            store = PostgresMemoryStore(pool=pool, embedder=NullEmbedder())
            await store.initialize()
            yield store, db_name
        finally:
            await pool.close()
    finally:
        try:
            await drop_database(db_name)
        finally:
            admin = await asyncpg.connect(**_PG)
            try:
                await admin.execute(f"DROP ROLE IF EXISTS {role}")
            finally:
                await admin.close()


@pytest.mark.parametrize(
    ("preinstalled", "bm25"),
    [((), False), (("pg_textsearch",), True)],
    ids=["no-text-search", "english-fallback"],
)
async def test_pg_memory_runs_on_whatever_text_search_a_managed_server_has(
    preinstalled: tuple[str, ...], bm25: bool
) -> None:
    async with _managed_server_store(preinstalled=preinstalled) as (store, _db_name):
        async with store._pool.acquire() as conn:  # type: ignore[union-attr]
            installed = {
                row["extname"]
                for row in await conn.fetch(
                    "SELECT extname FROM pg_extension "
                    "WHERE extname IN ('pg_textsearch', 'pg_jieba')"
                )
            }
        memory = Memory(store)
        await _remember(memory, "Answer in Chinese.", kind="preference")
        await _remember(memory, "Lives in Berlin.")
        await _remember(memory, "It is what it is.")
        shared_word = await memory.recall(owner_id="alpha", query="Is Berlin rainy?")
        restated = await memory.recall(owner_id="alpha", query="it is what it is.")

    # The role could create neither extension; recall used what the server had.
    assert installed == set(preinstalled)
    assert _bodies(shared_word.preferences) == ["Answer in Chinese."]
    assert _bodies(shared_word.facts) == (["Lives in Berlin."] if bm25 else [])
    assert _bodies(restated.facts) == ["It is what it is."]


async def test_pg_a_running_process_keeps_recalling_while_the_writer_rebuilds_bm25() -> None:
    async with _managed_server_store(preinstalled=("pg_textsearch",)) as (store, db_name):
        memory = Memory(store)
        await _remember(memory, "Lives in Berlin.")
        # pg_jieba arrives later, and a writer that can create it restarts.
        operator = await asyncpg.create_pool(**{**_PG, "database": db_name}, min_size=1, max_size=2)
        try:
            await PostgresMemoryStore(pool=operator, embedder=NullEmbedder()).initialize()
            async with operator.acquire() as conn:
                indexdef = await conn.fetchval(
                    "SELECT indexdef FROM pg_indexes WHERE indexname = $1", INDEX_NAME
                )
        finally:
            await operator.close()

        result = await memory.recall(owner_id="alpha", query="Is Berlin rainy?")

    assert "jiebacfg" in indexdef
    assert _bodies(result.facts) == ["Lives in Berlin."]


async def test_pg_an_owner_holds_one_active_record_per_body(store: PostgresMemoryStore) -> None:
    await _remember(Memory(store), "Prefers tea.", kind="preference")

    with pytest.raises(asyncpg.UniqueViolationError):
        await _plant(store, _record(body="  PREFERS tea. "))


async def test_pg_a_key_reused_after_its_journal_aged_out_never_drops_the_new_body(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)

    async def move(body: str) -> MemoryOperationReceipt:
        return await memory.remember(
            owner_id="alpha",
            kind="fact",
            body=body,
            provenance=_provenance(),
            idempotency_key="move",
        )

    await move("Lives in Berlin.")
    async with store._pool.acquire() as conn:  # type: ignore[union-attr]
        await conn.execute("DELETE FROM dlightrag_memory_operations")

    with pytest.raises(ValueError, match="already exists"):
        await move("Lives in Munich.")

    assert _bodies(await _store_active(store)) == ["Lives in Berlin."]


async def test_pg_undoing_a_supersede_conflicts_once_its_body_is_active_again(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)
    first = await _remember(memory, "Lives in Berlin.")
    replacement = await memory.remember(
        owner_id="alpha",
        kind="fact",
        body="Lives in Munich.",
        provenance=_provenance(),
        idempotency_key="munich",
        supersedes_id=first.memory_id,
    )
    await memory.remember(
        owner_id="alpha",
        kind="fact",
        body="Lives in Berlin.",
        provenance=_provenance(),
        idempotency_key="berlin-again",
    )

    undone = await _undo(memory, replacement.change_id, key="undo-1")

    assert undone.outcome == "conflict"
    assert sorted(_bodies(await _store_active(store))) == ["Lives in Berlin.", "Lives in Munich."]


async def test_pg_writers_starting_together_both_initialize() -> None:
    await skip_without_postgres()
    db_name = f"dlightrag_mem_{uuid.uuid4().hex[:12]}"
    admin = await asyncpg.connect(**_PG)
    try:
        await admin.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin.close()
    pool = await asyncpg.create_pool(**{**_PG, "database": db_name}, min_size=2, max_size=4)
    try:
        # The API and MCP processes are both writers and start at once.
        writers = [PostgresMemoryStore(pool=pool, embedder=NullEmbedder()) for _ in range(2)]
        await asyncio.gather(*(writer.initialize() for writer in writers))
    finally:
        await pool.close()
        await drop_database(db_name)


async def test_pg_bm25_index_is_the_one_served_config(store: PostgresMemoryStore) -> None:
    async with store._pool.acquire() as conn:  # type: ignore[union-attr]
        for config in ("simple", "english"):
            await conn.execute(
                f"CREATE INDEX {INDEX_NAME}_{config} ON dlightrag_memory_records "
                f"USING bm25(body) WITH (text_config='{config}')"
            )

    await store.initialize()

    async with store._pool.acquire() as conn:  # type: ignore[union-attr]
        rows = await conn.fetch(
            "SELECT indexname, indexdef FROM pg_indexes "
            "WHERE tablename = 'dlightrag_memory_records' AND indexname LIKE $1",
            f"{INDEX_NAME}%",
        )
    assert [str(row["indexname"]) for row in rows] == [INDEX_NAME]
    assert "jiebacfg" in str(rows[0]["indexdef"])


async def test_pg_restating_a_fact_as_a_preference_makes_it_stand(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)
    fact = await memory.remember(
        owner_id="alpha",
        kind="fact",
        body="Answer in Chinese.",
        provenance=_provenance(),
        idempotency_key="as-fact",
    )
    preference = await memory.remember(
        owner_id="alpha",
        kind="preference",
        body="answer in chinese.",
        provenance=_provenance(),
        idempotency_key="as-preference",
    )
    unrelated = "What is the capital of Australia?"

    assert preference.outcome == "changed"
    assert preference.supersedes_id == fact.memory_id
    assert _bodies((await memory.recall(owner_id="alpha", query=unrelated)).preferences) == [
        "answer in chinese."
    ]

    undone = await _undo(memory, preference.change_id, key="undo-kind")

    assert undone.outcome == "changed"
    assert (await memory.recall(owner_id="alpha", query=unrelated)).preferences == ()


async def _undo(memory: Memory, change_id: str, *, key: str):
    return await memory.undo(
        owner_id="alpha",
        change_id=change_id,
        provenance=MemoryProvenance(origin_kind="undo", origin_id=f"undo-{key}"),
        idempotency_key=key,
    )


async def _forget_a_row(memory: Memory) -> tuple[Any, MemoryOperationReceipt]:
    """Remember tea and trains, then forget tea by a restatement of its body."""
    tea = await _remember(memory, "Prefers tea.", kind="preference")
    await _remember(memory, "Likes trains.", kind="preference")
    forgotten = await memory.forget(
        owner_id="alpha",
        body="  prefers TEA.  ",
        provenance=_provenance(),
        idempotency_key="forget-1",
    )
    assert forgotten.outcome == "changed"
    assert forgotten.memory_ids == (tea.memory_id,)
    return forgotten, tea


async def _assert_undo_settled_nothing(store: PostgresMemoryStore, forgotten: Any) -> None:
    async with store._pool.acquire() as conn:  # type: ignore[union-attr]
        assert (
            await conn.fetchval(
                "SELECT COUNT(*) FROM dlightrag_memory_records "
                "WHERE owner_id = 'alpha' AND origin_kind = 'undo'"
            )
            == 0
        )
        assert (
            await conn.fetchval(
                "SELECT undone_by IS NULL FROM dlightrag_memory_operations "
                "WHERE owner_id = 'alpha' AND change_id = $1",
                uuid.UUID(forgotten.change_id),
            )
            is True
        )


async def _status(store: PostgresMemoryStore, memory_id: str | None) -> str | None:
    assert memory_id is not None
    record = await store.get(owner_id="alpha", memory_id=memory_id)
    return None if record is None else record.status


async def test_pg_forget_undo_restores_the_row(store: PostgresMemoryStore) -> None:
    memory = Memory(store)
    forgotten, tea = await _forget_a_row(memory)

    undone = await _undo(memory, forgotten.change_id, key="undo-1")

    assert undone.outcome == "changed"
    assert undone.memory_ids == (operation_record_id("alpha", undone.change_id, index=0),)
    restored = await store.get(owner_id="alpha", memory_id=undone.memory_ids[0])
    assert restored is not None
    assert (restored.body, restored.status) == ("Prefers tea.", "active")
    assert restored.supersedes_id == tea.memory_id
    assert restored.provenance.origin_kind == "undo"
    assert restored.created_at == restored.updated_at == undone.created_at
    assert await _status(store, tea.memory_id) == "forgotten"
    assert await store.count_active(owner_id="alpha") == 2


async def test_pg_forget_undo_conflicts_when_the_row_is_gone(store: PostgresMemoryStore) -> None:
    memory = Memory(store)
    forgotten, tea = await _forget_a_row(memory)
    assert tea.memory_id is not None
    async with store._pool.acquire() as conn:  # type: ignore[union-attr]
        await conn.execute(
            "DELETE FROM dlightrag_memory_records WHERE owner_id = 'alpha' AND memory_id = $1",
            uuid.UUID(tea.memory_id),
        )

    undone = await _undo(memory, forgotten.change_id, key="undo-1")

    assert undone.outcome == "conflict"
    assert await store.count_active(owner_id="alpha") == 1
    await _assert_undo_settled_nothing(store, forgotten)


async def test_pg_forget_undo_conflicts_while_its_body_is_active_again(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)
    forgotten, tea = await _forget_a_row(memory)
    external = await _remember(memory, "PREFERS TEA.", kind="preference")
    assert external.outcome == "changed"

    undone = await _undo(memory, forgotten.change_id, key="undo-1")

    assert undone.outcome == "conflict"
    assert await store.count_active(owner_id="alpha") == 2
    assert await _status(store, tea.memory_id) == "forgotten"
    await _assert_undo_settled_nothing(store, forgotten)

    # The target stays undoable once the restatement is gone.
    await memory.forget(
        owner_id="alpha",
        memory_id=external.memory_id,
        provenance=_provenance(),
        idempotency_key="forget-2",
    )
    retry = await _undo(memory, forgotten.change_id, key="undo-2")
    assert retry.outcome == "changed"
    assert await store.count_active(owner_id="alpha") == 2


async def test_pg_forget_undo_deterministic_id_collision_rolls_back(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)
    forgotten, tea = await _forget_a_row(memory)
    undo_change_id = operation_change_id(
        MemoryOperation(
            owner_id="alpha",
            idempotency_key="undo-1",
            action="undo",
            provenance=MemoryProvenance(origin_kind="undo", origin_id="undo-1"),
            target_change_id=forgotten.change_id,
        )
    )
    squatter_id = operation_record_id("alpha", undo_change_id, index=0)
    await _plant(store, _record(body="Squatter.", memory_id=squatter_id))

    with pytest.raises(ValueError, match="already exists"):
        await _undo(memory, forgotten.change_id, key="undo-1")

    # The transaction rolled back: no restored row, no undone_by mark, no new
    # journal entry, and the squatter and the forgotten target are untouched.
    squatter = await store.get(owner_id="alpha", memory_id=squatter_id)
    assert squatter is not None and squatter.body == "Squatter."
    assert await _status(store, tea.memory_id) == "forgotten"
    await _assert_undo_settled_nothing(store, forgotten)
    async with store._pool.acquire() as conn:  # type: ignore[union-attr]
        assert (
            await conn.fetchval(
                "SELECT COUNT(*) FROM dlightrag_memory_operations WHERE owner_id = 'alpha'"
            )
            == 3
        )

    # Clearing the collision leaves the same undo idempotency key settleable.
    async with store._pool.acquire() as conn:  # type: ignore[union-attr]
        await conn.execute(
            "DELETE FROM dlightrag_memory_records WHERE owner_id = 'alpha' AND memory_id = $1",
            uuid.UUID(squatter_id),
        )
    retry = await _undo(memory, forgotten.change_id, key="undo-1")
    assert retry.outcome == "changed"
    assert retry.change_id == undo_change_id
    assert await store.count_active(owner_id="alpha") == 2


async def test_pg_supersede_undo_deterministic_id_collision_rolls_back(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)
    original = await memory.remember(
        owner_id="alpha",
        kind="preference",
        body="Drinks tea.",
        provenance=_provenance(),
        idempotency_key="remember-1",
    )
    replacement = await memory.remember(
        owner_id="alpha",
        kind="preference",
        body="Drinks coffee.",
        provenance=_provenance(),
        idempotency_key="remember-2",
        supersedes_id=original.memory_id,
    )
    undo_change_id = operation_change_id(
        MemoryOperation(
            owner_id="alpha",
            idempotency_key="undo-1",
            action="undo",
            provenance=MemoryProvenance(origin_kind="undo", origin_id="undo-1"),
            target_change_id=replacement.change_id,
        )
    )
    squatter_id = operation_record_id("alpha", undo_change_id)
    await _plant(store, _record(body="Squatter.", memory_id=squatter_id))

    with pytest.raises(ValueError, match="already exists"):
        await _undo(memory, replacement.change_id, key="undo-1")

    # The transaction rolled back, including superseding the replacement: it is
    # still the active record, and nothing marks the remember undone.
    assert replacement.memory_id is not None
    current = await store.get(owner_id="alpha", memory_id=replacement.memory_id)
    squatter = await store.get(owner_id="alpha", memory_id=squatter_id)
    assert current is not None and current.status == "active"
    assert squatter is not None and squatter.body == "Squatter."
    async with store._pool.acquire() as conn:  # type: ignore[union-attr]
        assert (
            await conn.fetchval(
                "SELECT COUNT(*) FROM dlightrag_memory_records "
                "WHERE owner_id = 'alpha' AND origin_kind = 'undo'"
            )
            == 0
        )
        assert (
            await conn.fetchval(
                "SELECT undone_by IS NULL FROM dlightrag_memory_operations "
                "WHERE owner_id = 'alpha' AND change_id = $1",
                uuid.UUID(replacement.change_id),
            )
            is True
        )
        assert (
            await conn.fetchval(
                "SELECT COUNT(*) FROM dlightrag_memory_operations WHERE owner_id = 'alpha'"
            )
            == 2
        )

    # Clearing the collision leaves the same undo idempotency key settleable.
    async with store._pool.acquire() as conn:  # type: ignore[union-attr]
        await conn.execute(
            "DELETE FROM dlightrag_memory_records WHERE owner_id = 'alpha' AND memory_id = $1",
            uuid.UUID(squatter_id),
        )
    retry = await _undo(memory, replacement.change_id, key="undo-1")
    assert retry.outcome == "changed"
    assert retry.memory_ids == (squatter_id,)
    restored = await store.get(owner_id="alpha", memory_id=squatter_id)
    assert restored is not None and restored.body == "Drinks tea."


async def _rewrite_before_records(
    store: PostgresMemoryStore, forgotten: Any, before_records: list[dict[str, Any]]
) -> None:
    async with store._pool.acquire() as conn:  # type: ignore[union-attr]
        await conn.execute(
            "UPDATE dlightrag_memory_operations SET before_records = $2::jsonb "
            "WHERE owner_id = 'alpha' AND change_id = $1",
            uuid.UUID(forgotten.change_id),
            json.dumps(before_records),
        )


async def _journal_before(store: PostgresMemoryStore, forgotten: Any) -> list[dict[str, Any]]:
    async with store._pool.acquire() as conn:  # type: ignore[union-attr]
        value = await conn.fetchval(
            "SELECT before_records FROM dlightrag_memory_operations "
            "WHERE owner_id = 'alpha' AND change_id = $1",
            uuid.UUID(forgotten.change_id),
        )
    return json.loads(value) if isinstance(value, str) else value


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda before: [],
        lambda before: [{**before[0], "owner_id": "beta"}],
        lambda before: [*before, dict(before[0])],
        lambda before: [*before, {**before[0], "memory_id": str(uuid.uuid4())}],
    ],
    ids=["empty", "foreign-owner", "duplicate-id", "extra"],
)
async def test_pg_forget_undo_with_a_malformed_journal_conflicts_cleanly(
    store: PostgresMemoryStore,
    corrupt: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
) -> None:
    memory = Memory(store)
    forgotten, tea = await _forget_a_row(memory)
    original = await _journal_before(store, forgotten)
    await _rewrite_before_records(store, forgotten, corrupt(list(original)))

    undone = await _undo(memory, forgotten.change_id, key="undo-1")

    assert undone.outcome == "conflict"
    assert await store.count_active(owner_id="alpha") == 1
    assert await store.count_active(owner_id="beta") == 0
    assert await _status(store, tea.memory_id) == "forgotten"
    await _assert_undo_settled_nothing(store, forgotten)

    # Repairing the journal leaves the same target settleable.
    await _rewrite_before_records(store, forgotten, original)
    retry = await _undo(memory, forgotten.change_id, key="undo-2")
    assert retry.outcome == "changed"
    assert await store.count_active(owner_id="alpha") == 2


@pytest.mark.parametrize(
    "run_id, session_id",
    [("", None), (None, "")],
    ids=["empty-run-id", "empty-session-id"],
)
async def test_pg_forget_undo_preserves_exact_provenance(
    store: PostgresMemoryStore, run_id: str | None, session_id: str | None
) -> None:
    memory = Memory(store)
    forgotten, _tea = await _forget_a_row(memory)
    provenance = MemoryProvenance(
        origin_kind="undo", origin_id="undo-1", run_id=run_id, session_id=session_id
    )

    undone = await memory.undo(
        owner_id="alpha",
        change_id=forgotten.change_id,
        provenance=provenance,
        idempotency_key="undo-1",
    )

    assert undone.outcome == "changed"
    restored = await store.get(owner_id="alpha", memory_id=undone.memory_ids[0])
    assert restored is not None and restored.provenance == provenance


async def test_pg_concurrent_undo_has_one_winner(store: PostgresMemoryStore) -> None:
    memory = Memory(store)
    forgotten, _tea = await _forget_a_row(memory)

    first, second = await asyncio.gather(
        _undo(memory, forgotten.change_id, key="undo-a"),
        _undo(memory, forgotten.change_id, key="undo-b"),
    )

    assert sorted((first.outcome, second.outcome)) == ["changed", "conflict"]
    assert await store.count_active(owner_id="alpha") == 2
    async with store._pool.acquire() as conn:  # type: ignore[union-attr]
        assert (
            await conn.fetchval(
                "SELECT COUNT(*) FROM dlightrag_memory_records "
                "WHERE owner_id = 'alpha' AND origin_kind = 'undo' AND status = 'active'"
            )
            == 1
        )
        assert (
            await conn.fetchval(
                "SELECT undone_by IS NOT NULL FROM dlightrag_memory_operations "
                "WHERE owner_id = 'alpha' AND change_id = $1",
                uuid.UUID(forgotten.change_id),
            )
            is True
        )

    # A later repeated undo still conflicts via the undone_by mark.
    third = await _undo(memory, forgotten.change_id, key="undo-c")
    assert third.outcome == "conflict"


async def _dense_ids(store: PostgresMemoryStore, query: str) -> list[str]:
    candidates = await store.search_facts(owner_id="alpha", query=query, limit=10)
    return [candidate.record.memory_id for candidate in candidates if candidate.leg == "dense"]


async def _dense_state(store: PostgresMemoryStore, memory_id: str | None) -> tuple[Any, Any]:
    assert memory_id is not None
    async with store._pool.acquire() as conn:  # type: ignore[union-attr]
        row = await conn.fetchrow(
            "SELECT embedding_fingerprint, embedding::text AS embedding "
            "FROM dlightrag_memory_records WHERE owner_id = 'alpha' AND memory_id = $1",
            uuid.UUID(memory_id),
        )
    assert row is not None
    return row["embedding_fingerprint"], row["embedding"]


async def test_pg_dense_undo_of_forget_restores_the_forgotten_vector(
    dense_store: tuple[PostgresMemoryStore, _TopicEmbedder],
) -> None:
    store, embedder = dense_store
    memory = Memory(store)
    remembered = await memory.remember(
        owner_id="alpha",
        kind="fact",
        body="Prefers green tea.",
        provenance=_provenance(),
        idempotency_key="remember-1",
    )
    assert await _dense_ids(store, "tea") == [remembered.memory_id]
    forgotten = await memory.forget(
        owner_id="alpha",
        memory_id=remembered.memory_id,
        provenance=_provenance(),
        idempotency_key="forget-1",
    )
    assert await _dense_ids(store, "tea") == []
    document_calls = len(embedder.document_calls)

    undone = await _undo(memory, forgotten.change_id, key="undo-1")

    assert undone.outcome == "changed"
    assert await _dense_ids(store, "tea") == [undone.memory_id]
    fingerprint, vector = await _dense_state(store, undone.memory_id)
    assert fingerprint == embedder.embedding_fingerprint
    assert vector is not None
    assert (fingerprint, vector) == await _dense_state(store, remembered.memory_id)
    # The vector is inherited inside the settlement, not re-embedded.
    assert len(embedder.document_calls) == document_calls


async def test_pg_dense_undo_of_supersede_restores_the_original_vector(
    dense_store: tuple[PostgresMemoryStore, _TopicEmbedder],
) -> None:
    store, embedder = dense_store
    memory = Memory(store)
    original = await memory.remember(
        owner_id="alpha",
        kind="fact",
        body="Drinks tea.",
        provenance=_provenance(),
        idempotency_key="remember-1",
    )
    replacement = await memory.remember(
        owner_id="alpha",
        kind="fact",
        body="Drinks coffee.",
        provenance=_provenance(),
        idempotency_key="remember-2",
        supersedes_id=original.memory_id,
    )
    assert await _dense_ids(store, "coffee") == [replacement.memory_id]
    document_calls = len(embedder.document_calls)

    undone = await _undo(memory, replacement.change_id, key="undo-1")

    assert undone.outcome == "changed"
    assert await _dense_ids(store, "tea") == [undone.memory_id]
    assert await _dense_state(store, undone.memory_id) == await _dense_state(
        store, original.memory_id
    )
    assert len(embedder.document_calls) == document_calls


@pytest.mark.parametrize(
    "unreadable",
    [
        pytest.param("embedding_fingerprint = 'test:retired@local'", id="retired-model"),
        pytest.param("embedding_fingerprint = NULL, embedding = NULL", id="never-embedded"),
        # What earlier undo settlements wrote: the bound space's label, no vector.
        pytest.param("embedding = NULL", id="labelled-without-vector"),
    ],
)
async def test_pg_dense_undo_keeps_the_restored_vector_in_its_own_space(
    dense_store: tuple[PostgresMemoryStore, _TopicEmbedder], unreadable: str
) -> None:
    store, _embedder = dense_store
    memory = Memory(store)
    remembered = await _remember(memory, "Prefers tea.")
    assert remembered.memory_id is not None
    async with store._pool.acquire() as conn:  # type: ignore[union-attr]
        await conn.execute(
            f"UPDATE dlightrag_memory_records SET {unreadable} WHERE memory_id = $1",  # noqa: S608
            uuid.UUID(remembered.memory_id),
        )
    source_fingerprint, source_vector = await _dense_state(store, remembered.memory_id)
    forgotten = await memory.forget(
        owner_id="alpha",
        memory_id=remembered.memory_id,
        provenance=_provenance(),
        idempotency_key="forget-1",
    )

    undone = await _undo(memory, forgotten.change_id, key="undo-1")

    assert undone.outcome == "changed"
    restored_id = undone.memory_ids[0]
    assert await _dense_state(store, restored_id) == (
        source_fingerprint if source_vector is not None else None,
        source_vector,
    )
    # A vector the bound space cannot read stays unreachable after the undo.
    assert await _dense_ids(store, "tea") == []


@pytest.mark.parametrize(
    "unreadable",
    [
        # What earlier undo settlements wrote: the bound space's label, no vector.
        pytest.param("embedding = NULL", id="labelled-without-vector"),
        pytest.param("embedding_fingerprint = NULL, embedding = NULL", id="never-embedded"),
        pytest.param("embedding_fingerprint = 'test:retired@local'", id="retired-model"),
    ],
)
async def test_pg_restating_a_record_heals_a_vector_the_dense_leg_cannot_read(
    dense_store: tuple[PostgresMemoryStore, _TopicEmbedder], unreadable: str
) -> None:
    store, embedder = dense_store
    memory = Memory(store)
    remembered = await memory.remember(
        owner_id="alpha",
        kind="fact",
        body="Prefers green tea.",
        provenance=_provenance(),
        idempotency_key="remember-1",
    )
    async with store._pool.acquire() as conn:  # type: ignore[union-attr]
        await conn.execute(
            f"UPDATE dlightrag_memory_records SET {unreadable} WHERE memory_id = $1",  # noqa: S608
            uuid.UUID(remembered.memory_id),
        )
    assert await _dense_ids(store, "tea") == []

    restated = await memory.remember(
        owner_id="alpha",
        kind="fact",
        body="prefers  GREEN tea.",
        provenance=_provenance("run-2"),
        idempotency_key="remember-2",
    )

    assert restated.outcome == "unchanged"
    assert restated.memory_ids == (remembered.memory_id,)
    assert await _dense_ids(store, "tea") == [remembered.memory_id]
    fingerprint, vector = await _dense_state(store, remembered.memory_id)
    assert fingerprint == embedder.embedding_fingerprint
    assert vector is not None


async def test_pg_restating_a_record_keeps_a_vector_the_dense_leg_reads(
    dense_store: tuple[PostgresMemoryStore, _TopicEmbedder],
) -> None:
    store, _embedder = dense_store
    memory = Memory(store)
    remembered = await memory.remember(
        owner_id="alpha",
        kind="fact",
        body="Prefers green tea.",
        provenance=_provenance(),
        idempotency_key="remember-1",
    )
    async with store._pool.acquire() as conn:  # type: ignore[union-attr]
        await conn.execute(
            "UPDATE dlightrag_memory_records SET embedding = '[1,0.5,0.5]' WHERE memory_id = $1",
            uuid.UUID(remembered.memory_id),
        )
    before = await _dense_state(store, remembered.memory_id)

    restated = await memory.remember(
        owner_id="alpha",
        kind="fact",
        body="Prefers green tea.",
        provenance=_provenance("run-2"),
        idempotency_key="remember-2",
    )

    assert restated.outcome == "unchanged"
    assert await _dense_state(store, remembered.memory_id) == before


async def test_pg_list_active_page_traverses_ties_and_over_hundred_rows(
    store: PostgresMemoryStore,
) -> None:
    """Full newest-first traversal: same-timestamp ties, owner isolation, bounds."""
    anchor = datetime(2026, 3, 4, 5, 6, 7, tzinfo=UTC)
    records: list[MemoryRecord] = []
    for index in range(60):
        records.append(
            MemoryRecord(
                owner_id="alpha",
                memory_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"tie-a-{index}")),
                kind="preference",
                body=f"Tie A {index}.",
                provenance=_provenance(),
                created_at=anchor,
                updated_at=anchor,
            )
        )
    for index in range(50):
        records.append(
            MemoryRecord(
                owner_id="alpha",
                memory_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"tie-b-{index}")),
                kind="preference",
                body=f"Tie B {index}.",
                provenance=_provenance(),
                created_at=anchor - timedelta(hours=1),
                updated_at=anchor - timedelta(hours=1),
            )
        )
    for record in records:
        await _plant(store, record)
    await _plant(store, _record(owner="beta", body="Foreign."))

    def _key(record: MemoryRecord) -> tuple[datetime, str]:
        assert record.updated_at is not None
        return (record.updated_at, record.memory_id)

    expected = [_key(record) for record in sorted(records, key=_key, reverse=True)]
    observed: list[tuple[datetime, str]] = []
    after: tuple[datetime, str] | None = None
    while True:
        page, next_after = await store.list_active_page(owner_id="alpha", after=after, limit=40)
        assert len(page) <= 40
        observed.extend(_key(record) for record in page)
        if next_after is None:
            break
        assert page
        after = next_after
    assert observed == expected
    assert len(observed) == 110
    assert len(set(observed)) == 110

    # The exact paged-read index exists and matches the mixed-direction order.
    async with store._pool.acquire() as conn:  # type: ignore[union-attr]
        indexdef = await conn.fetchval(
            "SELECT indexdef FROM pg_indexes WHERE indexname = 'idx_dlightrag_memory_records_list'"
        )
        assert indexdef is not None
        normalized = " ".join(str(indexdef).split()).lower()
        assert "(owner_id, status, updated_at desc, memory_id desc)" in normalized


# ---------------------------------------------------------------------------
# The MCP host and the Research tools over the real store
# ---------------------------------------------------------------------------


async def test_pg_mcp_recall_returns_the_bound_subject_records(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)
    for owner_id, body in (("pi-user", "No email."), ("other-user", "No email at all.")):
        await memory.remember(
            owner_id=owner_id,
            kind="preference",
            body=body,
            provenance=MemoryProvenance(origin_kind="mcp", origin_id="seed"),
            idempotency_key="seed",
        )

    result = await mcp_recall(memory, subject="pi-user", query="email")

    assert [record["body"] for record in result["preferences"]] == ["No email."]
    assert result["preferences"][0]["memory_id"]
    assert result["facts"] == []
    assert result["recent"] == []


async def test_pg_mcp_remember_writes_mcp_provenance_and_replays(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)
    write: dict[str, Any] = {
        "subject": "pi-user",
        "kind": "fact",
        "body": "Project uses ruff.",
        "supersedes_id": None,
        "idempotency_key": "write-1",
    }
    stored = await mcp_remember(memory, **write)
    replay = await mcp_remember(memory, **write)

    assert stored["outcome"] == "changed"
    assert replay == stored
    (record,) = await _active(memory, owner_id="pi-user")
    assert record.provenance.origin_kind == "mcp"


async def test_pg_mcp_forget_and_undo_return_operation_receipts(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)
    stored = await mcp_remember(
        memory,
        subject="pi-user",
        kind="fact",
        body="Keep me.",
        supersedes_id=None,
        idempotency_key="write-1",
    )
    forgotten = await mcp_forget(
        memory,
        subject="pi-user",
        memory_id=stored["memory_ids"][0],
        body=None,
        idempotency_key="forget-1",
    )
    assert forgotten["outcome"] == "changed"

    undone = await mcp_undo(
        memory, subject="pi-user", change_id=forgotten["change_id"], idempotency_key="undo-1"
    )
    assert undone["outcome"] == "changed"
    assert [row.body for row in await _active(memory, owner_id="pi-user")] == ["Keep me."]


def _host(store: PostgresMemoryStore) -> MemoryHost:
    return MemoryHost(
        owner_id="o",
        auth_mode="jwt",
        run_id="11111111-1111-1111-1111-111111111111",
        session_id="22222222-2222-2222-2222-222222222222",
        memory=Memory(store),
    )


async def test_pg_remember_then_forget_tools_return_typed_receipts(
    store: PostgresMemoryStore,
) -> None:
    host = _host(store)
    remembered = await remember_tool(host=host).execute(
        RememberInput(kind="preference", body="No email."),
        tool_runtime(call_id="call-1"),
    )
    operation = (remembered.details or {})["memory_operation"]
    assert operation["outcome"] == "changed"
    memory_id = str(operation["memory_ids"][0])

    forgotten = await forget_tool(host=host).execute(
        ForgetInput(memory_id=memory_id), tool_runtime(call_id="call-2")
    )

    assert (forgotten.details or {})["memory_operation"]["outcome"] == "changed"
    assert await store.count_active(owner_id="o") == 0


async def test_pg_forget_tool_miss_is_unchanged(store: PostgresMemoryStore) -> None:
    result = await forget_tool(host=_host(store)).execute(
        ForgetInput(memory_id="33333333-3333-3333-3333-333333333333"),
        tool_runtime(),
    )

    assert (result.details or {})["memory_operation"]["outcome"] == "unchanged"


async def test_pg_recall_tool_lists_the_ids_a_correction_needs(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)
    provenance = MemoryProvenance(origin_kind="answer_run", origin_id="seed")
    berlin = await memory.remember(
        owner_id="o",
        kind="fact",
        body="Lives in Berlin.",
        provenance=provenance,
        idempotency_key="berlin",
    )
    # Ten newer preferences fill the standing section and the newest records.
    preferences = [
        await memory.remember(
            owner_id="o",
            kind="preference",
            body=f"Style rule {index}.",
            provenance=provenance,
            idempotency_key=f"style-{index}",
        )
        for index in range(RECALL_TOP_K)
    ]
    updates: list[ToolResult] = []

    # Nothing in the correction shares a word with the fact it replaces.
    result = await recall_memory_tool(host=_host(store)).execute(
        RecallInput(query="I moved to Munich"),
        recording_tool_runtime(updates, tool_name="recall_memory"),
    )

    assert result.text_content.splitlines() == [
        "Standing preferences:",
        *(f"- {p.memory_id} Style rule {index}." for index, p in enumerate(preferences)),
        "Other recent memories:",
        f"- {berlin.memory_id} (fact) Lives in Berlin.",
    ]
    assert [update.subject for update in updates] == ["I moved to Munich"]
