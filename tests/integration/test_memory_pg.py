# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL Profile Memory records, journal, atomic receipts, undo, and recall."""

import asyncio
import json
import re
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
from dlightrag_memory._storage.pg_bm25 import index_name
from dlightrag_memory.errors import MemoryWriteRejectedError
from dlightrag_memory.mcp_server import _forget as mcp_forget
from dlightrag_memory.mcp_server import _lookup as mcp_lookup
from dlightrag_memory.mcp_server import _remember as mcp_remember
from dlightrag_memory.mcp_server import _undo as mcp_undo
from dlightrag_memory.memory import recall_cost
from dlightrag_memory.normalize import normalized_body
from dlightrag_memory.policy import (
    MEMORY_QUOTE_LIMIT,
    RECALL_CHAR_BUDGET,
    RECALL_TOP_K,
    STANDING_FETCH_LIMIT,
)
from dlightrag_memory.ports import BM25Languages, ChineseAndEnglish, NullEmbedder, TextEmbedder
from dlightrag_memory.postgres import PostgresMemoryStore
from dlightrag_memory.store import operation_change_id, operation_record_id

from dlightrag.adapters.postgres.answer.memory_settings import (
    MEMORY_SETTINGS_DDL,
    PGMemorySettingsStore,
)
from dlightrag.application.memory.errors import MemoryDisabledError
from dlightrag.application.memory.memory_list import MemoryListPageRequest
from dlightrag.application.memory.service import MemoryService
from dlightrag.engine.agent.tools.contracts import ToolResult
from dlightrag.engine.answer.memory import render_auto_recall
from dlightrag.engine.answer.tools.memory import (
    ForgetInput,
    LookupInput,
    MemoryHost,
    RememberInput,
    UndoInput,
    forget_tool,
    lookup_tool,
    remember_tool,
    undo_tool,
)
from tests.support.pg import PG_CONN_KWARGS, drop_database, skip_without_postgres
from tests.support.tool_helpers import recording_tool_runtime, tool_runtime

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_PG: dict[str, Any] = PG_CONN_KWARGS


@asynccontextmanager
async def _scratch_store(
    embedder: TextEmbedder = NullEmbedder(),
    languages: BM25Languages = ChineseAndEnglish(),
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
        created = PostgresMemoryStore(pool=pool, embedder=embedder, languages=languages)
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


class _NoSessions:
    """This suite holds no Agent Sessions, so turning memory back on has none to restart."""

    async def restart_session_memory(self, *, owner_id: str) -> None:
        return None


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
        sessions=_NoSessions(),
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
    owner = {"owner_id": "alpha"}
    disabled = await service.set_enabled(**owner, enabled=False)

    assert disabled.enabled is False
    assert disabled.active_count is None
    assert disabled.epoch == 1
    assert (await service.settings(**owner)).active_count is None
    with pytest.raises(MemoryDisabledError):
        await service.list_active_page(owner_id="alpha")
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
    owner = {"owner_id": "alpha"}
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
        page = await service.list_active_page(owner_id="alpha", page=request)
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

    candidates = await store.search(owner_id="alpha", query="email", limit=10)

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

    assert _bodies(result.standing) == ["Answer in Chinese."]
    assert _bodies(result.relevant) == ["Works as a quantitative trader."]
    assert result.content_chars == sum(len(record.body) for record in result.records)


async def test_pg_recall_needs_a_shared_content_word_not_a_stopword(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)
    await _remember(memory, "The office is in Berlin.")

    result = await memory.recall(owner_id="alpha", query="What is the capital of Australia?")

    assert result.relevant == ()


async def test_pg_recall_needs_a_chinese_content_word_not_a_space_or_a_function_word(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)
    await _remember(memory, "周末 在 上海 跑步")

    unrelated = await memory.recall(owner_id="alpha", query="量化 交易 策略")
    function_words = await memory.recall(owner_id="alpha", query="你 在 做 什么")
    related = await memory.recall(owner_id="alpha", query="上海 天气 怎么样")

    # pg_jieba indexes the spaces as terms, and 在 is a function word.
    assert unrelated.relevant == ()
    assert function_words.relevant == ()
    assert _bodies(related.relevant) == ["周末 在 上海 跑步"]


async def test_pg_recall_finds_a_fact_restated_word_for_word(store: PostgresMemoryStore) -> None:
    memory = Memory(store)
    # Every word is a stopword, so only the exact leg can find it.
    await _remember(memory, "It is what it is.")

    result = await memory.recall(owner_id="alpha", query="it is what it is.")

    assert _bodies(result.relevant) == ["It is what it is."]


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

    assert _bodies(result.relevant) == recalled


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

    assert _bodies(result.relevant) == ["Milestone 3 shipped."]


async def test_pg_recall_caps_relevance_at_top_k_and_stands_what_the_budget_holds(
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

    # Short records: the count bounds relevance, and every preference stands.
    assert len(result.relevant) == RECALL_TOP_K
    assert all(record.kind == "fact" for record in result.relevant)
    assert _bodies(result.standing) == [f"Preference {index}." for index in range(15)]
    assert result.standing_omitted == 0


async def test_pg_recall_fills_relevance_first_and_standing_takes_the_rest(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)
    for index in range(18):
        await _remember(memory, f"{index:02d} " + "p" * 447, kind="preference")
    await _remember(memory, "Milestone shipped. " + "f" * 431)

    result = await memory.recall(owner_id="alpha", query="milestone")

    # The fact is relevant and is placed first; each line costs its 450 characters plus
    # RECALL_LINE_OVERHEAD, so fifteen preferences fill what the 8,000 budget has left
    # and three stand unshown.
    assert _bodies(result.relevant) == ["Milestone shipped. " + "f" * 431]
    assert [record.body[:2] for record in result.standing] == [f"{m:02d}" for m in range(3, 18)]
    assert result.standing_omitted == 3
    assert recall_cost(result.records) <= RECALL_CHAR_BUDGET


async def test_pg_recall_finds_a_crowded_out_preference_by_relevance(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)
    await _remember(memory, "Prefers a dark theme in every interface.", kind="preference")
    for index in range(STANDING_FETCH_LIMIT):
        await _remember(memory, f"Style rule {index}.", kind="preference")

    unrelated = await memory.recall(owner_id="alpha", query="What is the capital of Australia?")
    related = await memory.recall(owner_id="alpha", query="Which theme should the interface use?")

    # Newer preferences crowd the oldest out of the standing section ...
    assert "Prefers a dark theme in every interface." not in _bodies(unrelated.standing)
    assert unrelated.standing_omitted == 1
    # ... and a request that bears on it brings it back, shown once.
    assert _bodies(related.relevant) == ["Prefers a dark theme in every interface."]
    assert "Prefers a dark theme in every interface." not in _bodies(related.standing)
    assert related.standing_omitted == 0


async def test_pg_recall_stands_experience_beside_preference(store: PostgresMemoryStore) -> None:
    memory = Memory(store)
    await _remember(memory, "Answer in Chinese.", kind="preference")
    await _remember(
        memory, "The corpus needs pg_jieba before Chinese search works.", kind="experience"
    )
    await _remember(memory, "Lives in Gothenburg.")

    result = await memory.recall(owner_id="alpha", query="What is the capital of Australia?")

    assert {record.kind for record in result.standing} == {"preference", "experience"}
    assert result.relevant == ()


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
    ids=["no-text-search", "text-search-without-jieba"],
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
    assert _bodies(shared_word.standing) == ["Answer in Chinese."]
    assert _bodies(shared_word.relevant) == (["Lives in Berlin."] if bm25 else [])
    assert _bodies(restated.relevant) == ["It is what it is."]


async def test_pg_a_running_process_keeps_recalling_while_the_writer_rebuilds_bm25() -> None:
    async with _managed_server_store(preinstalled=("pg_textsearch",)) as (store, db_name):
        memory = Memory(store)
        await _remember(memory, "我住在柏林")
        # Without pg_jieba the Chinese index runs under simple, which cannot
        # split 我住在柏林; pg_jieba arrives, and a writer that can create it
        # restarts and rebuilds that index in place.
        before = await memory.recall(owner_id="alpha", query="柏林 天气 怎么样")
        operator = await asyncpg.create_pool(**{**_PG, "database": db_name}, min_size=1, max_size=2)
        try:
            await PostgresMemoryStore(pool=operator, embedder=NullEmbedder()).initialize()
            async with operator.acquire() as conn:
                indexdef = await conn.fetchval(
                    "SELECT indexdef FROM pg_indexes WHERE indexname = $1", index_name("zh")
                )
        finally:
            await operator.close()

        after = await memory.recall(owner_id="alpha", query="柏林 天气 怎么样")

    assert before.relevant == ()
    assert "jiebacfg" in indexdef
    assert _bodies(after.relevant) == ["我住在柏林"]


async def test_pg_facts_are_indexed_like_corpus_chunks_in_their_own_language() -> None:
    from dlightrag.engine.rag.retrieval.language import ProfileBM25Languages
    from dlightrag.engine.rag.workspace.settings import RetrievalSettings

    languages = ProfileBM25Languages(RetrievalSettings().bm25_profiles)
    async with _scratch_store(languages=languages) as store:
        memory = Memory(store)
        for body in (
            "Der Nutzer wohnt in Berlin.",
            "Has a daughter who is six.",
            "Codes in Python.",
        ):
            await _remember(memory, body)

        async def facts(query: str) -> list[str]:
            return _bodies((await memory.recall(owner_id="alpha", query=query)).relevant)

        # German function words never count; German content words do.
        assert await facts("Was ist der Unterschied zwischen Aktien und Anleihen?") == []
        assert await facts("Wo wohnt der Nutzer heute?") == ["Der Nutzer wohnt in Berlin."]
        # English facts are stemmed like English chunks.
        assert await facts("Gift ideas for my daughters?") == ["Has a daughter who is six."]
        # Every language the owner writes in is searched, so a Chinese question
        # naming Python meets the English fact.
        assert await facts("Python 的 GIL 怎么绕过？") == ["Codes in Python."]


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


async def test_pg_bm25_indexes_follow_the_served_languages(store: PostgresMemoryStore) -> None:
    async with store._pool.acquire() as conn:  # type: ignore[union-attr]
        # Names earlier releases served: one per configuration, then one in all.
        for name, config in (
            ("idx_dlightrag_memory_records_bm25", "public.jiebacfg"),
            ("idx_dlightrag_memory_records_bm25_jieba", "public.jiebacfg"),
        ):
            await conn.execute(
                f"CREATE INDEX {name} ON dlightrag_memory_records "
                f"USING bm25(body) WITH (text_config='{config}')"
            )

    await store.initialize()

    async with store._pool.acquire() as conn:  # type: ignore[union-attr]
        rows = await conn.fetch(
            "SELECT indexname, indexdef FROM pg_indexes "
            "WHERE tablename = 'dlightrag_memory_records' AND indexname LIKE $1",
            "idx_dlightrag_memory_records_bm25%",
        )
    indexdefs = {str(row["indexname"]): str(row["indexdef"]) for row in rows}
    assert set(indexdefs) == {index_name(language) for language in ("zh", "en", "simple")}
    assert "jiebacfg" in indexdefs[index_name("zh")]
    assert "english" in indexdefs[index_name("en")]


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
    assert _bodies((await memory.recall(owner_id="alpha", query=unrelated)).standing) == [
        "answer in chinese."
    ]

    undone = await _undo(memory, preference.change_id, key="undo-kind")

    assert undone.outcome == "changed"
    assert (await memory.recall(owner_id="alpha", query=unrelated)).standing == ()


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
    candidates = await store.search(owner_id="alpha", query=query, limit=10)
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


async def test_pg_mcp_lookup_returns_the_bound_subject_records(
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

    result = await mcp_lookup(memory, subject="pi-user", query="email")

    # The query bears on the preference, so it is shown once, as relevant.
    assert [record["body"] for record in result["relevant"]] == ["No email."]
    assert result["relevant"][0]["memory_id"]
    assert result["relevant"][0]["kind"] == "preference"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:00", result["relevant"][0]["written_at_utc"])
    assert result["standing"] == []
    assert result["standing_omitted"] == 0
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


async def test_pg_forget_tool_names_an_id_no_memory_has(store: PostgresMemoryStore) -> None:
    unknown = "33333333-3333-3333-3333-333333333333"

    result = await forget_tool(host=_host(store)).execute(
        ForgetInput(memory_id=unknown), tool_runtime()
    )

    assert result.is_error
    assert result.text_content == (
        f"No Profile Memory has id {unknown}. Call memory_lookup for current ids."
    )
    assert (result.details or {})["memory_operation"]["outcome"] == "unchanged"


async def test_pg_forget_tool_refuses_a_body_no_active_memory_has(
    store: PostgresMemoryStore,
) -> None:
    host = _host(store)
    await remember_tool(host=host).execute(
        RememberInput(kind="fact", body="Likes tea."), tool_runtime(call_id="call-1")
    )

    miss = await forget_tool(host=host).execute(
        ForgetInput(body="Likes coffee."), tool_runtime(call_id="call-2")
    )
    hit = await forget_tool(host=host).execute(
        ForgetInput(body="Likes tea."), tool_runtime(call_id="call-3")
    )
    again = await forget_tool(host=host).execute(
        ForgetInput(body="Likes tea."), tool_runtime(call_id="call-4")
    )

    assert miss.is_error
    assert miss.text_content.startswith("No active Profile Memory has exactly that body.")
    assert not hit.is_error
    # Gone is gone: a second forget of the same body finds no active memory to name.
    assert again.is_error


async def test_pg_forgetting_a_memory_twice_says_it_was_already_forgotten(
    store: PostgresMemoryStore,
) -> None:
    host = _host(store)
    remembered = await remember_tool(host=host).execute(
        RememberInput(kind="fact", body="Likes tea."), tool_runtime(call_id="call-1")
    )
    memory_id = str((remembered.details or {})["memory_operation"]["memory_ids"][0])
    await forget_tool(host=host).execute(
        ForgetInput(memory_id=memory_id), tool_runtime(call_id="call-2")
    )

    again = await forget_tool(host=host).execute(
        ForgetInput(memory_id=memory_id), tool_runtime(call_id="call-3")
    )

    assert not again.is_error
    assert again.text_content == f"Profile Memory {memory_id} was already forgotten."


async def test_pg_forgetting_a_replaced_memory_says_so_and_points_to_lookup(
    store: PostgresMemoryStore,
) -> None:
    host = _host(store)
    first = await remember_tool(host=host).execute(
        RememberInput(kind="fact", body="Lives in Berlin."), tool_runtime(call_id="call-1")
    )
    old_id = str((first.details or {})["memory_operation"]["memory_ids"][0])
    await remember_tool(host=host).execute(
        RememberInput(kind="fact", body="Lives in Munich.", supersedes_id=old_id),
        tool_runtime(call_id="call-2"),
    )

    result = await forget_tool(host=host).execute(
        ForgetInput(memory_id=old_id), tool_runtime(call_id="call-3")
    )

    assert not result.is_error
    assert "was already replaced by a newer memory" in result.text_content
    assert await store.count_active(owner_id="o") == 1


async def test_pg_replaying_a_forget_returns_its_original_receipt(
    store: PostgresMemoryStore,
) -> None:
    host = _host(store)
    remembered = await remember_tool(host=host).execute(
        RememberInput(kind="fact", body="Likes tea."), tool_runtime(call_id="call-1")
    )
    memory_id = str((remembered.details or {})["memory_operation"]["memory_ids"][0])

    first = await forget_tool(host=host).execute(
        ForgetInput(memory_id=memory_id), tool_runtime(call_id="call-2")
    )
    # The same call again, as a recovered Run replays it: the record is forgotten by now, yet
    # the answer is the original Forgot receipt, not a miss.
    replay = await forget_tool(host=host).execute(
        ForgetInput(memory_id=memory_id), tool_runtime(call_id="call-2")
    )

    assert first.text_content == f"Forgot Profile Memory {memory_id}."
    assert replay.text_content == first.text_content
    assert not replay.is_error
    assert (replay.details or {})["memory_operation"] == (first.details or {})["memory_operation"]


async def test_pg_undo_tool_names_an_id_no_memory_has(store: PostgresMemoryStore) -> None:
    unknown = "33333333-3333-3333-3333-333333333333"

    result = await undo_tool(host=_host(store)).execute(
        UndoInput(memory_id=unknown), tool_runtime()
    )

    assert result.is_error
    assert result.text_content == (
        f"No Profile Memory has id {unknown}. Call memory_lookup for current ids."
    )
    assert (result.details or {})["memory_operation"]["outcome"] == "conflict"


async def test_pg_undo_tool_refuses_a_body_no_active_memory_has(
    store: PostgresMemoryStore,
) -> None:
    host = _host(store)
    await remember_tool(host=host).execute(
        RememberInput(kind="fact", body="Likes tea."), tool_runtime(call_id="call-1")
    )

    miss = await undo_tool(host=host).execute(
        UndoInput(body="Likes coffee."), tool_runtime(call_id="call-2")
    )
    hit = await undo_tool(host=host).execute(
        UndoInput(body="Likes tea."), tool_runtime(call_id="call-3")
    )
    again = await undo_tool(host=host).execute(
        UndoInput(body="Likes tea."), tool_runtime(call_id="call-4")
    )

    assert miss.is_error
    assert miss.text_content == (
        "No active Profile Memory has exactly that body. Call memory_lookup to find its id."
    )
    assert not hit.is_error
    # Gone is gone: the undo forgot it, so a second undo of the same body finds no active memory.
    assert again.is_error
    assert again.text_content == miss.text_content


async def test_pg_undoing_a_forgotten_memory_says_it_is_not_active(
    store: PostgresMemoryStore,
) -> None:
    host = _host(store)
    remembered = await remember_tool(host=host).execute(
        RememberInput(kind="fact", body="Likes tea."), tool_runtime(call_id="call-1")
    )
    memory_id = str((remembered.details or {})["memory_operation"]["memory_ids"][0])
    await forget_tool(host=host).execute(
        ForgetInput(memory_id=memory_id), tool_runtime(call_id="call-2")
    )

    result = await undo_tool(host=host).execute(
        UndoInput(memory_id=memory_id), tool_runtime(call_id="call-3")
    )

    assert result.is_error
    assert f"Profile Memory {memory_id} was forgotten" in result.text_content
    assert "memory_undo names an active memory" in result.text_content
    assert "memory_lookup" in result.text_content
    assert await store.count_active(owner_id="o") == 0


async def test_pg_undoing_a_replaced_memory_says_so_and_points_to_lookup(
    store: PostgresMemoryStore,
) -> None:
    host = _host(store)
    first = await remember_tool(host=host).execute(
        RememberInput(kind="fact", body="Lives in Berlin."), tool_runtime(call_id="call-1")
    )
    old_id = str((first.details or {})["memory_operation"]["memory_ids"][0])
    await remember_tool(host=host).execute(
        RememberInput(kind="fact", body="Lives in Munich.", supersedes_id=old_id),
        tool_runtime(call_id="call-2"),
    )

    result = await undo_tool(host=host).execute(
        UndoInput(memory_id=old_id), tool_runtime(call_id="call-3")
    )

    assert result.is_error
    assert f"Profile Memory {old_id} was replaced by a newer memory" in result.text_content
    assert "memory_lookup" in result.text_content
    assert await store.count_active(owner_id="o") == 1


async def test_pg_replaying_an_undo_returns_its_original_receipt(
    store: PostgresMemoryStore,
) -> None:
    host = _host(store)
    remembered = await remember_tool(host=host).execute(
        RememberInput(kind="fact", body="Likes tea."), tool_runtime(call_id="call-1")
    )
    memory_id = str((remembered.details or {})["memory_operation"]["memory_ids"][0])

    first = await undo_tool(host=host).execute(
        UndoInput(memory_id=memory_id), tool_runtime(call_id="call-2")
    )
    # The same call again, as a recovered Run replays it: the record is forgotten by now, yet the
    # answer is the original Undid receipt, not a memory that is no longer active.
    replay = await undo_tool(host=host).execute(
        UndoInput(memory_id=memory_id), tool_runtime(call_id="call-2")
    )

    assert first.text_content == f"Undid Profile Memory {memory_id}."
    assert replay.text_content == first.text_content
    assert not replay.is_error
    assert (replay.details or {})["memory_operation"] == (first.details or {})["memory_operation"]


async def test_pg_replaying_an_undo_of_an_unknown_id_is_the_same_error(
    store: PostgresMemoryStore,
) -> None:
    host = _host(store)
    unknown = "33333333-3333-3333-3333-333333333333"

    first = await undo_tool(host=host).execute(
        UndoInput(memory_id=unknown), tool_runtime(call_id="call-1")
    )
    replay = await undo_tool(host=host).execute(
        UndoInput(memory_id=unknown), tool_runtime(call_id="call-1")
    )

    assert first.is_error
    assert replay.is_error
    assert replay.text_content == first.text_content
    assert (replay.details or {})["memory_operation"] == (first.details or {})["memory_operation"]


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
    result = await lookup_tool(host=_host(store)).execute(
        LookupInput(query="I moved to Munich"),
        recording_tool_runtime(updates, tool_name="memory_lookup"),
    )

    # Every line carries when it was last written, to the hour in UTC.
    when = r"\d{4}-\d{2}-\d{2} \d{2}:00"
    expected = [
        re.escape("Remembered about this owner (times are UTC, when each was last written):"),
        re.escape("Standing memories:"),
        *(
            re.escape(f"- {p.memory_id} (preference, ") + when + re.escape(f") Style rule {index}.")
            for index, p in enumerate(preferences)
        ),
        re.escape("Other recent memories:"),
        re.escape(f"- {berlin.memory_id} (fact, ") + when + re.escape(") Lives in Berlin."),
    ]
    lines = result.text_content.splitlines()
    assert len(lines) == len(expected)
    for line, pattern in zip(lines, expected, strict=True):
        assert re.fullmatch(pattern, line), line
    assert [update.subject for update in updates] == ["I moved to Munich"]


# Fuse, undo addressed by record, and the closed sets an older database widens
# ---------------------------------------------------------------------------


async def _fuse(
    memory: Memory,
    sources: Sequence[MemoryOperationReceipt],
    body: str,
    *,
    key: str = "fuse-1",
    kind: MemoryKind | None = None,
) -> MemoryOperationReceipt:
    return await memory.fuse(
        owner_id="alpha",
        memory_ids=[str(source.memory_id) for source in sources],
        body=body,
        kind=kind,
        provenance=_provenance(),
        idempotency_key=key,
    )


async def test_pg_fuse_replaces_its_sources_with_one_record_and_replays(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)
    chinese = await _remember(memory, "Answer in Chinese.", kind="preference")
    tables = await _remember(memory, "Put tables before prose.", kind="preference")

    fused = await _fuse(memory, (chinese, tables), "Answer in Chinese, tables before prose.")
    replay = await _fuse(memory, (chinese, tables), "Answer in Chinese, tables before prose.")

    assert fused.outcome == "changed"
    assert fused.action == "fuse"
    assert fused.kind == "preference"
    assert fused.supersedes_id is None
    assert fused.memory_ids[1:] == (chinese.memory_id, tables.memory_id)
    assert replay == fused
    assert _bodies(await _active(memory)) == ["Answer in Chinese, tables before prose."]


async def test_pg_fuse_of_mixed_kinds_must_say_the_kind(store: PostgresMemoryStore) -> None:
    memory = Memory(store)
    preference = await _remember(memory, "Answer in Chinese.", kind="preference")
    fact = await _remember(memory, "Lives in Shanghai.")

    with pytest.raises(MemoryWriteRejectedError, match="different kinds"):
        await _fuse(memory, (preference, fact), "Lives in Shanghai; answer in Chinese.")
    named = await _fuse(
        memory, (preference, fact), "Lives in Shanghai; answer in Chinese.", key="k", kind="fact"
    )

    assert named.outcome == "changed"
    assert named.kind == "fact"


async def test_pg_fuse_is_a_conflict_when_a_source_moved_or_the_body_is_taken(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)
    first = await _remember(memory, "Codes in Python.")
    second = await _remember(memory, "Writes tests first.")
    other = await _remember(memory, "Uses ruff.")
    await memory.forget(
        owner_id="alpha", memory_id=second.memory_id, provenance=_provenance(), idempotency_key="f"
    )

    moved = await _fuse(memory, (first, second), "Codes Python, test first.")
    taken = await _fuse(memory, (first, other), "Writes tests first.", key="taken")

    assert moved.outcome == "conflict"
    assert taken.outcome == "changed"  # the forgotten record no longer holds that body
    clash = await _remember(memory, "Prefers tabs.")
    keeper = await _remember(memory, "Prefers spaces.")
    collision = await _fuse(memory, (clash, taken), "Prefers spaces.", key="collide")
    assert collision.outcome == "conflict"
    assert keeper.memory_id in {record.memory_id for record in await _active(memory)}


async def test_pg_undo_of_a_fuse_brings_every_source_back(store: PostgresMemoryStore) -> None:
    memory = Memory(store)
    chinese = await _remember(memory, "Answer in Chinese.", kind="preference")
    tables = await _remember(memory, "Put tables before prose.", kind="preference")
    fused = await _fuse(memory, (chinese, tables), "Answer in Chinese, tables before prose.")

    undone = await memory.undo(
        owner_id="alpha",
        memory_id=fused.memory_id,
        provenance=_provenance(),
        idempotency_key="undo-fuse",
    )

    assert undone.outcome == "changed"
    assert undone.target_change_id == fused.change_id
    assert sorted(_bodies(await _active(memory))) == [
        "Answer in Chinese.",
        "Put tables before prose.",
    ]


async def test_pg_undo_addressed_by_record_compensates_its_last_change(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)
    berlin = await _remember(memory, "Lives in Berlin.")
    munich = await memory.remember(
        owner_id="alpha",
        kind="fact",
        body="Lives in Munich.",
        provenance=_provenance(),
        idempotency_key="moved",
        supersedes_id=berlin.memory_id,
    )

    by_body = await memory.undo(
        owner_id="alpha",
        body="lives in munich.",
        provenance=_provenance(),
        idempotency_key="undo-body",
    )
    restored = await _active(memory)
    again = await memory.undo(
        owner_id="alpha",
        memory_id=restored[0].memory_id,
        provenance=_provenance(),
        idempotency_key="undo-undo",
    )
    gone = await memory.undo(
        owner_id="alpha",
        memory_id=munich.memory_id,
        provenance=_provenance(),
        idempotency_key="undo-gone",
    )

    assert by_body.outcome == "changed"
    assert by_body.target_change_id == munich.change_id
    assert _bodies(restored) == ["Lives in Berlin."]
    # The restored record came from an undo, and an undo cannot itself be undone.
    assert again.outcome == "conflict"
    # A record no longer active is a conflict, not a guess at an earlier change.
    assert gone.outcome == "conflict"


async def test_pg_session_records_are_what_one_origin_wrote_for_one_session(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)
    passed = MemoryProvenance(
        origin_kind="session_pass", origin_id="pass-1", session_id="session-9"
    )
    await memory.remember(
        owner_id="alpha",
        kind="experience",
        body="The corpus needs pg_jieba before Chinese search works.",
        provenance=passed,
        idempotency_key="pass-1:0",
    )
    await _remember(memory, "Lives in Gothenburg.")

    records = await memory.session_records(
        owner_id="alpha", session_id="session-9", origin_kind="session_pass"
    )

    assert _bodies(records) == ["The corpus needs pg_jieba before Chinese search works."]
    assert records[0].provenance.origin_kind == "session_pass"


async def test_pg_fused_sources_name_the_fuse_that_retired_them(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)
    first = await _remember(memory, "Answer in Chinese.", kind="preference")
    second = await _remember(memory, "Tables before prose.", kind="preference")
    fused = await _fuse(memory, (first, second), "Chinese, tables before prose.")

    async with store._pool.acquire() as conn:  # type: ignore[union-attr]
        rows = await conn.fetch(
            "SELECT memory_id::text AS id, last_change_id::text AS change "
            "FROM dlightrag_memory_records WHERE owner_id = 'alpha'"
        )

    assert {row["id"]: row["change"] for row in rows} == {
        str(fused.memory_id): fused.change_id,
        str(first.memory_id): fused.change_id,
        str(second.memory_id): fused.change_id,
    }


# The owner's own words behind a record
# ---------------------------------------------------------------------------


async def _say(
    memory: Memory,
    body: str,
    quote: str | None,
    *,
    kind: MemoryKind = "preference",
    key: str | None = None,
    supersedes_id: str | None = None,
    provenance: MemoryProvenance | None = None,
    owner_id: str = "alpha",
) -> MemoryOperationReceipt:
    """One remember carrying the owner's words, as the pass writes it."""
    return await memory.apply(
        MemoryOperation(
            owner_id=owner_id,
            idempotency_key=key or f"say:{uuid.uuid5(uuid.NAMESPACE_URL, body)}",
            action="remember",
            provenance=provenance or _provenance(),
            kind=kind,
            body=body,
            supersedes_id=supersedes_id,
            quote=quote,
        )
    )


async def test_pg_a_quote_is_stored_with_its_record_and_every_read_returns_it(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)
    pass_run = MemoryProvenance(
        origin_kind="session_pass", origin_id="s:1-2", run_id="run-9", session_id="session-9"
    )
    said = await _say(
        memory,
        "The owner prefers charts in a dark theme.",
        "以后图表都用深色主题",
        provenance=pass_run,
    )
    bare = await _say(memory, "The owner wants short conclusions.", None, provenance=pass_run)

    by_id = {row.memory_id: row for row in await _active(memory)}
    assert by_id[str(said.memory_id)].quote == "以后图表都用深色主题"
    assert by_id[str(bare.memory_id)].quote is None
    got = await store.get(owner_id="alpha", memory_id=str(said.memory_id))
    assert got is not None and got.quote == "以后图表都用深色主题"
    # Every path a record is read by: the exact leg of search, the standing fetch, the lookup's
    # listing, one Session's records and one Run's changes.
    found = await memory.search(owner_id="alpha", query="The owner prefers charts in a dark theme.")
    assert {r.body: r.quote for r in found}["The owner prefers charts in a dark theme."] == (
        "以后图表都用深色主题"
    )
    recalled = await memory.recall(owner_id="alpha", query="unrelated question")
    assert {r.body: r.quote for r in recalled.standing} == {
        "The owner prefers charts in a dark theme.": "以后图表都用深色主题",
        "The owner wants short conclusions.": None,
    }
    session_records = await memory.session_records(
        owner_id="alpha", session_id="session-9", origin_kind="session_pass"
    )
    assert {r.body: r.quote for r in session_records} == {
        "The owner prefers charts in a dark theme.": "以后图表都用深色主题",
        "The owner wants short conclusions.": None,
    }
    changes = await memory.run_changes(owner_id="alpha", run_id="run-9")
    assert {c.record.body: c.record.quote for c in changes} == {
        "The owner prefers charts in a dark theme.": "以后图表都用深色主题",
        "The owner wants short conclusions.": None,
    }
    looked = await memory.lookup(owner_id="alpha", query="unrelated question")
    assert looked.standing == recalled.standing


async def test_pg_a_quote_is_recalled_into_the_block_and_the_lookup_the_agent_reads(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)
    await _say(
        memory, "The owner prefers charts in a dark theme.", "以后图表都用深色主题", owner_id="o"
    )

    recalled = await memory.recall(owner_id="o", query="how should charts look")
    # It is found by words it shares with the question (the sparse leg), not only stood.
    assert [r.quote for r in recalled.relevant] == ["以后图表都用深色主题"]
    block = render_auto_recall(recalled).splitlines()
    updates: list[ToolResult] = []
    looked = await lookup_tool(host=_host(store)).execute(
        LookupInput(query="how should charts look"),
        recording_tool_runtime(updates, tool_name="memory_lookup"),
    )
    mcp = await mcp_lookup(memory, subject="o", query="how should charts look")

    assert block[-2].startswith("- (preference, ")
    assert block[-1] == '  owner said: "以后图表都用深色主题"'
    # The Agent's lookup puts the second line under the first, which now carries the id.
    lookup_lines = looked.text_content.splitlines()
    assert lookup_lines[-1] == '  owner said: "以后图表都用深色主题"'
    assert re.match(r"- [0-9a-f-]{36} \(preference, ", lookup_lines[-2])
    # The MCP record carries it as a field.
    assert [record["quote"] for record in (*mcp["relevant"], *mcp["standing"])] == [
        "以后图表都用深色主题"
    ]


async def test_pg_an_update_keeps_its_own_quote_and_undoing_it_brings_back_the_old_one(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)
    old = await _say(memory, "Prefers short conclusions.", "结论短一点")
    new = await _say(
        memory,
        "Prefers conclusions of two sentences.",
        "最多两句话",
        supersedes_id=old.memory_id,
        key="update",
    )
    assert [(r.body, r.quote) for r in await _active(memory)] == [
        ("Prefers conclusions of two sentences.", "最多两句话")
    ]

    await memory.undo(
        owner_id="alpha",
        change_id=new.change_id,
        provenance=MemoryProvenance(origin_kind="undo", origin_id="undo-1"),
        idempotency_key="undo-1",
    )

    # The restored record is a new row with the words the old one stood on.
    assert [(r.body, r.quote) for r in await _active(memory)] == [
        ("Prefers short conclusions.", "结论短一点")
    ]


async def test_pg_forgetting_a_record_and_undoing_it_restores_its_words(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)
    said = await _say(memory, "Prefers short conclusions.", "结论短一点")
    forgotten = await memory.forget(
        owner_id="alpha", memory_id=said.memory_id, provenance=_provenance(), idempotency_key="f"
    )

    restored = await memory.undo(
        owner_id="alpha",
        change_id=forgotten.change_id,
        provenance=MemoryProvenance(origin_kind="undo", origin_id="undo-1"),
        idempotency_key="undo-f",
    )

    assert restored.outcome == "changed"
    assert [(r.body, r.quote) for r in await _active(memory)] == [
        ("Prefers short conclusions.", "结论短一点")
    ]


async def test_pg_a_fuse_carries_the_new_writes_own_quote_and_undoing_it_restores_the_sources(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)
    first = await _say(memory, "Answer in Chinese.", "用中文回答我")
    second = await _say(memory, "Put tables before prose.", "表格放前面")
    sources = [str(first.memory_id), str(second.memory_id)]

    fused = await memory.apply(
        MemoryOperation(
            owner_id="alpha",
            idempotency_key="fuse-quoted",
            action="fuse",
            provenance=_provenance(),
            kind="preference",
            body="Answer in Chinese, tables before prose.",
            source_ids=tuple(sources),
            quote="中文，表格放前面",
        )
    )

    # The merged record stands on the words of the write that merged it, not on its sources'.
    assert [(r.body, r.quote) for r in await _active(memory)] == [
        ("Answer in Chinese, tables before prose.", "中文，表格放前面")
    ]
    await memory.undo(
        owner_id="alpha",
        change_id=fused.change_id,
        provenance=MemoryProvenance(origin_kind="undo", origin_id="undo-1"),
        idempotency_key="undo-fuse",
    )
    assert {(r.body, r.quote) for r in await _active(memory)} == {
        ("Answer in Chinese.", "用中文回答我"),
        ("Put tables before prose.", "表格放前面"),
    }
    # A fuse with no words of its own has none, whatever its sources had.
    again_first = [r for r in await _active(memory) if r.body == "Answer in Chinese."][0]
    again_second = [r for r in await _active(memory) if r.body == "Put tables before prose."][0]
    bare = await memory.apply(
        MemoryOperation(
            owner_id="alpha",
            idempotency_key="fuse-bare",
            action="fuse",
            provenance=_provenance(),
            kind="preference",
            body="Chinese, tables first.",
            source_ids=(again_first.memory_id, again_second.memory_id),
        )
    )
    assert bare.outcome == "changed"
    assert [(r.body, r.quote) for r in await _active(memory)] == [("Chinese, tables first.", None)]


async def test_pg_a_key_replays_with_its_quote_and_is_refused_with_another(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)
    first = await _say(memory, "Prefers short conclusions.", "结论短一点", key="k")

    replay = await _say(memory, "Prefers short conclusions.", "结论短一点", key="k")
    with pytest.raises(MemoryWriteRejectedError, match="different input"):
        await _say(memory, "Prefers short conclusions.", "别写太长", key="k")
    with pytest.raises(MemoryWriteRejectedError, match="different input"):
        await _say(memory, "Prefers short conclusions.", None, key="k")

    assert replay == first
    assert [r.quote for r in await _active(memory)] == ["结论短一点"]


async def test_pg_a_quote_a_record_cannot_carry_is_refused_and_nothing_is_written(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)

    for quote in ("x" * (MEMORY_QUOTE_LIMIT + 1), "two\nlines", "see [1] here", ""):
        with pytest.raises(MemoryWriteRejectedError):
            await _say(memory, "Prefers short conclusions.", quote)

    assert await _active(memory) == ()


async def test_pg_recall_counts_a_quote_against_its_budget(store: PostgresMemoryStore) -> None:
    memory = Memory(store)
    for index in range(60):
        body = f"{index:02d} " + "记" * 97
        await _say(memory, body, "说" * MEMORY_QUOTE_LIMIT, key=f"quoted-{index}")

    result = await memory.recall(owner_id="alpha", query="unrelated question")

    # Each line costs its 100 characters and its 120 of words, each with its framing: 271, so
    # twenty-nine of the sixty fit where a hundred-character body alone would take fifty-nine.
    assert len(result.standing) == RECALL_CHAR_BUDGET // (100 + 34 + MEMORY_QUOTE_LIMIT + 17) == 29
    assert result.standing_omitted == 31
    assert recall_cost(result.records) <= RECALL_CHAR_BUDGET
    shown = sum(
        len(line) + 1
        for line in render_auto_recall(result).splitlines()
        if line.startswith(("- (", '  owner said: "'))
    )
    assert shown <= recall_cost(result.records)
