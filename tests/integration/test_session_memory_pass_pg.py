# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The Session Memory Pass over the real Run store, Session entries and memory store."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest
from dlightrag_memory import Memory, MemoryProvenance
from dlightrag_memory.consolidation import (
    ConsolidationVerdict,
    PlannedWrite,
    VerdictWrite,
    planned_operation,
)
from dlightrag_memory.postgres import PostgresMemoryStore

from dlightrag.adapters.postgres.answer.memory_settings import PGMemorySettingsStore
from dlightrag.adapters.postgres.runtime._session_memory import _SWEEP, _TURNS_BEFORE
from dlightrag.adapters.postgres.runtime.run_store import PGRunStore
from dlightrag.application.memory import session_pass as session_pass_module
from dlightrag.application.memory.service import MemoryService
from dlightrag.application.memory.session_pass import (
    SessionMemoryPass,
    SessionMemoryPassRunner,
    SessionPassSettings,
)
from dlightrag.engine.answer.memory import (
    MemoryCapability,
    PendingSession,
    SessionMemoryPlan,
    render_auto_recall,
)
from dlightrag.engine.runtime.coordinator import SettledRun
from tests.integration.run_runtime_pg_harness import run_envelope
from tests.support.pg import PG_CONN_KWARGS, drop_database, skip_without_postgres

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

OWNER = "owner-1"
RUN = "0190a1b2-0000-7000-8000-0000000000aa"


@pytest.fixture
async def pool() -> AsyncIterator[Any]:
    await skip_without_postgres()
    database = f"dlightrag_pass_{uuid.uuid4().hex[:12]}"
    admin = await asyncpg.connect(**PG_CONN_KWARGS)
    try:
        await admin.execute(f'CREATE DATABASE "{database}"')
    finally:
        await admin.close()
    created = await asyncpg.create_pool(
        **{**PG_CONN_KWARGS, "database": database}, min_size=1, max_size=6
    )
    try:
        await PGRunStore(pool=created).initialize()
        yield created
    finally:
        await created.close()
        await drop_database(database)


_NOTIFICATION = {"origin": "child_notification"}
_CHILD = "0190a1b2-0000-7000-8000-000000000001"


async def _session(
    pool: Any,
    *entries: tuple[str, str] | tuple[str, str, dict[str, Any]],
    owner: str = OWNER,
) -> str:
    """One Agent Session holding ``entries`` as (entry_type, content[, payload extras]) in order."""
    session_id = str(uuid.uuid4())
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO dlightrag_agent_sessions (owner_id, session_id, lease_run_id, "
            "fencing_epoch) VALUES ($1, $2, $3, 1)",
            owner,
            uuid.UUID(session_id),
            uuid.uuid4(),
        )
    await _append(pool, session_id, *entries, owner=owner)
    return session_id


async def _append(
    pool: Any,
    session_id: str,
    *entries: tuple[str, str] | tuple[str, str, dict[str, Any]],
    owner: str = OWNER,
) -> None:
    """Entries written after the Session's last one, in order, and its counter with them.

    The Session's `last_sequence` is bumped in the same breath as the entries are written, as the
    session repository does in one transaction, so it is always the latest entry's sequence.
    """
    async with pool.acquire() as conn:
        last = await conn.fetchval(
            "SELECT COALESCE(MAX(sequence), 0) FROM dlightrag_agent_session_entries "
            "WHERE owner_id = $1 AND session_id = $2",
            owner,
            uuid.UUID(session_id),
        )
        for sequence, entry in enumerate(entries, start=last + 1):
            payload = {"content": entry[1], **(entry[2] if len(entry) == 3 else {})}
            await conn.execute(
                "INSERT INTO dlightrag_agent_session_entries (owner_id, session_id, sequence, "
                "entry_id, entry_type, schema_version, timestamp, payload_json) "
                "VALUES ($1, $2, $3, $4, $5, 1, $6, $7::jsonb)",
                owner,
                uuid.UUID(session_id),
                sequence,
                uuid.uuid4(),
                entry[0],
                datetime.now(UTC),
                json.dumps(payload),
            )
        await conn.execute(
            "UPDATE dlightrag_agent_sessions SET last_sequence = $3 "
            "WHERE owner_id = $1 AND session_id = $2",
            owner,
            uuid.UUID(session_id),
            last + len(entries),
        )


async def _memory(pool: Any) -> tuple[Memory, MemoryService]:
    store = PostgresMemoryStore(pool=pool)
    await store.initialize()
    service = MemoryService(
        store,
        memory_list_cursor_secret=b"k" * 32,
        settings_store=PGMemorySettingsStore(pool=pool),
        sessions=PGRunStore(pool=pool),
    )
    return Memory(store), service


class _Model:
    """Answers each request with the next verdict, recording what it was asked."""

    def __init__(self, *verdicts: ConsolidationVerdict) -> None:
        self.verdicts = list(verdicts)
        self.asked: list[Any] = []

    async def __call__(self, messages: list[dict[str, Any]], **_kwargs: Any) -> str:
        self.asked.append(messages)
        return self.verdicts.pop(0).model_dump_json()

    async def aclose(self) -> None:
        return None


def _pass(
    pool: Any,
    memory: Memory,
    service: MemoryService,
    model: _Model,
    *,
    holder: str = "holder-a",
    batch: int = 40_000,
) -> SessionMemoryPass:
    return SessionMemoryPass(
        memory=memory,
        store=PGRunStore(pool=pool),
        gate=service,
        model=model,
        settings=SessionPassSettings(max_batch_tokens=batch, session_record_limit=5),
        holder=holder,
    )


_MUNICH = SessionMemoryPlan(
    read_from=1,
    read_through=1,
    writes=(PlannedWrite(action="add", kind="fact", body="Lives in Munich."),),
    run_id=RUN,
)


async def _row(pool: Any, session_id: str, *, owner: str = OWNER) -> Any:
    async with pool.acquire() as conn:
        return await conn.fetchrow(
            "SELECT * FROM dlightrag_session_memory_progress "
            "WHERE owner_id = $1 AND session_id = $2",
            owner,
            uuid.UUID(session_id),
        )


async def test_a_pass_reads_its_conversation_writes_its_verdict_and_advances(pool: Any) -> None:
    memory, service = await _memory(pool)
    older = await memory.remember(
        owner_id=OWNER,
        kind="preference",
        body="Answer in Chinese.",
        provenance=MemoryProvenance(origin_kind="management", origin_id="settings"),
        idempotency_key="older",
    )
    session_id = await _session(
        pool,
        ("user_message", "Please answer in Chinese, and always put the table first."),
        ("tool_result", "a long search result that the pass must never read"),
        ("assistant_message", "好的，我会先放表格。"),
    )
    verdict = ConsolidationVerdict(
        writes=[
            VerdictWrite(
                action="update",
                kind="preference",
                body="The owner wants answers in Chinese with the table first.",
                targets=[1],
            ),
            VerdictWrite(
                action="add",
                kind="experience",
                body="This owner's reports read best when a table leads.",
            ),
        ]
    )
    model = _Model(verdict)

    assert (
        await _pass(pool, memory, service, model).run(
            owner_id=OWNER, session_id=session_id, run_id=RUN
        )
        == "applied"
    )

    conversation = model.asked[0][1]["content"]
    assert "table first" in conversation
    # The model is shown the older memory as number 1, never by its id.
    assert "[1] (preference, " in conversation and "Answer in Chinese." in conversation
    assert str(older.memory_id) not in conversation
    assert "long search result" not in conversation
    contributed = await memory.session_records(
        owner_id=OWNER, session_id=session_id, origin_kind="session_pass"
    )
    assert sorted(record.kind for record in contributed) == ["experience", "preference"]
    standing = (await memory.recall(owner_id=OWNER, query="unrelated")).standing
    assert "Answer in Chinese." not in [record.body for record in standing]
    row = await _row(pool, session_id)
    assert row["read_through"] == 3
    assert row["plan_writes"] is None
    assert row["holder"] is None


async def test_what_a_pass_writes_is_attributed_to_the_run_that_asked_for_it(pool: Any) -> None:
    memory, service = await _memory(pool)
    await memory.remember(
        owner_id=OWNER,
        kind="fact",
        body="Lives in Berlin.",
        provenance=MemoryProvenance(origin_kind="management", origin_id="settings"),
        idempotency_key="older",
    )
    session_id = await _session(
        pool,
        ("user_message", "I moved from Berlin to Munich last month, and I like my tables first."),
        ("assistant_message", "Noted: Munich, and tables first."),
    )
    verdict = ConsolidationVerdict(
        writes=[
            VerdictWrite(
                action="update",
                kind="fact",
                body="Lives in Munich.",
                targets=[1],
            ),
            VerdictWrite(action="add", kind="preference", body="Likes tables first."),
        ]
    )

    assert (
        await _pass(pool, memory, service, _Model(verdict)).run(
            owner_id=OWNER, session_id=session_id, run_id=RUN
        )
        == "applied"
    )

    changes = await memory.run_changes(owner_id=OWNER, run_id=RUN)
    assert [(c.record.body, c.action) for c in changes] == [
        ("Lives in Munich.", "update"),
        ("Likes tables first.", "add"),
    ]
    assert {c.record.provenance.origin_kind for c in changes} == {"session_pass"}
    # Only the Run that asked sees them, and the plan's Run is not kept past the pass.
    assert await memory.run_changes(owner_id=OWNER, run_id="another-run") == ()
    assert (await _row(pool, session_id))["plan_run_id"] is None


async def test_a_plan_is_stored_with_the_run_it_is_attributed_to(pool: Any) -> None:
    session_id = await _session(pool, ("user_message", "I moved to Munich last month."))
    store = PGRunStore(pool=pool)
    await store.claim_session_memory(
        owner_id=OWNER, session_id=session_id, holder="a", lease_seconds=60
    )

    assert await store.store_session_memory_plan(
        owner_id=OWNER, session_id=session_id, holder="a", plan=_MUNICH
    )
    assert (await _row(pool, session_id))["plan_run_id"] == RUN
    await store.release_session_memory(owner_id=OWNER, session_id=session_id, holder="a")

    resumed = await store.claim_session_memory(
        owner_id=OWNER, session_id=session_id, holder="b", lease_seconds=60
    )
    assert resumed is not None and resumed.plan is not None
    assert resumed.plan.run_id == RUN
    assert resumed.plan.writes == _MUNICH.writes


async def test_a_resumed_plan_keeps_the_run_it_was_made_for_and_replays_what_was_applied(
    pool: Any,
) -> None:
    memory, service = await _memory(pool)
    session_id = await _session(pool, ("user_message", "I moved to Munich last month."))
    store = PGRunStore(pool=pool)
    plan = SessionMemoryPlan(
        read_from=1,
        read_through=1,
        writes=(
            PlannedWrite(action="add", kind="fact", body="Lives in Munich."),
            PlannedWrite(action="add", kind="preference", body="Likes tables first."),
        ),
        run_id=RUN,
    )
    await store.claim_session_memory(
        owner_id=OWNER, session_id=session_id, holder="dead", lease_seconds=60
    )
    assert await store.store_session_memory_plan(
        owner_id=OWNER, session_id=session_id, holder="dead", plan=plan
    )
    # The holder dies after the first write: it has applied under the plan's Run.
    span = f"{session_id}:1-1"
    first = planned_operation(
        plan.writes[0],
        index=0,
        owner_id=OWNER,
        provenance=MemoryProvenance(
            origin_kind="session_pass", origin_id=span, run_id=RUN, session_id=session_id
        ),
        pass_key=f"pass:{span}",
    )
    assert (await memory.apply(first)).outcome == "changed"
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE dlightrag_session_memory_progress "
            "SET claim_expires_at = NOW() - INTERVAL '1 second'"
        )

    # The next pass is asked for by a later Run, and does not ask its model again.
    model = _Model()
    assert (
        await _pass(pool, memory, service, model, holder="next").run(
            owner_id=OWNER, session_id=session_id, run_id="a-later-run"
        )
        == "resumed"
    )

    assert model.asked == []
    changes = await memory.run_changes(owner_id=OWNER, run_id=RUN)
    assert [c.record.body for c in changes] == ["Lives in Munich.", "Likes tables first."]
    assert await memory.run_changes(owner_id=OWNER, run_id="a-later-run") == ()
    assert (await _row(pool, session_id))["read_through"] == 1


async def test_a_claim_is_held_once_and_a_stale_holder_cannot_move_the_row(pool: Any) -> None:
    session_id = await _session(pool, ("user_message", "hello"))
    store = PGRunStore(pool=pool)

    first = await store.claim_session_memory(
        owner_id=OWNER, session_id=session_id, holder="a", lease_seconds=60
    )
    second = await store.claim_session_memory(
        owner_id=OWNER, session_id=session_id, holder="b", lease_seconds=60
    )
    stale = await store.advance_session_memory(
        owner_id=OWNER, session_id=session_id, holder="b", through=1
    )
    await store.release_session_memory(owner_id=OWNER, session_id=session_id, holder="a")
    third = await store.claim_session_memory(
        owner_id=OWNER, session_id=session_id, holder="b", lease_seconds=60
    )

    assert first is not None and first.read_through == 0
    assert second is None
    assert stale is None
    assert third is not None


async def test_a_holder_whose_claim_lapsed_cannot_store_a_plan(pool: Any) -> None:
    session_id = await _session(pool, ("user_message", "I moved to Munich last month."))
    store = PGRunStore(pool=pool)
    await store.claim_session_memory(
        owner_id=OWNER, session_id=session_id, holder="slow", lease_seconds=0
    )

    stored = await store.store_session_memory_plan(
        owner_id=OWNER, session_id=session_id, holder="slow", plan=_MUNICH
    )

    assert stored is False
    assert (await _row(pool, session_id))["plan_writes"] is None


async def test_a_plan_stored_under_a_live_claim_is_resumed_by_the_next_holder(pool: Any) -> None:
    memory, service = await _memory(pool)
    session_id = await _session(pool, ("user_message", "I moved to Munich last month."))
    store = PGRunStore(pool=pool)
    await store.claim_session_memory(
        owner_id=OWNER, session_id=session_id, holder="dead", lease_seconds=60
    )
    assert await store.store_session_memory_plan(
        owner_id=OWNER, session_id=session_id, holder="dead", plan=_MUNICH
    )
    async with pool.acquire() as conn:  # the dead holder's claim lapses
        await conn.execute(
            "UPDATE dlightrag_session_memory_progress "
            "SET claim_expires_at = NOW() - INTERVAL '1 second'"
        )
    model = _Model()

    assert (
        await _pass(pool, memory, service, model, holder="next").run(
            owner_id=OWNER, session_id=session_id, run_id=RUN
        )
        == "resumed"
    )
    assert model.asked == []
    contributed = await memory.session_records(
        owner_id=OWNER, session_id=session_id, origin_kind="session_pass"
    )
    assert [record.body for record in contributed] == ["Lives in Munich."]
    assert (await _row(pool, session_id))["read_through"] == 1


async def test_an_owner_who_switched_memory_off_is_never_read(pool: Any) -> None:
    memory, service = await _memory(pool)
    await service.set_enabled(owner_id=OWNER, enabled=False)
    session_id = await _session(
        pool, ("user_message", "Remember that I live in Gothenburg."), ("assistant_message", "ok")
    )
    model = _Model()
    session_pass = _pass(pool, memory, service, model)

    assert await session_pass.run(owner_id=OWNER, session_id=session_id, run_id=RUN) == "disabled"
    await service.set_enabled(owner_id=OWNER, enabled=True)

    assert (
        await session_pass.run(owner_id=OWNER, session_id=session_id, run_id=RUN) == "nothing_new"
    )
    assert model.asked == []
    assert (await _row(pool, session_id))["read_through"] == 2


async def test_the_progress_row_goes_with_its_session(pool: Any) -> None:
    session_id = await _session(pool, ("user_message", "hello"))
    store = PGRunStore(pool=pool)
    await store.claim_session_memory(
        owner_id=OWNER, session_id=session_id, holder="a", lease_seconds=60
    )

    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM dlightrag_agent_sessions WHERE owner_id = $1 AND session_id = $2",
            OWNER,
            uuid.UUID(session_id),
        )

    assert await _row(pool, session_id) is None
    gone = await store.claim_session_memory(
        owner_id=OWNER, session_id=session_id, holder="a", lease_seconds=60
    )
    assert gone is None


async def test_assistant_turns_that_only_called_tools_are_never_fetched(pool: Any) -> None:
    session_id = await _session(
        pool,
        ("user_message", "Compare the two quarterly reports."),
        ("assistant_message", ""),
        ("tool_result", "a report's text"),
        ("assistant_message", "   "),
        ("assistant_message", "Revenue rose eight percent."),
    )
    store = PGRunStore(pool=pool)

    turns = await store.session_turns_after(
        owner_id=OWNER, session_id=session_id, after=0, limit=10
    )

    assert [(turn.sequence, turn.role) for turn in turns] == [(1, "user"), (5, "assistant")]


async def test_a_childs_notification_is_never_fetched_between_turns_or_at_the_tail(
    pool: Any,
) -> None:
    session_id = await _session(
        pool,
        ("user_message", "Compare the two quarterly reports."),
        ("user_message", f"Child {_CHILD} [succeeded]: Revenue rose eight percent.", _NOTIFICATION),
        ("assistant_message", "Revenue rose eight percent."),
        (
            "user_message",
            f"Child session {_CHILD} asks for guidance (request_id=r1): which year?",
            _NOTIFICATION,
        ),
    )
    store = PGRunStore(pool=pool)

    turns = await store.session_turns_after(
        owner_id=OWNER, session_id=session_id, after=0, limit=10
    )

    assert [(turn.sequence, turn.role) for turn in turns] == [(1, "user"), (3, "assistant")]


async def test_a_pass_never_shows_its_model_a_childs_notification_and_reads_past_it(
    pool: Any,
) -> None:
    memory, service = await _memory(pool)
    session_id = await _session(
        pool,
        (
            "user_message",
            "I moved to Munich last month, so please answer me in German from now on.",
        ),
        (
            "user_message",
            f"Child {_CHILD} [succeeded]: " + "The quarterly filing shows growth. " * 40,
            _NOTIFICATION,
        ),
        ("assistant_message", "Verstanden, ab jetzt antworte ich Ihnen auf Deutsch."),
        (
            "user_message",
            f"Child session {_CHILD} asks for guidance (request_id=r1): which year?",
            _NOTIFICATION,
        ),
    )
    model = _Model(ConsolidationVerdict(), ConsolidationVerdict())
    session_pass = _pass(pool, memory, service, model)

    assert await session_pass.run(owner_id=OWNER, session_id=session_id, run_id=RUN) == "applied"

    first = model.asked[0][1]["content"]
    assert "[user] I moved to Munich" in first
    assert "[assistant] Verstanden" in first
    assert "quarterly filing" not in first
    assert "asks for guidance" not in first
    assert (await _row(pool, session_id))["read_through"] == 3

    await _append(
        pool,
        session_id,
        (
            "user_message",
            f"Child {_CHILD} [failed]: The second filing could not be read at all.",
            _NOTIFICATION,
        ),
        (
            "user_message",
            "My name is Anna, and I work as a nurse in a hospital near the river in Hamburg.",
        ),
    )

    assert await session_pass.run(owner_id=OWNER, session_id=session_id, run_id=RUN) == "applied"

    second = model.asked[1][1]["content"]
    context, newest = second.split("Newest part of this conversation")
    assert "[user] My name is Anna" in newest
    assert "I moved to" not in newest and "Verstanden" not in newest
    # What the first pass read is the second's context. A Child's notification is no context
    # either: not the one the first pass read past, nor the one before the new text.
    assert "[user] I moved to Munich" in context
    assert "[assistant] Verstanden" in context
    for hidden in ("quarterly filing", "asks for guidance", "could not be read"):
        assert hidden not in second
    assert (await _row(pool, session_id))["read_through"] == 6


def _weighty(marker: str) -> str:
    """About a hundred estimated tokens, so a batch of 150 holds exactly one such turn."""
    return f"{marker} " + "x" * 400


def _split(request: list[dict[str, str]]) -> tuple[str, str]:
    """The context and the newest part of one request's conversation."""
    shown = request[1]["content"].split("\n\nExisting memories that may bear on it")[0]
    context, _, newest = shown.partition("Newest part of this conversation")
    return context, newest


async def test_a_conversation_longer_than_one_call_is_read_to_its_end_in_one_pass(
    pool: Any,
) -> None:
    memory, service = await _memory(pool)
    session_id = await _session(
        pool,
        ("user_message", _weighty("I moved to Munich last month.")),
        ("tool_result", "a long search result that the pass must never read"),
        ("assistant_message", _weighty("Noted, Munich.")),
        ("user_message", f"Child {_CHILD} [succeeded]: a Child's result", _NOTIFICATION),
        ("user_message", _weighty("Please answer me in German from now on.")),
        ("assistant_message", _weighty("Verstanden.")),
    )
    model = _Model(
        ConsolidationVerdict(
            writes=[VerdictWrite(action="add", kind="fact", body="Lives in Munich.")]
        ),
        ConsolidationVerdict(),
        ConsolidationVerdict(
            writes=[
                VerdictWrite(
                    action="add", kind="preference", body="The owner wants answers in German."
                )
            ]
        ),
        ConsolidationVerdict(),
    )

    outcome = await _pass(pool, memory, service, model, batch=150).run(
        owner_id=OWNER, session_id=session_id, run_id=RUN
    )

    # Four text-bearing turns, a batch of one: four calls under one claim, to the very end.
    assert outcome == "applied"
    assert len(model.asked) == 4
    row = await _row(pool, session_id)
    assert (row["read_through"], row["plan_writes"], row["holder"]) == (6, None, None)
    parts = [_split(request) for request in model.asked]
    markers = ("Munich last month", "Noted, Munich", "German from now on", "Verstanden")
    for marker, (_, newest) in zip(markers, parts, strict=True):
        assert marker in newest
    # Every call after the first reads what came before it as context, nearest included...
    assert parts[0][0] == ""
    assert "Munich last month" in parts[1][0]
    assert "Noted, Munich" in parts[2][0] and "Munich last month" in parts[2][0]
    assert "German from now on" in parts[3][0]
    # ...but never a tool result or a Child's notification, as the part to judge or as context.
    for request in model.asked:
        assert "long search result" not in request[1]["content"]
        assert "Child's result" not in request[1]["content"]
    # And it sees what the batches before it wrote: they are this conversation's own memories.
    last = model.asked[3][1]["content"].split("Memories this conversation already contributed")[1]
    assert "Lives in Munich." in last and "The owner wants answers in German." in last
    # One record per write, each under its own batch's span, keys and mutation scope.
    contributed = await memory.session_records(
        owner_id=OWNER, session_id=session_id, origin_kind="session_pass"
    )
    assert {record.body: record.provenance.origin_id for record in contributed} == {
        "Lives in Munich.": f"{session_id}:1-1",
        "The owner wants answers in German.": f"{session_id}:4-5",
    }
    async with pool.acquire() as conn:
        journal = await conn.fetch(
            "SELECT idempotency_key, mutation_scope FROM dlightrag_memory_operations "
            "WHERE owner_id = $1 ORDER BY created_at",
            OWNER,
        )
    assert [(row["idempotency_key"], row["mutation_scope"]) for row in journal] == [
        (f"pass:{session_id}:1-1:0", f"pass:{session_id}:1-1"),
        (f"pass:{session_id}:4-5:0", f"pass:{session_id}:4-5"),
    ]


async def test_a_pass_longer_than_its_lease_keeps_its_claim_by_renewing_it(
    pool: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Four calls of 0.6 seconds under a lease of 1.5 seconds: only a renewed claim lasts."""
    monkeypatch.setattr(session_pass_module, "SESSION_PASS_CLAIM_SECONDS", 1.5)
    memory, service = await _memory(pool)
    session_id = await _session(
        pool,
        *(
            ("user_message" if index % 2 == 0 else "assistant_message", _weighty(f"turn {index}"))
            for index in range(4)
        ),
    )

    class _Slow(_Model):
        async def __call__(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
            await asyncio.sleep(0.6)
            return await super().__call__(messages, **kwargs)

    model = _Slow(*[ConsolidationVerdict()] * 4)

    outcome = await _pass(pool, memory, service, model, batch=150).run(
        owner_id=OWNER, session_id=session_id, run_id=RUN
    )

    assert outcome == "applied"
    assert len(model.asked) == 4
    assert (await _row(pool, session_id))["read_through"] == 4


async def test_a_pass_whose_claim_was_taken_over_stops_at_its_next_renewal(pool: Any) -> None:
    memory, service = await _memory(pool)
    session_id = await _session(
        pool,
        *(
            ("user_message" if index % 2 == 0 else "assistant_message", _weighty(f"turn {index}"))
            for index in range(3)
        ),
    )

    class _Takeover:
        """The owner's switch, and another process taking the claim once a batch is done."""

        taken = False

        async def capability(self, *, owner_id: str) -> MemoryCapability:
            if not self.taken and (await _row(pool, session_id))["read_through"] >= 1:
                self.taken = True
                async with pool.acquire() as conn:
                    await conn.execute(
                        "UPDATE dlightrag_session_memory_progress SET holder = 'other', "
                        "claim_expires_at = NOW() + INTERVAL '60 seconds'"
                    )
            return await service.capability(owner_id=owner_id)

        async def capability_current(
            self, *, owner_id: str, epoch: int, settlement: object | None = None
        ) -> bool:
            return await service.capability_current(
                owner_id=owner_id, epoch=epoch, settlement=settlement
            )

    model = _Model(*[ConsolidationVerdict()] * 3)
    session_pass = SessionMemoryPass(
        memory=memory,
        store=PGRunStore(pool=pool),
        gate=_Takeover(),
        model=model,
        settings=SessionPassSettings(max_batch_tokens=150, session_record_limit=5),
        holder="holder-a",
    )

    outcome = await session_pass.run(owner_id=OWNER, session_id=session_id, run_id=RUN)

    # The first batch stands. The next renewal finds the claim is another's: nothing more is
    # asked or moved, and letting go of its own claim leaves the other process's alone.
    assert outcome == "superseded"
    assert len(model.asked) == 1
    row = await _row(pool, session_id)
    assert (row["read_through"], row["holder"]) == (1, "other")


async def test_a_stored_plan_is_resumed_and_the_pass_then_reads_what_came_after_it(
    pool: Any,
) -> None:
    memory, service = await _memory(pool)
    session_id = await _session(
        pool,
        ("user_message", "I moved to Munich last month."),
        ("assistant_message", "Noted."),
        ("user_message", "Please answer me in German from now on."),
        ("assistant_message", "Verstanden."),
    )
    store = PGRunStore(pool=pool)
    await store.claim_session_memory(
        owner_id=OWNER, session_id=session_id, holder="dead", lease_seconds=60
    )
    plan = SessionMemoryPlan(
        read_from=1,
        read_through=2,
        writes=(PlannedWrite(action="add", kind="fact", body="Lives in Munich."),),
        run_id=RUN,
    )
    assert await store.store_session_memory_plan(
        owner_id=OWNER, session_id=session_id, holder="dead", plan=plan
    )
    async with pool.acquire() as conn:  # the dead holder's claim lapses
        await conn.execute(
            "UPDATE dlightrag_session_memory_progress "
            "SET claim_expires_at = NOW() - INTERVAL '1 second'"
        )
    model = _Model(
        ConsolidationVerdict(
            writes=[
                VerdictWrite(
                    action="add", kind="preference", body="The owner wants answers in German."
                )
            ]
        )
    )

    outcome = await _pass(pool, memory, service, model, holder="next").run(
        owner_id=OWNER, session_id=session_id, run_id="a-later-run"
    )

    # The plan settles without the model; the model is asked once, about what followed it,
    # with the range the plan covered as context.
    assert outcome == "applied"
    (request,) = model.asked
    context, newest = _split(request)
    assert "German from now on" in newest and "Munich" not in newest
    assert "Munich last month" in context
    attributed = {
        record.body: record.provenance.run_id
        for record in await memory.session_records(
            owner_id=OWNER, session_id=session_id, origin_kind="session_pass"
        )
    }
    assert attributed == {
        "Lives in Munich.": RUN,
        "The owner wants answers in German.": "a-later-run",
    }
    row = await _row(pool, session_id)
    assert (row["read_through"], row["plan_writes"], row["holder"]) == (4, None, None)


async def test_only_the_live_holder_can_renew_a_claim(pool: Any) -> None:
    session_id = await _session(pool, ("user_message", "hello"))
    store = PGRunStore(pool=pool)
    arguments = {"owner_id": OWNER, "session_id": session_id}
    await store.claim_session_memory(**arguments, holder="a", lease_seconds=60)
    claimed = (await _row(pool, session_id))["claim_expires_at"]

    # Another holder cannot extend it; the holder can.
    assert await store.renew_session_memory(**arguments, holder="b", lease_seconds=3600) is False
    assert (await _row(pool, session_id))["claim_expires_at"] == claimed
    assert await store.renew_session_memory(**arguments, holder="a", lease_seconds=3600) is True
    renewed = (await _row(pool, session_id))["claim_expires_at"]
    assert renewed > claimed + timedelta(minutes=30)
    assert (await _row(pool, session_id))["holder"] == "a"

    # A claim that lapsed is not revived, even by its own holder.
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE dlightrag_session_memory_progress "
            "SET claim_expires_at = NOW() - INTERVAL '1 second'"
        )
    assert await store.renew_session_memory(**arguments, holder="a", lease_seconds=60) is False

    # Once another process has taken it over, the first holder finds out the same way.
    assert await store.claim_session_memory(**arguments, holder="b", lease_seconds=60) is not None
    assert await store.renew_session_memory(**arguments, holder="a", lease_seconds=60) is False
    assert await store.renew_session_memory(**arguments, holder="b", lease_seconds=60) is True

    # And a released claim is nobody's to renew.
    await store.release_session_memory(**arguments, holder="b")
    assert await store.renew_session_memory(**arguments, holder="b", lease_seconds=60) is False
    assert (await _row(pool, session_id))["holder"] is None


def _seen(turns: Any) -> list[tuple[int, str, str]]:
    return [(turn.sequence, turn.role, turn.text) for turn in turns]


async def test_the_turns_before_a_point_are_its_last_rounds_oldest_first(pool: Any) -> None:
    session_id = await _session(
        pool,
        ("assistant_message", "before any user turn"),
        ("user_message", "one"),
        ("assistant_message", ""),
        ("tool_result", "a report's text"),
        ("user_message", f"Child {_CHILD} [succeeded]: Revenue rose.", _NOTIFICATION),
        ("assistant_message", "two"),
        ("assistant_message", "two again"),
        ("user_message", "three"),
        ("assistant_message", "four"),
    )
    elsewhere = await _session(pool, ("user_message", "another session's turn"))
    store = PGRunStore(pool=pool)

    # Tool results, an assistant entry with no text and a Child's notification are not turns, and
    # every assistant turn of a round is in it; fewer than the rounds asked for gives them all.
    # An assistant turn before the first user turn belongs to no round.
    everything = await store.session_turns_before(
        owner_id=OWNER, session_id=session_id, before=10, rounds=5
    )
    assert _seen(everything) == [
        (2, "user", "one"),
        (6, "assistant", "two"),
        (7, "assistant", "two again"),
        (8, "user", "three"),
        (9, "assistant", "four"),
    ]
    # Fewer rounds start at the user turn that opens the last of them.
    last = await store.session_turns_before(
        owner_id=OWNER, session_id=session_id, before=10, rounds=1
    )
    assert _seen(last) == [(8, "user", "three"), (9, "assistant", "four")]
    # The point itself is not before it, and nothing precedes the first entry.
    exclusive = await store.session_turns_before(
        owner_id=OWNER, session_id=session_id, before=8, rounds=5
    )
    assert [turn.sequence for turn in exclusive] == [2, 6, 7]
    assert not await store.session_turns_before(
        owner_id=OWNER, session_id=session_id, before=2, rounds=5
    )
    # Only this owner's, only this Session's.
    assert not await store.session_turns_before(
        owner_id="someone-else", session_id=session_id, before=10, rounds=5
    )
    assert _seen(
        await store.session_turns_before(owner_id=OWNER, session_id=elsewhere, before=10, rounds=5)
    ) == [(1, "user", "another session's turn")]


async def test_exactly_the_last_five_rounds_are_taken_and_their_assistant_turns_all_come(
    pool: Any,
) -> None:
    session_id = await _session(
        pool,
        ("user_message", "u1"),
        ("assistant_message", "a1"),
        ("user_message", "u2"),
        ("assistant_message", "a2"),
        ("assistant_message", "a2 again"),
        ("user_message", "u3"),
        ("user_message", "u4"),
        ("assistant_message", "a4"),
        ("user_message", "u5"),
        ("assistant_message", "a5"),
        ("user_message", "u6"),
        ("assistant_message", "a6"),
        ("assistant_message", "a6 again"),
    )
    store = PGRunStore(pool=pool)

    turns = await store.session_turns_before(
        owner_id=OWNER, session_id=session_id, before=14, rounds=5
    )

    # The first round is the sixth from the end: out, with its assistant turn.
    assert [turn.text for turn in turns] == [
        "u2",
        "a2",
        "a2 again",
        "u3",
        "u4",
        "a4",
        "u5",
        "a5",
        "u6",
        "a6",
        "a6 again",
    ]


async def test_the_context_floor_cuts_the_rounds_short(pool: Any) -> None:
    session_id = await _session(
        pool,
        ("user_message", "u1"),
        ("assistant_message", "a1"),
        ("user_message", "u2"),
        ("assistant_message", "a2"),
        ("user_message", "u3"),
        ("assistant_message", "a3"),
    )
    store = PGRunStore(pool=pool)
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO dlightrag_session_memory_progress "
            "(owner_id, session_id, read_through, context_floor) VALUES ($1, $2, 0, 3)",
            OWNER,
            uuid.UUID(session_id),
        )

    turns = await store.session_turns_before(
        owner_id=OWNER, session_id=session_id, before=7, rounds=5
    )

    # Nothing at or below the floor is a turn, so u2 (sequence 3) opens no round, and the
    # assistant turn above the floor that follows it (sequence 4) belongs to no round either.
    assert _seen(turns) == [(5, "user", "u3"), (6, "assistant", "a3")]


async def test_the_turns_before_a_point_walk_the_session_key_by_index(pool: Any) -> None:
    """Among many Sessions of one owner, the last rounds come off the Session's own key."""
    session_id = await _session(pool, ("user_message", "one"), ("assistant_message", "two"))
    async with pool.acquire() as conn:
        for _ in range(30):  # the owner's other Sessions, each a long run of tool results
            other = uuid.uuid4()
            await conn.execute(
                "INSERT INTO dlightrag_agent_sessions (owner_id, session_id, lease_run_id, "
                "fencing_epoch) VALUES ($1, $2, $3, 1)",
                OWNER,
                other,
                uuid.uuid4(),
            )
            await conn.execute(
                "INSERT INTO dlightrag_agent_session_entries (owner_id, session_id, sequence, "
                "entry_id, entry_type, schema_version, timestamp, payload_json) "
                "SELECT $1, $2, number, gen_random_uuid(), 'tool_result', 1, NOW(), $3::jsonb "
                "FROM generate_series(1, 200) AS number",
                OWNER,
                other,
                json.dumps({"content": "result"}),
            )

    async with pool.acquire() as conn, conn.transaction():
        await conn.execute("ANALYZE dlightrag_agent_session_entries")
        await conn.execute("SET LOCAL enable_seqscan = off")
        rows = await conn.fetch(
            f"EXPLAIN (COSTS OFF) {_TURNS_BEFORE}", OWNER, uuid.UUID(session_id), 3, 5
        )

    plan = "\n".join(str(row["QUERY PLAN"]) for row in rows)
    assert "Index Scan Backward using dlightrag_agent_session_entries_pkey" in plan
    assert "Index Scan using dlightrag_agent_session_entries_pkey" in plan
    assert "Seq Scan on dlightrag_agent_session_entries" not in plan
    assert "Sort" not in plan


async def _child_session(pool: Any, *entries: tuple[str, str]) -> str:
    """A Child Session of a Run of the owner, holding ``entries``: a session a pass never reads."""
    run_id = str(uuid.uuid7())
    store = PGRunStore(pool=pool)
    await store.accept_run(
        envelope=run_envelope("retrieval", key=f"parent-{run_id}", owner=OWNER), run_id=run_id
    )
    child_id = await _session(pool, *entries)
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO dlightrag_answer_child_sessions (owner_id, run_id, child_session_id, "
            "parent_session_id, parent_call_id, status, depth, context_snapshot_json) "
            "VALUES ($1, $2, $3, $4, 'call-1', 'running', 1, '{}'::jsonb)",
            OWNER,
            uuid.UUID(run_id),
            uuid.UUID(child_id),
            uuid.uuid4(),
        )
    return child_id


async def test_restarting_an_owners_sessions_moves_every_watermark_and_floor_to_its_latest_entry(
    pool: Any,
) -> None:
    store = PGRunStore(pool=pool)
    texts = [("user_message", "one"), ("assistant_message", "two"), ("user_message", "three")]
    with_row = await _session(pool, *texts)
    await store.claim_session_memory(
        owner_id=OWNER, session_id=with_row, holder="a", lease_seconds=60
    )
    assert await store.advance_session_memory(
        owner_id=OWNER, session_id=with_row, holder="a", through=1
    )
    await store.release_session_memory(owner_id=OWNER, session_id=with_row, holder="a")
    without_row = await _session(pool, *texts, ("assistant_message", "four"))
    empty = await _session(pool)
    child = await _child_session(pool, ("user_message", "a Child's turn"))
    elsewhere = await _session(pool, *texts, owner="someone-else")

    await store.restart_session_memory(owner_id=OWNER)

    # Every Session of the owner that holds entries starts at its latest one, whether or not it
    # had a row; one with nothing yet has nothing to skip, and a Child's has no pass to restart.
    row = await _row(pool, with_row)
    assert (row["read_through"], row["context_floor"]) == (3, 3)
    row = await _row(pool, without_row)
    assert (row["read_through"], row["context_floor"]) == (4, 4)
    assert await _row(pool, empty) is None
    assert await _row(pool, child) is None
    # Another owner's Sessions are not this owner's to restart.
    assert await _row(pool, elsewhere, owner="someone-else") is None
    # What was there up to the restart is no context; what follows it is.
    await _append(pool, with_row, ("user_message", "five"), ("assistant_message", "six"))
    turns = await store.session_turns_before(
        owner_id=OWNER, session_id=with_row, before=6, rounds=5
    )
    assert [turn.sequence for turn in turns] == [4, 5]
    turns = await store.session_turns_before(
        owner_id=OWNER, session_id=with_row, before=4, rounds=5
    )
    assert turns == ()


async def test_a_restart_never_moves_a_watermark_or_a_floor_back_and_leaves_a_claim_and_plan_alone(
    pool: Any,
) -> None:
    store = PGRunStore(pool=pool)
    arguments = {"owner_id": OWNER}
    ahead = await _session(pool, ("user_message", "one"), ("assistant_message", "two"))
    async with pool.acquire() as conn:  # a row already past the Session's latest entry
        await conn.execute(
            "INSERT INTO dlightrag_session_memory_progress "
            "(owner_id, session_id, read_through, context_floor) VALUES ($1, $2, 9, 8)",
            OWNER,
            uuid.UUID(ahead),
        )
    in_flight = await _session(pool, ("user_message", "one"), ("assistant_message", "two"))
    await store.claim_session_memory(
        **arguments, session_id=in_flight, holder="a", lease_seconds=60
    )
    assert await store.store_session_memory_plan(
        **arguments, session_id=in_flight, holder="a", plan=_MUNICH
    )
    before = await _row(pool, in_flight)

    await store.restart_session_memory(**arguments)
    await store.restart_session_memory(**arguments)  # and again: it settles, it does not stack

    row = await _row(pool, ahead)
    assert (row["read_through"], row["context_floor"]) == (9, 8)
    # The pass in flight keeps its claim and its plan, and reads on from the moved watermark.
    row = await _row(pool, in_flight)
    assert (row["read_through"], row["context_floor"]) == (2, 2)
    for kept in ("holder", "claim_expires_at", "plan_from", "plan_through", "plan_writes"):
        assert row[kept] == before[kept], kept
    assert (
        await store.advance_session_memory(**arguments, session_id=in_flight, holder="a", through=1)
        == 2
    )


async def test_text_written_while_memory_was_off_is_neither_read_nor_shown_as_context(
    pool: Any,
) -> None:
    memory, service = await _memory(pool)
    session_id = await _session(
        pool,
        ("user_message", "before-the-switch: I live in Gothenburg."),
        ("assistant_message", "before-the-switch: noted."),
    )
    model = _Model(*[ConsolidationVerdict()] * 3)
    session_pass = _pass(pool, memory, service, model)
    arguments = {"owner_id": OWNER, "session_id": session_id, "run_id": RUN}

    # Read while memory was on.
    assert await session_pass.run(**arguments) == "applied"
    # The owner turns memory off. Runs fail and write entries: no pass is scheduled for a failed
    # Run, so none ever sees memory off.
    await service.set_enabled(owner_id=OWNER, enabled=False)
    await _append(
        pool,
        session_id,
        ("user_message", "while-off: my doctor told me something private."),
        ("assistant_message", "while-off: understood."),
    )
    # The owner turns it back on: every Session starts afresh at its latest entry.
    await service.set_enabled(owner_id=OWNER, enabled=True)
    row = await _row(pool, session_id)
    assert (row["read_through"], row["context_floor"]) == (4, 4)
    # A Run succeeds. Its pass reads what followed, and shows nothing from before as context.
    await _append(
        pool,
        session_id,
        ("user_message", "after-on: please answer me in German."),
        ("assistant_message", "after-on: gerne."),
    )
    assert await session_pass.run(**arguments) == "applied"
    await _append(pool, session_id, ("user_message", "after-on: and put a table first."))
    assert await session_pass.run(**arguments) == "applied"

    first, right_after, later = model.asked
    assert "while-off" not in "".join(m["content"] for request in model.asked for m in request)
    context, newest = _split(right_after)
    assert context == "" and "after-on: please answer me in German." in newest
    # Then context works as it always did, from text written while memory was on, and no further.
    context, newest = _split(later)
    assert "after-on: please answer me in German." in context
    assert "before-the-switch" not in context
    assert "after-on: and put a table first." in newest
    assert "before-the-switch: I live in Gothenburg." in _split(first)[1]


async def test_turning_memory_on_restarts_every_session_and_only_from_off(pool: Any) -> None:
    memory, service = await _memory(pool)
    first = await _session(pool, ("user_message", "one"), ("assistant_message", "two"))
    second = await _session(pool, ("user_message", "uno"))
    await service.set_enabled(owner_id=OWNER, enabled=False)
    await _append(pool, first, ("user_message", "while-off"))
    await _append(pool, second, ("user_message", "while-off"), ("assistant_message", "while-off"))

    # On, from off: both Sessions start at their latest entries.
    await service.set_enabled(owner_id=OWNER, enabled=True)
    rows = [await _row(pool, session) for session in (first, second)]
    assert [(row["read_through"], row["context_floor"]) for row in rows] == [(3, 3), (3, 3)]

    # On, from on: nothing moves, however much the Sessions have grown since.
    await _append(pool, first, ("user_message", "after-on one"), ("assistant_message", "reply"))
    await _append(pool, second, ("user_message", "after-on two"))
    await service.set_enabled(owner_id=OWNER, enabled=True)
    rows = [await _row(pool, session) for session in (first, second)]
    assert [(row["read_through"], row["context_floor"]) for row in rows] == [(3, 3), (3, 3)]
    # Off, from on: nothing moves either.
    await service.set_enabled(owner_id=OWNER, enabled=False)
    rows = [await _row(pool, session) for session in (first, second)]
    assert [(row["read_through"], row["context_floor"]) for row in rows] == [(3, 3), (3, 3)]
    # And the next time round it starts afresh again, further on.
    await service.set_enabled(owner_id=OWNER, enabled=True)
    rows = [await _row(pool, session) for session in (first, second)]
    assert [(row["read_through"], row["context_floor"]) for row in rows] == [(5, 5), (4, 4)]


async def test_a_pass_in_flight_when_memory_goes_off_and_on_writes_nothing_and_reads_on_from_there(
    pool: Any,
) -> None:
    memory, service = await _memory(pool)
    session_id = await _session(
        pool,
        ("user_message", _weighty("one")),
        ("assistant_message", _weighty("two")),
        ("user_message", _weighty("three")),
    )

    class _TurnsMemoryOffAndOn(_Model):
        async def __call__(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
            # While the first batch is being judged, the owner turns memory off and on again.
            if not self.asked:
                await service.set_enabled(owner_id=OWNER, enabled=False)
                await service.set_enabled(owner_id=OWNER, enabled=True)
            return await super().__call__(messages, **kwargs)

    model = _TurnsMemoryOffAndOn(
        ConsolidationVerdict(
            writes=[VerdictWrite(action="add", kind="fact", body="Lives in Munich.")]
        ),
        *[ConsolidationVerdict()] * 2,
    )

    outcome = await _pass(pool, memory, service, model, batch=150).run(
        owner_id=OWNER, session_id=session_id, run_id=RUN
    )

    # The batch was read under an epoch the switch has since moved past, so its write is refused.
    # The restart moved the watermark to the Session's latest entry and the store said so when the
    # batch was applied: the pass reads on from there, so what the restart skipped is never asked
    # about, and nothing read under the stale epoch is read again.
    assert outcome == "applied"
    assert len(model.asked) == 1
    assert not await memory.session_records(
        owner_id=OWNER, session_id=session_id, origin_kind="session_pass"
    )
    row = await _row(pool, session_id)
    assert (row["read_through"], row["context_floor"], row["holder"]) == (3, 3, None)


async def test_a_clear_while_the_model_thinks_leaves_nothing_for_its_batch_to_write_back(
    pool: Any,
) -> None:
    memory, service = await _memory(pool)
    await memory.remember(
        owner_id=OWNER,
        kind="preference",
        body="The owner wants conclusions kept short.",
        provenance=MemoryProvenance(origin_kind="management", origin_id="settings"),
        idempotency_key="older",
    )
    session_id = await _session(
        pool, ("user_message", "I moved to Munich."), ("assistant_message", "Noted.")
    )

    class _ClearsMemory(_Model):
        async def __call__(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
            await service.clear(owner_id=OWNER)  # the owner clears their profile mid-thought
            return await super().__call__(messages, **kwargs)

    model = _ClearsMemory(
        ConsolidationVerdict(
            writes=[VerdictWrite(action="add", kind="fact", body="Lives in Munich.")]
        )
    )

    outcome = await _pass(pool, memory, service, model).run(
        owner_id=OWNER, session_id=session_id, run_id=RUN
    )

    # The profile the owner cleared stays cleared: the batch was read before the clear, so its
    # write is refused. The watermark moved past the batch all the same.
    assert outcome == "applied"
    assert await memory.count_active(owner_id=OWNER) == 0
    assert (await _row(pool, session_id))["read_through"] == 2


async def test_a_short_reply_is_searched_by_what_it_answers_and_finds_the_related_memory(
    pool: Any,
) -> None:
    memory, service = await _memory(pool)
    await memory.remember(
        owner_id=OWNER,
        kind="preference",
        body="The owner wants conclusions kept short.",
        provenance=MemoryProvenance(origin_kind="management", origin_id="settings"),
        idempotency_key="older",
    )
    session_id = await _session(
        pool,
        ("user_message", "Please keep your conclusions short."),
        ("assistant_message", "Understood, shorter conclusions from now on."),
    )
    model = _Model(ConsolidationVerdict(), ConsolidationVerdict())
    session_pass = _pass(pool, memory, service, model)
    arguments = {"owner_id": OWNER, "session_id": session_id, "run_id": RUN}
    assert await session_pass.run(**arguments) == "applied"

    # Said on its own, the reply names nothing the memory is about.
    assert not await memory.search(owner_id=OWNER, query="yes, exactly that.", limit=20)
    await _append(pool, session_id, ("user_message", "yes, exactly that."))
    assert await session_pass.run(**arguments) == "applied"

    # Searched with what it answers, it finds the memory, and the model is shown it to compare.
    asked = model.asked[1][1]["content"]
    assert "[user] yes, exactly that." in asked
    assert "The owner wants conclusions kept short." in asked


async def test_the_owners_words_reach_the_model_whole_however_long_and_assistant_context_is_cut(
    pool: Any,
) -> None:
    memory, service = await _memory(pool)
    context_user = "CTX-USER-HEAD " + "u" * 3_000 + " CTX-USER-TAIL"
    context_assistant = "CTX-ASSISTANT-HEAD " + "a" * 3_000 + " CTX-ASSISTANT-TAIL"
    huge_user = "NEW-HEAD " + "n" * 25_000 + " NEW-MIDDLE " + "n" * 25_000 + " NEW-TAIL"
    session_id = await _session(
        pool,
        ("user_message", context_user),
        ("assistant_message", context_assistant),
    )
    model = _Model(*[ConsolidationVerdict()] * 3)
    # 5,000 tokens is about 20,000 characters: far below the 50,000-character turn below.
    session_pass = _pass(pool, memory, service, model, batch=5_000)
    arguments = {"owner_id": OWNER, "session_id": session_id, "run_id": RUN}

    # The earlier rounds are read first, so that they are past the watermark: context only.
    assert await session_pass.run(**arguments) == "applied"
    assert len(model.asked) == 1
    await _append(
        pool,
        session_id,
        ("user_message", huge_user),
        ("assistant_message", "Understood, noted."),
    )
    model.asked.clear()

    assert await session_pass.run(**arguments) == "applied"

    # The turn is larger than a batch: it is read alone, never split, and the reply after it
    # is the next call.
    assert len(model.asked) == 2
    huge_call, reply_call = (_split(request) for request in model.asked)
    contents = [request[1]["content"] for request in model.asked]
    for content in contents:
        # A user turn in the context is whole, in every call that shows it...
        assert f"[user] {context_user}" in content
        # ...and an assistant turn there is its first 500 characters under an ellipsis.
        assert f"[assistant] {context_assistant[:500]}…" in content
        assert "CTX-ASSISTANT-TAIL" not in content
    # The huge user turn is whole in the one call that judges it, and in no other newest part.
    assert f"[user] {huge_user}" in huge_call[1]
    assert all(marker in huge_call[1] for marker in ("NEW-HEAD", "NEW-MIDDLE", "NEW-TAIL"))
    assert "NEW-HEAD" not in reply_call[1] and "Understood, noted." in reply_call[1]
    assert "NEW-HEAD" not in huge_call[0]
    # Shown as context to the call after it, it is still whole.
    assert f"[user] {huge_user}" in reply_call[0]
    row = await _row(pool, session_id)
    assert (row["read_through"], row["plan_writes"], row["holder"]) == (4, None, None)


# ---------------------------------------------------------------------------------------- sweep


async def _answered(
    pool: Any,
    session_id: str,
    *statuses: str,
    ages: tuple[float, ...] = (),
    owner: str = OWNER,
) -> list[str]:
    """One Answer Run of the Session per status, routing included; a terminal Run finished
    ``ages[i]`` seconds ago (by default, in the order given, a second apart). Returns the Run ids."""
    run_ids: list[str] = []
    async with pool.acquire() as conn:
        for index, status in enumerate(statuses):
            run_id = uuid.uuid4()
            age = ages[index] if index < len(ages) else float(len(statuses) - index)
            await conn.execute(
                """
                INSERT INTO dlightrag_runs (
                    owner_id, run_id, run_kind, lane, submitted_by, access_scope_kind,
                    submission_key, prepared_input_json, accepted_input_json,
                    request_fingerprint, status, result_json, error_kind, finished_at,
                    retention_seconds
                )
                VALUES ($1, $2, 'answer', 'query', $1, 'owner', $3,
                    CASE WHEN $4 IN ('queued', 'running') THEN '{"query": "q"}'::jsonb END,
                    '{}'::jsonb, 'fingerprint', $4,
                    CASE WHEN $4 = 'succeeded' THEN '{"answer": "x"}'::jsonb END,
                    CASE WHEN $4 = 'failed' THEN 'run_failed' END,
                    CASE WHEN $4 IN ('succeeded', 'failed', 'cancelled')
                         THEN NOW() - make_interval(secs => $5::float8) END,
                    31536000)
                """,
                owner,
                run_id,
                f"{run_id}",
                status,
                age,
            )
            await conn.execute(
                """
                INSERT INTO dlightrag_answer_run_routing (
                    owner_id, run_id, requested_mode, valid_modes, context_policy_revision,
                    agent_session_id, agent_lane_id
                )
                VALUES ($1, $2, 'auto', ARRAY['fast', 'research']::text[], 'revision', $3, 'default')
                """,
                owner,
                run_id,
                uuid.UUID(session_id),
            )
            run_ids.append(str(run_id))
    return run_ids


async def _progress(
    pool: Any,
    session_id: str,
    *,
    read_through: int = 0,
    failures: int = 0,
    retry_in: float | None = None,
    owner: str = OWNER,
) -> None:
    """A progress row as an earlier pass left it; ``retry_in`` seconds from now (negative: past)."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO dlightrag_session_memory_progress
                (owner_id, session_id, read_through, failure_count, retry_after)
            VALUES ($1, $2, $3, $4,
                    CASE WHEN $5::float8 IS NULL THEN NULL
                         ELSE NOW() + make_interval(secs => $5::float8) END)
            """,
            owner,
            uuid.UUID(session_id),
            read_through,
            failures,
            retry_in,
        )


async def _seconds_until_retry(pool: Any, session_id: str) -> float:
    async with pool.acquire() as conn:
        return float(
            await conn.fetchval(
                "SELECT EXTRACT(EPOCH FROM (retry_after - NOW())) "
                "FROM dlightrag_session_memory_progress WHERE owner_id = $1 AND session_id = $2",
                OWNER,
                uuid.UUID(session_id),
            )
        )


class _FailingModel(_Model):
    """A model that is down: every request raises."""

    async def __call__(self, messages: list[dict[str, Any]], **_kwargs: Any) -> str:
        self.asked.append(messages)
        raise RuntimeError("provider down")


async def _caught_up(pool: Any, session_id: str, *, read_through: int) -> None:
    async with asyncio.timeout(10):
        while True:
            row = await _row(pool, session_id)
            if row is not None and row["read_through"] == read_through and row["holder"] is None:
                return
            await asyncio.sleep(0.02)


async def test_a_pass_lost_with_the_process_is_found_by_the_sweep_and_read(pool: Any) -> None:
    memory, service = await _memory(pool)
    session_id = await _session(
        pool,
        ("user_message", "以后都用中文回答我。"),
        ("tool_result", "a long search result the pass must never read"),
        ("assistant_message", "好的。"),
    )
    older, latest = await _answered(pool, session_id, "succeeded", "succeeded", ages=(3600, 5))
    store = PGRunStore(pool=pool)

    # The Run succeeded and the process died before its pass ran: there is no progress row at all,
    # and the tables alone say that the tail was never read, and which Run it belongs to.
    assert await _row(pool, session_id) is None
    assert await store.pending_session_memory(limit=20) == (
        PendingSession(owner_id=OWNER, session_id=session_id, run_id=latest),
    )

    verdict = ConsolidationVerdict(
        writes=[
            VerdictWrite(action="add", kind="preference", body="The owner prefers Chinese answers.")
        ]
    )
    model = _Model(verdict)
    runner = SessionMemoryPassRunner(
        session_pass=_pass(pool, memory, service, model), store=PGRunStore(pool=pool)
    )
    assert await runner.sweep() == 1
    await _caught_up(pool, session_id, read_through=3)
    await runner.aclose()

    changes = await memory.run_changes(owner_id=OWNER, run_id=latest)
    assert [(c.record.body, c.action) for c in changes] == [
        ("The owner prefers Chinese answers.", "add")
    ]
    assert await memory.run_changes(owner_id=OWNER, run_id=older) == ()
    assert "long search result" not in model.asked[0][1]["content"]
    # Caught up, it is not offered again.
    assert await store.pending_session_memory(limit=20) == ()
    assert len(model.asked) == 1


async def test_a_restart_that_cancels_a_pass_leaves_its_session_for_the_next_process(
    pool: Any,
) -> None:
    memory, service = await _memory(pool)
    session_id = await _session(
        pool, ("user_message", "Please always answer in German."), ("assistant_message", "Ja.")
    )
    (run_id,) = await _answered(pool, session_id, "succeeded")

    class _Hangs(_Model):
        async def __call__(self, messages: list[dict[str, Any]], **_kwargs: Any) -> str:
            self.asked.append(messages)
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    first_process = SessionMemoryPassRunner(
        session_pass=_pass(pool, memory, service, _Hangs(), holder="first"),
        store=PGRunStore(pool=pool),
    )
    first_process.run_settled(
        SettledRun(owner_id=OWNER, run_id=run_id, run_kind="answer", status="succeeded")
    )
    async with asyncio.timeout(10):
        while not (row := await _row(pool, session_id)) or row["holder"] != "first":
            await asyncio.sleep(0.01)
    await first_process.aclose()  # the deploy: the in-flight pass is cancelled mid-model-call

    # Nothing was read, and the cancelled pass let its claim go and counted no failure.
    row = await _row(pool, session_id)
    assert (row["read_through"], row["holder"], row["failure_count"]) == (0, None, 0)
    store = PGRunStore(pool=pool)
    assert [p.run_id for p in await store.pending_session_memory(limit=20)] == [run_id]

    second_process = SessionMemoryPassRunner(
        session_pass=_pass(
            pool,
            memory,
            service,
            _Model(
                ConsolidationVerdict(
                    writes=[
                        VerdictWrite(
                            action="add", kind="preference", body="The owner prefers German."
                        )
                    ]
                )
            ),
            holder="second",
        ),
        store=PGRunStore(pool=pool),
    )
    assert await second_process.sweep() == 1
    await _caught_up(pool, session_id, read_through=2)
    await second_process.aclose()

    (record,) = await memory.session_records(
        owner_id=OWNER, session_id=session_id, origin_kind="session_pass"
    )
    assert (record.body, record.provenance.run_id) == ("The owner prefers German.", run_id)


async def test_a_sweep_offers_only_the_sessions_a_pass_should_read(pool: Any) -> None:
    _memory_store, service = await _memory(pool)
    store = PGRunStore(pool=pool)
    other = "owner-2"
    await service.set_enabled(owner_id=other, enabled=False)
    labels: dict[str, str] = {}

    async def scenario(
        label: str,
        *entries: tuple[str, str] | tuple[str, str, dict[str, Any]],
        runs: tuple[str, ...] = ("succeeded",),
        owner: str = OWNER,
        read_through: int | None = None,
        failures: int = 0,
        retry_in: float | None = None,
        claim: float | None = None,
    ) -> None:
        session_id = await _session(pool, *entries, owner=owner)
        labels[session_id] = label
        await _answered(pool, session_id, *runs, owner=owner)
        if read_through is not None:
            await _progress(
                pool,
                session_id,
                read_through=read_through,
                failures=failures,
                retry_in=retry_in,
                owner=owner,
            )
        if claim is not None:
            await store.claim_session_memory(
                owner_id=owner, session_id=session_id, holder="someone", lease_seconds=claim
            )

    text = ("user_message", "I live in Gothenburg."), ("assistant_message", "Noted.")
    await scenario("behind: no pass ever ran", *text)
    await scenario(
        "behind: past the watermark", *text, ("user_message", "And I like tea."), read_through=2
    )
    await scenario("behind: its wait is over", *text, read_through=0, failures=2, retry_in=-1)
    await scenario("behind: its claim lapsed", *text, claim=0)
    await scenario(
        "behind: a failed Run after a succeeded one", *text, runs=("succeeded", "failed")
    )
    await scenario("behind: another owner, memory on", *text, owner="owner-3")

    await scenario("caught up", *text, read_through=2)
    await scenario(
        "only tool results past the watermark",
        *text,
        ("tool_result", "a search result"),
        ("tool_result", "another"),
        read_through=2,
    )
    await scenario(
        "only an assistant turn with no text past the watermark",
        *text,
        ("assistant_message", "  \n "),
        read_through=2,
    )
    await scenario(
        "only a Child's notification past the watermark",
        *text,
        ("user_message", "Child result", _NOTIFICATION),
        read_through=2,
    )
    await scenario("a Run is running", *text, runs=("succeeded", "running"))
    await scenario("a Run is queued", *text, runs=("succeeded", "queued"))
    await scenario("no Run succeeded", *text, runs=("failed", "cancelled"))
    await scenario("no Run at all (a Child's Session)", *text, runs=())
    await scenario("its owner's memory is off", *text, owner=other)
    await scenario("a pass holds it", *text, claim=300)
    await scenario("it is backing off", *text, read_through=0, failures=2, retry_in=3600)

    offered = {labels[p.session_id] for p in await store.pending_session_memory(limit=100)}

    assert offered == {label for label in labels.values() if label.startswith("behind")}


async def test_a_pass_is_attributed_to_the_sessions_newest_succeeded_run(pool: Any) -> None:
    session_id = await _session(pool, ("user_message", "hello there"))
    run_ids = await _answered(
        pool,
        session_id,
        "succeeded",
        "succeeded",
        "failed",
        "cancelled",
        ages=(7200, 60, 30, 10),
    )

    (pending,) = await PGRunStore(pool=pool).pending_session_memory(limit=20)

    # The newest that succeeded, not the newest: a Run that failed or was cancelled later does not
    # count, and neither does an older one that succeeded.
    assert pending.run_id == run_ids[1]


async def test_a_sweep_is_bounded_and_takes_the_newest_conversations_first(pool: Any) -> None:
    sessions: list[str] = []
    for index in range(25):
        session_id = await _session(pool, ("user_message", f"conversation {index}"))
        await _answered(pool, session_id, "succeeded")
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE dlightrag_agent_sessions SET updated_at = NOW() - make_interval(mins => $3) "
                "WHERE owner_id = $1 AND session_id = $2",
                OWNER,
                uuid.UUID(session_id),
                index,
            )
        sessions.append(session_id)
    store = PGRunStore(pool=pool)

    assert [p.session_id for p in await store.pending_session_memory(limit=20)] == sessions[:20]
    assert [p.session_id for p in await store.pending_session_memory(limit=3)] == sessions[:3]


async def test_a_failed_pass_backs_its_session_off_and_a_success_forgives_it(pool: Any) -> None:
    memory, service = await _memory(pool)
    session_id = await _session(
        pool, ("user_message", "Always cite the filing."), ("assistant_message", "Understood.")
    )
    (run_id,) = await _answered(pool, session_id, "succeeded")
    store = PGRunStore(pool=pool)
    arguments = {"owner_id": OWNER, "session_id": session_id, "run_id": run_id}
    failing = _pass(pool, memory, service, _FailingModel())

    # The first failure: counted, and the Session is not offered for ten minutes.
    assert await failing.run(**arguments) == "model_failed"
    row = await _row(pool, session_id)
    assert (row["failure_count"], row["read_through"], row["holder"]) == (1, 0, None)
    assert 595 <= await _seconds_until_retry(pool, session_id) <= 600
    assert await store.pending_session_memory(limit=20) == ()

    # Its wait over, it is offered again; failing again doubles the wait.
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE dlightrag_session_memory_progress SET retry_after = NOW() - INTERVAL '1 second'"
        )
    assert [p.run_id for p in await store.pending_session_memory(limit=20)] == [run_id]
    assert await failing.run(**arguments) == "model_failed"
    assert (await _row(pool, session_id))["failure_count"] == 2
    assert 1195 <= await _seconds_until_retry(pool, session_id) <= 1200

    # A pass that moves the watermark forgives all of it.
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE dlightrag_session_memory_progress SET retry_after = NOW() - INTERVAL '1 second'"
        )
    working = _pass(pool, memory, service, _Model(ConsolidationVerdict()))
    assert await working.run(**arguments) == "applied"
    row = await _row(pool, session_id)
    assert (row["failure_count"], row["retry_after"], row["read_through"]) == (0, None, 2)
    assert await store.pending_session_memory(limit=20) == ()


async def test_the_wait_doubles_with_each_failure_and_stops_at_six_hours(pool: Any) -> None:
    session_id = await _session(pool, ("user_message", "hello"))
    store = PGRunStore(pool=pool)
    waits: list[float] = []

    async def fail_once() -> None:
        await store.claim_session_memory(
            owner_id=OWNER, session_id=session_id, holder="a", lease_seconds=60
        )
        await store.back_off_session_memory(
            owner_id=OWNER, session_id=session_id, holder="a", first_seconds=600, max_seconds=21_600
        )
        await store.release_session_memory(owner_id=OWNER, session_id=session_id, holder="a")
        waits.append(await _seconds_until_retry(pool, session_id))

    for _ in range(8):
        await fail_once()

    # 10 min, 20, 40, 80, 160, 320 (past the ceiling), so 6 h from then on.
    expected = [600, 1200, 2400, 4800, 9600, 19_200, 21_600, 21_600]
    assert [round(wait / 60) for wait in waits] == [round(wait / 60) for wait in expected]
    # However long it has failed, the arithmetic neither overflows nor passes the ceiling.
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE dlightrag_session_memory_progress SET failure_count = 2000000000"
        )
    await fail_once()
    assert 21_590 <= waits[-1] <= 21_600
    assert (await _row(pool, session_id))["failure_count"] == 2_000_000_001


async def test_a_failure_is_counted_only_by_the_holder_of_the_claim(pool: Any) -> None:
    session_id = await _session(pool, ("user_message", "hello"))
    store = PGRunStore(pool=pool)
    await store.claim_session_memory(
        owner_id=OWNER, session_id=session_id, holder="a", lease_seconds=60
    )

    async def back_off(holder: str) -> None:
        await store.back_off_session_memory(
            owner_id=OWNER,
            session_id=session_id,
            holder=holder,
            first_seconds=600,
            max_seconds=21_600,
        )

    await back_off("b")  # a pass whose claim was taken over, or never held, records nothing
    assert (await _row(pool, session_id))["failure_count"] == 0
    await back_off("a")
    assert (await _row(pool, session_id))["failure_count"] == 1
    await store.release_session_memory(owner_id=OWNER, session_id=session_id, holder="a")
    await back_off("a")  # and neither does one that has let go of it
    assert (await _row(pool, session_id))["failure_count"] == 1


async def test_a_pass_a_run_schedules_ignores_the_wait_a_sweep_keeps(pool: Any) -> None:
    memory, service = await _memory(pool)
    session_id = await _session(
        pool, ("user_message", "I live in Gothenburg."), ("assistant_message", "Noted.")
    )
    (run_id,) = await _answered(pool, session_id, "succeeded")
    await _progress(pool, session_id, failures=3, retry_in=3600)
    store = PGRunStore(pool=pool)
    model = _Model(ConsolidationVerdict())

    # A sweep leaves the Session alone for the hour; a new Run is new evidence, and its pass runs.
    assert await store.pending_session_memory(limit=20) == ()
    runner = SessionMemoryPassRunner(
        session_pass=_pass(pool, memory, service, model), store=PGRunStore(pool=pool)
    )
    runner.run_settled(
        SettledRun(owner_id=OWNER, run_id=run_id, run_kind="answer", status="succeeded")
    )
    await _caught_up(pool, session_id, read_through=2)
    await runner.aclose()

    row = await _row(pool, session_id)
    assert (row["failure_count"], row["retry_after"]) == (0, None)
    assert len(model.asked) == 1


_SEED_SESSIONS = """
INSERT INTO dlightrag_agent_sessions (owner_id, session_id, lease_run_id, fencing_epoch,
                                         last_sequence)
SELECT $1, md5($3 || n)::uuid, gen_random_uuid(), 1, $4 FROM generate_series(1, $2) AS n
"""
_SEED_ENTRIES = """
INSERT INTO dlightrag_agent_session_entries
    (owner_id, session_id, sequence, entry_id, entry_type, schema_version, timestamp, payload_json)
SELECT $1, md5($3 || n)::uuid, sequence, gen_random_uuid(),
       CASE WHEN sequence = 1 THEN 'user_message' ELSE 'assistant_message' END, 1, NOW(),
       jsonb_build_object('content', 'text ' || n)
FROM generate_series(1, $2) AS n, generate_series(1, $4) AS sequence
"""
_SEED_PROGRESS = """
INSERT INTO dlightrag_session_memory_progress (owner_id, session_id, read_through)
SELECT $1, md5($3 || n)::uuid, $4 FROM generate_series(1, $2) AS n
"""
_SEED_RUNS = """
INSERT INTO dlightrag_runs (
    owner_id, run_id, run_kind, lane, submitted_by, access_scope_kind, submission_key,
    prepared_input_json, accepted_input_json, request_fingerprint, status, result_json,
    finished_at, retention_seconds
)
SELECT $1, md5('run-' || $3 || n)::uuid, 'answer', 'query', $1, 'owner', $3 || n,
       NULL, '{}'::jsonb, 'fingerprint', 'succeeded', '{"answer": "x"}'::jsonb,
       NOW() - make_interval(secs => n), 31536000
FROM generate_series(1, $2) AS n
"""
_SEED_ROUTING = """
INSERT INTO dlightrag_answer_run_routing (
    owner_id, run_id, requested_mode, valid_modes, context_policy_revision, agent_session_id,
    agent_lane_id
)
SELECT $1, md5('run-' || $3 || n)::uuid, 'auto', ARRAY['fast', 'research']::text[], 'revision',
       md5($3 || n)::uuid, 'default'
FROM generate_series(1, $2) AS n
"""


async def test_the_sweep_probes_indexes_only_for_the_sessions_that_are_behind(pool: Any) -> None:
    async with pool.acquire() as conn:
        # Thousands of conversations a pass has caught up with, each with a Run and its entries.
        for table_sql in (_SEED_SESSIONS, _SEED_ENTRIES, _SEED_PROGRESS):
            await conn.execute(table_sql, OWNER, 3_000, "settled-", 2)
        await conn.execute(_SEED_RUNS, OWNER, 3_000, "settled-")
        await conn.execute(_SEED_ROUTING, OWNER, 3_000, "settled-")
        # Thousands of Child Sessions: rows of the same table, never given a progress row or routing.
        await conn.execute(_SEED_SESSIONS, OWNER, 3_000, "child-", 5)
        await conn.execute(_SEED_ENTRIES, OWNER, 3_000, "child-", 5)
        # A few conversations whose pass was lost.
        await conn.execute(_SEED_SESSIONS, OWNER, 4, "lost-", 2)
        await conn.execute(_SEED_ENTRIES, OWNER, 4, "lost-", 2)
        await conn.execute(_SEED_RUNS, OWNER, 4, "lost-")
        await conn.execute(_SEED_ROUTING, OWNER, 4, "lost-")
        # As autovacuum keeps them, so the planner chooses by what the tables hold.
        for table in (
            "dlightrag_agent_sessions",
            "dlightrag_agent_session_entries",
            "dlightrag_session_memory_progress",
            "dlightrag_runs",
            "dlightrag_answer_run_routing",
        ):
            await conn.execute(f"VACUUM (ANALYZE) {table}")

        lines = await conn.fetch(f"EXPLAIN (ANALYZE, COSTS OFF) {_SWEEP}", 20)  # noqa: S608
        found = await conn.fetch(_SWEEP, 20)
    plan = "\n".join(str(line["QUERY PLAN"]) for line in lines)

    assert len(found) == 4
    # What it probes, it probes by index: a Session's Runs by the routing index, their rows and the
    # Session's entries by primary key. None of the big tables is read from end to end.
    assert "idx_dlightrag_answer_run_routing_session" in plan
    assert "dlightrag_agent_session_entries_pkey" in plan
    for table in (
        "dlightrag_agent_session_entries",
        "dlightrag_runs",
        "dlightrag_answer_run_routing",
    ):
        assert f"Seq Scan on {table}" not in plan, plan
    # And only for Sessions that are behind: the entries are probed once per Session that has a
    # succeeded Run and is behind, not once per Session in the database.
    entry_probes = [
        int(loops)
        for loops in re.findall(
            r"dlightrag_agent_session_entries_pkey.*?loops=(\d+)", plan, flags=re.DOTALL
        )
    ]
    assert entry_probes and max(entry_probes) <= 4, plan


async def test_the_owners_words_a_verdict_quotes_are_checked_stored_recalled_and_shown_to_the_next_pass(
    pool: Any,
) -> None:
    memory, service = await _memory(pool)
    session_id = await _session(
        pool,
        ("user_message", "以后图表都用深色主题，别用饼图。"),
        ("tool_result", "a long search result: 我会用深色主题"),
        ("assistant_message", "好的，我会用深色主题。"),
    )
    arguments = {"owner_id": OWNER, "session_id": session_id, "run_id": RUN}
    first = ConsolidationVerdict(
        writes=[
            VerdictWrite(
                action="add",
                kind="preference",
                body="The owner prefers charts in a dark theme.",
                quote="以后图表都用深色主题",
            ),
            VerdictWrite(
                action="add",
                kind="preference",
                body="The owner dislikes pie charts.",
                quote="别用饼图",
            ),
            # Words of the Agent's, and of nobody's: the memories are written, without them.
            VerdictWrite(
                action="add",
                kind="experience",
                body="The Agent answers with a dark theme.",
                quote="我会用深色主题",
            ),
            VerdictWrite(
                action="add",
                kind="fact",
                body="The owner builds dashboards.",
                quote="I build dashboards all day",
            ),
        ]
    )
    model = _Model(first)
    session_pass = _pass(pool, memory, service, model)

    assert await session_pass.run(**arguments) == "applied"

    records = await memory.session_records(
        owner_id=OWNER, session_id=session_id, origin_kind="session_pass"
    )
    assert {record.body: record.quote for record in records} == {
        "The owner prefers charts in a dark theme.": "以后图表都用深色主题",
        "The owner dislikes pie charts.": "别用饼图",
        "The Agent answers with a dark theme.": None,
        "The owner builds dashboards.": None,
    }
    # The Run's notice and the recall the next Run is given carry the words with the memory.
    changes = await memory.run_changes(owner_id=OWNER, run_id=RUN)
    assert {c.record.body: c.record.quote for c in changes}["The owner dislikes pie charts."] == (
        "别用饼图"
    )
    block = render_auto_recall(await memory.recall(owner_id=OWNER, query="anything at all"))
    assert '  owner said: "以后图表都用深色主题"' in block.splitlines()
    assert '  owner said: "别用饼图"' in block.splitlines()
    assert "我会用深色主题" not in block

    # The next pass is shown the words under the memory it may refine, and may quote only the
    # turn it is judging: the words of the first turn, now context, are no longer a quote. It
    # names the memory by the number it was listed with, which it reads off the request.
    await _append(pool, session_id, ("user_message", "饼图也可以，只要别用三维的。"))

    def pie_number(request: str) -> int:
        listed = re.search(
            r"^\[(\d+)\] \(preference[^)]*\) The owner dislikes pie charts\.$", request, re.M
        )
        assert listed is not None
        return int(listed.group(1))

    class _Reads(_Model):
        async def __call__(self, messages: list[dict[str, Any]], **_kwargs: Any) -> str:
            self.asked.append(messages)
            return ConsolidationVerdict(
                writes=[
                    VerdictWrite(
                        action="update",
                        kind="preference",
                        body="The owner accepts pie charts when they are not three-dimensional.",
                        targets=[pie_number(messages[1]["content"])],
                        quote="只要别用三维的",
                    ),
                    VerdictWrite(
                        action="add",
                        kind="preference",
                        body="The owner wants a dark theme everywhere.",
                        quote="以后图表都用深色主题",
                    ),
                ]
            ).model_dump_json()

    reads = _Reads()
    assert await _pass(pool, memory, service, reads).run(**arguments) == "applied"

    asked = reads.asked[0][1]["content"]
    assert "Newest part" in asked and '  owner said: "别用饼图"' in asked
    refined = await memory.session_records(
        owner_id=OWNER, session_id=session_id, origin_kind="session_pass"
    )
    quotes = {record.body: record.quote for record in refined}
    assert (
        quotes["The owner accepts pie charts when they are not three-dimensional."]
        == "只要别用三维的"
    )
    assert quotes["The owner wants a dark theme everywhere."] is None
    assert "The owner dislikes pie charts." not in quotes  # superseded by the refinement
