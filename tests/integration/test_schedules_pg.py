# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Integration tests for Schedules and their Fires on PostgreSQL 18 (ADR 0103).

The queue is the database, so the contract worth testing is the one only a database can keep: that
two processes claiming at the same moment write one Fire for a slot, that a Fire's ending and its
Schedule's pause are one transaction, that Fires trim and cascade, and that the spend the Run
Inspector states for a Run is the spend a Schedule is held to. Every test runs in a throwaway
database; skipped if PostgreSQL is unavailable.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest

from dlightrag.adapters.postgres.runtime.run_store import PGRunStore
from dlightrag.adapters.postgres.runtime.schedules import PGScheduleStore
from dlightrag.application.access.principal import UserContext, owner_id_from_user
from dlightrag.application.schedules.cadence import Cadence, CronExpression
from dlightrag.application.schedules.errors import ScheduleLimitError
from dlightrag.application.schedules.model import (
    FireState,
    Schedule,
    ScheduleState,
    Target,
    plan_due,
)
from tests.support.pg import PG_CONN_KWARGS, drop_database, skip_without_postgres

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

NOW = datetime(2026, 12, 7, 12, 0, 20, tzinfo=UTC)
USER = UserContext(
    user_id="dave", auth_mode="jwt", claims={"iss": "https://issuer", "groups": ["a"]}
)
OWNER = owner_id_from_user(USER)
OTHER = UserContext(user_id="erin", auth_mode="jwt", claims={"iss": "https://issuer"})


@pytest.fixture
async def db_name() -> AsyncIterator[str]:
    await skip_without_postgres()
    name = f"dlightrag_schedules_{uuid.uuid4().hex[:12]}"
    admin = await asyncpg.connect(**PG_CONN_KWARGS)
    try:
        await admin.execute(f'CREATE DATABASE "{name}"')
    finally:
        await admin.close()
    try:
        yield name
    finally:
        await drop_database(name)


@pytest.fixture
async def pool(db_name: str) -> AsyncIterator[Any]:
    created = await asyncpg.create_pool(
        **{**PG_CONN_KWARGS, "database": db_name}, min_size=1, max_size=12
    )
    try:
        yield created
    finally:
        await created.close()


@pytest.fixture
async def store(pool: Any) -> PGScheduleStore:
    await PGRunStore(pool=pool).initialize()
    return PGScheduleStore(pool=pool)


def _schedule(
    *, cron: str = "0 * * * *", owner: str = OWNER, now: datetime = NOW - timedelta(hours=2)
) -> Schedule:
    return Schedule.create(
        owner_id=owner,
        principal=USER,
        name="Weekly review",
        prompt="Summarize.",
        mode="research",
        workspaces=["alpha", "beta"],
        target=Target.NEW_CONVERSATION,
        cadence=Cadence("UTC", CronExpression.parse(cron), None),
        max_tokens_per_day=1_000,
        now=now,
    )


def _plan(now: datetime = NOW) -> Any:
    return lambda schedule: plan_due(schedule.cadence, schedule.next_fire_at or now, now)


async def _fires(pool: Any, schedule: Schedule) -> list[Any]:
    async with pool.acquire() as conn:
        return await conn.fetch(
            "SELECT * FROM dlightrag_schedule_fires WHERE schedule_id = $1::uuid ORDER BY slot",
            schedule.schedule_id,
        )


async def test_an_owner_keeps_one_identity_that_the_next_human_turn_replaces(
    store: PGScheduleStore, pool: Any
) -> None:
    assert await store.principal_of(OWNER) is None

    await store.remember_principal(OWNER, USER)
    await store.remember_principal("someone-else", OTHER)
    assert await store.principal_of(OWNER) == USER

    later = UserContext(user_id="dave", auth_mode="jwt", claims={"iss": "https://issuer", "x": 2})
    await store.remember_principal(OWNER, later)

    assert await store.principal_of(OWNER) == later
    assert await store.principal_of("someone-else") == OTHER

    # An identity no turn has refreshed for a day is no one's latest turn: the Agent acts under none.
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE dlightrag_owner_principals SET updated_at = NOW() - INTERVAL '2 days' "
            "WHERE owner_id = $1",
            OWNER,
        )
    assert await store.principal_of(OWNER) is None
    async with pool.acquire() as conn:
        rows = await conn.fetchval(
            "SELECT count(*) FROM dlightrag_owner_principals WHERE owner_id = $1", OWNER
        )
    assert rows == 1


async def test_keeping_an_identity_drops_the_records_nobody_can_use_any_more(
    store: PGScheduleStore, pool: Any
) -> None:
    other = owner_id_from_user(OTHER)
    await store.remember_principal(other, OTHER)
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE dlightrag_owner_principals SET updated_at = NOW() - INTERVAL '2 days'"
        )

    await store.remember_principal(OWNER, USER)

    async with pool.acquire() as conn:
        kept = await conn.fetch("SELECT owner_id FROM dlightrag_owner_principals")
    assert [row["owner_id"] for row in kept] == [OWNER]


async def test_a_schedule_says_who_last_wrote_it_and_only_two_values_hold(
    store: PGScheduleStore, pool: Any
) -> None:
    made = await store.create(replace(_schedule(), changed_by="agent"), max_active=20)
    assert made.changed_by == "agent"

    paused = await store.pause(OWNER, made.schedule_id, reason=None)
    assert paused is not None and paused.changed_by == "agent"
    edited = await store.edit(replace(paused, changed_by="owner"), reschedule=False)
    assert edited is not None and edited.changed_by == "owner"
    await store.pause(OWNER, made.schedule_id, reason=None)
    resumed = await store.resume(
        OWNER, made.schedule_id, principal=USER, max_active=20, changed_by="agent"
    )
    assert resumed is not None and resumed.changed_by == "agent"
    assert (await store.get(OWNER, made.schedule_id)) == resumed

    async with pool.acquire() as conn:
        with pytest.raises(asyncpg.CheckViolationError, match="changed_by_check"):
            await conn.execute(
                "UPDATE dlightrag_schedules SET changed_by = 'script' WHERE schedule_id = $1::uuid",
                made.schedule_id,
            )


async def test_a_schedule_round_trips_with_the_identity_it_acts_as(store: PGScheduleStore) -> None:
    created = await store.create(_schedule(), max_active=20)

    loaded = await store.get(OWNER, created.schedule_id)

    assert loaded == created
    assert loaded is not None
    assert loaded.principal == USER
    assert loaded.workspaces == ("alpha", "beta")
    assert loaded.cadence.cron == "0 * * * *"
    assert await store.get(owner_id_from_user(OTHER), created.schedule_id) is None
    assert await store.list(OWNER) == (created,)
    assert await store.list(owner_id_from_user(OTHER)) == ()


async def test_the_stores_now_is_the_databases_clock_in_utc(
    pool: Any, store: PGScheduleStore
) -> None:
    async with pool.acquire() as conn:
        database_now = await conn.fetchval("SELECT now()")

    stored = await store.now()

    assert stored.utcoffset() == timedelta(0)
    assert abs(stored - database_now) < timedelta(seconds=5)


async def test_two_processes_claiming_at_once_write_one_fire_for_a_slot(
    store: PGScheduleStore, pool: Any
) -> None:
    created = await store.create(_schedule(), max_active=20)
    assert created.next_fire_at is not None and created.next_fire_at < NOW

    claims = await asyncio.gather(
        *(store.claim_due(NOW, limit=20, fires_kept=200, plan=_plan()) for _ in range(8))
    )

    started = [due for claim in claims for due in claim.due]
    rows = await _fires(pool, created)
    # Two slots were due (11:00 and 12:00): one gap, and the latest, claimed by one process only.
    assert sorted(str(row["state"]) for row in rows) == ["missed", "pending"]
    assert len(started) == 1
    assert started[0].fire.slot == datetime(2026, 12, 7, 12, 0, tzinfo=UTC)
    assert started[0].schedule.next_fire_at == datetime(2026, 12, 7, 13, 0, tzinfo=UTC)
    reloaded = await store.get(OWNER, created.schedule_id)
    assert reloaded is not None
    assert reloaded.next_fire_at == datetime(2026, 12, 7, 13, 0, tzinfo=UTC)


async def test_the_same_slot_cannot_hold_two_fires(store: PGScheduleStore) -> None:
    created = await store.create(_schedule(), max_active=20)

    first = await store.insert_manual_fire(created, NOW, fires_kept=200)
    second = await store.insert_manual_fire(created, NOW, fires_kept=200)

    assert first is not None and first.fire.state is FireState.PENDING
    assert second is None


async def test_a_single_time_completes_only_when_its_fire_started_a_run(
    store: PGScheduleStore,
) -> None:
    at = NOW - timedelta(minutes=5)
    once = Schedule.create(
        owner_id=OWNER,
        principal=USER,
        name="Once",
        prompt="Just once.",
        mode="fast",
        workspaces=["alpha"],
        target=Target.SAME_CONVERSATION,
        cadence=Cadence("UTC", None, at),
        max_tokens_per_day=1_000,
        now=at - timedelta(hours=1),
    )
    created = await store.create(once, max_active=20)

    claim = await store.claim_due(NOW, limit=20, fires_kept=200, plan=_plan())

    assert [due.fire.slot for due in claim.due] == [at]
    in_flight = await store.get(OWNER, created.schedule_id)
    assert in_flight is not None
    assert (in_flight.state, in_flight.next_fire_at) == (ScheduleState.ACTIVE, None)
    assert (await store.claim_due(NOW, limit=20, fires_kept=200, plan=_plan())).schedules == 0

    await store.finish_fire(
        OWNER,
        claim.due[0].fire.fire_id,
        expected=(FireState.PENDING,),
        state=FireState.STARTED,
        run_id=str(uuid.uuid4()),
        failure_streak=3,
    )
    done = await store.get(OWNER, created.schedule_id)
    assert done is not None
    assert (done.state, done.next_fire_at) == (ScheduleState.COMPLETED, None)


async def test_a_single_time_whose_fire_started_nothing_waits_paused_with_the_reason(
    store: PGScheduleStore,
) -> None:
    at = NOW - timedelta(minutes=5)
    once = Schedule.create(
        owner_id=OWNER,
        principal=USER,
        name="Once",
        prompt="Just once.",
        mode="fast",
        workspaces=["alpha"],
        target=Target.NEW_CONVERSATION,
        cadence=Cadence("UTC", None, at),
        max_tokens_per_day=1_000,
        now=at - timedelta(hours=1),
    )
    created = await store.create(once, max_active=20)
    claim = await store.claim_due(NOW, limit=20, fires_kept=200, plan=_plan())

    await store.finish_fire(
        OWNER,
        claim.due[0].fire.fire_id,
        expected=(FireState.PENDING,),
        state=FireState.SKIPPED_BUSY,
        reason="its conversation is already working on another run",
        failure_streak=3,
    )

    waiting = await store.get(OWNER, created.schedule_id)
    assert waiting is not None
    assert (waiting.state, waiting.next_fire_at) == (ScheduleState.PAUSED, None)
    assert waiting.pause_reason == "its conversation is already working on another run"


async def test_a_single_time_that_was_too_late_when_claimed_is_paused_at_once(
    store: PGScheduleStore,
) -> None:
    at = NOW - timedelta(hours=3)
    once = Schedule.create(
        owner_id=OWNER,
        principal=USER,
        name="Once",
        prompt="Just once.",
        mode="fast",
        workspaces=["alpha"],
        target=Target.NEW_CONVERSATION,
        cadence=Cadence("UTC", None, at),
        max_tokens_per_day=1_000,
        now=at - timedelta(hours=1),
    )
    created = await store.create(once, max_active=20)

    claim = await store.claim_due(NOW, limit=20, fires_kept=200, plan=_plan())

    assert claim.due == ()
    waiting = await store.get(OWNER, created.schedule_id)
    assert waiting is not None and waiting.state is ScheduleState.PAUSED
    assert "too late" in (waiting.pause_reason or "")


async def test_a_paused_schedule_is_not_claimed_and_resume_keeps_its_place(
    store: PGScheduleStore,
) -> None:
    created = await store.create(_schedule(), max_active=20)
    paused = await store.pause(OWNER, created.schedule_id, reason="paused by its owner")
    assert paused is not None and paused.state is ScheduleState.PAUSED
    assert paused.next_fire_at == created.next_fire_at

    assert (await store.claim_due(NOW, limit=20, fires_kept=200, plan=_plan())).schedules == 0
    assert await store.pause(OWNER, created.schedule_id, reason="again") is None

    resumed = await store.resume(OWNER, created.schedule_id, principal=OTHER, max_active=20)
    assert resumed is not None
    assert (resumed.state, resumed.pause_reason, resumed.principal) == (
        ScheduleState.ACTIVE,
        None,
        OTHER,
    )
    assert resumed.next_fire_at == created.next_fire_at
    assert (await store.claim_due(NOW, limit=20, fires_kept=200, plan=_plan())).schedules == 1


async def test_an_owner_holds_at_most_the_allowed_active_schedules(store: PGScheduleStore) -> None:
    first = await store.create(_schedule(), max_active=2)
    await store.create(_schedule(), max_active=2)
    with pytest.raises(ScheduleLimitError):
        await store.create(_schedule(), max_active=2)
    # Another owner is not counted, and a paused Schedule makes room that resume then needs back.
    await store.create(_schedule(owner=owner_id_from_user(OTHER)), max_active=2)
    await store.pause(OWNER, first.schedule_id, reason="paused by its owner")
    await store.create(_schedule(), max_active=2)
    with pytest.raises(ScheduleLimitError):
        await store.resume(OWNER, first.schedule_id, principal=USER, max_active=2)


async def test_an_edit_moves_the_next_slot_only_when_asked(store: PGScheduleStore) -> None:
    created = await store.create(_schedule(), max_active=20)
    renamed = created.edit(principal=OTHER, now=NOW, name="Renamed")

    kept = await store.edit(renamed, reschedule=False)

    assert kept is not None
    assert (kept.name, kept.principal, kept.next_fire_at) == (
        "Renamed",
        OTHER,
        created.next_fire_at,
    )
    moved = created.edit(
        principal=USER,
        now=NOW,
        cadence=Cadence("UTC", CronExpression.parse("30 9 * * *"), None),
    )
    rescheduled = await store.edit(moved, reschedule=True)
    assert rescheduled is not None
    assert rescheduled.next_fire_at == datetime(2026, 12, 8, 9, 30, tzinfo=UTC)


async def test_fires_beyond_the_kept_number_are_trimmed_and_a_delete_takes_them_all(
    store: PGScheduleStore, pool: Any
) -> None:
    created = await store.create(_schedule(), max_active=20)
    for hour in range(5):
        await store.insert_manual_fire(created, NOW - timedelta(days=1, hours=hour), fires_kept=200)

    await store.claim_due(NOW, limit=20, fires_kept=3, plan=_plan())

    slots = [row["slot"] for row in await _fires(pool, created)]
    assert len(slots) == 3
    assert slots[-1] == datetime(2026, 12, 7, 12, 0, tzinfo=UTC)

    assert await store.delete(OWNER, created.schedule_id) is True
    assert await _fires(pool, created) == []
    assert await store.delete(OWNER, created.schedule_id) is False


async def _end(
    store: PGScheduleStore,
    schedule: Schedule,
    slot: datetime,
    state: FireState,
    *,
    pause_reason: str | None = None,
) -> None:
    """Write a manual Fire for ``slot`` and end it in ``state``, as the firing rules would."""
    due = await store.insert_manual_fire(schedule, slot, fires_kept=200)
    assert due is not None
    if state is FireState.SUCCEEDED:
        await store.finish_fire(
            OWNER,
            due.fire.fire_id,
            expected=(FireState.PENDING,),
            state=FireState.STARTED,
            run_id=str(uuid.uuid4()),
            failure_streak=3,
        )
        expected = (FireState.STARTED,)
    else:
        expected = (FireState.PENDING,)
    ended = await store.finish_fire(
        OWNER,
        due.fire.fire_id,
        expected=expected,
        state=state,
        reason=f"{state} at {slot:%H:%M}",
        pause_reason=pause_reason,
        failure_streak=3,
    )
    assert ended is not None and ended.state is state


async def test_three_bad_endings_in_a_row_pause_the_schedule_and_a_success_resets_the_count(
    store: PGScheduleStore,
) -> None:
    created = await store.create(_schedule(), max_active=20)
    base = NOW - timedelta(days=1)
    for step, state in enumerate((FireState.FAILED, FireState.FAILED, FireState.SUCCEEDED)):
        await _end(store, created, base + timedelta(minutes=step), state)
    after_success = await store.get(OWNER, created.schedule_id)
    assert after_success is not None and after_success.state is ScheduleState.ACTIVE

    # A skip is neither a bad ending nor a break in the streak.
    await _end(store, created, base + timedelta(minutes=3), FireState.FAILED)
    await _end(store, created, base + timedelta(minutes=4), FireState.SKIPPED_BUSY)
    await _end(store, created, base + timedelta(minutes=5), FireState.REFUSED)
    still = await store.get(OWNER, created.schedule_id)
    assert still is not None and still.state is ScheduleState.ACTIVE
    await _end(store, created, base + timedelta(minutes=6), FireState.FAILED)

    paused = await store.get(OWNER, created.schedule_id)
    assert paused is not None
    assert paused.state is ScheduleState.PAUSED
    assert paused.pause_reason is not None
    assert "last 3 fires failed or were refused" in paused.pause_reason
    assert "failed at" in paused.pause_reason


async def test_a_refusal_the_owner_must_fix_pauses_at_once_with_its_reason(
    store: PGScheduleStore,
) -> None:
    created = await store.create(_schedule(), max_active=20)

    await _end(store, created, NOW, FireState.REFUSED, pause_reason="no readable workspace")

    paused = await store.get(OWNER, created.schedule_id)
    assert paused is not None
    assert (paused.state, paused.pause_reason) == (ScheduleState.PAUSED, "no readable workspace")


async def test_a_fire_moves_once_and_a_started_fire_keeps_its_schedules_conversation(
    store: PGScheduleStore,
) -> None:
    same = Schedule.create(
        owner_id=OWNER,
        principal=USER,
        name="Same",
        prompt="Again.",
        mode="research",
        workspaces=["alpha"],
        target=Target.SAME_CONVERSATION,
        cadence=Cadence("UTC", CronExpression.parse("0 * * * *"), None),
        max_tokens_per_day=1_000,
        now=NOW,
    )
    created = await store.create(same, max_active=20)
    due = await store.insert_manual_fire(created, NOW, fires_kept=200)
    assert due is not None
    run_id, conversation_id = str(uuid.uuid4()), str(uuid.uuid4())

    started = await store.finish_fire(
        OWNER,
        due.fire.fire_id,
        expected=(FireState.PENDING,),
        state=FireState.STARTED,
        run_id=run_id,
        conversation_id=conversation_id,
        failure_streak=3,
    )
    again = await store.finish_fire(
        OWNER,
        due.fire.fire_id,
        expected=(FireState.PENDING,),
        state=FireState.SKIPPED_BUSY,
        failure_streak=3,
    )

    assert started is not None and (started.run_id, started.conversation_id) == (
        run_id,
        conversation_id,
    )
    assert again is None
    kept = await store.get(OWNER, created.schedule_id)
    assert kept is not None and kept.conversation_id == conversation_id
    found = await store.fire_for_run(OWNER, run_id)
    assert found is not None and found.fire_id == due.fire.fire_id
    assert await store.fire_for_run(owner_id_from_user(OTHER), run_id) is None
    previous = await store.previous_fire(OWNER, created.schedule_id, NOW + timedelta(hours=1))
    assert previous is not None and previous.fire_id == due.fire.fire_id
    assert await store.previous_fire(OWNER, created.schedule_id, NOW) is None


async def test_pending_fires_are_retried_once_per_lease_and_started_ones_are_swept(
    store: PGScheduleStore,
) -> None:
    created = await store.create(_schedule(), max_active=20)
    claim = await store.claim_due(NOW, limit=20, fires_kept=200, plan=_plan())
    (due,) = claim.due

    assert (
        await store.claim_pending(NOW + timedelta(seconds=30), lease=timedelta(minutes=1), limit=10)
        == ()
    )
    retried = await asyncio.gather(
        *(
            store.claim_pending(NOW + timedelta(minutes=2), lease=timedelta(minutes=1), limit=10)
            for _ in range(4)
        )
    )
    taken = [item for batch in retried for item in batch]
    assert [item.fire.fire_id for item in taken] == [due.fire.fire_id]
    assert taken[0].schedule.schedule_id == created.schedule_id

    await store.finish_fire(
        OWNER,
        due.fire.fire_id,
        expected=(FireState.PENDING,),
        state=FireState.STARTED,
        run_id=str(uuid.uuid4()),
        failure_streak=3,
    )
    swept = await store.unsettled_fires(
        older_than=datetime.now(UTC) + timedelta(minutes=1), limit=10
    )
    assert [fire.fire_id for fire in swept] == [due.fire.fire_id]
    assert (
        await store.unsettled_fires(older_than=datetime.now(UTC) - timedelta(days=1), limit=10)
        == ()
    )


async def test_fires_are_listed_newest_first_and_only_to_their_owner(
    store: PGScheduleStore,
) -> None:
    created = await store.create(_schedule(), max_active=20)
    for hour in range(3):
        await store.insert_manual_fire(created, NOW - timedelta(hours=hour), fires_kept=200)

    fires = await store.list_fires(OWNER, created.schedule_id, limit=2)

    assert fires is not None
    assert [fire.slot for fire in fires] == [NOW, NOW - timedelta(hours=1)]
    assert await store.list_fires(owner_id_from_user(OTHER), created.schedule_id, limit=2) is None


_INSERT_RUN = """
INSERT INTO dlightrag_runs (
    owner_id, run_id, run_kind, lane, submitted_by, access_scope_kind, submission_key,
    accepted_input_json, request_fingerprint, status, result_json, retention_seconds,
    created_at, finished_at
)
VALUES ($1, $2::uuid, 'answer', 'query', $1, 'owner', $2, $3::jsonb, $2, 'succeeded', $4::jsonb,
        31536000, $5, $5)
"""

_INSERT_ROUTING = """
INSERT INTO dlightrag_answer_run_routing (
    owner_id, run_id, requested_mode, valid_modes, context_policy_revision,
    agent_session_id, agent_lane_id
)
VALUES ($1, $2::uuid, 'research', ARRAY['research'], 'test', $3::uuid, 'main')
"""


async def _insert_run(
    pool: Any,
    *,
    session: str,
    at: datetime,
    usage: dict[str, Any],
    kind: str | None = None,
    owner: str = OWNER,
) -> str:
    run_id = str(uuid.uuid4())
    async with pool.acquire() as conn:
        await conn.execute(
            _INSERT_RUN,
            owner,
            run_id,
            json.dumps({"query": "q", "continuation_kind": kind} if kind else {"query": "q"}),
            json.dumps({"answer": "a", "usage": usage}),
            at,
        )
        await conn.execute(_INSERT_ROUTING, owner, run_id, session)
    return run_id


async def test_spend_is_what_the_inspector_states_for_the_fired_runs_and_their_goal_rounds(
    store: PGScheduleStore, pool: Any
) -> None:
    created = await store.create(_schedule(), max_active=20)
    session, other_session = str(uuid.uuid4()), str(uuid.uuid4())
    since = NOW - timedelta(hours=24)

    fired = await _insert_run(
        pool,
        session=session,
        at=NOW - timedelta(hours=5),
        usage={
            "usage_details": {"total_tokens": 100},
            "inclusive_usage_details": {"total_tokens": 400},
        },
    )
    fast = await _insert_run(
        pool, session=str(uuid.uuid4()), at=NOW - timedelta(hours=3), usage={"total_tokens": 7}
    )
    # Goal rounds the fired Run's Session ran afterwards are the Schedule's spend; a human turn in
    # the same Session, a round before the fire, another Session and another owner's are not.
    await _insert_run(
        pool,
        session=session,
        at=NOW - timedelta(hours=4),
        kind="goal_round",
        usage={"inclusive_usage_details": {"total_tokens": 1000}},
    )
    await _insert_run(
        pool,
        session=session,
        at=NOW - timedelta(hours=4),
        usage={"inclusive_usage_details": {"total_tokens": 5}},
    )
    await _insert_run(
        pool,
        session=session,
        at=NOW - timedelta(hours=6),
        kind="goal_round",
        usage={"inclusive_usage_details": {"total_tokens": 9}},
    )
    await _insert_run(
        pool,
        session=other_session,
        at=NOW - timedelta(hours=4),
        kind="goal_round",
        usage={"inclusive_usage_details": {"total_tokens": 9}},
    )
    await _insert_run(
        pool,
        session=session,
        at=NOW - timedelta(hours=4),
        kind="goal_round",
        usage={"inclusive_usage_details": {"total_tokens": 9}},
        owner=owner_id_from_user(OTHER),
    )
    for slot, run_id in ((NOW - timedelta(hours=5), fired), (NOW - timedelta(hours=3), fast)):
        due = await store.insert_manual_fire(created, slot, fires_kept=200)
        assert due is not None
        await store.finish_fire(
            OWNER,
            due.fire.fire_id,
            expected=(FireState.PENDING,),
            state=FireState.STARTED,
            run_id=run_id,
            failure_streak=3,
        )

    assert await store.spent_tokens(OWNER, created.schedule_id, since=since) == 400 + 7 + 1000
    # A Run older than the window is out of it, and so is the round that followed only that Run.
    assert (
        await store.spent_tokens(OWNER, created.schedule_id, since=NOW - timedelta(hours=4))
        == 7 + 1000
    )
    assert await store.spent_tokens(OWNER, str(uuid.uuid4()), since=since) == 0


async def test_a_goal_schedule_keeps_its_round_cap_through_an_edit(store: PGScheduleStore) -> None:
    goal = Schedule.create(
        owner_id=OWNER,
        principal=USER,
        name="Goal review",
        prompt="Finish the review.",
        mode="research",
        workspaces=["alpha"],
        target=Target.SAME_CONVERSATION,
        cadence=Cadence("UTC", CronExpression.parse("0 * * * *"), None),
        max_tokens_per_day=1_000,
        now=NOW,
        goal_max_rounds=4,
    )
    created = await store.create(goal, max_active=20)
    plain = await store.create(_schedule(), max_active=20)

    raised = await store.edit(
        created.edit(principal=USER, now=NOW, goal_max_rounds=9), reschedule=False
    )

    assert (await store.get(OWNER, created.schedule_id)) == raised
    assert raised is not None and raised.goal_max_rounds == 9
    assert (await store.get(OWNER, plain.schedule_id)) == plain
    assert plain.goal_max_rounds is None


async def test_the_store_refuses_a_goal_schedule_that_is_not_research(
    store: PGScheduleStore, pool: Any
) -> None:
    created = await store.create(_schedule(), max_active=20)

    with pytest.raises(asyncpg.CheckViolationError, match="dlightrag_schedules_goal_check"):
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE dlightrag_schedules SET mode = 'fast', goal_max_rounds = 2 "
                "WHERE schedule_id = $1::uuid",
                created.schedule_id,
            )


async def test_anthropic_shaped_usage_without_a_total_is_counted(
    store: PGScheduleStore, pool: Any
) -> None:
    """A provider that states no ``total_tokens`` still spends: prompt (cache included) and output."""
    created = await store.create(_schedule(), max_active=20)
    usage = {
        "usage_details": {
            "input_tokens": 400_000,
            "output_tokens": 50_000,
            "cache_read_input_tokens": 1_000_000,
        },
        "inclusive_usage_details": {
            "input_tokens": 900_000,
            "output_tokens": 90_000,
            "cache_read_input_tokens": 2_000_000,
            "cache_creation_input_tokens": 10_000,
        },
    }
    run_id = await _insert_run(
        pool, session=str(uuid.uuid4()), at=NOW - timedelta(hours=1), usage=usage
    )
    due = await store.insert_manual_fire(created, NOW - timedelta(hours=1), fires_kept=200)
    assert due is not None
    await store.finish_fire(
        OWNER,
        due.fire.fire_id,
        expected=(FireState.PENDING,),
        state=FireState.STARTED,
        run_id=run_id,
        failure_streak=3,
    )

    spent = await store.spent_tokens(OWNER, created.schedule_id, since=NOW - timedelta(hours=24))

    assert spent == 900_000 + 2_000_000 + 10_000 + 90_000


_INSERT_SESSION = """
INSERT INTO dlightrag_agent_sessions (owner_id, session_id, lease_run_id, commit_sequence,
                                        fencing_epoch)
VALUES ($1, $2::uuid, $3::uuid, 0, 1)
ON CONFLICT DO NOTHING
"""

_INSERT_CALL = """
INSERT INTO dlightrag_agent_session_entries (
    owner_id, session_id, sequence, entry_id, entry_type, schema_version, timestamp,
    payload_json, run_id
)
SELECT $1, $2::uuid, coalesce(max(sequence), 0) + 1, $3::uuid, 'assistant_message', 2, $4,
       $5::jsonb, $6::uuid
FROM dlightrag_agent_session_entries
WHERE owner_id = $1 AND session_id = $2::uuid
"""


async def test_a_failed_runs_model_calls_count_even_though_it_left_no_usage_record(
    store: PGScheduleStore, pool: Any
) -> None:
    created = await store.create(_schedule(), max_active=20)
    session = str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO dlightrag_runs (
                owner_id, run_id, run_kind, lane, submitted_by, access_scope_kind,
                submission_key, accepted_input_json, request_fingerprint, status, error_kind,
                retention_seconds, created_at, finished_at
            )
            VALUES ($1, $2::uuid, 'answer', 'query', $1, 'owner', $2, '{}'::jsonb, $2, 'failed',
                    'model_error', 31536000, $3, $3)
            """,
            OWNER,
            run_id,
            NOW - timedelta(hours=2),
        )
        await conn.execute(_INSERT_SESSION, OWNER, session, run_id)
        for number, usage in enumerate(
            (
                {"prompt_tokens": 1_000, "completion_tokens": 200, "total_tokens": 1_200},
                {"input_tokens": 3_000, "output_tokens": 300, "cache_read_input_tokens": 500},
            )
        ):
            await conn.execute(
                _INSERT_CALL,
                OWNER,
                session,
                str(uuid.uuid4()),
                NOW - timedelta(hours=2, minutes=-number),
                json.dumps({"content": "", "usage": usage}),
                run_id,
            )
    due = await store.insert_manual_fire(created, NOW - timedelta(hours=2), fires_kept=200)
    assert due is not None
    await store.finish_fire(
        OWNER,
        due.fire.fire_id,
        expected=(FireState.PENDING,),
        state=FireState.STARTED,
        run_id=run_id,
        failure_streak=3,
    )

    spent = await store.spent_tokens(OWNER, created.schedule_id, since=NOW - timedelta(hours=24))

    assert spent == 1_200 + 3_000 + 500 + 300


async def test_claim_pending_hands_back_the_fires_of_a_paused_schedule_for_the_firer_to_judge(
    store: PGScheduleStore,
) -> None:
    created = await store.create(_schedule(), max_active=20)
    claim = await store.claim_due(NOW, limit=20, fires_kept=200, plan=_plan())
    assert len(claim.due) == 1
    await store.pause(OWNER, created.schedule_id, reason="paused by its owner")

    retried = await store.claim_pending(
        NOW + timedelta(minutes=5), lease=timedelta(seconds=60), limit=50
    )

    assert [item.schedule.state for item in retried] == [ScheduleState.PAUSED]
    assert [item.fire.manual for item in retried] == [False]


async def test_a_manual_fire_is_marked_manual_and_trims_to_the_number_kept(
    store: PGScheduleStore, pool: Any
) -> None:
    created = await store.create(_schedule(), max_active=20)
    for hour in range(5):
        await store.insert_manual_fire(created, NOW - timedelta(hours=hour + 1), fires_kept=3)

    due = await store.insert_manual_fire(created, NOW, fires_kept=3)

    assert due is not None and due.fire.manual is True
    rows = await _fires(pool, created)
    assert len(rows) == 3
    assert rows[-1]["manual"] is True


async def test_open_fires_are_the_pending_and_started_ones_and_an_earlier_pending_one_is_found(
    store: PGScheduleStore,
) -> None:
    created = await store.create(_schedule(), max_active=20)
    assert await store.has_open_fire(OWNER, created.schedule_id) is False
    earlier = await store.insert_manual_fire(created, NOW - timedelta(minutes=2), fires_kept=200)
    later = await store.insert_manual_fire(created, NOW, fires_kept=200)
    assert earlier is not None and later is not None

    assert await store.has_open_fire(OWNER, created.schedule_id) is True
    found = await store.earlier_pending_fire(
        OWNER, created.schedule_id, later.fire.slot, exclude=later.fire.fire_id
    )
    assert found is not None and found.fire_id == earlier.fire.fire_id
    assert (
        await store.earlier_pending_fire(
            OWNER, created.schedule_id, earlier.fire.slot, exclude=earlier.fire.fire_id
        )
        is None
    )

    for item in (earlier, later):
        await store.finish_fire(
            OWNER,
            item.fire.fire_id,
            expected=(FireState.PENDING,),
            state=FireState.SKIPPED_LIMIT,
            failure_streak=3,
        )
    assert await store.has_open_fire(OWNER, created.schedule_id) is False


async def test_a_fire_missed_after_an_error_counts_toward_the_streak_and_a_plain_gap_does_not(
    store: PGScheduleStore,
) -> None:
    created = await store.create(_schedule(), max_active=20)
    base = NOW - timedelta(days=1)
    for step in range(3):
        due = await store.insert_manual_fire(
            created, base + timedelta(minutes=step), fires_kept=200
        )
        assert due is not None
        if step < 2:  # the third is missed with no error behind it: a gap, not a failure
            await store.record_error(OWNER, due.fire.fire_id, "RuntimeError: unreadable")
        await store.finish_fire(
            OWNER,
            due.fire.fire_id,
            expected=(FireState.PENDING,),
            state=FireState.MISSED,
            reason="it could not be started while it was still worth starting",
            failure_streak=3,
        )
    still = await store.get(OWNER, created.schedule_id)
    assert still is not None and still.state is ScheduleState.ACTIVE

    due = await store.insert_manual_fire(created, base + timedelta(minutes=3), fires_kept=200)
    assert due is not None
    await store.record_error(OWNER, due.fire.fire_id, "RuntimeError: unreadable")
    await store.finish_fire(
        OWNER,
        due.fire.fire_id,
        expected=(FireState.PENDING,),
        state=FireState.MISSED,
        reason="it could not be started: RuntimeError: unreadable",
        failure_streak=3,
    )
    # Newest first: missed with an error, a plain miss, then two with errors. The plain miss is
    # not in the streak, so the three error misses are, and the Schedule pauses.
    again = await store.get(OWNER, created.schedule_id)
    assert again is not None and again.state is ScheduleState.PAUSED
    assert "RuntimeError: unreadable" in (again.pause_reason or "")


async def test_resuming_starts_the_failure_count_again(store: PGScheduleStore) -> None:
    created = await store.create(_schedule(), max_active=20)
    # Two failures from before the pause: with one more they would be three in a row.
    old = datetime(2020, 1, 1, tzinfo=UTC)
    for step in range(2):
        await _end(store, created, old + timedelta(minutes=step), FireState.FAILED)
    await store.pause(OWNER, created.schedule_id, reason="paused by its owner")
    resumed = await store.resume(OWNER, created.schedule_id, principal=USER, max_active=20)
    assert resumed is not None and resumed.resumed_at is not None

    after = datetime.now(UTC) + timedelta(seconds=5)
    for step in range(2):
        await _end(store, created, after + timedelta(minutes=step), FireState.FAILED)
        running = await store.get(OWNER, created.schedule_id)
        assert running is not None and running.state is ScheduleState.ACTIVE
    await _end(store, created, after + timedelta(minutes=2), FireState.FAILED)

    paused = await store.get(OWNER, created.schedule_id)
    assert paused is not None and paused.state is ScheduleState.PAUSED


async def test_resume_stamps_the_identity_the_schedule_acts_as_afresh(
    store: PGScheduleStore,
) -> None:
    created = await store.create(
        _schedule(now=datetime.now(UTC) - timedelta(hours=2)), max_active=20
    )
    await store.pause(OWNER, created.schedule_id, reason="paused by its owner")

    resumed = await store.resume(OWNER, created.schedule_id, principal=OTHER, max_active=20)

    assert resumed is not None
    assert resumed.principal == OTHER
    assert resumed.principal_stamped_at > created.principal_stamped_at
    assert resumed.resumed_at == resumed.principal_stamped_at


async def test_a_started_fire_a_run_is_still_working_on_does_not_pin_the_sweep(
    store: PGScheduleStore,
) -> None:
    created = await store.create(_schedule(), max_active=20)
    fires = []
    for step in range(3):
        due = await store.insert_manual_fire(
            created, NOW - timedelta(hours=3 - step), fires_kept=200
        )
        assert due is not None
        await store.finish_fire(
            OWNER,
            due.fire.fire_id,
            expected=(FireState.PENDING,),
            state=FireState.STARTED,
            run_id=str(uuid.uuid4()),
            failure_streak=3,
        )
        fires.append(due.fire)
    later = datetime.now(UTC) + timedelta(minutes=1)
    first = await store.unsettled_fires(older_than=later, limit=2)
    assert len(first) == 2

    for fire in first:
        await store.touch_fire(OWNER, fire.fire_id)

    untouched = ({fire.fire_id for fire in fires} - {fire.fire_id for fire in first}).pop()
    after_touch = await store.unsettled_fires(older_than=later, limit=2)
    assert after_touch[0].fire_id == untouched
    assert len(await store.unsettled_fires(older_than=later, limit=10)) == 3
    assert (
        await store.unsettled_fires(older_than=datetime.now(UTC) - timedelta(hours=1), limit=10)
        == ()
    )


async def test_a_stored_cadence_that_no_longer_loads_pauses_that_schedule_and_not_the_batch(
    store: PGScheduleStore, pool: Any
) -> None:
    broken = await store.create(_schedule(), max_active=20)
    healthy = await store.create(_schedule(), max_active=20)
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE dlightrag_schedules SET cron = 'not a cron' WHERE schedule_id = $1::uuid",
            broken.schedule_id,
        )

    claim = await store.claim_due(NOW, limit=20, fires_kept=200, plan=_plan())

    assert [due.schedule.schedule_id for due in claim.due] == [healthy.schedule_id]
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT state, pause_reason FROM dlightrag_schedules WHERE schedule_id = $1::uuid",
            broken.schedule_id,
        )
    assert row is not None and row["state"] == "paused"
    assert "can no longer be read" in row["pause_reason"]


async def test_ending_a_fire_while_its_schedule_is_deleted_never_deadlocks(
    store: PGScheduleStore,
) -> None:
    """A delete takes the Schedule and then its Fires; ending a Fire takes them in that order too."""
    for _ in range(15):
        created = await store.create(_schedule(), max_active=20)
        due = await store.insert_manual_fire(created, NOW, fires_kept=200)
        assert due is not None

        results = await asyncio.gather(
            store.delete(OWNER, created.schedule_id),
            store.finish_fire(
                OWNER,
                due.fire.fire_id,
                expected=(FireState.PENDING,),
                state=FireState.REFUSED,
                reason="no readable workspace",
                pause_reason="no readable workspace",
                failure_streak=3,
            ),
            return_exceptions=True,
        )

        assert not [item for item in results if isinstance(item, BaseException)], results
