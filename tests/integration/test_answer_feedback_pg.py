# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Answer feedback on PostgreSQL: which messages can be judged, and what a judgement keeps.

A judgement hangs off the Run that wrote the message and is keyed by the message's own place in
its agent's Session. These tests hold the store to that: only an assistant message the Run itself
wrote is accepted, an owner reaches only their own judgements, a judgement replaces the one before
it, and deleting the Run takes it along.

Every test runs in a throwaway database created and dropped per test; skipped without PostgreSQL.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

import asyncpg
import pytest

from dlightrag.engine.answer.feedback import FEEDBACK_NOTE_LIMIT, FEEDBACK_REASONS
from tests.conftest import FingerprintingRunStore
from tests.support.child_sessions import spawn_child
from tests.support.pg import PG_CONN_KWARGS, delete_runs, drop_database, skip_without_postgres

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_OWNER = "owner-alpha"
_OTHER_OWNER = "owner-beta"
_WORKER = "worker-1"

# What each Run wrote, by its place in the Session it wrote it in. The conversation's main Session
# holds two Runs: `earlier` wrote 1-2 and `run` wrote 3-5. `run`'s Child wrote 1-4 of its own, two
# questions and the answer to each, and its second Child wrote 1-3, a question, a tool result and
# the answer.
_USER_MESSAGE = 3
_TOOL_RESULT = 4
_ANSWER = 5
_EARLIER_ANSWER = 2
_CHILD_ANSWER = 2
_CHILD_SECOND_ANSWER = 4
_SECOND_CHILD_ANSWER = 3


@pytest.fixture
async def pool() -> AsyncIterator[Any]:
    await skip_without_postgres()
    database = f"dlightrag_feedback_{uuid.uuid4().hex[:12]}"
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


@dataclass(frozen=True)
class _Scene:
    """Three conversations, each with the messages its Runs wrote.

    ``earlier`` and ``run`` are two Runs of one conversation, so one main Session holds both their
    messages, and ``run`` spawned ``child`` and then ``second_child``. ``elsewhere`` is another
    conversation of the same owner, with a Child of its own, and ``foreign`` is another owner's.
    """

    store: FingerprintingRunStore
    pool: Any
    earlier: str
    run: str
    child: str
    second_child: str
    elsewhere: str
    elsewhere_child: str
    foreign: str


async def _write(
    pool: Any, *, owner: str, session: str, run_id: str, entries: Sequence[tuple[int, str]]
) -> None:
    """Session entries as their writer states them: each names the Run that wrote it."""
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO dlightrag_agent_sessions (owner_id, session_id, lease_run_id, "
            "fencing_epoch) VALUES ($1, $2, $3, 1) ON CONFLICT DO NOTHING",
            owner,
            uuid.UUID(session),
            uuid.UUID(run_id),
        )
        await conn.executemany(
            "INSERT INTO dlightrag_agent_session_entries (owner_id, session_id, sequence, "
            "entry_id, entry_type, schema_version, timestamp, payload_json, search_text, run_id) "
            "VALUES ($1, $2, $3, $4, $5, 1, NOW(), '{}'::jsonb, '', $6)",
            [
                (owner, uuid.UUID(session), sequence, uuid.uuid4(), entry_type, uuid.UUID(run_id))
                for sequence, entry_type in entries
            ],
        )


@pytest.fixture
async def scene(pool: Any) -> _Scene:
    store = FingerprintingRunStore(pool=pool)
    await store.initialize()

    async def start(owner: str, session: str) -> tuple[str, int]:
        """A Run of ``session``, accepted and claimed, as the Run that writes its messages."""
        created = await store.create_run(
            owner_id=owner,
            request={
                "query": "why",
                "workspaces": ["alpha"],
                "agent_session_id": session,
                "agent_lane_id": "main",
            },
        )
        claim = await store.claim_next(worker_id=_WORKER)
        assert claim is not None and claim.run.run_id == created.run.run_id
        return created.run.run_id, claim.run.fencing_epoch

    conversation, other, foreign_conversation = (str(uuid.uuid7()) for _ in range(3))
    earlier, _ = await start(_OWNER, conversation)
    elsewhere, elsewhere_epoch = await start(_OWNER, other)
    run, epoch = await start(_OWNER, conversation)
    foreign, _ = await start(_OTHER_OWNER, foreign_conversation)
    child = await spawn_child(
        store, owner_id=_OWNER, run_id=run, worker_id=_WORKER, fencing_epoch=epoch
    )
    second_child = await spawn_child(
        store, owner_id=_OWNER, run_id=run, worker_id=_WORKER, fencing_epoch=epoch
    )
    elsewhere_child = await spawn_child(
        store,
        owner_id=_OWNER,
        run_id=elsewhere,
        worker_id=_WORKER,
        fencing_epoch=elsewhere_epoch,
    )

    question = (1, "user_message"), (2, "assistant_message")
    await _write(pool, owner=_OWNER, session=conversation, run_id=earlier, entries=question)
    await _write(
        pool,
        owner=_OWNER,
        session=conversation,
        run_id=run,
        entries=(
            (_USER_MESSAGE, "user_message"),
            (_TOOL_RESULT, "tool_result"),
            (_ANSWER, "assistant_message"),
        ),
    )
    await _write(
        pool,
        owner=_OWNER,
        session=child.child_session_id,
        run_id=run,
        entries=(*question, (3, "user_message"), (4, "assistant_message")),
    )
    await _write(
        pool,
        owner=_OWNER,
        session=second_child.child_session_id,
        run_id=run,
        entries=(
            (1, "user_message"),
            (2, "tool_result"),
            (_SECOND_CHILD_ANSWER, "assistant_message"),
        ),
    )
    await _write(pool, owner=_OWNER, session=other, run_id=elsewhere, entries=question)
    await _write(
        pool,
        owner=_OWNER,
        session=elsewhere_child.child_session_id,
        run_id=elsewhere,
        entries=question,
    )
    await _write(
        pool, owner=_OTHER_OWNER, session=foreign_conversation, run_id=foreign, entries=question
    )
    return _Scene(
        store=store,
        pool=pool,
        earlier=earlier,
        run=run,
        child=child.child_session_id,
        second_child=second_child.child_session_id,
        elsewhere=elsewhere,
        elsewhere_child=elsewhere_child.child_session_id,
        foreign=foreign,
    )


async def _put(
    store: FingerprintingRunStore,
    run_id: str,
    *,
    agent: str = "main",
    sequence: int = _ANSWER,
    value: str = "up",
    reasons: Sequence[str] = (),
    note: str | None = None,
    owner: str = _OWNER,
):
    return await store.put_answer_feedback(
        owner_id=owner,
        run_id=run_id,
        agent=agent,
        sequence=sequence,
        value=value,
        reasons=reasons,
        note=note,
    )


async def _kept(pool: Any) -> int:
    async with pool.acquire() as conn:
        return int(await conn.fetchval("SELECT count(*) FROM dlightrag_answer_feedback"))


async def _first_given(pool: Any, run_id: str) -> Any:
    """When the judgement of the Run's answer was first given: the training record's column,
    which nothing the product reads carries."""
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT created_at FROM dlightrag_answer_feedback"
            " WHERE run_id = $1 AND agent = 'main' AND sequence = $2",
            uuid.UUID(run_id),
            _ANSWER,
        )


async def test_a_judgement_replaces_the_one_before_it_and_keeps_when_it_was_first_given(
    scene: _Scene,
) -> None:
    down = await _put(
        scene.store,
        scene.run,
        value="down",
        reasons=("unsupported", "too_verbose"),
        note="too long",
    )
    assert down is not None
    assert (down.agent, down.sequence, down.value, down.reasons, down.note) == (
        "main",
        _ANSWER,
        "down",
        ("unsupported", "too_verbose"),
        "too long",
    )
    assert await _first_given(scene.pool, scene.run) == down.updated_at

    up = await _put(scene.store, scene.run, value="up")

    assert up is not None
    assert (up.value, up.reasons, up.note) == ("up", (), None), "nothing of the old one stays"
    assert up.updated_at > down.updated_at
    assert await _first_given(scene.pool, scene.run) == down.updated_at, "the first time stays"
    assert await scene.store.list_answer_feedback(owner_id=_OWNER, run_id=scene.run) == (up,)


async def test_only_an_assistant_message_the_run_wrote_in_that_agents_session_is_judged(
    scene: _Scene,
) -> None:
    """Each of these is a message that is not there to judge. Refusing changes nothing, so one
    conversation serves them all."""
    refused: dict[str, tuple[str, str, str, int]] = {
        "a user message": (_OWNER, scene.run, "main", _USER_MESSAGE),
        "a tool result": (_OWNER, scene.run, "main", _TOOL_RESULT),
        "a message nobody wrote": (_OWNER, scene.run, "main", 99),
        "a message an earlier Run of the Session wrote": (
            _OWNER,
            scene.run,
            "main",
            _EARLIER_ANSWER,
        ),
        "the main agent's place named as a Child's": (_OWNER, scene.run, scene.child, _ANSWER),
        "a Child's user message": (_OWNER, scene.run, scene.child, 1),
        "a Child named in capitals": (_OWNER, scene.run, scene.child.upper(), _CHILD_ANSWER),
        "an agent that is no id": (_OWNER, scene.run, "someone", _CHILD_ANSWER),
        "a Child of another Run": (_OWNER, scene.run, scene.elsewhere_child, _CHILD_ANSWER),
        "another owner judging this owner's Run": (_OTHER_OWNER, scene.run, "main", _ANSWER),
        "this owner judging another owner's Run": (
            _OWNER,
            scene.foreign,
            "main",
            _EARLIER_ANSWER,
        ),
        "a Run that is not there": (_OWNER, str(uuid.uuid4()), "main", _ANSWER),
        "a Run id that is no id": (_OWNER, "nope", "main", _ANSWER),
    }

    for case, (owner, run_id, agent, sequence) in refused.items():
        judged = await _put(scene.store, run_id, agent=agent, sequence=sequence, owner=owner)
        assert judged is None, case

    assert await _kept(scene.pool) == 0, "a message that is not there leaves no judgement behind"


async def test_each_run_judges_the_messages_it_wrote_by_the_agent_that_wrote_them(
    scene: _Scene,
) -> None:
    # The Children's ids order their groups, and ids made one after the other ascend.
    assert scene.child < scene.second_child
    for run_id, agent, sequence in (
        (scene.run, scene.second_child, _SECOND_CHILD_ANSWER),
        (scene.run, scene.child, _CHILD_SECOND_ANSWER),
        (scene.run, "main", _ANSWER),
        (scene.run, scene.child, _CHILD_ANSWER),
        (scene.earlier, "main", _EARLIER_ANSWER),
    ):
        assert await _put(scene.store, run_id, agent=agent, sequence=sequence) is not None

    run = await scene.store.list_answer_feedback(owner_id=_OWNER, run_id=scene.run)
    earlier = await scene.store.list_answer_feedback(owner_id=_OWNER, run_id=scene.earlier)

    # The Run's own agent first, then each Child grouped, the oldest message first within an
    # agent: the numbers of different Sessions are unrelated, so they are never compared. The
    # second Child's 3 numbers between the first one's 2 and 4 and still comes after both.
    assert [(item.agent, item.sequence) for item in run] == [
        ("main", _ANSWER),
        (scene.child, _CHILD_ANSWER),
        (scene.child, _CHILD_SECOND_ANSWER),
        (scene.second_child, _SECOND_CHILD_ANSWER),
    ]
    assert [(item.agent, item.sequence) for item in earlier] == [("main", _EARLIER_ANSWER)]


async def test_an_owner_reaches_only_the_judgements_they_gave(scene: _Scene) -> None:
    judged = await _put(scene.store, scene.run, value="down", reasons=("off_topic",))
    assert judged is not None

    assert await scene.store.list_answer_feedback(owner_id=_OTHER_OWNER, run_id=scene.run) == ()
    await scene.store.delete_answer_feedback(
        owner_id=_OTHER_OWNER, run_id=scene.run, agent="main", sequence=_ANSWER
    )
    assert await scene.store.list_answer_feedback(owner_id=_OWNER, run_id=scene.run) == (judged,)
    assert await scene.store.list_answer_feedback(owner_id=_OWNER, run_id=scene.elsewhere) == ()


async def test_retracting_a_judgement_is_idempotent(scene: _Scene) -> None:
    assert await _put(scene.store, scene.run, value="down", reasons=("off_topic",)) is not None
    retract = {"owner_id": _OWNER, "run_id": scene.run, "agent": "main", "sequence": _ANSWER}

    await scene.store.delete_answer_feedback(**retract)
    assert await scene.store.list_answer_feedback(owner_id=_OWNER, run_id=scene.run) == ()

    # Taking back what is already gone, or what a Run that is not there never held, is no error.
    await scene.store.delete_answer_feedback(**retract)
    await scene.store.delete_answer_feedback(**{**retract, "run_id": "nope"})
    assert await scene.store.list_answer_feedback(owner_id=_OWNER, run_id=scene.run) == ()


async def test_deleting_a_run_takes_its_judgements_with_it(scene: _Scene) -> None:
    assert await _put(scene.store, scene.run) is not None
    assert await _put(scene.store, scene.elsewhere, sequence=_EARLIER_ANSWER) is not None

    await delete_runs(scene.pool, scene.store, owner_id=_OWNER, run_ids=[scene.run])

    assert await scene.store.list_answer_feedback(owner_id=_OWNER, run_id=scene.run) == ()
    assert len(await scene.store.list_answer_feedback(owner_id=_OWNER, run_id=scene.elsewhere)) == 1


async def test_the_table_holds_the_limits_the_engine_states(scene: _Scene) -> None:
    """The Engine shapes a judgement before it is stored; the table is the last word, and agrees.

    The longest note the Engine allows, counted in characters, is kept, and one more is refused.
    A thumb down says why in a reason or in words: one that carries words alone is kept, and one
    that carries neither is refused.
    """
    longest = "字" * FEEDBACK_NOTE_LIMIT
    kept = await _put(scene.store, scene.run, value="down", reasons=FEEDBACK_REASONS, note=longest)
    assert kept is not None and kept.note == longest and kept.reasons == FEEDBACK_REASONS
    words_alone = await _put(
        scene.store, scene.run, value="down", reasons=(), note="it answers a different question"
    )
    assert words_alone is not None
    assert (words_alone.reasons, words_alone.note) == ((), "it answers a different question")
    assert await _put(scene.store, scene.run, value="up") is not None
    cannot_be: tuple[tuple[str, tuple[str, tuple[str, ...], str | None]], ...] = (
        ("dlightrag_answer_feedback_says_why_check", ("down", (), None)),
        ("dlightrag_answer_feedback_says_why_check", ("down", (), "   ")),
        ("dlightrag_answer_feedback_note_check", ("down", ("off_topic",), longest + "字")),
        ("dlightrag_answer_feedback_value_check", ("sideways", ("off_topic",), None)),
    )

    for constraint, (value, reasons, note) in cannot_be:
        with pytest.raises(asyncpg.CheckViolationError, match=constraint):
            await _put(scene.store, scene.run, value=value, reasons=reasons, note=note)

    (stored,) = await scene.store.list_answer_feedback(owner_id=_OWNER, run_id=scene.run)
    assert (stored.value, stored.reasons, stored.note) == ("up", (), None), "a refusal changes none"


async def test_judgements_made_at_once_leave_one_whole_judgement(scene: _Scene) -> None:
    submitted = [
        ("down", (reason,), f"note {number}")
        for number, reason in enumerate(
            ("unsupported", "factual_error", "too_verbose", "off_topic") * 2
        )
    ]

    kept = await asyncio.gather(
        *(
            _put(scene.store, scene.run, value=value, reasons=reasons, note=note)
            for value, reasons, note in submitted
        )
    )

    assert all(item is not None for item in kept)
    (stored,) = await scene.store.list_answer_feedback(owner_id=_OWNER, run_id=scene.run)
    assert (stored.value, stored.reasons, stored.note) in submitted, (
        "the survivor is one judgement as it was given, never a mixture of two"
    )


async def _until_a_statement_waits_on_a_lock(pool: Any) -> None:
    async with pool.acquire() as conn:
        for _ in range(200):
            if await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM pg_stat_activity "
                "WHERE datname = current_database() AND wait_event_type = 'Lock')"
            ):
                return
            await asyncio.sleep(0.05)
    raise TimeoutError("no statement waited on a lock")


async def test_a_judgement_written_while_its_run_is_deleted_reads_as_unknown(
    scene: _Scene,
) -> None:
    """The Run's row is held while a judgement is written, so a deletion in flight is waited for.

    Without the hold the write passes its checks, then fails the foreign key once the deletion
    commits: a reader rating an answer in one tab while deleting its conversation in another
    would be told the server failed instead of that the answer is gone.
    """
    async with scene.pool.acquire() as deleting:
        deletion = deleting.transaction()
        await deletion.start()
        await deleting.execute(
            "DELETE FROM dlightrag_runs WHERE owner_id = $1 AND run_id = $2",
            _OWNER,
            uuid.UUID(scene.run),
        )
        judging = asyncio.ensure_future(_put(scene.store, scene.run))
        try:
            await _until_a_statement_waits_on_a_lock(scene.pool)
        finally:
            await deletion.commit()

    assert await judging is None
    assert await _kept(scene.pool) == 0
