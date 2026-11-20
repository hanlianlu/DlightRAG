# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The Session Memory Pass over in-memory doubles: claim, batches, two phases, switch.

The pass against the real Run store and memory store runs in
tests/integration/test_session_memory_pass_pg.py.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import pytest
from dlightrag_memory import (
    MemoryOperation,
    MemoryOperationReceipt,
    MemoryProvenance,
    MemoryRecord,
    MemoryWriteRejectedError,
)
from dlightrag_memory.consolidation import (
    CONTEXT_ROUNDS,
    ConsolidationVerdict,
    ConversationTurn,
    PlannedWrite,
    VerdictWrite,
    consolidation_request,
)

from dlightrag.application.memory import session_pass as module
from dlightrag.application.memory.session_pass import (
    SESSION_BACKOFF_FIRST_SECONDS,
    SESSION_BACKOFF_MAX_SECONDS,
    SESSION_SWEEP_BATCH,
    SessionMemoryPass,
    SessionMemoryPassRunner,
    SessionPassSettings,
)
from dlightrag.engine.answer.memory import (
    MemoryCapability,
    PendingSession,
    SessionMemoryPlan,
    SessionMemoryProgress,
    SessionTurn,
)
from dlightrag.engine.runtime.contracts import RunKind
from dlightrag.engine.runtime.coordinator import SettledRun

OWNER = "owner"
SESSION = "22222222-2222-2222-2222-222222222222"
RUN = "run-1"


@dataclass
class _Store:
    """The Run store's Session view, kept in memory and recording what a pass did to it."""

    turns: list[SessionTurn] = field(default_factory=list)
    row: SessionMemoryProgress = field(default_factory=lambda: SessionMemoryProgress(0))
    held_by_other: bool = False
    events: list[str] = field(default_factory=list)
    plans: list[SessionMemoryPlan] = field(default_factory=list)
    # An operation named here succeeds that many times and then refuses, as a store does once
    # the claim has lapsed or been taken over: {"renew": 1} lets one renewal through.
    refuses_after: dict[str, int] = field(default_factory=dict)
    # The (before, rounds) of every read of the turns before a batch.
    context_reads: list[tuple[int, int]] = field(default_factory=list)
    # The entry at which the owner last turned memory back on; no turn up to there is context.
    floor: int = 0
    # What the store offers a sweep, and the limit of each look it was asked for.
    pending: list[PendingSession] = field(default_factory=list)
    looks: list[int] = field(default_factory=list)
    look_errors: list[Exception] = field(default_factory=list)
    # The (first, max) seconds of every failure a pass counted, and an error recording one raises.
    backoffs: list[tuple[float, float]] = field(default_factory=list)
    backoff_error: Exception | None = None
    _calls: dict[str, int] = field(default_factory=dict)

    def _holds(self, operation: str) -> bool:
        self._calls[operation] = self._calls.get(operation, 0) + 1
        allowed = self.refuses_after.get(operation)
        return allowed is None or self._calls[operation] <= allowed

    async def load_routing(self, *, owner_id: str, run_id: str) -> Any:
        if run_id.endswith("no-session"):
            return None
        return type("Routing", (), {"agent_session_id": SESSION})()

    async def claim_session_memory(self, *, owner_id, session_id, holder, lease_seconds):
        if self.held_by_other:
            return None
        self.events.append("claim")
        return self.row

    async def renew_session_memory(self, *, owner_id, session_id, holder, lease_seconds) -> bool:
        self.events.append("renew")
        return self._holds("renew")

    async def release_session_memory(self, *, owner_id, session_id, holder) -> None:
        self.events.append("release")

    async def store_session_memory_plan(self, *, owner_id, session_id, holder, plan) -> bool:
        self.events.append(f"plan {plan.read_from}-{plan.read_through} {len(plan.writes)}")
        if not self._holds("plan"):
            return False
        self.plans.append(plan)
        self.row = replace(self.row, plan=plan)
        return True

    async def advance_session_memory(self, *, owner_id, session_id, holder, through):
        self.events.append(f"advance {through}")
        if not self._holds("advance"):
            return None
        # Where the watermark stands: past `through` if a restart moved it there meanwhile.
        self.row = SessionMemoryProgress(read_through=max(self.row.read_through, through))
        return self.row.read_through

    def restart(self) -> None:
        """What the Run store does when the owner turns memory back on: start afresh at the latest."""
        latest = max((turn.sequence for turn in self.turns), default=0)
        self.row = replace(self.row, read_through=max(self.row.read_through, latest))
        self.floor = max(self.floor, latest)

    async def pending_session_memory(self, *, limit: int) -> list[PendingSession]:
        self.looks.append(limit)
        if self.look_errors:
            raise self.look_errors.pop(0)
        return list(self.pending)

    async def back_off_session_memory(
        self, *, owner_id, session_id, holder, first_seconds, max_seconds
    ) -> None:
        self.events.append("back off")
        self.backoffs.append((first_seconds, max_seconds))
        if self.backoff_error is not None:
            raise self.backoff_error

    async def session_turns_after(self, *, owner_id, session_id, after, limit):
        return [turn for turn in self.turns if turn.sequence > after][:limit]

    async def session_turns_before(self, *, owner_id, session_id, before, rounds):
        self.context_reads.append((before, rounds))
        own = [turn for turn in self.turns if self.floor < turn.sequence < before]
        users = [turn.sequence for turn in own if turn.role == "user"]
        first = users[-rounds:][0] if users else before
        return [turn for turn in own if turn.sequence >= first]


@dataclass
class _Memory:
    applied: list[MemoryOperation] = field(default_factory=list)
    refuse: bool = False
    candidates: tuple[MemoryRecord, ...] = ()
    own: tuple[MemoryRecord, ...] = ()
    after_apply: Callable[[], None] | None = None
    queries: list[str] = field(default_factory=list)

    async def search(self, *, owner_id: str, query: str, limit: int) -> tuple[MemoryRecord, ...]:
        self.queries.append(query)
        return self.candidates

    async def session_records(self, *, owner_id: str, session_id: str, origin_kind: str):
        return self.own

    async def apply(self, operation: MemoryOperation, *, guard=None) -> MemoryOperationReceipt:
        if guard is not None:
            await guard(None)
        if self.refuse:
            raise MemoryWriteRejectedError("refused")
        self.applied.append(operation)
        if self.after_apply is not None:
            self.after_apply()
        return MemoryOperationReceipt(
            change_id="c",
            action=operation.action,
            outcome="changed",
            memory_ids=("m",),
            provenance=operation.provenance,
        )


@dataclass
class _Gate:
    """The owner's switch and epoch, as the service keeps them: a write holds only while both do."""

    enabled: bool = True
    epoch: int = 4
    # The epoch each write the pass tried was guarded by.
    guarded: list[int] = field(default_factory=list)

    async def capability(self, *, owner_id: str) -> MemoryCapability:
        return MemoryCapability(enabled=self.enabled, epoch=self.epoch)

    async def capability_current(self, *, owner_id, epoch, settlement=None) -> bool:
        self.guarded.append(epoch)
        return self.enabled and epoch == self.epoch


@dataclass
class _Model:
    """Answers each request with the next verdict, or raises it if it is an exception."""

    verdicts: list[ConsolidationVerdict | Exception] = field(default_factory=list)
    asked: list[list[dict[str, Any]]] = field(default_factory=list)
    closed: bool = False
    # What happens in the world while the model thinks: it runs on every call, before the answer.
    during_call: Callable[[], object] | None = None

    async def __call__(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        assert kwargs["structured_output"].name == "session_memory_verdict"
        self.asked.append(messages)
        if self.during_call is not None:
            self.during_call()
        answer = self.verdicts.pop(0) if self.verdicts else ConsolidationVerdict()
        if isinstance(answer, Exception):
            raise answer
        return answer.model_dump_json()

    async def aclose(self) -> None:
        self.closed = True


def _pass(
    store: _Store,
    memory: _Memory | None = None,
    *,
    model: _Model | None = None,
    gate: _Gate | None = None,
    batch: int = 1_000,
) -> SessionMemoryPass:
    return SessionMemoryPass(
        memory=memory or _Memory(),  # type: ignore[arg-type]
        store=store,
        gate=gate or _Gate(),
        model=model or _Model(),
        settings=SessionPassSettings(max_batch_tokens=batch, session_record_limit=5),
        holder="holder",
    )


def _verdict(*bodies: str) -> ConsolidationVerdict:
    return ConsolidationVerdict(
        writes=[VerdictWrite(action="add", kind="preference", body=body) for body in bodies]
    )


def _stored(memory_id: str, body: str) -> MemoryRecord:
    return MemoryRecord(
        owner_id=OWNER,
        memory_id=memory_id,
        kind="preference",
        body=body,
        provenance=MemoryProvenance(origin_kind="session_pass", origin_id="p", session_id=SESSION),
    )


def _turns(*texts: str) -> list[SessionTurn]:
    return [
        SessionTurn(sequence=index + 1, role="user" if index % 2 == 0 else "assistant", text=text)
        for index, text in enumerate(texts)
    ]


async def _run(session_pass: SessionMemoryPass, *, run_id: str = RUN) -> str:
    return await session_pass.run(owner_id=OWNER, session_id=SESSION, run_id=run_id)


def _sections(request: list[dict[str, Any]]) -> tuple[str, str]:
    """The earlier turns shown as context, and the newest part, of one request's turns."""
    shown = request[1]["content"].split("\n\nExisting memories that may bear on it")[0]
    context, _, newest = shown.partition("Newest part of this conversation")
    return context, newest


async def test_a_one_line_conversation_is_judged_however_short() -> None:
    store, model = _Store(turns=_turns("hi")), _Model()

    assert await _run(_pass(store, model=model)) == "applied"
    assert len(model.asked) == 1
    assert "[user] hi" in model.asked[0][1]["content"]
    assert store.row.read_through == 1


async def test_a_session_with_nothing_past_the_watermark_asks_nothing() -> None:
    for turns, row in ((_turns("hi", "hello"), SessionMemoryProgress(2)), ([], None)):
        store = _Store(turns=turns, row=row or SessionMemoryProgress(0))
        model = _Model()

        assert await _run(_pass(store, model=model)) == "nothing_new"
        assert model.asked == []
        assert store.events == ["claim", "release"]


async def test_a_pass_stores_its_plan_before_it_writes_and_then_advances() -> None:
    store = _Store(turns=_turns("以后都用中文回答我，表格放在最前面。", "好的，我记住了。"))
    memory = _Memory()
    model = _Model([_verdict("The owner prefers answers in Chinese.")])

    assert await _run(_pass(store, memory, model=model)) == "applied"

    assert "[user] 以后都用中文回答我" in model.asked[0][1]["content"]
    assert store.events == ["claim", "plan 1-2 1", "advance 2", "release"]
    (operation,) = memory.applied
    assert operation.provenance.origin_kind == "session_pass"
    assert operation.provenance.session_id == SESSION
    assert operation.idempotency_key == f"pass:{SESSION}:1-2:0"
    assert operation.mutation_scope == f"pass:{SESSION}:1-2"


async def test_a_verdict_names_memories_by_number_and_the_stored_plan_holds_their_ids() -> None:
    older = "0b1f6a52-0e44-4f5d-9d57-7a3c1d9f0a01"
    mine = "c4a8e1d2-77b0-4c1e-8a6f-2f6a9b3d5e02"
    store = _Store(turns=_turns("结论再短一点，也别铺垫。", "好的。"))
    memory = _Memory(
        candidates=(
            _stored(older, "The owner prefers short conclusions."),
            _stored(mine, "The owner wants no preamble."),
        ),
        own=(_stored(mine, "The owner wants no preamble."),),
    )
    verdict = ConsolidationVerdict(
        writes=[
            VerdictWrite(
                action="fuse",
                kind="preference",
                body="The owner prefers short conclusions without preamble.",
                targets=[1, 2],
            ),
            VerdictWrite(action="update", kind="preference", body="Ghost.", targets=[3]),
        ]
    )
    model = _Model([verdict])

    assert await _run(_pass(store, memory, model=model)) == "applied"

    # The model saw numbers, never an id; the plan, which resumes and keys the writes, holds ids.
    request = "\n".join(message["content"] for message in model.asked[0])
    assert older not in request and mine not in request
    assert "[1] (preference) The owner prefers short conclusions." in request
    assert "[2] (preference) The owner wants no preamble." in request
    (write,) = store.plans[0].writes
    assert write.target_ids == (older, mine)
    (operation,) = memory.applied
    assert operation.action == "fuse" and operation.source_ids == (older, mine)


async def test_a_pass_attributes_its_writes_to_the_run_that_asked_and_stores_it_with_the_plan() -> (
    None
):
    store = _Store(turns=_turns("以后都用中文回答我，表格放在最前面。", "好的，我记住了。"))
    memory = _Memory()
    model = _Model([_verdict("The owner prefers answers in Chinese.")])

    assert await _run(_pass(store, memory, model=model), run_id="run-7") == "applied"

    (operation,) = memory.applied
    assert operation.provenance.run_id == "run-7"
    # The plan the store kept is the one that names the Run, so a resume cannot lose it.
    assert store.plans[0].run_id == "run-7"
    # Attribution is not part of what a write is keyed by.
    assert operation.idempotency_key == f"pass:{SESSION}:1-2:0"


async def test_a_stored_plan_is_resumed_without_asking_the_model_again() -> None:
    plan = SessionMemoryPlan(
        read_from=1,
        read_through=1,
        writes=(PlannedWrite(action="add", kind="fact", body="Lives in Gothenburg."),),
        run_id="run-0",
    )
    store = _Store(turns=_turns("x" * 400), row=SessionMemoryProgress(0, plan))
    memory, model = _Memory(), _Model()

    assert await _run(_pass(store, memory, model=model)) == "resumed"
    assert model.asked == []
    assert [operation.body for operation in memory.applied] == ["Lives in Gothenburg."]
    assert store.events == ["claim", "advance 1", "release"]


async def test_a_resumed_plan_applies_under_the_run_it_was_stored_for() -> None:
    write = PlannedWrite(action="add", kind="fact", body="Lives in Gothenburg.")
    plan = SessionMemoryPlan(read_from=1, read_through=1, writes=(write,), run_id="run-0")
    store = _Store(turns=_turns("x" * 400), row=SessionMemoryProgress(0, plan))
    memory = _Memory()

    # Another Run asked for this pass, but the plan is the earlier Run's and keeps its keys.
    assert await _run(_pass(store, memory), run_id="run-9") == "resumed"

    (operation,) = memory.applied
    assert operation.provenance.run_id == "run-0"
    assert operation.idempotency_key == f"pass:{SESSION}:1-1:0"


async def test_memory_switched_off_reads_nothing_and_moves_nothing() -> None:
    store, model = _Store(turns=_turns("x" * 400, "y" * 400)), _Model()

    assert await _run(_pass(store, model=model, gate=_Gate(enabled=False))) == "disabled"

    # Nothing is asked and nothing is advanced: turning memory back on starts the Session afresh.
    assert model.asked == []
    assert store.events == ["claim", "release"]
    assert store.row.read_through == 0


async def test_a_batch_is_guarded_by_the_epoch_taken_before_it_was_read() -> None:
    gate = _Gate()
    store, memory = _Store(turns=_turns("x" * 400)), _Memory()
    model = _Model([_verdict("The owner prefers tea.")])

    assert await _run(_pass(store, memory, model=model, gate=gate)) == "applied"

    # Nothing changed while the model thought, so the write holds.
    assert [operation.body for operation in memory.applied] == ["The owner prefers tea."]
    assert gate.guarded == [4]


@pytest.mark.parametrize(
    ("disturbance", "ended"),
    [
        pytest.param(lambda gate: setattr(gate, "epoch", gate.epoch + 1), "applied", id="cleared"),
        pytest.param(
            lambda gate: (setattr(gate, "epoch", gate.epoch + 1), setattr(gate, "enabled", False)),
            "disabled",
            id="switched-off",
        ),
        pytest.param(
            lambda gate: (setattr(gate, "epoch", gate.epoch + 1), setattr(gate, "enabled", True)),
            "applied",
            id="switched-off-and-on",
        ),
    ],
)
async def test_an_epoch_that_moved_while_the_model_thought_refuses_the_batch_but_it_advances(
    disturbance: Callable[[_Gate], object], ended: str
) -> None:
    gate = _Gate()
    store, memory = _Store(turns=_turns("x" * 400, "y" * 400)), _Memory()
    model = _Model(
        [_verdict("The owner prefers tea.", "The owner lives in Gothenburg.")],
        during_call=lambda: disturbance(gate),
    )
    session_pass = _pass(store, memory, model=model, gate=gate)

    # A pass whose owner has left the switch off ends there; one whose epoch merely moved is done.
    assert await _run(session_pass) == ended

    # Every write was held to the epoch taken before the batch was read, not the one at apply
    # time, so none landed. The watermark still moved past the batch: text read under a stale
    # epoch is never read again, and the next pass has nothing new to ask about.
    assert memory.applied == []
    assert gate.guarded == [4, 4]
    assert store.events[:3] == ["claim", "plan 1-2 2", "advance 2"]
    assert store.row.read_through == 2
    gate.enabled = True
    assert await _run(session_pass) == "nothing_new"
    assert len(model.asked) == 1


async def test_each_batch_is_guarded_by_the_epoch_taken_before_that_batch() -> None:
    gate = _Gate()
    store, memory = _Store(turns=_turns("a" * 400, "b" * 400, "c" * 400)), _Memory()
    asked: list[int] = []

    def clear_during_the_first_call() -> None:
        asked.append(1)
        if len(asked) == 1:
            gate.epoch += 1

    model = _Model(
        [_verdict("First."), _verdict("Second."), _verdict("Third.")],
        during_call=clear_during_the_first_call,
    )

    assert await _run(_pass(store, memory, model=model, gate=gate, batch=150)) == "applied"

    # The first batch was read under epoch 4 and refused; the others were read under 5 and hold.
    assert [operation.body for operation in memory.applied] == ["Second.", "Third."]
    assert gate.guarded == [4, 5, 5]
    assert store.row.read_through == 3


async def test_a_resumed_plan_is_guarded_by_the_epoch_at_resume_time() -> None:
    plan = SessionMemoryPlan(
        read_from=1,
        read_through=1,
        writes=(PlannedWrite(action="add", kind="fact", body="Lives in Gothenburg."),),
        run_id="run-0",
    )
    gate = _Gate(epoch=7)
    store = _Store(turns=_turns("x" * 400), row=SessionMemoryProgress(0, plan))
    memory = _Memory()

    assert await _run(_pass(store, memory, gate=gate)) == "resumed"

    # The plan carries no epoch (it was read by a pass that is gone), so it is held to the epoch
    # as it stands when it is resumed, and nothing new is stored with it.
    assert [operation.body for operation in memory.applied] == ["Lives in Gothenburg."]
    assert gate.guarded == [7]
    assert store.plans == []


async def test_a_failed_or_malformed_answer_leaves_the_watermark_for_the_next_run() -> None:
    for failure in (RuntimeError("provider down"), ValueError("not a verdict")):
        store = _Store(turns=_turns("x" * 400))
        assert await _run(_pass(store, model=_Model([failure]))) == "model_failed"
        # The failure is counted against the Session before the claim is let go.
        assert store.events == ["claim", "back off", "release"]
        assert store.row.read_through == 0


async def test_a_pass_another_process_holds_is_left_alone() -> None:
    store, model = _Store(turns=_turns("x" * 400), held_by_other=True), _Model()

    assert await _run(_pass(store, model=model)) == "busy"
    assert model.asked == []


async def test_a_conversation_longer_than_one_call_is_read_to_its_end_in_one_pass() -> None:
    store = _Store(turns=_turns("a" * 400, "b" * 400, "c" * 400))
    memory = _Memory()
    model = _Model([_verdict("First."), _verdict("Second."), _verdict("Third.")])

    # Each turn weighs 100 estimated tokens, so a batch of 150 holds one: three model calls.
    assert await _run(_pass(store, memory, model=model, batch=150)) == "applied"

    # One claim. Each batch's plan is stored before it applies and the watermark moves before
    # the next is read, and the claim is renewed before every batch after the first.
    assert store.events == [
        "claim",
        "plan 1-1 1",
        "advance 1",
        "renew",
        "plan 2-2 1",
        "advance 2",
        "renew",
        "plan 3-3 1",
        "advance 3",
        "release",
    ]
    spans = [f"{SESSION}:{n}-{n}" for n in (1, 2, 3)]
    assert [operation.idempotency_key for operation in memory.applied] == [
        f"pass:{span}:0" for span in spans
    ]
    assert [operation.mutation_scope for operation in memory.applied] == [
        f"pass:{span}" for span in spans
    ]
    assert [operation.provenance.origin_id for operation in memory.applied] == spans
    assert {operation.provenance.run_id for operation in memory.applied} == {RUN}
    assert [plan.run_id for plan in store.plans] == [RUN, RUN, RUN]
    # Each call judges its own batch and reads the earlier ones as context, never as its own.
    parts = [_sections(request) for request in model.asked]
    assert [("a" * 400 in new, "b" * 400 in new, "c" * 400 in new) for _, new in parts] == [
        (True, False, False),
        (False, True, False),
        (False, False, True),
    ]
    assert [("a" * 400 in old, "b" * 400 in old) for old, _ in parts] == [
        (False, False),
        (True, False),
        (True, True),
    ]
    assert store.context_reads == [(n, CONTEXT_ROUNDS) for n in (1, 2, 3)]
    # Each batch is searched by what came before it as well as by itself.
    assert memory.queries == [
        "a" * 400,
        "a" * 400 + "\n" + "b" * 400,
        "a" * 400 + "\n" + "b" * 400 + "\n" + "c" * 400,
    ]
    assert store.row.read_through == 3


async def test_a_stored_plan_is_resumed_first_and_the_pass_then_reads_on() -> None:
    plan = SessionMemoryPlan(
        read_from=1,
        read_through=1,
        writes=(PlannedWrite(action="add", kind="fact", body="Lives in Gothenburg."),),
        run_id="run-0",
    )
    store = _Store(
        turns=_turns("a" * 400, "b" * 400, "c" * 400), row=SessionMemoryProgress(0, plan)
    )
    memory, model = _Memory(), _Model([_verdict("Prefers tea.")])

    assert await _run(_pass(store, memory, model=model), run_id="run-9") == "applied"

    # The plan settles without the model, under the Run it was stored for and its own keys;
    # what follows it is asked once, as one batch, and attributed to the Run that asked.
    assert store.events == ["claim", "advance 1", "plan 2-3 1", "advance 3", "release"]
    resumed, fresh = memory.applied
    assert (resumed.provenance.run_id, resumed.idempotency_key) == (
        "run-0",
        f"pass:{SESSION}:1-1:0",
    )
    assert (fresh.provenance.run_id, fresh.idempotency_key) == ("run-9", f"pass:{SESSION}:2-3:0")
    (request,) = model.asked
    context, newest = _sections(request)
    assert "b" * 400 in newest and "c" * 400 in newest and "a" * 400 not in newest
    assert "a" * 400 in context


async def test_a_conversation_longer_than_one_fetch_window_is_read_to_its_end() -> None:
    store = _Store(turns=_turns(*["ok"] * module._TURN_FETCH_LIMIT, "later text"))
    model = _Model()

    assert await _run(_pass(store, model=model, batch=1_000_000)) == "applied"
    assert len(model.asked) == 2
    assert store.row.read_through == module._TURN_FETCH_LIMIT + 1


async def test_a_claim_that_cannot_be_renewed_stops_the_pass_before_it_asks_again(
    caplog: pytest.LogCaptureFixture,
) -> None:
    for allowed, asked, read_through in ((0, 1, 1), (1, 2, 2)):
        store = _Store(
            turns=_turns("a" * 400, "b" * 400, "c" * 400), refuses_after={"renew": allowed}
        )
        model = _Model()
        caplog.clear()

        with caplog.at_level(logging.WARNING, logger=module.logger.name):
            assert await _run(_pass(store, model=model, batch=150)) == "superseded"

        # The batches before the refused renewal stand; nothing is asked, stored or advanced after.
        assert len(model.asked) == asked
        assert store.row.read_through == read_through
        assert store.events[-2:] == ["renew", "release"]
        # And the lost claim is said once, with the Session it was lost for.
        (warning,) = [r for r in caplog.records if "lost its claim" in r.getMessage()]
        assert warning.session_id == SESSION  # type: ignore[attr-defined]


async def test_a_plan_or_watermark_the_store_refuses_ends_the_pass() -> None:
    for refused, allowed, asked in (("plan", 0, 1), ("advance", 1, 2)):
        store = _Store(
            turns=_turns("a" * 400, "b" * 400, "c" * 400), refuses_after={refused: allowed}
        )
        memory, model = _Memory(), _Model([_verdict("x."), _verdict("y."), _verdict("z.")])

        assert await _run(_pass(store, memory, model=model, batch=150)) == "superseded"

        # A pass whose claim is gone neither reads nor asks on: it was told so by the store.
        assert len(model.asked) == asked
        assert store.events[-1] == "release"
        assert "plan 3-3 1" not in store.events


async def test_a_switch_turned_off_between_batches_ends_the_pass_before_the_next_is_read() -> None:
    gate = _Gate()
    store = _Store(turns=_turns("a" * 400, "b" * 400, "c" * 400))
    memory = _Memory(after_apply=lambda: setattr(gate, "enabled", False))
    model = _Model([_verdict("First."), _verdict("Second.")])

    assert await _run(_pass(store, memory, model=model, gate=gate, batch=150)) == "disabled"

    # The first batch was written while the switch was on; the owner then turned it off, so the
    # next batch is never read. The rest is not mined later either: turning memory back on
    # restarts the Session at its latest entry.
    assert len(model.asked) == 1
    assert [operation.body for operation in memory.applied] == ["First."]
    assert store.events == ["claim", "plan 1-1 1", "advance 1", "release"]
    assert store.row.read_through == 1


async def test_a_cancelled_pass_releases_its_claim_and_keeps_the_batches_it_applied() -> None:
    store = _Store(turns=_turns("a" * 400, "b" * 400, "c" * 400))
    memory = _Memory()
    second_call = asyncio.Event()

    class _Hangs(_Model):
        async def __call__(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
            if self.asked:  # the second call never answers
                second_call.set()
                await asyncio.Event().wait()
            return await super().__call__(messages, **kwargs)

    task = asyncio.create_task(
        _run(_pass(store, memory, model=_Hangs([_verdict("First.")]), batch=150))
    )
    await asyncio.wait_for(second_call.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The first batch is applied and advanced; the second was never stored; the claim is let go.
    assert [operation.body for operation in memory.applied] == ["First."]
    assert (store.row.read_through, store.row.plan) == (1, None)
    assert store.events == ["claim", "plan 1-1 1", "advance 1", "renew", "release"]


async def test_writes_the_store_refuses_in_one_batch_do_not_stop_the_next() -> None:
    store = _Store(turns=_turns("a" * 400, "b" * 400, "c" * 400))
    model = _Model([_verdict("x."), _verdict("y."), _verdict("z.")])

    assert await _run(_pass(store, _Memory(refuse=True), model=model, batch=150)) == "applied"

    assert len(model.asked) == 3
    assert store.row.read_through == 3


async def test_a_stored_plan_found_while_memory_is_off_writes_nothing() -> None:
    plan = SessionMemoryPlan(
        read_from=1,
        read_through=1,
        writes=(PlannedWrite(action="add", kind="fact", body="Lives in Gothenburg."),),
        run_id="run-0",
    )
    store = _Store(turns=_turns("a" * 400, "b" * 400), row=SessionMemoryProgress(0, plan))
    memory, model = _Memory(), _Model()

    assert await _run(_pass(store, memory, model=model, gate=_Gate(enabled=False))) == "disabled"

    assert memory.applied == [] and model.asked == []
    assert store.events == ["claim", "advance 1", "release"]


async def test_text_from_while_memory_was_off_is_neither_read_nor_shown_as_context() -> None:
    gate = _Gate()
    store = _Store(turns=_turns("before-the-switch one", "before-the-switch two"))
    model = _Model()
    session_pass = _pass(store, model=model, gate=gate)

    # Read while memory was on.
    assert await _run(session_pass) == "applied"
    # The owner turns memory off. Runs fail and write entries, so no pass is scheduled and none
    # sees memory off. The owner turns it back on, and the Run store restarts the Session.
    gate.enabled = False
    store.turns += [
        SessionTurn(sequence=3, role="user", text="while-off one"),
        SessionTurn(sequence=4, role="assistant", text="while-off two"),
    ]
    gate.enabled = True
    store.restart()
    # A Run succeeds after that; its pass reads only what followed.
    store.turns += [
        SessionTurn(sequence=5, role="user", text="after-on one"),
        SessionTurn(sequence=6, role="assistant", text="after-on two"),
    ]
    assert await _run(session_pass) == "applied"
    store.turns += [SessionTurn(sequence=7, role="user", text="after-on three")]
    assert await _run(session_pass) == "applied"

    first, right_after, later = model.asked
    assert "while-off" not in "".join(m["content"] for request in model.asked for m in request)
    # Right after the restart nothing precedes the new text that may be shown as context.
    assert _sections(right_after)[0] == ""
    assert "after-on one" in _sections(right_after)[1]
    # Then context works as it always did, from text written while memory was on, and no further.
    context, newest = _sections(later)
    assert "after-on one" in context and "after-on two" in context
    assert "before-the-switch" not in context and "after-on three" in newest
    assert "before-the-switch" in _sections(first)[1]


async def test_a_restart_during_a_pass_moves_where_it_reads_next() -> None:
    store = _Store(turns=_turns("a" * 400, "b" * 400, "c" * 400))
    # The owner turns memory off and back on while the first batch is being applied.
    memory = _Memory(after_apply=store.restart)
    model = _Model([_verdict("First.")])

    assert await _run(_pass(store, memory, model=model, batch=150)) == "applied"

    # The store answered the advance with the watermark the restart had moved to, and the pass
    # read on from there: the rest of what the restart skipped is never asked about.
    assert len(model.asked) == 1
    assert store.events == ["claim", "plan 1-1 1", "advance 1", "release"]
    assert store.row.read_through == 3


async def test_a_short_reply_is_searched_by_the_context_it_answers() -> None:
    store = _Store(
        turns=[
            *_turns("Keep every conclusion to three sentences.", "Understood, three at most."),
            SessionTurn(sequence=3, role="user", text="yes, exactly that"),
        ],
        row=SessionMemoryProgress(2),
    )
    memory = _Memory()

    assert await _run(_pass(store, memory, model=_Model())) == "applied"

    assert memory.queries == [
        "Keep every conclusion to three sentences.\nUnderstood, three at most.\nyes, exactly that"
    ]


async def test_the_search_query_is_the_last_two_thousand_characters_of_context_and_batch() -> None:
    async def query_for(*texts: str, read_through: int) -> str:
        turns = [
            SessionTurn(sequence=index, role="user", text=text)
            for index, text in enumerate(texts, start=1)
        ]
        store = _Store(turns=turns, row=SessionMemoryProgress(read_through))
        memory = _Memory()
        await _run(_pass(store, memory, model=_Model()))
        (query,) = memory.queries
        return query

    # The tail of both, so a short batch keeps the end of what came before it...
    assert (
        await query_for("o" * 1500, "n" * 800, read_through=1)
        == ("o" * 1500 + "\n" + "n" * 800)[-2_000:]
    )
    # ...a batch long enough on its own is searched by itself alone...
    assert await query_for("o" * 1500, "n" * 2_500, read_through=1) == "n" * 2_000
    # ...and with nothing before it, as ever, by the batch.
    assert await query_for("only text", read_through=0) == "only text"


async def test_a_pass_stops_at_its_batch_bound_and_the_next_one_carries_on(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(module, "_MAX_BATCHES_PER_PASS", 2)
    store = _Store(turns=_turns("a" * 400, "b" * 400, "c" * 400))
    model = _Model()
    session_pass = _pass(store, model=model, batch=150)

    with caplog.at_level(logging.WARNING, logger=module.logger.name):
        assert await _run(session_pass) == "applied"

    assert len(model.asked) == 2
    assert store.row.read_through == 2
    (warning,) = [r for r in caplog.records if "batch bound" in r.getMessage()]
    assert warning.session_id == SESSION  # type: ignore[attr-defined]

    caplog.clear()
    assert await _run(session_pass) == "applied"
    assert store.row.read_through == 3
    assert [r for r in caplog.records if "batch bound" in r.getMessage()] == []


async def test_a_pass_that_catches_up_at_its_bound_has_not_hit_it(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(module, "_MAX_BATCHES_PER_PASS", 2)
    store = _Store(turns=_turns("a" * 400, "b" * 400))

    with caplog.at_level(logging.WARNING, logger=module.logger.name):
        assert await _run(_pass(store, batch=150)) == "applied"

    assert store.row.read_through == 2
    assert [r for r in caplog.records if "batch bound" in r.getMessage()] == []


async def test_a_failed_batch_leaves_the_batches_before_it_applied() -> None:
    store = _Store(turns=_turns("a" * 400, "b" * 400, "c" * 400))
    model = _Model([_verdict("First."), RuntimeError("provider down")])
    memory = _Memory()

    assert await _run(_pass(store, memory, model=model, batch=150)) == "model_failed"

    assert [operation.body for operation in memory.applied] == ["First."]
    assert store.row.read_through == 1
    assert store.events[-1] == "release"


async def test_a_pass_reads_over_entries_its_store_withholds_instead_of_waiting_for_them() -> None:
    """The store never returns a Child's notification, so the sequences it holds are gaps."""
    store = _Store(
        turns=[
            SessionTurn(sequence=1, role="user", text="x" * 400),
            SessionTurn(sequence=3, role="assistant", text="y" * 400),
        ]
    )
    model = _Model()
    session_pass = _pass(store, model=model)

    assert await _run(session_pass) == "applied"
    store.turns.append(SessionTurn(sequence=5, role="user", text="z" * 400))
    assert await _run(session_pass) == "applied"

    assert [event for event in store.events if event.startswith(("plan", "advance"))] == [
        "plan 1-3 0",
        "advance 3",
        "plan 4-5 0",
        "advance 5",
    ]
    assert "z" * 400 not in model.asked[0][1]["content"]
    # What the first pass judged is context for the second now, and no longer the part to judge.
    context, newest = _sections(model.asked[1])
    assert "x" * 400 not in newest and "z" * 400 in newest
    assert "x" * 400 in context


async def test_a_long_user_turn_in_the_context_is_shown_whole_and_an_assistant_one_is_cut() -> None:
    long_user = "USER-HEAD " + "u" * 3_000 + " USER-TAIL"
    long_assistant = "ASSISTANT-HEAD " + "a" * 3_000 + " ASSISTANT-TAIL"
    store = _Store(
        turns=_turns(long_user, long_assistant, "Please answer in German."),
        row=SessionMemoryProgress(2),
    )
    model = _Model()

    assert await _run(_pass(store, model=model)) == "applied"

    (request,) = model.asked
    context, newest = _sections(request)
    assert f"[user] {long_user}" in context
    assert f"[assistant] {long_assistant[:500]}…" in context
    assert "ASSISTANT-TAIL" not in request[1]["content"]
    assert "Please answer in German." in newest


async def test_a_write_the_store_refuses_does_not_stop_the_rest() -> None:
    store = _Store(turns=_turns("x" * 400))
    model = _Model([_verdict("a.", "b.")])

    assert await _run(_pass(store, _Memory(refuse=True), model=model)) == "applied"
    assert store.row.read_through == 1


async def test_the_owners_words_a_verdict_quotes_reach_the_store_only_if_they_are_theirs() -> None:
    said = "以后图表都用深色主题，别用饼图。"
    store = _Store(
        turns=[
            SessionTurn(sequence=1, role="user", text="先给我看上季度的数据"),
            SessionTurn(sequence=2, role="assistant", text="好的，我会用深色主题。"),
            SessionTurn(sequence=3, role="user", text=said),
        ],
        row=SessionMemoryProgress(2),
    )
    memory = _Memory()
    writes = [
        ("The owner prefers charts in a dark theme.", "以后图表都用深色主题"),
        ("The owner dislikes pie charts.", "别用饼图"),
        # The Agent's words, an earlier turn's words and words nobody said: the writes stand, bare.
        ("The Agent switched to a dark theme.", "我会用深色主题"),
        ("The owner reads quarterly data first.", "先给我看上季度的数据"),
        ("The owner loves dark mode.", "I love dark mode so much"),
    ]
    verdict = ConsolidationVerdict(
        writes=[
            VerdictWrite(action="add", kind="preference", body=body, quote=quote)
            for body, quote in writes
        ]
    )

    assert await _run(_pass(store, memory, model=_Model([verdict]), batch=1_000)) == "applied"

    expected = ["以后图表都用深色主题", "别用饼图", None, None, None]
    # What the plan keeps, and so what a resume would apply, is only what the host verified.
    assert [write.quote for write in store.plans[0].writes] == expected
    assert [operation.quote for operation in memory.applied] == expected
    assert [operation.body for operation in memory.applied] == [body for body, _ in writes]


async def test_a_quote_over_the_bound_costs_the_write_its_quote_and_nothing_else() -> None:
    long = "字" * 200
    store = _Store(turns=_turns(long))
    memory = _Memory()
    verdict = ConsolidationVerdict(
        writes=[VerdictWrite(action="add", kind="fact", body="Writes long turns.", quote=long)]
    )

    assert await _run(_pass(store, memory, model=_Model([verdict]))) == "applied"

    (operation,) = memory.applied
    assert (operation.body, operation.quote) == ("Writes long turns.", None)
    assert store.row.read_through == 1


async def test_a_resumed_plan_applies_the_quote_it_stored() -> None:
    plan = SessionMemoryPlan(
        read_from=1,
        read_through=1,
        writes=(
            PlannedWrite(
                action="add", kind="fact", body="Lives in Gothenburg.", quote="I live in Gothenburg"
            ),
            # A write the owner did not quote carries none.
            PlannedWrite(action="add", kind="fact", body="Works in finance."),
        ),
        run_id="run-0",
    )
    store = _Store(turns=_turns("I live in Gothenburg"), row=SessionMemoryProgress(0, plan))
    memory = _Memory()

    assert await _run(_pass(store, memory)) == "resumed"

    assert [operation.quote for operation in memory.applied] == ["I live in Gothenburg", None]


def _settled(
    run_id: str,
    *,
    kind: RunKind = "answer",
    status: Literal["succeeded", "failed", "cancelled"] = "succeeded",
) -> SettledRun:
    return SettledRun(owner_id=OWNER, run_id=run_id, run_kind=kind, status=status)


async def test_the_runner_runs_one_pass_per_session_and_one_more_for_a_run_that_waited() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()
    runs: list[tuple[str, str]] = []
    model = _Model()

    class _Pass(SessionMemoryPass):
        async def run(self, *, owner_id: str, session_id: str, run_id: str) -> Any:
            runs.append((session_id, run_id))
            started.set()
            await release.wait()
            if len(runs) == 2:
                finished.set()
            return "applied"

    store = _Store()
    session_pass = _Pass(
        memory=_Memory(),  # type: ignore[arg-type]
        store=store,
        gate=_Gate(),
        model=model,
        settings=SessionPassSettings(max_batch_tokens=1, session_record_limit=5),
        holder="h",
    )
    runner = SessionMemoryPassRunner(session_pass=session_pass, store=store)
    runner.run_settled(_settled("answer-1"))
    await started.wait()
    runner.run_settled(_settled("answer-2"))
    runner.run_settled(_settled("answer-3"))
    runner.run_settled(_settled("answer-no-session"))
    runner.run_settled(_settled("retrieval-1", kind="retrieval"))
    await asyncio.sleep(0)
    release.set()
    await asyncio.wait_for(finished.wait(), timeout=2)
    await runner.aclose()

    # The first pass, then exactly one more for the Runs that settled meanwhile; a Run of
    # another kind, or one with no Agent Session, schedules nothing. The pass the waiting
    # Runs share is attributed to the latest of them: it reads everything they wrote.
    assert runs == [(SESSION, "answer-1"), (SESSION, "answer-3")]
    # Closing the runner closes the model its pass owns, and schedules nothing more.
    assert model.closed
    runner.run_settled(_settled("answer-4"))
    assert runs == [(SESSION, "answer-1"), (SESSION, "answer-3")]


async def test_a_model_failure_is_counted_against_the_session_with_the_first_wait_and_the_ceiling() -> (
    None
):
    store = _Store(turns=_turns("x" * 400))

    assert await _run(_pass(store, model=_Model([RuntimeError("provider down")]))) == "model_failed"

    # Counted while the claim is still held, so the store can fence it by the holder.
    assert store.events == ["claim", "back off", "release"]
    assert store.backoffs == [(SESSION_BACKOFF_FIRST_SECONDS, SESSION_BACKOFF_MAX_SECONDS)]
    assert (SESSION_BACKOFF_FIRST_SECONDS, SESSION_BACKOFF_MAX_SECONDS) == (600.0, 21_600.0)


async def test_a_batch_that_fails_after_batches_that_advanced_is_counted_after_them() -> None:
    store = _Store(turns=_turns("a" * 400, "b" * 400, "c" * 400))
    model = _Model([_verdict("First."), RuntimeError("provider down")])

    assert await _run(_pass(store, model=model, batch=150)) == "model_failed"

    # The batch that advanced cleared whatever was counted before it; the failure counts anew.
    assert store.events == [
        "claim",
        "plan 1-1 1",
        "advance 1",
        "renew",
        "back off",
        "release",
    ]


async def test_an_error_nothing_expected_is_counted_too_and_still_raised() -> None:
    class _BrokenSearch(_Memory):
        async def search(
            self, *, owner_id: str, query: str, limit: int
        ) -> tuple[MemoryRecord, ...]:
            raise RuntimeError("memory store down")

    store = _Store(turns=_turns("x" * 400))

    with pytest.raises(RuntimeError, match="memory store down"):
        await _run(_pass(store, _BrokenSearch()))

    assert store.events == ["claim", "back off", "release"]
    assert store.row.read_through == 0


@pytest.mark.parametrize(
    "case",
    ["applied", "nothing_new", "disabled", "busy", "superseded", "resumed"],
)
async def test_only_a_failure_of_the_session_is_counted_against_it(case: str) -> None:
    plan = SessionMemoryPlan(
        read_from=1,
        read_through=1,
        writes=(PlannedWrite(action="add", kind="fact", body="Lives in Gothenburg."),),
        run_id="run-0",
    )
    gate = _Gate(enabled=case != "disabled")
    store = _Store(
        turns=_turns("x" * 400),
        row=SessionMemoryProgress(0, plan) if case == "resumed" else SessionMemoryProgress(0),
        held_by_other=case == "busy",
        refuses_after={"plan": 0} if case == "superseded" else {},
    )
    if case == "nothing_new":
        store.row = SessionMemoryProgress(1)

    assert await _run(_pass(store, gate=gate)) == case

    # Memory off, a claim another process holds, a claim lost, and nothing to read are no fault of the
    # Session's; and a pass that moved the watermark has no failure to count.
    assert store.backoffs == []


async def test_a_failure_that_cannot_be_recorded_is_logged_and_hides_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = _Store(turns=_turns("x" * 400), backoff_error=RuntimeError("database down"))

    with caplog.at_level(logging.WARNING, logger=module.logger.name):
        assert (
            await _run(_pass(store, model=_Model([RuntimeError("provider down")])))
            == "model_failed"
        )

    # The pass ends as it would have, and lets its claim go; the missed count is said once.
    assert store.events == ["claim", "back off", "release"]
    (warning,) = [r for r in caplog.records if "could not record its failure" in r.getMessage()]
    assert warning.session_id == SESSION  # type: ignore[attr-defined]


OTHER_SESSION = "33333333-3333-3333-3333-333333333333"


class _RecordingPass(SessionMemoryPass):
    """Stands in for the pass: records each (Session, Run) asked of it, and waits to be let go."""

    def __init__(self, store: _Store, *, release: asyncio.Event | None = None) -> None:
        super().__init__(
            memory=_Memory(),  # type: ignore[arg-type]
            store=store,
            gate=_Gate(),
            model=_Model(),
            settings=SessionPassSettings(max_batch_tokens=1, session_record_limit=5),
            holder="h",
        )
        self.runs: list[tuple[str, str]] = []
        self.inside = 0
        self.most_inside = 0
        self._release = release

    async def run(self, *, owner_id: str, session_id: str, run_id: str) -> Any:
        self.runs.append((session_id, run_id))
        self.inside += 1
        self.most_inside = max(self.most_inside, self.inside)
        try:
            if self._release is not None:
                await self._release.wait()
        finally:
            self.inside -= 1
        return "applied"


async def _until(condition: Callable[[], bool]) -> None:
    async with asyncio.timeout(2):
        while not condition():
            await asyncio.sleep(0.001)


def _pending(session_id: str, run_id: str) -> PendingSession:
    return PendingSession(owner_id=OWNER, session_id=session_id, run_id=run_id)


async def test_a_sweep_hands_each_pending_session_to_a_pass_attributed_to_its_latest_run() -> None:
    store = _Store(pending=[_pending(SESSION, "run-8"), _pending(OTHER_SESSION, "run-9")])
    session_pass = _RecordingPass(store)
    runner = SessionMemoryPassRunner(session_pass=session_pass, store=store)

    assert await runner.sweep() == 2
    await _until(lambda: len(session_pass.runs) == 2)
    await runner.aclose()

    # The store names the Run, and the pass is asked as if that Run's success had just scheduled it.
    assert sorted(session_pass.runs) == [(SESSION, "run-8"), (OTHER_SESSION, "run-9")]
    # A look is bounded, whatever the store holds.
    assert store.looks == [SESSION_SWEEP_BATCH] == [20]


async def test_a_sweep_with_nothing_pending_schedules_nothing() -> None:
    store = _Store()
    session_pass = _RecordingPass(store)
    runner = SessionMemoryPassRunner(session_pass=session_pass, store=store)

    assert await runner.sweep() == 0
    await runner.aclose()

    assert session_pass.runs == []


async def test_swept_passes_share_the_per_process_bound_with_the_passes_runs_schedule() -> None:
    release = asyncio.Event()
    sessions = [f"00000000-0000-0000-0000-00000000000{n}" for n in range(1, 6)]
    store = _Store(pending=[_pending(session, f"run-{n}") for n, session in enumerate(sessions)])
    session_pass = _RecordingPass(store, release=release)
    runner = SessionMemoryPassRunner(session_pass=session_pass, store=store)

    assert await runner.sweep() == 5
    await _until(lambda: session_pass.inside == 2)
    await asyncio.sleep(0.02)
    # Five were handed over and two run: the rest wait for a slot, as passes after Runs do.
    assert (session_pass.inside, len(session_pass.runs)) == (2, 2)
    release.set()
    await _until(lambda: len(session_pass.runs) == 5)
    await runner.aclose()

    assert session_pass.most_inside == 2


async def test_a_sweep_leaves_a_session_whose_pass_is_running_here_alone() -> None:
    release = asyncio.Event()
    store = _Store(pending=[_pending(SESSION, "run-8")])
    session_pass = _RecordingPass(store, release=release)
    runner = SessionMemoryPassRunner(session_pass=session_pass, store=store)
    runner.run_settled(_settled("answer-1"))
    await _until(lambda: session_pass.inside == 1)

    # The running pass reads to the end by itself, so a sweep neither doubles it nor queues another.
    assert await runner.sweep() == 0
    release.set()
    await asyncio.sleep(0.02)
    await runner.aclose()

    assert session_pass.runs == [(SESSION, "answer-1")]


async def test_a_run_that_settles_during_a_swept_pass_still_gets_the_pass_it_asks_for() -> None:
    release = asyncio.Event()
    store = _Store(pending=[_pending(SESSION, "run-8")])
    session_pass = _RecordingPass(store, release=release)
    runner = SessionMemoryPassRunner(session_pass=session_pass, store=store)
    assert await runner.sweep() == 1
    await _until(lambda: session_pass.inside == 1)

    # A Run is new text; unlike a sweep it asks for one more pass once the running one ends.
    runner.run_settled(_settled("answer-2"))
    await asyncio.sleep(0.01)
    release.set()
    await _until(lambda: len(session_pass.runs) == 2)
    await runner.aclose()

    assert session_pass.runs == [(SESSION, "run-8"), (SESSION, "answer-2")]


async def test_the_sweeper_looks_shortly_after_it_starts_and_then_on_its_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "SESSION_SWEEP_FIRST_SECONDS", 0.01)
    monkeypatch.setattr(module, "SESSION_SWEEP_INTERVAL_SECONDS", 0.03)
    store = _Store()
    runner = SessionMemoryPassRunner(session_pass=_RecordingPass(store), store=store)

    assert store.looks == []
    runner.start()
    started = runner._sweeper
    runner.start()  # idempotent: one sweeper, not two
    assert runner._sweeper is started
    await _until(lambda: len(store.looks) >= 3)
    await runner.aclose()

    assert set(store.looks) == {SESSION_SWEEP_BATCH}


def test_a_process_first_looks_half_a_minute_in_and_then_every_ten_minutes() -> None:
    assert module.SESSION_SWEEP_FIRST_SECONDS == 30.0
    assert module.SESSION_SWEEP_INTERVAL_SECONDS == 600.0


async def test_a_look_that_fails_is_logged_and_the_sweeper_looks_again(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(module, "SESSION_SWEEP_FIRST_SECONDS", 0.005)
    monkeypatch.setattr(module, "SESSION_SWEEP_INTERVAL_SECONDS", 0.01)
    store = _Store(pending=[_pending(SESSION, "run-8")], look_errors=[RuntimeError("db down")])
    session_pass = _RecordingPass(store)
    runner = SessionMemoryPassRunner(session_pass=session_pass, store=store)

    with caplog.at_level(logging.WARNING, logger=module.logger.name):
        runner.start()
        await _until(lambda: len(session_pass.runs) >= 1)
    await runner.aclose()

    assert [r for r in caplog.records if "sweep failed" in r.getMessage()]
    # The first look failed and the second found the Session: one failure never ends the sweeper.
    assert len(store.looks) >= 2
    assert session_pass.runs[0] == (SESSION, "run-8")


async def test_closing_stops_the_sweeper_and_the_passes_it_started() -> None:
    release = asyncio.Event()
    store = _Store(pending=[_pending(SESSION, "run-8")])
    session_pass = _RecordingPass(store, release=release)
    runner = SessionMemoryPassRunner(session_pass=session_pass, store=store)
    runner.start()
    assert await runner.sweep() == 1
    await _until(lambda: session_pass.inside == 1)
    sweeper = runner._sweeper
    assert sweeper is not None

    await runner.aclose()

    # The sweeper and the pass it started are cancelled and awaited; nothing looks again.
    assert sweeper.cancelled() and session_pass.inside == 0
    looks = len(store.looks)
    assert await runner.sweep() == 0
    runner.start()
    assert runner._sweeper is None
    await asyncio.sleep(0.02)
    assert len(store.looks) == looks + 1  # the direct sweep() above asked; nothing else did


class _DerivedStore(_Store):
    """Answers a sweep as the database does: a Session is pending while its turns are past its watermark."""

    async def pending_session_memory(self, *, limit: int) -> list[PendingSession]:
        self.looks.append(limit)
        if any(turn.sequence > self.row.read_through for turn in self.turns):
            return [_pending(SESSION, "run-latest")]
        return []


async def test_a_session_a_sweep_found_is_read_by_the_pass_and_is_not_found_again() -> None:
    store = _DerivedStore(turns=_turns("以后都用中文回答我。", "好的。"))
    memory = _Memory()
    model = _Model([_verdict("The owner prefers answers in Chinese.")])
    runner = SessionMemoryPassRunner(session_pass=_pass(store, memory, model=model), store=store)

    # A Run succeeded and its pass was lost with the process: the tail is still unread.
    assert await runner.sweep() == 1
    await _until(lambda: store.events[-1:] == ["release"])

    (operation,) = memory.applied
    assert operation.provenance.run_id == "run-latest"
    assert store.row.read_through == 2
    # Caught up, it is no longer pending; a second look finds nothing and schedules nothing.
    assert await runner.sweep() == 0
    await runner.aclose()
    assert len(model.asked) == 1


def test_the_prompt_states_the_verdict_contract_the_parser_enforces() -> None:
    """The pass can be asked for json_object with no schema, so the prompt is the contract.

    A provider told only "return JSON" has no way to know the verdict's shape unless the
    prompt says it; the model then guesses keys, and `extra="forbid"` rejects the guess —
    which is exactly what a deployment on a json_object role measured. This pins that the
    prompt names every field of the shape the parser accepts, and the empty verdict.
    """
    messages = consolidation_request(
        turns=[ConversationTurn(role="user", text="one")],
        candidates=[],
        session_records=[],
        record_limit=5,
    ).messages
    prompt = messages[0]["content"]
    prompt = " ".join(prompt.split())  # one flow, so a re-wrapped line never breaks a pin
    for contract_word in (
        "writes",
        "action",
        "kind",
        "body",
        "targets",
        '"add"',
        '"update"',
        '"fuse"',
        '"preference"',
        '"fact"',
        '"experience"',
    ):
        assert contract_word in prompt, contract_word
    assert '{"writes": []}' in prompt
    # The exemplar an add copies must name no targets, or the planner drops the write.
    assert (
        '"action": "add", "kind": "preference", "body": "The owner prefers ...", "targets": []'
        in prompt
    )
    # The model names memories by the number they are listed with, never by an id.
    assert "by the numbers they are listed with" in prompt
    assert "target_ids" not in prompt
    assert (
        "Existing: [1] (preference, 2026-03-02 10:00) The owner prefers short conclusions."
        in prompt
    )
    # Times come from the store, and the more recent of two conflicting memories is likelier current.
    assert "the more recently written one is likelier current" in prompt
    assert "never put a date or time in a memory's body" in prompt
    # The decision rules the examples teach, from the sources the prompt was written against.
    assert "One write holds one statement" in prompt
    assert "No change is not a write" in prompt
    assert "Save a statement only when it is all three" in prompt
    assert "a one-off instruction stays one-off" in prompt
    assert "The memories shown span this and earlier conversations alike" in prompt
    assert "Updating or merging existing records never counts" in prompt
    assert 'Answer: {"writes": []}' in prompt


def test_the_prompt_directs_the_model_at_how_the_owner_wants_work_done() -> None:
    """The three directions the 2026-10-17 eval (docs/memory-pass-eval.md) added to the prompt.

    A pass that does not reason found none of the owner's real preferences; with the prompt
    below it recalls them. These are what the revision changed, so a later edit cannot drop one
    without the eval's result going stale.
    """
    messages = consolidation_request(
        turns=[ConversationTurn(role="user", text="one")],
        candidates=[],
        session_records=[],
        record_limit=5,
    ).messages
    prompt = " ".join(messages[0]["content"].split())
    # The owner's reactions to how the work is done are where a preference usually is.
    assert "Look hardest at the owner's own reactions to the work" in prompt
    assert "What the task is about" in prompt
    # A third party's taste is theirs, not the owner's.
    assert "a third party's wish" in prompt
    # The Agent's own workspace is the same for every owner, so it is nobody's experience.
    assert "The Agent's own workspace" in prompt
    assert "nobody's memory" in prompt


def test_the_prompt_says_the_earlier_turns_are_context_and_were_already_judged() -> None:
    """The one sentence the earlier turns added: what they are for, and what they are not."""
    prompt = " ".join(
        consolidation_request(
            turns=[ConversationTurn(role="user", text="one")],
            candidates=[],
            session_records=[],
            record_limit=5,
        )
        .messages[0]["content"]
        .split()
    )

    assert "A few turns just before the newest part may come first as context only" in prompt
    assert 'what the newest part refers to (what "这样" or "that" points at)' in prompt
    assert "what they said on their own was already judged by an earlier pass" in prompt
