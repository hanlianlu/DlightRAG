# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The goal round driver continues durable active goals and records how a round ended (ADR 0077)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, cast

import pytest

from dlightrag.application.answer.goal_rounds import (
    GoalRoundDriver,
    GoalRoundReservation,
    goal_round_prompt,
)
from dlightrag.application.answer_runs.service import (
    ActiveRunLimitError,
    AnswerRequest,
    AnswerService,
)
from dlightrag.application.runs import RunService
from dlightrag.engine.answer.goal import Goal, GoalPhase
from dlightrag.engine.answer.goal_service import (
    MODEL_OUTPUT_LIMIT_REASON,
    ROUND_LIMIT_REASON,
    AnswerGoalService,
)
from dlightrag.engine.runtime.coordinator import SettledRun
from dlightrag.engine.runtime.records import SessionWorkBusyError
from tests.support.goals import FakeGoalStore

_OWNER = "owner-1"
_SESSION = "session-1"
_RUN = "run-1"
_CONVERSATION = "conversation-1"


@dataclass
class FakeAnswers:
    requests: list[tuple[AnswerRequest, str | None]] = field(default_factory=list)
    raises: Exception | None = None

    async def create(
        self,
        *,
        request: AnswerRequest,
        owner_id: str,
        caller: object,
        idempotency_key: str | None = None,
    ) -> object:
        del owner_id, caller
        if self.raises is not None:
            raise self.raises
        self.requests.append((request, idempotency_key))
        return SimpleNamespace(run_id="next-run")


@dataclass
class FakeRuns:
    views: dict[str, Any] = field(default_factory=dict)
    listed: tuple[Any, ...] = ()
    list_calls: list[dict[str, object]] = field(default_factory=list)

    async def get(self, *, owner_id: str, run_id: str) -> Any | None:
        del owner_id
        return self.views.get(run_id)

    async def list(
        self,
        *,
        owner_id: str,
        after_run_id: str | None = None,
        limit: int = 50,
        newest_first: bool = False,
    ) -> tuple[Any, ...]:
        self.list_calls.append({"limit": limit, "newest_first": newest_first})
        del owner_id, after_run_id
        return self.listed


def _view(
    *,
    terminal: bool = True,
    run_kind: str = "answer",
    session_id: str = _SESSION,
    continuation_kind: str | None = None,
    mode: str = "research",
    requested_skill: str | None = None,
    access_scope_kind: str = "owner",
    goal_id: str | None = None,
    goal_revision: int | None = None,
    goal_round: int | None = None,
    status: str = "succeeded",
    error_kind: str | None = None,
    result: dict[str, Any] | None = None,
) -> Any:
    request: dict[str, Any] = {
        "agent_session_id": session_id,
        "workspaces": ["workspace-a"],
        "mode": mode,
        "agent_lane_id": "main",
        "requested_skill": requested_skill,
        "continuation_kind": continuation_kind,
    }
    if goal_id is not None:
        request["goal_id"] = goal_id
    if goal_revision is not None:
        request["goal_revision"] = goal_revision
    if goal_round is not None:
        request["goal_round"] = goal_round
    return SimpleNamespace(
        run_id=_RUN,
        run_kind=run_kind,
        submitted_by=_OWNER,
        access_scope_kind=access_scope_kind,
        request=request,
        events_trimmed_at=None,
        terminal=terminal,
        status=status,
        error_kind=error_kind,
        result=result,
    )


def _settled(status: str = "succeeded", *, stop_reason: str | None = None) -> SettledRun:
    return SettledRun(
        owner_id=_OWNER,
        run_id=_RUN,
        run_kind="answer",
        status=status,  # type: ignore[arg-type]
        stop_reason=stop_reason,
    )


def _driver(
    *,
    goal_service: AnswerGoalService,
    answers: FakeAnswers,
    runs: FakeRuns,
    round_turns: Any | None = None,
) -> GoalRoundDriver:
    return GoalRoundDriver(
        goals=goal_service,
        answers=cast(AnswerService, answers),
        runs=cast(RunService, runs),
        round_turns=round_turns,
    )


async def _goal(
    service: AnswerGoalService,
    *,
    max_rounds: int = 4,
    objective: str = "finish the model",
    session_id: str = _SESSION,
) -> Goal:
    return await service.create(
        owner_id=_OWNER,
        session_id=session_id,
        run_id=_RUN,
        objective=objective,
        max_rounds=max_rounds,
    )


async def _admitted(service: AnswerGoalService, goal: Goal) -> Goal:
    started, _ = await service.admit_round(
        owner_id=_OWNER,
        session_id=_SESSION,
        run_id=_RUN,
        goal_id=goal.id,
        revision=goal.revision,
    )
    return started


class FakeTurns:
    """The conversation side of the continuation: it holds the prompt for delivery."""

    def __init__(
        self,
        *,
        accepts: bool = True,
        pending: bool = False,
        accepts_now: bool = True,
        resumes_session: bool = True,
    ) -> None:
        self.accepts = accepts
        self.pending = pending
        self.accepts_now = accepts_now
        self.resumes_session = resumes_session
        self.calls: list[str] = []
        self.request_calls: list[dict[str, Any]] = []
        self.now_calls: list[dict[str, Any]] = []
        self.session_calls: list[tuple[str, str]] = []
        self.withdraws = True
        self.withdraw_calls = 0
        self.withdraw_args: list[dict[str, Any]] = []
        self.reservations: list[GoalRoundReservation] = []

    async def request_goal_round(
        self,
        *,
        owner_id: str,
        parent_run_id: str,
        prompt: str,
        workspaces: tuple[str, ...] = (),
        goal_id: str,
        goal_revision: int,
        goal_round: int,
    ) -> GoalRoundReservation | None:
        self.calls.append(prompt)
        self.request_calls.append(
            {
                "owner_id": owner_id,
                "parent_run_id": parent_run_id,
                "workspaces": workspaces,
                "goal_id": goal_id,
                "goal_revision": goal_revision,
                "goal_round": goal_round,
            }
        )
        if not self.accepts:
            return None
        reservation = GoalRoundReservation(
            request_id=f"reservation-{len(self.reservations) + 1}",
            prompt_digest="d" * 64,
        )
        self.reservations.append(reservation)
        return reservation

    async def goal_round_pending(self, *, owner_id: str, parent_run_id: str) -> bool:
        del owner_id, parent_run_id
        return self.pending

    async def withdraw_goal_round(
        self,
        *,
        owner_id: str,
        parent_run_id: str,
        request_id: str,
        prompt_digest: str,
    ) -> bool:
        self.withdraw_calls += 1
        self.withdraw_args.append(
            {
                "owner_id": owner_id,
                "parent_run_id": parent_run_id,
                "request_id": request_id,
                "prompt_digest": prompt_digest,
            }
        )
        return self.withdraws

    async def accept_goal_round_now(
        self,
        *,
        owner_id: str,
        conversation_id: str,
        prompt: str,
        workspaces: tuple[str, ...] = (),
        goal_id: str,
        goal_revision: int,
        goal_round: int,
    ) -> bool:
        self.now_calls.append(
            {
                "owner_id": owner_id,
                "conversation_id": conversation_id,
                "prompt": prompt,
                "workspaces": workspaces,
                "goal_id": goal_id,
                "goal_revision": goal_revision,
                "goal_round": goal_round,
            }
        )
        return self.accepts_now

    async def continue_session_goal(self, *, owner_id: str, session_id: str) -> bool:
        self.session_calls.append((owner_id, session_id))
        return self.resumes_session


def _runs(view: Any | None = None, *, listed: tuple[Any, ...] = ()) -> FakeRuns:
    return FakeRuns(views={_RUN: view or _view()}, listed=listed)


@pytest.mark.asyncio
async def test_a_successful_run_starts_one_round_with_the_record_it_can_correct() -> None:
    store = FakeGoalStore()
    goals = AnswerGoalService(store=store)
    goal = await _goal(goals)
    answers = FakeAnswers()
    driver = _driver(
        goal_service=goals,
        answers=answers,
        runs=_runs(_view(mode="auto", requested_skill="sealed-report")),
    )

    await driver.settle(_settled())

    assert len(answers.requests) == 1
    request, key = answers.requests[0]
    assert key is not None and uuid.UUID(key)
    assert request.continuation_kind == "goal_round"
    assert request.mode == "research"  # a fast or auto parent becomes a research round
    assert request.requested_skill == "sealed-report"
    assert request.parent_run_id == _RUN
    assert request.agent_session_id == _SESSION
    assert request.workspaces == ("workspace-a",)
    assert "<goal_round>" in request.query
    assert f'Objective: "{goal.objective}"' in request.query
    assert "Round: 1/4" in request.query
    assert store.rows[(_OWNER, _SESSION)].rounds_started == 1


@pytest.mark.asyncio
async def test_a_failed_goal_round_blocks_the_goal_instead_of_retrying() -> None:
    store = FakeGoalStore()
    goals = AnswerGoalService(store=store)
    goal = await _goal(goals)
    answers = FakeAnswers()
    driver = _driver(
        goal_service=goals,
        answers=answers,
        runs=_runs(_view(continuation_kind="goal_round", goal_id=goal.id)),
    )

    await driver.settle(_settled("failed"))

    final = store.rows[(_OWNER, _SESSION)]
    assert final.phase is GoalPhase.BLOCKED
    assert final.blocked_reason is not None
    assert "failed" in final.blocked_reason
    assert "resume to retry" in final.blocked_reason
    assert answers.requests == []


@pytest.mark.asyncio
async def test_a_human_runs_failure_does_not_change_the_goal() -> None:
    store = FakeGoalStore()
    goals = AnswerGoalService(store=store)
    goal = await _goal(goals)

    await _driver(goal_service=goals, answers=FakeAnswers(), runs=_runs()).settle(
        _settled("failed")
    )

    assert store.rows[(_OWNER, _SESSION)] == goal


@pytest.mark.asyncio
async def test_a_cancelled_goal_round_pauses_the_goal() -> None:
    store = FakeGoalStore()
    goals = AnswerGoalService(store=store)
    goal = await _goal(goals)
    answers = FakeAnswers()
    driver = _driver(
        goal_service=goals,
        answers=answers,
        runs=_runs(_view(continuation_kind="goal_round", goal_id=goal.id)),
    )

    await driver.settle(_settled("cancelled"))

    assert store.rows[(_OWNER, _SESSION)].phase is GoalPhase.PAUSED
    assert answers.requests == []


@pytest.mark.asyncio
async def test_a_cancelled_queued_goal_round_pauses_the_goal() -> None:
    store = FakeGoalStore()
    goals = AnswerGoalService(store=store)
    goal = await _goal(goals)
    view = _view(continuation_kind="goal_round", goal_id=goal.id)

    await _driver(goal_service=goals, answers=FakeAnswers(), runs=_runs(view)).run_cancelled(view)

    assert store.rows[(_OWNER, _SESSION)].phase is GoalPhase.PAUSED


@pytest.mark.asyncio
async def test_a_cancelled_human_run_does_not_pause_the_goal() -> None:
    store = FakeGoalStore()
    goals = AnswerGoalService(store=store)
    goal = await _goal(goals)
    view = _view()

    await _driver(goal_service=goals, answers=FakeAnswers(), runs=_runs(view)).run_cancelled(view)

    assert store.rows[(_OWNER, _SESSION)] == goal


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_reason", ["length", "max-tokens"])
async def test_a_round_that_ended_on_the_model_token_limit_blocks_the_goal(
    stop_reason: str,
) -> None:
    store = FakeGoalStore()
    goals = AnswerGoalService(store=store)
    goal = await _goal(goals)
    answers = FakeAnswers()
    driver = _driver(
        goal_service=goals,
        answers=answers,
        runs=_runs(_view(continuation_kind="goal_round", goal_id=goal.id)),
    )

    await driver.settle(_settled(stop_reason=stop_reason))

    final = store.rows[(_OWNER, _SESSION)]
    assert final.phase is GoalPhase.BLOCKED
    assert final.blocked_reason == MODEL_OUTPUT_LIMIT_REASON
    assert answers.requests == []


@pytest.mark.asyncio
async def test_a_paused_goal_settles_its_round_without_starting_another() -> None:
    """An already-paused goal whose round settles starts no next round."""
    store = FakeGoalStore()
    goals = AnswerGoalService(store=store)
    goal = await _goal(goals)
    started = await _admitted(goals, goal)
    paused = await goals.update(
        owner_id=_OWNER,
        session_id=_SESSION,
        run_id=_RUN,
        goal_id=started.id,
        revision=started.revision,
        action="pause",
    )
    answers = FakeAnswers()
    driver = _driver(
        goal_service=goals,
        answers=answers,
        runs=_runs(_view(continuation_kind="goal_round")),
    )

    await driver.settle(_settled())

    assert answers.requests == []
    final = store.rows[(_OWNER, _SESSION)]
    assert final == paused
    assert final.rounds_started == 1


@pytest.mark.asyncio
async def test_a_succeeded_goal_round_starts_the_next_round() -> None:
    store = FakeGoalStore()
    goals = AnswerGoalService(store=store)
    goal = await _goal(goals)
    started = await _admitted(goals, goal)
    answers = FakeAnswers()
    driver = _driver(
        goal_service=goals,
        answers=answers,
        runs=_runs(_view(continuation_kind="goal_round")),
    )

    await driver.settle(_settled())

    assert len(answers.requests) == 1
    request, _ = answers.requests[0]
    assert "Round: 2/4" in request.query
    assert store.rows[(_OWNER, _SESSION)].rounds_started == 2
    assert store.rows[(_OWNER, _SESSION)].phase is GoalPhase.ACTIVE
    assert started.phase is GoalPhase.ACTIVE


@pytest.mark.asyncio
async def test_a_paused_goal_does_not_continue_after_a_succeeded_run() -> None:
    store = FakeGoalStore()
    goals = AnswerGoalService(store=store)
    goal = await _goal(goals)
    await goals.update(
        owner_id=_OWNER,
        session_id=_SESSION,
        run_id=_RUN,
        goal_id=goal.id,
        revision=goal.revision,
        action="pause",
    )
    answers = FakeAnswers()

    await _driver(goal_service=goals, answers=answers, runs=_runs()).settle(_settled())

    assert answers.requests == []


@pytest.mark.asyncio
async def test_an_exhausted_cap_ends_blocked_not_silently() -> None:
    store = FakeGoalStore()
    goals = AnswerGoalService(store=store)
    goal = await _goal(goals, max_rounds=1)
    await _admitted(goals, goal)
    answers = FakeAnswers()
    driver = _driver(goal_service=goals, answers=answers, runs=_runs())

    await driver.settle(_settled())

    assert answers.requests == []
    final = store.rows[(_OWNER, _SESSION)]
    assert final.phase is GoalPhase.BLOCKED
    assert final.blocked_reason == ROUND_LIMIT_REASON


@pytest.mark.asyncio
async def test_a_human_run_already_in_flight_wins_at_the_front_door() -> None:
    store = FakeGoalStore()
    goals = AnswerGoalService(store=store)
    goal = await _goal(goals)
    answers = FakeAnswers(raises=SessionWorkBusyError("run-human"))
    human = _view(terminal=False)
    driver = _driver(goal_service=goals, answers=answers, runs=_runs(listed=(human,)))

    await driver.settle(_settled())

    assert answers.requests == []
    stored = store.rows[(_OWNER, _SESSION)]
    assert stored.rounds_started == 0  # no round was admitted
    assert (stored.id, stored.phase) == (goal.id, GoalPhase.ACTIVE)


@pytest.mark.asyncio
async def test_precedence_is_decided_by_the_front_door_not_by_a_run_scan() -> None:
    """The driver no longer scans recent Runs: the Session work head is the one authority."""
    goals = AnswerGoalService(store=FakeGoalStore())
    await _goal(goals)
    runs = _runs()
    driver = _driver(goal_service=goals, answers=FakeAnswers(), runs=runs)

    await driver.settle(_settled())

    assert runs.list_calls == []


@pytest.mark.asyncio
async def test_a_transient_refusal_gives_the_round_back() -> None:
    store = FakeGoalStore()
    goals = AnswerGoalService(store=store)
    goal = await _goal(goals)
    answers = FakeAnswers(raises=ActiveRunLimitError(1))
    driver = _driver(goal_service=goals, answers=answers, runs=_runs())

    await driver.settle(_settled())

    stored = store.rows[(_OWNER, _SESSION)]
    assert stored.rounds_started == 0
    assert (stored.id, stored.phase) == (goal.id, GoalPhase.ACTIVE)


@pytest.mark.asyncio
async def test_a_refused_round_creation_gives_the_round_back_and_keeps_the_goal() -> None:
    store = FakeGoalStore()
    goals = AnswerGoalService(store=store)
    goal = await _goal(goals)
    answers = FakeAnswers(raises=RuntimeError("the queue is down"))
    driver = _driver(goal_service=goals, answers=answers, runs=_runs())

    await driver.settle(_settled())

    stored = store.rows[(_OWNER, _SESSION)]
    assert stored.rounds_started == 0
    assert (stored.id, stored.phase) == (goal.id, GoalPhase.ACTIVE)


@pytest.mark.asyncio
async def test_a_web_conversation_holds_the_round_as_its_next_turn() -> None:
    goals = AnswerGoalService(store=FakeGoalStore())
    await _goal(goals)
    answers = FakeAnswers()
    turns = FakeTurns()
    delivered: list[tuple[str, str]] = []

    async def deliver(owner_id: str, parent_run_id: str) -> bool:
        delivered.append((owner_id, parent_run_id))
        return True

    driver = GoalRoundDriver(
        goals=goals,
        answers=cast(AnswerService, answers),
        runs=cast(RunService, _runs()),
        round_turns=turns,
    )
    driver.bind_pending_delivery(deliver)

    await driver.settle(_settled())

    assert len(turns.calls) == 1
    assert "<goal_round>" in turns.calls[0]
    assert answers.requests == []  # the delivery path admits the turn, not a second producer
    assert delivered == [(_OWNER, _RUN)], "a settled Session is asked to move at once"


@pytest.mark.asyncio
async def test_an_undelivered_continuation_is_not_admitted_again() -> None:
    store = FakeGoalStore()
    goals = AnswerGoalService(store=store)
    goal = await _goal(goals)
    turns = FakeTurns(pending=True)
    driver = GoalRoundDriver(
        goals=goals,
        answers=cast(AnswerService, FakeAnswers()),
        runs=cast(RunService, _runs()),
        round_turns=turns,
    )

    await driver.settle(_settled())

    assert turns.calls == []
    assert store.rows[(_OWNER, _SESSION)].rounds_started == 0
    assert store.rows[(_OWNER, _SESSION)] == goal


@pytest.mark.asyncio
async def test_a_run_without_a_conversation_falls_back_to_plain_acceptance() -> None:
    goals = AnswerGoalService(store=FakeGoalStore())
    await _goal(goals)
    answers = FakeAnswers()
    turns = FakeTurns(accepts=False)
    driver = GoalRoundDriver(
        goals=goals,
        answers=cast(AnswerService, answers),
        runs=cast(RunService, _runs()),
        round_turns=turns,
    )

    await driver.settle(_settled())

    assert len(turns.calls) == 1
    assert len(answers.requests) == 1


# -- startup recovery ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_restart_continues_an_active_goal_from_its_last_run() -> None:
    """The durable row is the recovery record: a new process takes up the work, not a human."""
    store = FakeGoalStore()
    creation = AnswerGoalService(store=store)
    goal = await _goal(creation)
    store.parent_runs[(_OWNER, _SESSION)] = _RUN
    restarted = AnswerGoalService(store=store)
    answers = FakeAnswers()
    driver = _driver(goal_service=restarted, answers=answers, runs=_runs(_view(terminal=True)))

    await driver.resume_after_restart()

    assert len(answers.requests) == 1
    stored = await restarted.get(owner_id=_OWNER, session_id=_SESSION)
    assert stored is not None and stored.rounds_started == 1
    assert stored.id == goal.id


@pytest.mark.asyncio
async def test_a_restart_delivers_a_pending_continuation_instead_of_admitting_one() -> None:
    store = FakeGoalStore()
    creation = AnswerGoalService(store=store)
    await _goal(creation)
    store.parent_runs[(_OWNER, _SESSION)] = _RUN
    restarted = AnswerGoalService(store=store)
    turns = FakeTurns(pending=True)
    delivered: list[tuple[str, str]] = []

    async def deliver(owner_id: str, parent_run_id: str) -> bool:
        delivered.append((owner_id, parent_run_id))
        return True

    driver = GoalRoundDriver(
        goals=restarted,
        answers=cast(AnswerService, FakeAnswers()),
        runs=cast(RunService, _runs(_view(terminal=True))),
        round_turns=turns,
    )
    driver.bind_pending_delivery(deliver)

    await driver.resume_after_restart()

    assert delivered == [(_OWNER, _RUN)]
    assert turns.calls == []
    stored = await restarted.get(owner_id=_OWNER, session_id=_SESSION)
    assert stored is not None and stored.rounds_started == 0


@pytest.mark.asyncio
async def test_a_restart_leaves_a_session_with_a_running_run_to_its_settlement() -> None:
    """A Run is already advancing the Session: the settlement is the trigger, not a scan."""
    store = FakeGoalStore()
    creation = AnswerGoalService(store=store)
    await _goal(creation)
    store.parent_runs[(_OWNER, _SESSION)] = _RUN
    restarted = AnswerGoalService(store=store)
    answers = FakeAnswers()
    driver = _driver(goal_service=restarted, answers=answers, runs=_runs(_view(terminal=False)))

    await driver.resume_after_restart()

    assert answers.requests == []
    stored = await restarted.get(owner_id=_OWNER, session_id=_SESSION)
    assert stored is not None and stored.rounds_started == 0


@pytest.mark.asyncio
async def test_a_restart_ends_a_capped_goal_the_way_settlement_would() -> None:
    store = FakeGoalStore()
    creation = AnswerGoalService(store=store)
    goal = await _goal(creation, max_rounds=1)
    await _admitted(creation, goal)
    store.parent_runs[(_OWNER, _SESSION)] = _RUN
    restarted = AnswerGoalService(store=store)
    answers = FakeAnswers()
    driver = _driver(goal_service=restarted, answers=answers, runs=_runs(_view(terminal=True)))

    await driver.resume_after_restart()

    assert answers.requests == []
    stored = await restarted.get(owner_id=_OWNER, session_id=_SESSION)
    assert stored is not None and stored.phase is GoalPhase.BLOCKED
    assert stored.blocked_reason == ROUND_LIMIT_REASON


@pytest.mark.asyncio
async def test_a_restart_asks_the_conversation_to_start_a_goal_that_has_no_run() -> None:
    """A `/goal` crash window: the goal is durable and active but no Run exists to settle."""
    store = FakeGoalStore()
    creation = AnswerGoalService(store=store)
    await _goal(creation)
    restarted = AnswerGoalService(store=store)
    answers = FakeAnswers()
    turns = FakeTurns(resumes_session=True)

    await _driver(
        goal_service=restarted, answers=answers, runs=FakeRuns(), round_turns=turns
    ).resume_after_restart()

    assert turns.session_calls == [(_OWNER, _SESSION)]
    assert answers.requests == []
    stored = await restarted.get(owner_id=_OWNER, session_id=_SESSION)
    assert stored is not None and stored.phase is GoalPhase.ACTIVE


@pytest.mark.asyncio
async def test_a_restart_skips_a_no_run_goal_with_no_web_conversation() -> None:
    """A non-Web Session has no conversation that could start the round, so recovery leaves it."""
    store = FakeGoalStore()
    creation = AnswerGoalService(store=store)
    await _goal(creation)
    restarted = AnswerGoalService(store=store)
    answers = FakeAnswers()

    await _driver(goal_service=restarted, answers=answers, runs=FakeRuns()).resume_after_restart()

    assert answers.requests == []
    stored = await restarted.get(owner_id=_OWNER, session_id=_SESSION)
    assert stored is not None and stored.phase is GoalPhase.ACTIVE


@pytest.mark.asyncio
async def test_a_restart_replays_a_lost_failed_settlement_as_blocked() -> None:
    """The observer's task was lost; recovery records the failure instead of retrying the goal."""
    store = FakeGoalStore()
    creation = AnswerGoalService(store=store)
    goal = await _goal(creation)
    store.parent_runs[(_OWNER, _SESSION)] = _RUN
    restarted = AnswerGoalService(store=store)
    answers = FakeAnswers()
    view = _view(
        continuation_kind="goal_round",
        goal_id=goal.id,
        goal_revision=goal.revision,
        status="failed",
        error_kind="answer_failed",
    )

    await _driver(goal_service=restarted, answers=answers, runs=_runs(view)).resume_after_restart()

    stored = store.rows[(_OWNER, _SESSION)]
    assert stored.phase is GoalPhase.BLOCKED
    assert stored.blocked_reason is not None
    assert "failed" in stored.blocked_reason
    assert "resume to retry" in stored.blocked_reason
    assert answers.requests == []


@pytest.mark.asyncio
async def test_a_restart_replays_a_lost_cancelled_settlement_as_paused() -> None:
    store = FakeGoalStore()
    creation = AnswerGoalService(store=store)
    goal = await _goal(creation)
    store.parent_runs[(_OWNER, _SESSION)] = _RUN
    restarted = AnswerGoalService(store=store)
    answers = FakeAnswers()
    view = _view(
        continuation_kind="goal_round",
        goal_id=goal.id,
        goal_revision=goal.revision,
        status="cancelled",
    )

    await _driver(goal_service=restarted, answers=answers, runs=_runs(view)).resume_after_restart()

    assert store.rows[(_OWNER, _SESSION)].phase is GoalPhase.PAUSED
    assert answers.requests == []


@pytest.mark.asyncio
async def test_a_restart_replays_a_lost_token_limit_settlement_as_blocked() -> None:
    store = FakeGoalStore()
    creation = AnswerGoalService(store=store)
    goal = await _goal(creation)
    store.parent_runs[(_OWNER, _SESSION)] = _RUN
    restarted = AnswerGoalService(store=store)
    answers = FakeAnswers()
    view = _view(
        continuation_kind="goal_round",
        goal_id=goal.id,
        goal_revision=goal.revision,
        result={"trace": {"agent_stop_reason": "length"}},
    )

    await _driver(goal_service=restarted, answers=answers, runs=_runs(view)).resume_after_restart()

    stored = store.rows[(_OWNER, _SESSION)]
    assert stored.phase is GoalPhase.BLOCKED
    assert stored.blocked_reason == MODEL_OUTPUT_LIMIT_REASON
    assert answers.requests == []


@pytest.mark.asyncio
async def test_a_restart_continues_a_goal_after_a_failed_human_turn() -> None:
    """A human turn's failure does not stop the goal; the idle Session is continued."""
    store = FakeGoalStore()
    creation = AnswerGoalService(store=store)
    goal = await _goal(creation)
    store.parent_runs[(_OWNER, _SESSION)] = _RUN
    restarted = AnswerGoalService(store=store)
    answers = FakeAnswers()
    view = _view(
        continuation_kind="follow_up",
        goal_id=goal.id,
        goal_revision=goal.revision,
        status="failed",
        error_kind="answer_failed",
    )

    await _driver(goal_service=restarted, answers=answers, runs=_runs(view)).resume_after_restart()

    stored = store.rows[(_OWNER, _SESSION)]
    assert stored.phase is GoalPhase.ACTIVE
    assert len(answers.requests) == 1


@pytest.mark.asyncio
async def test_a_restart_continues_after_a_successful_goal_round() -> None:
    """A normally succeeded goal round is not an abnormal stop: recovery starts the next round."""
    store = FakeGoalStore()
    creation = AnswerGoalService(store=store)
    goal = await _goal(creation)
    started = await _admitted(creation, goal)
    store.parent_runs[(_OWNER, _SESSION)] = _RUN
    restarted = AnswerGoalService(store=store)
    answers = FakeAnswers()
    view = _view(
        continuation_kind="goal_round",
        goal_id=goal.id,
        goal_revision=started.revision,
    )

    await _driver(goal_service=restarted, answers=answers, runs=_runs(view)).resume_after_restart()

    assert len(answers.requests) == 1
    assert store.rows[(_OWNER, _SESSION)].rounds_started == 2
    assert store.rows[(_OWNER, _SESSION)].phase is GoalPhase.ACTIVE


@pytest.mark.asyncio
async def test_a_restart_after_a_resume_starts_a_round_instead_of_replaying_a_stale_stop() -> None:
    """The round predates the resume, so its cancellation is stale; recovery continues the goal."""
    store = FakeGoalStore()
    creation = AnswerGoalService(store=store)
    goal = await _goal(creation)
    started = await _admitted(creation, goal)
    paused = await creation.update(
        owner_id=_OWNER,
        session_id=_SESSION,
        run_id=_RUN,
        goal_id=started.id,
        revision=started.revision,
        action="pause",
    )
    resumed = await creation.update(
        owner_id=_OWNER,
        session_id=_SESSION,
        run_id=_RUN,
        goal_id=paused.id,
        revision=paused.revision,
        action="resume",
        human_control=True,
    )
    store.parent_runs[(_OWNER, _SESSION)] = _RUN
    restarted = AnswerGoalService(store=store)
    answers = FakeAnswers()
    view = _view(
        continuation_kind="goal_round",
        goal_id=goal.id,
        goal_revision=started.revision,
        status="cancelled",
    )

    await _driver(goal_service=restarted, answers=answers, runs=_runs(view)).resume_after_restart()

    stored = store.rows[(_OWNER, _SESSION)]
    assert stored.phase is GoalPhase.ACTIVE
    assert stored.id == resumed.id
    assert stored.rounds_started == 2
    assert len(answers.requests) == 1


# -- /goal create and human resume ---------------------------------------------------------


@pytest.mark.asyncio
async def test_goal_create_starts_the_first_round_immediately_in_a_fresh_conversation() -> None:
    """The human asked for work now: no parent Run exists to settle and start it later."""
    store = FakeGoalStore()
    goals = AnswerGoalService(store=store)
    goal = await _goal(goals)
    answers = FakeAnswers()
    turns = FakeTurns()
    driver = _driver(goal_service=goals, answers=answers, runs=FakeRuns(), round_turns=turns)

    started = await driver.continue_created_goal(
        owner_id=_OWNER,
        session_id=_SESSION,
        conversation_id=_CONVERSATION,
        workspaces=("alpha", "beta"),
    )

    assert started is True
    assert len(turns.now_calls) == 1
    call = turns.now_calls[0]
    assert call["owner_id"] == _OWNER
    assert call["conversation_id"] == _CONVERSATION
    assert call["workspaces"] == ("alpha", "beta")
    assert call["goal_id"] == goal.id
    assert call["goal_round"] == 1
    assert f'Objective: "{goal.objective}"' in call["prompt"]
    assert "Round: 1/4" in call["prompt"]
    assert answers.requests == []
    stored = await goals.get(owner_id=_OWNER, session_id=_SESSION)
    assert stored is not None and stored.rounds_started == 1
    assert call["goal_revision"] == stored.revision


@pytest.mark.asyncio
async def test_goal_create_acceptance_refusal_gives_the_round_back() -> None:
    store = FakeGoalStore()
    goals = AnswerGoalService(store=store)
    goal = await _goal(goals)
    turns = FakeTurns(accepts_now=False)
    driver = _driver(goal_service=goals, answers=FakeAnswers(), runs=FakeRuns(), round_turns=turns)

    started = await driver.continue_created_goal(
        owner_id=_OWNER, session_id=_SESSION, conversation_id=_CONVERSATION
    )

    assert started is False
    stored = await goals.get(owner_id=_OWNER, session_id=_SESSION)
    assert stored is not None
    assert (stored.id, stored.phase, stored.rounds_started) == (goal.id, GoalPhase.ACTIVE, 0)


@pytest.mark.asyncio
async def test_goal_create_does_not_start_a_round_for_a_paused_goal() -> None:
    store = FakeGoalStore()
    goals = AnswerGoalService(store=store)
    goal = await _goal(goals)
    await goals.update(
        owner_id=_OWNER,
        session_id=_SESSION,
        run_id=_RUN,
        goal_id=goal.id,
        revision=goal.revision,
        action="pause",
    )
    turns = FakeTurns()
    driver = _driver(goal_service=goals, answers=FakeAnswers(), runs=FakeRuns(), round_turns=turns)

    started = await driver.continue_created_goal(
        owner_id=_OWNER, session_id=_SESSION, conversation_id=_CONVERSATION
    )

    assert started is False
    assert turns.now_calls == []


@pytest.mark.asyncio
async def test_goal_create_without_a_web_conversation_starts_nothing() -> None:
    goals = AnswerGoalService(store=FakeGoalStore())
    await _goal(goals)
    driver = _driver(goal_service=goals, answers=FakeAnswers(), runs=FakeRuns())

    assert (
        await driver.continue_created_goal(
            owner_id=_OWNER, session_id=_SESSION, conversation_id=_CONVERSATION
        )
        is False
    )


@pytest.mark.asyncio
async def test_a_human_resume_starts_the_next_round_of_an_idle_goal() -> None:
    store = FakeGoalStore()
    goals = AnswerGoalService(store=store)
    goal = await _goal(goals)
    answers = FakeAnswers()
    driver = _driver(goal_service=goals, answers=answers, runs=_runs(listed=(_view(),)))

    started = await driver.continue_after_resume(
        owner_id=_OWNER, session_id=_SESSION, parent_run_id=_RUN
    )

    assert started is True
    assert len(answers.requests) == 1
    request, key = answers.requests[0]
    assert key is not None and uuid.UUID(key)
    assert request.continuation_kind == "goal_round"
    assert request.parent_run_id == _RUN
    assert request.agent_session_id == _SESSION
    assert "<goal_round>" in request.query
    assert f'Objective: "{goal.objective}"' in request.query
    assert "Round: 1/4" in request.query
    assert store.rows[(_OWNER, _SESSION)].rounds_started == 1


@pytest.mark.asyncio
async def test_a_resume_does_not_continue_while_a_run_is_in_flight() -> None:
    goals = AnswerGoalService(store=FakeGoalStore())
    await _goal(goals)
    answers = FakeAnswers()
    driver = _driver(goal_service=goals, answers=answers, runs=_runs(_view(terminal=False)))

    started = await driver.continue_after_resume(
        owner_id=_OWNER, session_id=_SESSION, parent_run_id=_RUN
    )

    assert started is False
    assert answers.requests == []


@pytest.mark.asyncio
async def test_a_resumed_round_loses_to_a_run_the_front_door_already_admitted() -> None:
    store = FakeGoalStore()
    goals = AnswerGoalService(store=store)
    goal = await _goal(goals)
    answers = FakeAnswers(raises=SessionWorkBusyError("run-live"))
    driver = _driver(
        goal_service=goals,
        answers=answers,
        runs=_runs(listed=(_view(terminal=False),)),
    )

    started = await driver.continue_after_resume(
        owner_id=_OWNER, session_id=_SESSION, parent_run_id=_RUN
    )

    assert started is False
    # The refused round is given back: no admission is burned on the front-door refusal.
    stored = store.rows[(_OWNER, _SESSION)]
    assert stored.rounds_started == 0
    assert (stored.id, stored.phase) == (goal.id, GoalPhase.ACTIVE)


@pytest.mark.asyncio
async def test_a_cancelled_round_from_before_a_resume_does_not_pause_the_resumed_goal() -> None:
    """A human resume between admission and settlement wins; the stale cancellation is ignored."""
    store = FakeGoalStore()
    goals = AnswerGoalService(store=store)
    goal = await _goal(goals)
    started = await _admitted(goals, goal)
    paused = await goals.update(
        owner_id=_OWNER,
        session_id=_SESSION,
        run_id=_RUN,
        goal_id=started.id,
        revision=started.revision,
        action="pause",
    )
    resumed = await goals.update(
        owner_id=_OWNER,
        session_id=_SESSION,
        run_id=_RUN,
        goal_id=paused.id,
        revision=paused.revision,
        action="resume",
        human_control=True,
    )
    view = _view(
        continuation_kind="goal_round",
        goal_id=goal.id,
        goal_revision=started.revision,
        status="cancelled",
    )

    await _driver(goal_service=goals, answers=FakeAnswers(), runs=_runs(view)).settle(
        _settled("cancelled")
    )

    stored = store.rows[(_OWNER, _SESSION)]
    assert stored == resumed
    assert stored.phase is GoalPhase.ACTIVE


# -- goal identity on rounds and settlements -----------------------------------------------


@pytest.mark.asyncio
async def test_a_goal_round_request_carries_the_goal_identity() -> None:
    """The fallback Run names the goal and admitted round it serves, for its own settlement."""
    store = FakeGoalStore()
    goals = AnswerGoalService(store=store)
    goal = await _goal(goals)
    answers = FakeAnswers()

    await _driver(goal_service=goals, answers=answers, runs=_runs()).settle(_settled())

    request, _ = answers.requests[0]
    stored = store.rows[(_OWNER, _SESSION)]
    assert request.goal_id == goal.id
    assert request.goal_revision == stored.revision
    assert request.goal_round == 1
    assert request.continuation_kind == "goal_round"


@pytest.mark.asyncio
async def test_the_web_continuation_request_carries_the_goal_identity() -> None:
    goals = AnswerGoalService(store=FakeGoalStore())
    goal = await _goal(goals)
    turns = FakeTurns()

    driver = GoalRoundDriver(
        goals=goals,
        answers=cast(AnswerService, FakeAnswers()),
        runs=cast(RunService, _runs()),
        round_turns=turns,
    )
    await driver.settle(_settled())

    call = turns.request_calls[0]
    stored = await goals.get(owner_id=_OWNER, session_id=_SESSION)
    assert stored is not None
    assert call["goal_id"] == goal.id
    # The reservation is written before the admission, so it names the revision and round the
    # admission will give the goal.
    assert call["goal_revision"] == goal.revision + 1
    assert call["goal_round"] == goal.rounds_started + 1
    assert call["workspaces"] == ("workspace-a",)


@pytest.mark.asyncio
async def test_the_reservation_is_written_before_the_round_is_admitted() -> None:
    """A settlement/sweep interleaving between the two must see the row and stay out."""

    order: list[str] = []
    goals = AnswerGoalService(store=FakeGoalStore())
    await _goal(goals)
    turns = FakeTurns()
    reserve = turns.request_goal_round

    async def recording_request(**kwargs: Any) -> GoalRoundReservation | None:
        order.append("reserve")
        return await reserve(**kwargs)

    turns.request_goal_round = recording_request  # pyright: ignore[reportAttributeAccessIssue]
    original_admit = goals.admit_round

    async def recording_admit(**kwargs: Any) -> Any:
        order.append("admit")
        return await original_admit(**kwargs)

    goals.admit_round = recording_admit  # pyright: ignore[reportAttributeAccessIssue]
    delivered: list[str] = []

    async def deliver(owner_id: str, parent_run_id: str) -> bool:
        delivered.append(parent_run_id)
        return True

    driver = GoalRoundDriver(
        goals=goals,
        answers=cast(AnswerService, FakeAnswers()),
        runs=cast(RunService, _runs()),
        round_turns=turns,
    )
    driver.bind_pending_delivery(deliver)

    await driver.settle(_settled())

    assert order == ["reserve", "admit"]
    assert delivered == [_RUN]


@pytest.mark.asyncio
async def test_a_second_settlement_does_not_admit_beside_a_waiting_reservation() -> None:
    """Sweep-vs-settlement: once a reservation exists, no second admission is taken."""
    store = FakeGoalStore()
    goals = AnswerGoalService(store=store)
    await _goal(goals)
    turns = FakeTurns()
    delivered: list[str] = []

    async def deliver(owner_id: str, parent_run_id: str) -> bool:
        # The Session is busy: the reservation stays pending for the next settlement.
        delivered.append(parent_run_id)
        return False

    driver = GoalRoundDriver(
        goals=goals,
        answers=cast(AnswerService, FakeAnswers()),
        runs=cast(RunService, _runs()),
        round_turns=turns,
    )
    driver.bind_pending_delivery(deliver)

    await driver.settle(_settled())
    first = store.rows[(_OWNER, _SESSION)].rounds_started
    # The reservation is now visible to the next trigger: it must stop before admitting.
    turns.pending = True
    await driver.settle(_settled())

    stored = store.rows[(_OWNER, _SESSION)]
    assert first == 1
    assert stored.rounds_started == 1, "one reservation, one admission"
    assert len(turns.request_calls) == 1


@pytest.mark.asyncio
async def test_a_lost_admission_withdraws_the_reservation() -> None:
    goals = AnswerGoalService(store=FakeGoalStore())
    await _goal(goals)
    turns = FakeTurns()

    async def losing_admit(**kwargs: Any) -> Any:
        from dlightrag.engine.answer.goal import GOAL_STALE_REVISION, GoalError

        raise GoalError(GOAL_STALE_REVISION, "the goal changed")

    goals.admit_round = losing_admit  # pyright: ignore[reportAttributeAccessIssue]
    driver = GoalRoundDriver(
        goals=goals,
        answers=cast(AnswerService, FakeAnswers()),
        runs=cast(RunService, _runs()),
        round_turns=turns,
    )

    await driver.settle(_settled())

    assert len(turns.request_calls) == 1
    assert turns.withdraw_calls == 1
    reservation = turns.reservations[0]
    assert turns.withdraw_args == [
        {
            "owner_id": _OWNER,
            "parent_run_id": _RUN,
            "request_id": reservation.request_id,
            "prompt_digest": reservation.prompt_digest,
        }
    ]


@pytest.mark.asyncio
async def test_a_lost_admission_leaves_a_round_a_delivery_already_started() -> None:
    """The delivery consumed the reservation and admitted the round: this call must not refund."""
    goals = AnswerGoalService(store=FakeGoalStore())
    await _goal(goals)
    turns = FakeTurns()
    turns.withdraws = False  # the row was already consumed by the delivery

    async def losing_admit(**kwargs: Any) -> Any:
        from dlightrag.engine.answer.goal import GOAL_STALE_REVISION, GoalError

        raise GoalError(GOAL_STALE_REVISION, "the goal changed")

    goals.admit_round = losing_admit  # pyright: ignore[reportAttributeAccessIssue]
    releases: list[str] = []
    original_release = goals.release_round

    async def recording_release(**kwargs: Any) -> Any:
        releases.append(str(kwargs["goal_id"]))
        return await original_release(**kwargs)

    goals.release_round = recording_release  # pyright: ignore[reportAttributeAccessIssue]
    driver = GoalRoundDriver(
        goals=goals,
        answers=cast(AnswerService, FakeAnswers()),
        runs=cast(RunService, _runs()),
        round_turns=turns,
    )

    await driver.settle(_settled())

    assert turns.withdraw_calls == 1
    # It still withdraws by its own identity: a consumed or replaced row is reported gone,
    # never silently deleted by conversation.
    assert turns.withdraw_args[0]["request_id"] == turns.reservations[0].request_id
    assert turns.withdraw_args[0]["prompt_digest"] == turns.reservations[0].prompt_digest
    assert releases == [], "a delivery-started round is not refunded"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "stop_reason"),
    [("failed", None), ("cancelled", None), ("succeeded", "length")],
)
async def test_a_stop_from_another_goal_does_not_change_the_current_goal(
    status: str, stop_reason: str | None
) -> None:
    """A round admitted for a cleared/replaced goal must not pause or block its replacement."""
    store = FakeGoalStore()
    goals = AnswerGoalService(store=store)
    goal = await _goal(goals)
    driver = _driver(
        goal_service=goals,
        answers=FakeAnswers(),
        runs=_runs(_view(continuation_kind="goal_round", goal_id="goal-replaced", goal_round=1)),
    )

    await driver.settle(_settled(status, stop_reason=stop_reason))

    assert store.rows[(_OWNER, _SESSION)] == goal
    assert goal.phase is GoalPhase.ACTIVE


@pytest.mark.asyncio
async def test_a_stop_without_round_identity_does_not_change_the_current_goal() -> None:
    """A round admitted before identity was recorded is no authority over the current goal."""
    store = FakeGoalStore()
    goals = AnswerGoalService(store=store)
    goal = await _goal(goals)
    driver = _driver(
        goal_service=goals,
        answers=FakeAnswers(),
        runs=_runs(_view(continuation_kind="goal_round")),
    )

    await driver.settle(_settled("failed"))

    assert store.rows[(_OWNER, _SESSION)] == goal


@pytest.mark.asyncio
async def test_a_cancelled_queued_round_from_another_goal_does_not_pause_the_current_goal() -> None:
    store = FakeGoalStore()
    goals = AnswerGoalService(store=store)
    goal = await _goal(goals)
    view = _view(continuation_kind="goal_round", goal_id="goal-replaced")

    await _driver(goal_service=goals, answers=FakeAnswers(), runs=_runs(view)).run_cancelled(view)

    assert store.rows[(_OWNER, _SESSION)] == goal


@pytest.mark.asyncio
async def test_a_successful_stale_round_continues_the_current_active_goal() -> None:
    """Success carries no durable stop: the current active goal continues regardless of identity."""
    store = FakeGoalStore()
    goals = AnswerGoalService(store=store)
    goal = await _goal(goals)
    answers = FakeAnswers()
    driver = _driver(
        goal_service=goals,
        answers=answers,
        runs=_runs(_view(continuation_kind="goal_round", goal_id="goal-replaced")),
    )

    await driver.settle(_settled())

    assert len(answers.requests) == 1
    assert answers.requests[0][0].goal_id == goal.id
    assert store.rows[(_OWNER, _SESSION)].rounds_started == 1


# -- the round prompt ----------------------------------------------------------------------


def test_the_prompt_carries_the_objective_and_the_round() -> None:
    goal = Goal.create("finish the sealed model", max_rounds=3)

    prompt = goal_round_prompt(goal=goal, round_number=2)

    assert f'Objective: "{goal.objective}"' in prompt
    assert "Round: 2/3" in prompt
    assert prompt.startswith("<goal_round>")
    assert prompt.endswith("</goal_round>")


def test_the_round_prompt_continues_the_session_and_never_claims_a_new_context() -> None:
    """A round is a budget boundary: same Session, workspace, notes and history (ADR 0077)."""
    goal = Goal.create("finish the sealed model", max_rounds=3)

    prompt = goal_round_prompt(goal=goal, round_number=2)

    assert "same Session" in prompt
    assert "do not redo work an earlier round already completed" in prompt
    assert "notes/goal.md" in prompt
    assert "run `pwd` before file commands" in prompt
    assert "relative to that root" in prompt
    assert "never reuse an absolute path copied from earlier-round history" in prompt
    assert "do not hand-write a deliverable a loaded Skill produces" in prompt
    assert "load_skill call in this Run plus the Skill's own verification passing" in prompt
    assert "Call get_goal before updating or completing the goal" in prompt
    assert "leave the goal active" in prompt
    folded = prompt.casefold()
    for stale in (
        "new context",
        "fresh context",
        "do not carry over",
        "load the agent skills this objective depends on in this round",
        "disarm",
        "re-arm",
        "rearm",
        "armed",
        "activation",
        "inactive goal",
    ):
        assert stale not in folded, f"the round prompt still claims {stale!r}"
