# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The goal service is authority and compare-and-set over a narrow store (ADR 0060, 0077)."""

from __future__ import annotations

from functools import partial
from typing import Any, cast

import pytest

from dlightrag.engine.answer.goal import (
    GOAL_INVALID_TRANSITION,
    GOAL_NOT_FOUND,
    GOAL_ROUNDS_EXHAUSTED,
    GOAL_STALE_REVISION,
    Goal,
    GoalAction,
    GoalError,
    GoalPhase,
)
from dlightrag.engine.answer.goal_service import (
    BLOCKED_AFTER_ROUNDS,
    HUMAN_ONLY_ACTIONS,
    MODEL_OUTPUT_LIMIT_REASON,
    ROUND_LIMIT_REASON,
    AnswerGoalService,
)
from tests.support.goals import FakeGoalStore

_OWNER = "owner-1"
_SESSION = "session-1"
_RUN = "run-1"


def _service() -> tuple[AnswerGoalService, FakeGoalStore]:
    store = FakeGoalStore()
    return AnswerGoalService(store=store), store


async def _refused(code: str, call: object) -> None:
    with pytest.raises(GoalError) as raised:
        await call()  # type: ignore[operator]
    assert raised.value.code == code


async def _created(
    service: AnswerGoalService,
    *,
    session_id: str = _SESSION,
    objective: str = "finish the sealed Polestar model",
    max_rounds: int = 4,
) -> Goal:
    return await service.create(
        owner_id=_OWNER,
        session_id=session_id,
        run_id=_RUN,
        objective=objective,
        max_rounds=max_rounds,
    )


async def _paused(service: AnswerGoalService) -> Goal:
    goal = await _created(service)
    return await service.update(
        owner_id=_OWNER,
        session_id=_SESSION,
        run_id=_RUN,
        goal_id=goal.id,
        revision=goal.revision,
        action="pause",
    )


async def _stop(
    service: AnswerGoalService,
    method: str,
    goal: Goal,
    *,
    revision: int | None = None,
    run_id: str = _RUN,
    reason: str = "",
) -> Goal | None:
    kwargs: dict[str, Any] = {
        "owner_id": _OWNER,
        "session_id": _SESSION,
        "run_id": run_id,
        "goal_id": goal.id,
        "revision": goal.revision if revision is None else revision,
    }
    if method == "stop_round_failed":
        kwargs["reason"] = reason or "the search provider returned 503"
    return await getattr(service, method)(**kwargs)


@pytest.mark.asyncio
async def test_create_makes_the_session_goal() -> None:
    service, _ = _service()

    goal = await _created(service)

    assert goal.phase == "active"
    assert goal.revision == 1
    assert await service.get(owner_id=_OWNER, session_id=_SESSION) == goal


@pytest.mark.asyncio
async def test_a_second_goal_is_refused_while_the_first_is_unfinished() -> None:
    for phase in ("active", "paused"):
        service, _ = _service()
        goal = await _created(service)
        if phase == "paused":
            goal = await service.update(
                owner_id=_OWNER,
                session_id=_SESSION,
                run_id=_RUN,
                goal_id=goal.id,
                revision=goal.revision,
                action="pause",
            )
        with pytest.raises(GoalError) as raised:
            await _created(service)
        assert raised.value.code == GOAL_INVALID_TRANSITION
        assert "unfinished goal" in str(raised.value)
        assert "pause, block or clear" in str(raised.value)

    service, store = _service()
    goal = await _created(service, max_rounds=6)
    started = goal
    for _ in range(BLOCKED_AFTER_ROUNDS):
        started, _ = await service.admit_round(
            owner_id=_OWNER,
            session_id=_SESSION,
            run_id=_RUN,
            goal_id=started.id,
            revision=started.revision,
        )
    blocked = await service.update(
        owner_id=_OWNER,
        session_id=_SESSION,
        run_id=_RUN,
        goal_id=started.id,
        revision=started.revision,
        action="blocked",
        blocked_reason="the filings portal is unreachable",
        goal_round=BLOCKED_AFTER_ROUNDS,
    )
    assert blocked.phase == "blocked"
    await _refused(GOAL_INVALID_TRANSITION, lambda: _created(service))


@pytest.mark.asyncio
async def test_a_goal_round_cannot_create_one() -> None:
    service, _ = _service()

    await _refused(
        GOAL_INVALID_TRANSITION,
        lambda: service.create(
            owner_id=_OWNER,
            session_id=_SESSION,
            run_id=_RUN,
            objective="another",
            max_rounds=2,
            goal_round=1,
        ),
    )


@pytest.mark.asyncio
async def test_create_replaces_a_completed_goal() -> None:
    service, store = _service()
    first = await _created(service, max_rounds=2)
    completed = await service.update(
        owner_id=_OWNER,
        session_id=_SESSION,
        run_id=_RUN,
        goal_id=first.id,
        revision=first.revision,
        action="complete",
    )

    second = await _created(service, objective="value Volvo", max_rounds=6)

    assert second.id != completed.id
    assert second.revision == 1
    assert second.rounds_started == 0
    assert second.phase == GoalPhase.ACTIVE
    assert second.objective == "value Volvo"
    assert second.max_rounds == 6
    assert store.rows[(_OWNER, _SESSION)] == second


@pytest.mark.asyncio
async def test_create_over_a_completed_goal_that_moved_is_a_stale_refusal() -> None:
    class _RacingStore(FakeGoalStore):
        async def replace_completed(self, **_: Any) -> Goal | None:
            return None

    store = _RacingStore()
    service = AnswerGoalService(store=store)
    first = await _created(service)
    await service.update(
        owner_id=_OWNER,
        session_id=_SESSION,
        run_id=_RUN,
        goal_id=first.id,
        revision=first.revision,
        action="complete",
    )

    await _refused(GOAL_STALE_REVISION, lambda: _created(service))


@pytest.mark.asyncio
async def test_a_goal_round_can_complete_or_block_but_not_edit_pause_or_resume() -> None:
    service, store = _service()
    goal = await _created(service)
    for action in HUMAN_ONLY_ACTIONS:
        call = partial(
            service.update,
            owner_id=_OWNER,
            session_id=_SESSION,
            run_id=_RUN,
            goal_id=goal.id,
            revision=goal.revision,
            action=cast(GoalAction, action),
            objective="new",
            max_rounds=2,
            goal_round=1,
        )
        await _refused(GOAL_INVALID_TRANSITION, call)
    assert store.rows[(_OWNER, _SESSION)] == goal

    completed = await service.update(
        owner_id=_OWNER,
        session_id=_SESSION,
        run_id=_RUN,
        goal_id=goal.id,
        revision=goal.revision,
        action="complete",
        goal_round=1,
    )
    assert completed.phase == "complete"


@pytest.mark.asyncio
async def test_a_goal_round_blocks_only_after_the_reference_threshold_of_rounds() -> None:
    service, store = _service()
    goal = await _created(service)

    def block(round_number: int) -> object:
        return lambda: service.update(
            owner_id=_OWNER,
            session_id=_SESSION,
            run_id=_RUN,
            goal_id=goal.id,
            revision=goal.revision,
            action="blocked",
            blocked_reason="the filings portal is unreachable",
            goal_round=round_number,
        )

    await _refused(GOAL_INVALID_TRANSITION, block(1))

    matured = Goal(
        id=goal.id,
        revision=goal.revision,
        objective=goal.objective,
        phase=goal.phase,
        max_rounds=goal.max_rounds,
        rounds_started=BLOCKED_AFTER_ROUNDS,
    )
    store.rows[(_OWNER, _SESSION)] = matured
    blocked = await service.update(
        owner_id=_OWNER,
        session_id=_SESSION,
        run_id=_RUN,
        goal_id=matured.id,
        revision=matured.revision,
        action="blocked",
        blocked_reason="the filings portal is unreachable",
        goal_round=BLOCKED_AFTER_ROUNDS,
    )
    assert blocked.phase == "blocked"
    assert blocked.blocked_reason == "the filings portal is unreachable"


@pytest.mark.asyncio
async def test_a_human_turn_may_block_immediately() -> None:
    service, _ = _service()
    goal = await _created(service)

    blocked = await service.update(
        owner_id=_OWNER,
        session_id=_SESSION,
        run_id=_RUN,
        goal_id=goal.id,
        revision=goal.revision,
        action="blocked",
        blocked_reason="the human stopped this line of work",
    )

    assert blocked.phase == "blocked"


@pytest.mark.asyncio
async def test_pause_and_human_resume_move_the_phase() -> None:
    service, _ = _service()
    goal = await _created(service)

    paused = await service.update(
        owner_id=_OWNER,
        session_id=_SESSION,
        run_id=_RUN,
        goal_id=goal.id,
        revision=goal.revision,
        action="pause",
    )
    assert paused.phase == "paused"

    resumed = await service.update(
        owner_id=_OWNER,
        session_id=_SESSION,
        run_id=_RUN,
        goal_id=paused.id,
        revision=paused.revision,
        action="resume",
        human_control=True,
    )
    assert resumed.phase == "active"


@pytest.mark.asyncio
async def test_the_model_cannot_resume_a_paused_goal() -> None:
    """Only the human's goal bar passes ``human_control``; the tool path is refused (ADR 0077)."""
    service, store = _service()
    paused = await _paused(service)

    with pytest.raises(GoalError) as raised:
        await service.update(
            owner_id=_OWNER,
            session_id=_SESSION,
            run_id=_RUN,
            goal_id=paused.id,
            revision=paused.revision,
            action="resume",
        )
    assert raised.value.code == GOAL_INVALID_TRANSITION
    assert "resumed by the human" in str(raised.value)
    assert store.rows[(_OWNER, _SESSION)] == paused


@pytest.mark.asyncio
async def test_a_blocked_goal_may_be_resumed_by_the_model_on_a_human_turn() -> None:
    """A blocking condition waits for the human's turn, not the human's button (ADR 0077)."""
    service, _ = _service()
    goal = await _created(service, max_rounds=6)
    started = goal
    for _ in range(BLOCKED_AFTER_ROUNDS):
        started, _ = await service.admit_round(
            owner_id=_OWNER,
            session_id=_SESSION,
            run_id=_RUN,
            goal_id=started.id,
            revision=started.revision,
        )
    blocked = await service.update(
        owner_id=_OWNER,
        session_id=_SESSION,
        run_id=_RUN,
        goal_id=started.id,
        revision=started.revision,
        action="blocked",
        blocked_reason="the filings portal is unreachable",
        goal_round=BLOCKED_AFTER_ROUNDS,
    )

    resumed = await service.update(
        owner_id=_OWNER,
        session_id=_SESSION,
        run_id=_RUN,
        goal_id=blocked.id,
        revision=blocked.revision,
        action="resume",
    )

    assert resumed.phase == "active"
    assert resumed.blocked_reason is None


@pytest.mark.asyncio
async def test_resume_of_an_active_goal_is_an_idempotent_no_op() -> None:
    service, _ = _service()
    goal = await _created(service)

    resumed = await service.update(
        owner_id=_OWNER,
        session_id=_SESSION,
        run_id=_RUN,
        goal_id=goal.id,
        revision=goal.revision,
        action="resume",
        human_control=True,
    )

    assert resumed == goal


@pytest.mark.asyncio
async def test_resume_refuses_a_goal_whose_rounds_are_all_started() -> None:
    service, _ = _service()
    goal = await _created(service, max_rounds=1)
    started, _ = await service.admit_round(
        owner_id=_OWNER, session_id=_SESSION, run_id=_RUN, goal_id=goal.id, revision=goal.revision
    )
    paused = await service.update(
        owner_id=_OWNER,
        session_id=_SESSION,
        run_id=_RUN,
        goal_id=started.id,
        revision=started.revision,
        action="pause",
    )

    await _refused(
        GOAL_ROUNDS_EXHAUSTED,
        lambda: service.update(
            owner_id=_OWNER,
            session_id=_SESSION,
            run_id=_RUN,
            goal_id=paused.id,
            revision=paused.revision,
            action="resume",
            human_control=True,
        ),
    )


@pytest.mark.asyncio
async def test_admit_round_counts_rounds_and_respects_the_cap() -> None:
    service, _ = _service()
    goal = await _created(service, max_rounds=1)

    started, number = await service.admit_round(
        owner_id=_OWNER,
        session_id=_SESSION,
        run_id=_RUN,
        goal_id=goal.id,
        revision=goal.revision,
    )
    assert (number, started.rounds_started) == (1, 1)

    await _refused(
        GOAL_ROUNDS_EXHAUSTED,
        lambda: service.admit_round(
            owner_id=_OWNER,
            session_id=_SESSION,
            run_id=_RUN,
            goal_id=started.id,
            revision=started.revision,
        ),
    )


@pytest.mark.asyncio
async def test_another_process_admits_a_round_from_the_durable_phase_alone() -> None:
    """The stored phase is the permission: a new process needs no activation of its own."""
    store = FakeGoalStore()
    first = AnswerGoalService(store=store)
    goal = await _created(first)

    restarted = AnswerGoalService(store=store)

    started, number = await restarted.admit_round(
        owner_id=_OWNER,
        session_id=_SESSION,
        run_id=_RUN,
        goal_id=goal.id,
        revision=goal.revision,
    )
    assert number == 1
    assert started.phase == "active"
    assert started.rounds_started == 1


@pytest.mark.asyncio
async def test_a_lost_compare_and_set_is_a_stale_refusal() -> None:
    service, _ = _service()
    goal = await _created(service)
    stale = goal.revision
    await service.update(
        owner_id=_OWNER,
        session_id=_SESSION,
        run_id=_RUN,
        goal_id=goal.id,
        revision=stale,
        action="pause",
    )

    await _refused(
        GOAL_STALE_REVISION,
        lambda: service.update(
            owner_id=_OWNER,
            session_id=_SESSION,
            run_id=_RUN,
            goal_id=goal.id,
            revision=stale,
            action="complete",
        ),
    )


@pytest.mark.asyncio
async def test_an_unknown_goal_id_is_not_the_stale_revision_complaint() -> None:
    service, _ = _service()
    goal = await _created(service)

    await _refused(
        GOAL_NOT_FOUND,
        lambda: service.update(
            owner_id=_OWNER,
            session_id=_SESSION,
            run_id=_RUN,
            goal_id="goal-other",
            revision=goal.revision,
            action="complete",
        ),
    )


@pytest.mark.asyncio
async def test_a_round_whose_run_never_existed_is_given_back() -> None:
    service, store = _service()
    goal = await _created(service, max_rounds=2)
    started, _ = await service.admit_round(
        owner_id=_OWNER, session_id=_SESSION, run_id=_RUN, goal_id=goal.id, revision=goal.revision
    )

    released = await service.release_round(
        owner_id=_OWNER,
        session_id=_SESSION,
        run_id=_RUN,
        goal_id=started.id,
        revision=started.revision,
    )

    assert released is not None and released.rounds_started == 0
    assert store.rows[(_OWNER, _SESSION)].rounds_started == 0


@pytest.mark.asyncio
async def test_the_round_cap_ends_as_blocked() -> None:
    service, store = _service()
    goal = await _created(service, max_rounds=1)
    started, _ = await service.admit_round(
        owner_id=_OWNER, session_id=_SESSION, run_id=_RUN, goal_id=goal.id, revision=goal.revision
    )

    blocked = await service.block_round_limit(
        owner_id=_OWNER,
        session_id=_SESSION,
        run_id=_RUN,
        goal_id=started.id,
        revision=started.revision,
    )

    assert blocked is not None
    assert blocked.phase == "blocked"
    assert blocked.blocked_reason == ROUND_LIMIT_REASON
    assert store.rows[(_OWNER, _SESSION)].phase == "blocked"


@pytest.mark.asyncio
async def test_clear_removes_the_goal() -> None:
    service, store = _service()
    await _created(service)

    assert await service.clear(owner_id=_OWNER, session_id=_SESSION, run_id=_RUN) is True
    assert store.rows.get((_OWNER, _SESSION)) is None
    assert await service.clear(owner_id=_OWNER, session_id=_SESSION, run_id=_RUN) is False


@pytest.mark.asyncio
async def test_active_goals_reads_the_durable_rows_without_side_effects() -> None:
    store = FakeGoalStore()
    service = AnswerGoalService(store=store)
    active = await _created(service)
    paused = await _created(service, session_id="session-paused")
    await service.update(
        owner_id=_OWNER,
        session_id="session-paused",
        run_id=_RUN,
        goal_id=paused.id,
        revision=paused.revision,
        action="pause",
    )
    store.parent_runs[(_OWNER, _SESSION)] = _RUN
    rows_before = dict(store.rows)

    first = await service.active_goals()
    second = await service.active_goals()

    assert first == second
    assert [(entry.session_id, entry.goal) for entry in first] == [(_SESSION, active)]
    assert first[0].parent_run_id == _RUN
    assert store.rows == rows_before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "phase", "expected_reason"),
    [
        ("stop_round_failed", GoalPhase.BLOCKED, "the search provider returned 503"),
        ("stop_round_cancelled", GoalPhase.PAUSED, None),
        ("stop_round_token_limit", GoalPhase.BLOCKED, MODEL_OUTPUT_LIMIT_REASON),
    ],
)
async def test_a_harness_stop_records_the_rounds_observed_outcome(
    method: str, phase: GoalPhase, expected_reason: str | None
) -> None:
    service, store = _service()
    goal = await _created(service)

    stopped = await _stop(service, method, goal)

    assert stopped is not None
    assert stopped.phase is phase
    assert stopped.blocked_reason == expected_reason
    assert store.rows[(_OWNER, _SESSION)] == stopped


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method", ["stop_round_failed", "stop_round_cancelled", "stop_round_token_limit"]
)
async def test_a_harness_stop_is_a_no_op_when_the_goal_is_no_longer_active(method: str) -> None:
    service, store = _service()
    paused = await _paused(service)

    assert await _stop(service, method, paused) is None
    assert store.rows[(_OWNER, _SESSION)] == paused


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method", ["stop_round_failed", "stop_round_cancelled", "stop_round_token_limit"]
)
async def test_a_harness_stop_does_not_override_a_human_revision(method: str) -> None:
    """A human edit/resume after the round was admitted wins; the old round's outcome is stale."""
    service, store = _service()
    goal = await _created(service)
    stale = goal.revision
    edited = await service.update(
        owner_id=_OWNER,
        session_id=_SESSION,
        run_id=_RUN,
        goal_id=goal.id,
        revision=stale,
        action="edit",
        objective="finish the revised sealed model",
    )

    assert await _stop(service, method, goal, revision=stale) is None
    assert store.rows[(_OWNER, _SESSION)] == edited
    assert edited.phase is GoalPhase.ACTIVE


@pytest.mark.asyncio
async def test_a_harness_stop_does_not_retry_a_lost_compare_and_set() -> None:
    class _OneLostCasStore(FakeGoalStore):
        lost = False

        async def replace(self, **kwargs: Any) -> Goal | None:
            if not self.lost:
                self.lost = True
                return None
            return await super().replace(**kwargs)

    store = _OneLostCasStore()
    service = AnswerGoalService(store=store)
    goal = await _created(service)

    assert await _stop(service, "stop_round_failed", goal) is None
    assert store.rows[(_OWNER, _SESSION)].phase is GoalPhase.ACTIVE


@pytest.mark.asyncio
async def test_a_harness_stop_for_an_unknown_goal_is_a_no_op() -> None:
    service, _ = _service()
    unknown = Goal(
        id="goal-other",
        revision=1,
        objective="another",
        phase=GoalPhase.ACTIVE,
        max_rounds=2,
    )

    assert await _stop(service, "stop_round_failed", unknown) is None


async def test_a_goal_created_with_a_chosen_id_is_stored_under_it() -> None:
    from tests.support.goals import FakeGoalStore

    service = AnswerGoalService(store=FakeGoalStore())

    created = await service.create(
        owner_id="owner",
        session_id="session",
        run_id="",
        objective="finish the model",
        max_rounds=3,
        goal_id="goal-chosen",
    )

    assert created.id == "goal-chosen"
    stored = await service.get(owner_id="owner", session_id="session")
    assert stored is not None and stored.id == "goal-chosen"
