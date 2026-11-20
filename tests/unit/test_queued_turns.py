# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The queued turn driver delivers a reader's waiting messages before the goal continues."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Literal, cast

from dlightrag.application.answer.queued_turns import QueuedTurnDriver
from dlightrag.application.runs import RunService
from dlightrag.engine.runtime.contracts import RunKind
from dlightrag.engine.runtime.coordinator import SettledRun

_OWNER = "owner-1"
_RUN = "run-1"


@dataclass
class FakeTurns:
    result: bool = True
    #: Per-call results; after they run out ``result`` answers. The driver asks twice: once for
    #: human words and once for the goal continuation the settlement may have just requested.
    results: list[bool] = field(default_factory=list)
    raises: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)
    order: list[str] | None = None

    async def accept_queued_turn(
        self,
        owner_id: str,
        *,
        parent_run_id: str,
        workspaces: tuple[str, ...],
    ) -> bool:
        self.calls.append(
            {"owner_id": owner_id, "parent_run_id": parent_run_id, "workspaces": workspaces}
        )
        if self.order is not None:
            self.order.append("queue")
        if self.raises is not None:
            raise self.raises
        if self.results:
            return self.results.pop(0)
        return self.result


@dataclass
class FakeRuns:
    view: Any = None

    async def get(self, *, owner_id: str, run_id: str) -> Any | None:
        del owner_id, run_id
        return self.view


@dataclass
class FakeGoalRounds:
    order: list[str] = field(default_factory=list)
    settled: list[SettledRun] = field(default_factory=list)
    cancelled: list[Any] = field(default_factory=list)

    async def settle(self, settled: SettledRun) -> None:
        self.settled.append(settled)
        self.order.append("goal")

    async def run_cancelled(self, view: Any) -> None:
        self.cancelled.append(view)


def _view(*, run_kind: str = "answer") -> Any:
    return SimpleNamespace(
        run_id=_RUN,
        run_kind=run_kind,
        submitted_by=_OWNER,
        request={"workspaces": ["workspace-a"]},
    )


def _settled(
    *,
    run_kind: RunKind = "answer",
    status: Literal["succeeded", "failed", "cancelled"] = "succeeded",
) -> SettledRun:
    return SettledRun(owner_id=_OWNER, run_id=_RUN, run_kind=run_kind, status=status)


def _driver(*, turns: FakeTurns, runs: FakeRuns, goals: FakeGoalRounds | None) -> QueuedTurnDriver:
    return QueuedTurnDriver(
        turns=cast(Any, turns),
        runs=cast(RunService, runs),
        goal_rounds=cast(Any, goals),
    )


async def test_a_settled_run_delivers_the_queue_before_the_goal_continues() -> None:
    order: list[str] = []
    turns = FakeTurns(order=order, results=[True, False])
    goals = FakeGoalRounds(order=order)
    driver = _driver(turns=turns, runs=FakeRuns(view=_view()), goals=goals)

    await driver.settle(_settled())

    assert order == ["queue", "goal", "queue"], (
        "human words, then the goal's settlement, then the continuation attempt"
    )
    assert turns.calls[0] == {
        "owner_id": _OWNER,
        "parent_run_id": _RUN,
        "workspaces": ("workspace-a",),
    }
    assert len(turns.calls) == 2, (
        "the second attempt admits the continuation the goal just asked for"
    )
    assert goals.settled == [_settled()]


async def test_no_queued_work_still_hands_the_settlement_to_the_goal() -> None:
    turns = FakeTurns(result=False)
    goals = FakeGoalRounds()
    driver = _driver(turns=turns, runs=FakeRuns(view=_view()), goals=goals)

    await driver.settle(_settled())

    assert goals.settled == [_settled()]


async def test_a_delivery_failure_never_blocks_the_goal() -> None:
    turns = FakeTurns(raises=RuntimeError("queue unavailable"))
    goals = FakeGoalRounds()
    driver = _driver(turns=turns, runs=FakeRuns(view=_view()), goals=goals)

    await driver.settle(_settled())

    assert goals.settled == [_settled()]


async def test_a_run_that_is_not_an_answer_delivers_nothing() -> None:
    turns = FakeTurns()
    goals = FakeGoalRounds()
    driver = _driver(turns=turns, runs=FakeRuns(view=_view(run_kind="retrieval")), goals=goals)

    await driver.settle(_settled(run_kind="retrieval"))

    assert turns.calls == []
    assert goals.settled == [_settled(run_kind="retrieval")]


async def test_a_cancelled_run_still_releases_the_queue_and_the_goal() -> None:
    turns = FakeTurns()
    goals = FakeGoalRounds()
    view = _view()

    await _driver(turns=turns, runs=FakeRuns(view=view), goals=goals).run_cancelled(view)

    assert turns.calls == [
        {"owner_id": _OWNER, "parent_run_id": _RUN, "workspaces": ("workspace-a",)}
    ]
    assert goals.cancelled == [view]


async def test_the_settlement_callback_schedules_without_blocking() -> None:
    turns = FakeTurns()
    driver = _driver(turns=turns, runs=FakeRuns(view=_view()), goals=FakeGoalRounds())

    driver.run_settled(_settled())
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert len(turns.calls) == 2  # human words, then the continuation attempt


async def test_a_settlement_without_a_matching_run_delivers_nothing() -> None:
    turns = FakeTurns()
    goals = FakeGoalRounds()
    driver = _driver(turns=turns, runs=FakeRuns(view=None), goals=goals)

    await driver.settle(_settled())

    assert turns.calls == []
    assert goals.settled == [_settled()]


async def test_a_requested_continuation_is_admitted_through_the_same_delivery() -> None:
    turns = FakeTurns(result=False, results=[True])
    driver = _driver(turns=turns, runs=FakeRuns(view=_view()), goals=FakeGoalRounds())

    delivered = await driver.deliver_pending_goal_round(_OWNER, _RUN)

    assert delivered is True
    assert turns.calls == [
        {"owner_id": _OWNER, "parent_run_id": _RUN, "workspaces": ("workspace-a",)}
    ]


async def test_a_continuation_request_for_another_run_is_not_delivered() -> None:
    turns = FakeTurns()
    driver = _driver(turns=turns, runs=FakeRuns(view=None), goals=FakeGoalRounds())

    assert await driver.deliver_pending_goal_round(_OWNER, "run-gone") is False
    assert turns.calls == []
