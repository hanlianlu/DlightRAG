# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Deliver the messages a reader queued while a Run was working (S0.3).

The settlement observer used to be the goal round driver, so a settlement could start a goal
round while the reader's queued messages were still waiting. This driver makes the order
explicit instead of racing two observers: it offers the queue first, then hands the same
settlement to the goal round driver. A goal round cannot start on top of the queued turn,
because the goal driver's own session-busy check sees the Run this driver just accepted.

Delivery is best-effort in the same sense as the goal round driver: a failure here never
reaches the Run it is about, the queue rows are left where they are, and the next settlement
( or a new send) tries again. The queue rows are removed only after the accept succeeds, so a
crash between the accept and the remove replays the same turn instead of duplicating it.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from dlightrag.application.runs import RunService, RunView
from dlightrag.engine.runtime.coordinator import SettledRun

logger = logging.getLogger(__name__)

#: The Run kind whose settlement can deliver queued conversation work.
_ANSWER = "answer"


class QueuedTurnTurns(Protocol):
    """The conversation surface that accepts a coalesced queued turn."""

    async def accept_queued_turn(
        self,
        owner_id: str,
        *,
        parent_run_id: str,
        workspaces: tuple[str, ...],
    ) -> bool: ...


class QueuedTurnContinuation(Protocol):
    """The goal round driver, called after the queue had its chance."""

    async def settle(self, settled: SettledRun) -> None: ...

    async def run_cancelled(self, view: RunView) -> None: ...


class QueuedTurnDriver:
    """The one settlement observer: human queued work first, then the goal round."""

    def __init__(
        self,
        *,
        turns: QueuedTurnTurns,
        runs: RunService,
        goal_rounds: QueuedTurnContinuation | None = None,
    ) -> None:
        self._turns = turns
        self._runs = runs
        self._goal_rounds = goal_rounds
        self._tasks: set[asyncio.Task[None]] = set()

    def run_settled(self, settled: SettledRun) -> None:
        """The coordinator's synchronous, schedule-only callback."""
        if settled.run_kind != _ANSWER:
            return
        try:
            task = asyncio.get_running_loop().create_task(self.settle(settled))
        except RuntimeError:  # no loop: the composition is shutting down
            return
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def settle(self, settled: SettledRun) -> None:
        """Deliver queued work for a settled answer Run, then let the goal continue.

        Human queued words are delivered first. The goal driver then requests its next-turn
        continuation, and the second delivery attempt admits it if nothing else took the
        Session — one settlement, at most one new turn.
        """
        view = None
        try:
            view = await self._runs.get(owner_id=settled.owner_id, run_id=settled.run_id)
            if view is not None and view.run_kind == _ANSWER:
                await self._deliver(settled.owner_id, view)
        except Exception:
            logger.warning(
                "Queued turn delivery failed after run %s settled",
                settled.run_id,
                exc_info=True,
            )
        if self._goal_rounds is not None:
            await self._goal_rounds.settle(settled)
        if view is not None and view.run_kind == _ANSWER:
            try:
                await self._deliver(settled.owner_id, view)
            except Exception:
                logger.warning(
                    "Goal continuation delivery failed after run %s settled",
                    settled.run_id,
                    exc_info=True,
                )

    async def deliver_pending_goal_round(self, owner_id: str, parent_run_id: str) -> bool:
        """Admit a goal continuation requested for a settled parent Run, if the Session is free."""
        view = await self._runs.get(owner_id=owner_id, run_id=parent_run_id)
        if view is None or view.run_kind != _ANSWER:
            return False
        return await self._deliver(owner_id, view)

    async def run_cancelled(self, view: RunView) -> None:
        """A queued Run cancelled before execution still releases the reader's queue."""
        if view.run_kind == _ANSWER:
            try:
                await self._deliver(view.submitted_by, view)
            except Exception:
                logger.warning(
                    "Queued turn delivery failed after run %s was cancelled",
                    view.run_id,
                    exc_info=True,
                )
        if self._goal_rounds is not None:
            await self._goal_rounds.run_cancelled(view)

    async def _deliver(self, owner_id: str, view: RunView) -> bool:
        workspaces = tuple(str(item) for item in (view.request.get("workspaces") or ()))
        delivered = await self._turns.accept_queued_turn(
            owner_id,
            parent_run_id=view.run_id,
            workspaces=workspaces or ("default",),
        )
        if delivered:
            logger.info("Delivered the next queued turn that followed run %s", view.run_id)
        return delivered


__all__ = ["QueuedTurnContinuation", "QueuedTurnDriver", "QueuedTurnTurns"]
