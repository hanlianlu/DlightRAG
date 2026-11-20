# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Goal operations for one Agent Session (ADR 0060, ADR 0077).

The service is the authority layer the harness tools and the round driver call. It owns two
things the domain deliberately does not:

- **Authority.** Creating, editing, pausing and resuming are a direct human turn's decisions;
  a goal round may only complete or block, and from a round a block is accepted only after the
  reference shape's threshold of started rounds. A durable ``paused`` goal is resumed by the
  human: the model's tool path cannot resume it, whatever turn it runs in.
- **Durability.** Every mutation is a compare-and-set on the revision the caller read; a lost
  race is a stale refusal, never a silent last-writer-wins. The durable row is also the whole
  continuation state: there is no process-local activation, so any process may pick an active
  goal up and a restart continues it (ADR 0077).

The store is a narrow port (load/insert/replace/replace-completed/delete/list-active) so the
PostgreSQL adapter and the unit fake are interchangeable. This module names no Skill and no
Tool: it is harness state.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from dlightrag.engine.answer.goal import (
    BLOCKED_AFTER_ROUNDS,
    GOAL_INVALID_TRANSITION,
    GOAL_NOT_FOUND,
    GOAL_STALE_REVISION,
    Goal,
    GoalAction,
    GoalError,
    GoalPhase,
)

logger = logging.getLogger(__name__)

#: Actions only a direct human turn may take. ``complete`` and ``blocked`` are the model's.
HUMAN_ONLY_ACTIONS = frozenset({"edit", "pause", "resume"})

#: What a goal carries when the round cap, rather than the model, ends the continuation.
ROUND_LIMIT_REASON = (
    "the goal reached its round cap; edit the cap or scope a new objective to continue"
)

#: What a goal carries when a round ended at the model's token limit; the work is resumable.
MODEL_OUTPUT_LIMIT_REASON = "the goal round stopped at the model token limit; resume to continue"


@dataclass(frozen=True, slots=True)
class ActiveGoal:
    """One active goal a process must take up, with the Run that last carried its Session.

    ``parent_run_id`` is the newest answer Run of the Session, whatever its status: a terminal one
    is the Run whose settlement starts the next round, and a non-terminal one is already advancing
    the Session (its own settlement will continue the goal). It is ``None`` only for a Session that
    has never run an answer turn.
    """

    owner_id: str
    session_id: str
    goal: Goal
    parent_run_id: str | None


class GoalStore(Protocol):
    """One Session's current goal row, with compare-and-set semantics.

    ``insert``, ``replace`` and ``replace_completed`` return ``None`` when the write lost: a goal
    already exists, or the row is no longer at the revision the caller read. All are refusals,
    never exceptions.
    """

    async def load(self, *, owner_id: str, session_id: str) -> Goal | None: ...

    async def insert(
        self, *, owner_id: str, session_id: str, goal: Goal, run_id: str
    ) -> Goal | None: ...

    async def replace(
        self,
        *,
        owner_id: str,
        session_id: str,
        previous: Goal,
        goal: Goal,
        run_id: str,
    ) -> Goal | None: ...

    async def replace_completed(
        self,
        *,
        owner_id: str,
        session_id: str,
        previous: Goal,
        goal: Goal,
        run_id: str,
    ) -> Goal | None: ...

    async def delete(self, *, owner_id: str, session_id: str, previous: Goal) -> bool: ...

    async def list_active(self) -> tuple[ActiveGoal, ...]: ...


class AnswerGoalService:
    """The durable goal service a Run's tools and the round driver call."""

    def __init__(self, *, store: GoalStore) -> None:
        self._store = store

    async def get(self, *, owner_id: str, session_id: str) -> Goal | None:
        return await self._store.load(owner_id=owner_id, session_id=session_id)

    async def active_goals(self) -> tuple[ActiveGoal, ...]:
        """Every durable active goal, read-only; the startup recovery feeds these to the driver."""
        return await self._store.list_active()

    async def create(
        self,
        *,
        owner_id: str,
        session_id: str,
        run_id: str,
        objective: str,
        max_rounds: int,
        goal_round: int | None = None,
    ) -> Goal:
        """Create the session's one goal on a direct human turn.

        A completed goal is replaced atomically by the new one; an unfinished goal must be
        paused, blocked or cleared first, so a second objective never silently overwrites work
        the driver may still be continuing.
        """
        if goal_round is not None:
            raise GoalError(
                GOAL_INVALID_TRANSITION,
                "a goal is created on a direct human turn, not from a goal round",
            )
        goal = Goal.create(objective, max_rounds=max_rounds)
        current = await self._store.load(owner_id=owner_id, session_id=session_id)
        if current is not None:
            return await self._replace_completed(
                current, goal, owner_id=owner_id, session_id=session_id, run_id=run_id
            )
        stored = await self._store.insert(
            owner_id=owner_id, session_id=session_id, goal=goal, run_id=run_id
        )
        if stored is not None:
            return stored
        # A concurrent turn inserted a goal after this one's read: decide from the row that
        # exists now instead of reporting a loss the caller cannot see.
        current = await self._store.load(owner_id=owner_id, session_id=session_id)
        if current is None:
            raise GoalError(
                GOAL_STALE_REVISION, "the goal changed while it was created; read it again"
            )
        return await self._replace_completed(
            current, goal, owner_id=owner_id, session_id=session_id, run_id=run_id
        )

    async def update(
        self,
        *,
        owner_id: str,
        session_id: str,
        run_id: str,
        goal_id: str,
        revision: int,
        action: GoalAction,
        objective: str | None = None,
        max_rounds: int | None = None,
        blocked_reason: str | None = None,
        goal_round: int | None = None,
        human_control: bool = False,
    ) -> Goal:
        """Move the stored goal through one domain transition, or refuse it.

        ``human_control`` is the Web goal bar's authority; the model's tool path leaves it false,
        so a durable ``paused`` goal cannot be resumed by an Agent on its own.
        """
        current = await self._store.load(owner_id=owner_id, session_id=session_id)
        if current is None or current.id != goal_id:
            raise GoalError(GOAL_NOT_FOUND, "this conversation carries no goal with that id")
        if action in HUMAN_ONLY_ACTIONS and goal_round is not None:
            raise GoalError(
                GOAL_INVALID_TRANSITION,
                f"{action} is a direct human turn's action, not a goal round's",
            )
        if action == "resume" and current.phase == GoalPhase.ACTIVE:
            # An active goal already continues: a human resume of it is an idempotent no-op, not
            # a phase change, so the stored row keeps its revision and the driver keeps working.
            if revision != current.revision:
                raise GoalError(
                    GOAL_STALE_REVISION,
                    "the goal changed while this turn read it; read it again",
                )
            return current
        if action == "resume" and current.phase == GoalPhase.PAUSED and not human_control:
            raise GoalError(
                GOAL_INVALID_TRANSITION,
                "a paused goal is resumed by the human, not by a goal round or an Agent tool",
            )
        if (
            action == "blocked"
            and goal_round is not None
            and current.rounds_started < BLOCKED_AFTER_ROUNDS
        ):
            raise GoalError(
                GOAL_INVALID_TRANSITION,
                f"blocked needs {BLOCKED_AFTER_ROUNDS} rounds started; "
                f"this goal has {current.rounds_started}",
            )
        changed = self._transition(
            current,
            action=action,
            revision=revision,
            objective=objective,
            max_rounds=max_rounds,
            blocked_reason=blocked_reason,
        )
        stored = await self._store.replace(
            owner_id=owner_id,
            session_id=session_id,
            previous=current,
            goal=changed,
            run_id=run_id,
        )
        if stored is None:
            raise GoalError(
                GOAL_STALE_REVISION, "the goal changed while this turn read it; read it again"
            )
        return stored

    async def admit_round(
        self,
        *,
        owner_id: str,
        session_id: str,
        run_id: str,
        goal_id: str,
        revision: int,
    ) -> tuple[Goal, int]:
        """Start one round of an active goal; return the stored goal and the round number."""
        current = await self._store.load(owner_id=owner_id, session_id=session_id)
        if current is None or current.id != goal_id:
            raise GoalError(GOAL_NOT_FOUND, "this conversation carries no goal with that id")
        started, number = current.admit_round(revision=revision)
        stored = await self._store.replace(
            owner_id=owner_id,
            session_id=session_id,
            previous=current,
            goal=started,
            run_id=run_id,
        )
        if stored is None:
            raise GoalError(
                GOAL_STALE_REVISION, "the goal changed while this round was admitted; read it again"
            )
        return stored, number

    async def release_round(
        self,
        *,
        owner_id: str,
        session_id: str,
        run_id: str,
        goal_id: str,
        revision: int,
    ) -> Goal | None:
        """Return an admitted round whose Run could not be created; ``None`` when the goal moved."""
        current = await self._store.load(owner_id=owner_id, session_id=session_id)
        if current is None or current.id != goal_id:
            return None
        released = current.release_round(revision=revision)
        return await self._store.replace(
            owner_id=owner_id,
            session_id=session_id,
            previous=current,
            goal=released,
            run_id=run_id,
        )

    async def block_round_limit(
        self,
        *,
        owner_id: str,
        session_id: str,
        run_id: str,
        goal_id: str,
        revision: int,
    ) -> Goal | None:
        """End an active goal at its round cap the way the reference shape does: ``blocked``."""
        return await self._stop_active(
            owner_id=owner_id,
            session_id=session_id,
            run_id=run_id,
            goal_id=goal_id,
            revision=revision,
            transition=lambda goal, fresh: goal.block(revision=fresh, reason=ROUND_LIMIT_REASON),
        )

    async def stop_round_failed(
        self,
        *,
        owner_id: str,
        session_id: str,
        run_id: str,
        goal_id: str,
        revision: int,
        reason: str,
    ) -> Goal | None:
        """Harness-only stop: an active goal whose round failed becomes ``blocked`` (retryable).

        This is not model authority: it records what the round driver observed, so a failed
        continuation is visible and waits for a human resume instead of silently retrying.
        """
        return await self._stop_active(
            owner_id=owner_id,
            session_id=session_id,
            run_id=run_id,
            goal_id=goal_id,
            revision=revision,
            transition=lambda goal, fresh: goal.block(revision=fresh, reason=reason),
        )

    async def stop_round_cancelled(
        self,
        *,
        owner_id: str,
        session_id: str,
        run_id: str,
        goal_id: str,
        revision: int,
    ) -> Goal | None:
        """Harness-only stop: an active goal whose round was cancelled becomes ``paused``."""
        return await self._stop_active(
            owner_id=owner_id,
            session_id=session_id,
            run_id=run_id,
            goal_id=goal_id,
            revision=revision,
            transition=lambda goal, fresh: goal.pause(revision=fresh),
        )

    async def stop_round_token_limit(
        self,
        *,
        owner_id: str,
        session_id: str,
        run_id: str,
        goal_id: str,
        revision: int,
    ) -> Goal | None:
        """Harness-only stop: a round that ended at the model token limit becomes ``blocked``."""
        return await self._stop_active(
            owner_id=owner_id,
            session_id=session_id,
            run_id=run_id,
            goal_id=goal_id,
            revision=revision,
            transition=lambda goal, fresh: goal.block(
                revision=fresh, reason=MODEL_OUTPUT_LIMIT_REASON
            ),
        )

    async def clear(self, *, owner_id: str, session_id: str, run_id: str) -> bool:
        """Remove the session's current goal on a direct human request."""
        current = await self._store.load(owner_id=owner_id, session_id=session_id)
        if current is None:
            return False
        return await self._store.delete(owner_id=owner_id, session_id=session_id, previous=current)

    # -- internals ---------------------------------------------------------------------------

    async def _replace_completed(
        self,
        current: Goal,
        goal: Goal,
        *,
        owner_id: str,
        session_id: str,
        run_id: str,
    ) -> Goal:
        """Replace a completed row with a fresh goal, or refuse with the phase that blocks it."""
        if current.phase is not GoalPhase.COMPLETE:
            raise GoalError(
                GOAL_INVALID_TRANSITION,
                "this conversation already carries an unfinished goal; pause, block or clear it "
                "before creating a new one",
            )
        stored = await self._store.replace_completed(
            owner_id=owner_id,
            session_id=session_id,
            previous=current,
            goal=goal,
            run_id=run_id,
        )
        if stored is None:
            raise GoalError(
                GOAL_STALE_REVISION,
                "the goal changed while it was being replaced; read it again",
            )
        return stored

    async def _stop_active(
        self,
        *,
        owner_id: str,
        session_id: str,
        run_id: str,
        goal_id: str,
        revision: int,
        transition: Callable[[Goal, int], Goal],
    ) -> Goal | None:
        """Apply one harness stop to an active goal; ``None`` when the goal already moved.

        The stop is a snapshot of a finished round, not an authority over a later human decision.
        It applies only to the exact revision the round served: a row that paused, completed, was
        replaced, or was edited/resumed by a human is the winner, and the stop is a no-op. There
        is no retry — a lost compare-and-set means a newer decision exists, and re-applying the old
        round's outcome would silently undo it (the pause -> resume race this fence closes).
        """
        current = await self._store.load(owner_id=owner_id, session_id=session_id)
        if current is None or current.id != goal_id or current.phase is not GoalPhase.ACTIVE:
            return None
        try:
            changed = transition(current, revision)
        except GoalError:
            return None
        return await self._store.replace(
            owner_id=owner_id,
            session_id=session_id,
            previous=current,
            goal=changed,
            run_id=run_id,
        )

    @staticmethod
    def _transition(
        current: Goal,
        *,
        action: str,
        revision: int,
        objective: str | None,
        max_rounds: int | None,
        blocked_reason: str | None,
    ) -> Goal:
        if action == "edit":
            return current.edit(revision=revision, objective=objective, max_rounds=max_rounds)
        if action == "pause":
            return current.pause(revision=revision)
        if action == "resume":
            return current.resume(revision=revision)
        if action == "complete":
            return current.complete(revision=revision)
        if action == "blocked":
            if blocked_reason is None:
                raise GoalError(
                    GOAL_INVALID_TRANSITION,
                    "blocked needs the concrete condition in blocked_reason",
                )
            return current.block(revision=revision, reason=blocked_reason)
        raise GoalError(
            GOAL_INVALID_TRANSITION,
            f"unknown action {action!r}; use edit, pause, resume, complete or blocked",
        )
