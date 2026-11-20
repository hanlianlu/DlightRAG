# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""A Goal store stand-in that reports the PostgreSQL adapter's compare-and-set outcomes."""

from __future__ import annotations

from dlightrag.engine.answer.goal import Goal, GoalPhase
from dlightrag.engine.answer.goal_service import ActiveGoal


class FakeGoalStore:
    """One row per session; a write reports ``None`` exactly when the store would refuse it."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], Goal] = {}
        #: The Run a restart resumes each Session from; a real store reads it from the run rows.
        self.parent_runs: dict[tuple[str, str], str | None] = {}

    async def load(self, *, owner_id: str, session_id: str) -> Goal | None:
        return self.rows.get((owner_id, session_id))

    async def list_active(self) -> tuple[ActiveGoal, ...]:
        return tuple(
            ActiveGoal(
                owner_id=owner_id,
                session_id=session_id,
                goal=goal,
                parent_run_id=self.parent_runs.get((owner_id, session_id)),
            )
            for (owner_id, session_id), goal in self.rows.items()
            if goal.phase == GoalPhase.ACTIVE
        )

    async def insert(
        self, *, owner_id: str, session_id: str, goal: Goal, run_id: str
    ) -> Goal | None:
        del run_id
        if (owner_id, session_id) in self.rows:
            return None
        self.rows[(owner_id, session_id)] = goal
        return goal

    async def delete(self, *, owner_id: str, session_id: str, previous: Goal) -> bool:
        current = self.rows.get((owner_id, session_id))
        if current is None or current.id != previous.id or current.revision != previous.revision:
            return False
        del self.rows[(owner_id, session_id)]
        return True

    async def replace(
        self,
        *,
        owner_id: str,
        session_id: str,
        previous: Goal,
        goal: Goal,
        run_id: str,
    ) -> Goal | None:
        del run_id
        current = self.rows.get((owner_id, session_id))
        if current is None or current.id != previous.id or current.revision != previous.revision:
            return None
        self.rows[(owner_id, session_id)] = goal
        return goal

    async def replace_completed(
        self,
        *,
        owner_id: str,
        session_id: str,
        previous: Goal,
        goal: Goal,
        run_id: str,
    ) -> Goal | None:
        """Replace a completed row atomically; any other phase refuses like the PostgreSQL adapter."""
        del run_id
        current = self.rows.get((owner_id, session_id))
        if (
            current is None
            or current.id != previous.id
            or current.revision != previous.revision
            or current.phase != GoalPhase.COMPLETE
        ):
            return None
        self.rows[(owner_id, session_id)] = goal
        return goal


__all__ = ["FakeGoalStore"]
