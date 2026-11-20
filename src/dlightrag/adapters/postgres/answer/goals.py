# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""A Session's current Goal in PostgreSQL (ADR 0060, ADR 0077).

One row per Agent Session, cascading with the Session like its notes. The row is the whole
durable goal: objective, phase, round cap and rounds started. Every mutation is a compare-and-set
on ``revision``, so a driver and a human turn that read the same goal cannot both overwrite it;
the loser is refused as stale and re-reads. A completed goal may be replaced by a fresh one in
the same conversation, which is one atomic compare-and-set on the completed row. This module owns
the DDL, the schema declaration the ``runs`` scope verifies, and the adapter the application's
goal service calls.
"""

from __future__ import annotations

from typing import Any

from dlightrag.adapters.postgres.core._migrations import ForeignKeyRequirement, TableRequirement
from dlightrag.adapters.postgres.core._operations import ConnectionPool, PostgresOperationRunner
from dlightrag.engine.answer.goal import Goal, GoalPhase
from dlightrag.engine.answer.goal_service import ActiveGoal

ANSWER_GOALS_DDL = (
    """
CREATE TABLE IF NOT EXISTS dlightrag_answer_goals (
    owner_id         TEXT        NOT NULL,
    session_id       UUID        NOT NULL,
    goal_id          TEXT        NOT NULL,
    revision         BIGINT      NOT NULL,
    objective        TEXT        NOT NULL,
    phase            TEXT        NOT NULL,
    max_rounds       BIGINT      NOT NULL,
    rounds_started   INTEGER     NOT NULL DEFAULT 0,
    blocked_reason   TEXT,
    written_by_run_id TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id, session_id),
    CONSTRAINT dlightrag_answer_goals_goal_id_key UNIQUE (owner_id, goal_id),
    CONSTRAINT dlightrag_answer_goals_session_fkey
        FOREIGN KEY (owner_id, session_id)
        REFERENCES dlightrag_agent_sessions (owner_id, session_id) ON DELETE CASCADE,
    CONSTRAINT dlightrag_answer_goals_revision_check CHECK (revision >= 1),
    CONSTRAINT dlightrag_answer_goals_phase_check
        CHECK (phase IN ('active', 'paused', 'blocked', 'complete')),
    CONSTRAINT dlightrag_answer_goals_rounds_check
        CHECK (max_rounds >= 1
               AND rounds_started >= 0 AND rounds_started <= max_rounds),
    CONSTRAINT dlightrag_answer_goals_objective_check
        CHECK (char_length(objective) BETWEEN 1 AND 2000),
    CONSTRAINT dlightrag_answer_goals_blocked_check
        CHECK ((phase = 'blocked') = (blocked_reason IS NOT NULL)),
    CONSTRAINT dlightrag_answer_goals_blocked_reason_check
        CHECK (blocked_reason IS NULL OR char_length(blocked_reason) BETWEEN 1 AND 2000)
)
""",
)

ANSWER_GOALS_SCHEMA_TABLE = TableRequirement(
    name="dlightrag_answer_goals",
    columns=(
        "owner_id",
        "session_id",
        "goal_id",
        "revision",
        "objective",
        "phase",
        "max_rounds",
        "rounds_started",
        "blocked_reason",
        "written_by_run_id",
        "created_at",
        "updated_at",
    ),
    primary_key=("owner_id", "session_id"),
    unique=(("owner_id", "goal_id"),),
    foreign_keys=(
        ForeignKeyRequirement(
            columns=("owner_id", "session_id"), references="dlightrag_agent_sessions"
        ),
    ),
    checks=(
        "dlightrag_answer_goals_revision_check",
        "dlightrag_answer_goals_phase_check",
        "dlightrag_answer_goals_rounds_check",
        "dlightrag_answer_goals_objective_check",
        "dlightrag_answer_goals_blocked_check",
        "dlightrag_answer_goals_blocked_reason_check",
    ),
)

_READ_GOAL = """
SELECT goal_id, revision, objective, phase, max_rounds, rounds_started, blocked_reason
FROM dlightrag_answer_goals
WHERE owner_id = $1 AND session_id = $2::uuid
"""

#: Every goal the database still records as active, with the newest answer Run that carried its
#: Session. The Run is the one the round driver resumes from after a restart: a terminal Run's
#: settlement starts the next round, a running one is already advancing the Session.
_LIST_ACTIVE_GOALS = """
SELECT goal.owner_id,
       goal.session_id,
       goal.goal_id,
       goal.revision,
       goal.objective,
       goal.phase,
       goal.max_rounds,
       goal.rounds_started,
       goal.blocked_reason,
       run.run_id AS parent_run_id
FROM dlightrag_answer_goals AS goal
LEFT JOIN LATERAL (
    SELECT candidate.run_id
    FROM dlightrag_runs AS candidate
    JOIN dlightrag_answer_run_routing AS routing
      ON routing.owner_id = candidate.owner_id
     AND routing.run_id = candidate.run_id
    WHERE candidate.owner_id = goal.owner_id
      AND routing.agent_session_id = goal.session_id
      AND candidate.run_kind = 'answer'
    ORDER BY candidate.created_at DESC
    LIMIT 1
) AS run ON TRUE
WHERE goal.phase = 'active'
ORDER BY goal.owner_id, goal.session_id
"""

_INSERT_GOAL = """
INSERT INTO dlightrag_answer_goals (
    owner_id, session_id, goal_id, revision, objective, phase, max_rounds,
    rounds_started, blocked_reason, written_by_run_id, created_at, updated_at
) VALUES ($1, $2::uuid, $3, $4, $5, $6, $7, $8, $9, $10, NOW(), NOW())
ON CONFLICT (owner_id, session_id) DO NOTHING
RETURNING goal_id, revision, objective, phase, max_rounds, rounds_started, blocked_reason
"""

_DELETE_GOAL = """
DELETE FROM dlightrag_answer_goals
WHERE owner_id = $1 AND session_id = $2::uuid AND goal_id = $3 AND revision = $4
RETURNING goal_id
"""

_REPLACE_GOAL = """
UPDATE dlightrag_answer_goals
SET revision = $4,
    objective = $5,
    phase = $6,
    max_rounds = $7,
    rounds_started = $8,
    blocked_reason = $9,
    written_by_run_id = $10,
    updated_at = NOW()
WHERE owner_id = $1 AND session_id = $2::uuid AND goal_id = $3 AND revision = $11
RETURNING goal_id, revision, objective, phase, max_rounds, rounds_started, blocked_reason
"""

#: Replace a completed goal with a fresh one in one statement: the new row keeps its own id,
#: revision 1 and zero rounds, and the write only lands while the stored row is still the
#: completed row the caller read. It is the same compare-and-set every other mutation uses.
_REPLACE_COMPLETED_GOAL = """
UPDATE dlightrag_answer_goals
SET goal_id = $4,
    revision = $5,
    objective = $6,
    phase = $7,
    max_rounds = $8,
    rounds_started = $9,
    blocked_reason = $10,
    written_by_run_id = $11,
    created_at = NOW(),
    updated_at = NOW()
WHERE owner_id = $1 AND session_id = $2::uuid AND goal_id = $3
  AND revision = $12 AND phase = 'complete'
RETURNING goal_id, revision, objective, phase, max_rounds, rounds_started, blocked_reason
"""


def _goal_from_row(row: Any) -> Goal:
    return Goal(
        id=str(row["goal_id"]),
        revision=int(row["revision"]),
        objective=str(row["objective"]),
        phase=GoalPhase(str(row["phase"])),
        max_rounds=int(row["max_rounds"]),
        rounds_started=int(row["rounds_started"]),
        blocked_reason=None if row["blocked_reason"] is None else str(row["blocked_reason"]),
    )


class PGAnswerGoalStore(PostgresOperationRunner):
    """The compare-and-set store behind the application's goal service."""

    def __init__(self, *, pool: ConnectionPool | None = None) -> None:
        super().__init__(pool=pool)

    async def load(self, *, owner_id: str, session_id: str) -> Goal | None:
        async def operation(conn: Any) -> Goal | None:
            row = await conn.fetchrow(_READ_GOAL, owner_id, session_id)
            return None if row is None else _goal_from_row(row)

        return await self._run(operation)

    async def list_active(self) -> tuple[ActiveGoal, ...]:
        """Every active goal, with the newest answer Run that carried its Session."""

        async def operation(conn: Any) -> tuple[ActiveGoal, ...]:
            rows = await conn.fetch(_LIST_ACTIVE_GOALS)
            return tuple(
                ActiveGoal(
                    owner_id=str(row["owner_id"]),
                    session_id=str(row["session_id"]),
                    goal=_goal_from_row(row),
                    parent_run_id=(
                        None if row["parent_run_id"] is None else str(row["parent_run_id"])
                    ),
                )
                for row in rows
            )

        return await self._run(operation)

    async def insert(
        self, *, owner_id: str, session_id: str, goal: Goal, run_id: str
    ) -> Goal | None:
        """Write a fresh goal; ``None`` means this Session already carries one."""

        async def operation(conn: Any) -> Goal | None:
            row = await conn.fetchrow(
                _INSERT_GOAL,
                owner_id,
                session_id,
                goal.id,
                goal.revision,
                goal.objective,
                str(goal.phase),
                goal.max_rounds,
                goal.rounds_started,
                goal.blocked_reason,
                run_id,
            )
            return None if row is None else _goal_from_row(row)

        return await self._run_once(operation)

    async def delete(self, *, owner_id: str, session_id: str, previous: Goal) -> bool:
        """Delete the stored goal only if it is still at ``previous``'s revision."""

        async def operation(conn: Any) -> bool:
            row = await conn.fetchrow(
                _DELETE_GOAL, owner_id, session_id, previous.id, previous.revision
            )
            return row is not None

        return await self._run_once(operation)

    async def replace(
        self,
        *,
        owner_id: str,
        session_id: str,
        previous: Goal,
        goal: Goal,
        run_id: str,
    ) -> Goal | None:
        """Move the stored goal from ``previous``'s revision to ``goal``; ``None`` means stale."""

        async def operation(conn: Any) -> Goal | None:
            row = await conn.fetchrow(
                _REPLACE_GOAL,
                owner_id,
                session_id,
                previous.id,
                goal.revision,
                goal.objective,
                str(goal.phase),
                goal.max_rounds,
                goal.rounds_started,
                goal.blocked_reason,
                run_id,
                previous.revision,
            )
            return None if row is None else _goal_from_row(row)

        return await self._run_once(operation)

    async def replace_completed(
        self,
        *,
        owner_id: str,
        session_id: str,
        previous: Goal,
        goal: Goal,
        run_id: str,
    ) -> Goal | None:
        """Replace a completed row with a fresh goal atomically.

        ``None`` means the row was no longer the completed goal the caller read, or another
        mutation won the race first.
        """

        async def operation(conn: Any) -> Goal | None:
            row = await conn.fetchrow(
                _REPLACE_COMPLETED_GOAL,
                owner_id,
                session_id,
                previous.id,
                goal.id,
                goal.revision,
                goal.objective,
                str(goal.phase),
                goal.max_rounds,
                goal.rounds_started,
                goal.blocked_reason,
                run_id,
                previous.revision,
            )
            return None if row is None else _goal_from_row(row)

        return await self._run_once(operation)


__all__ = [
    "ANSWER_GOALS_DDL",
    "ANSWER_GOALS_SCHEMA_TABLE",
    "PGAnswerGoalStore",
]
