# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Session Memory Pass progress mixed into PGRunStore (ADR 0045).

One row per Agent Session, cascading with it like the Session's notes. The row is a
pass's whole durable state: the entry sequence read through, a claim that lets one
pass at a time run across processes, and a plan awaiting application, so a pass that
dies halfway resumes the same writes instead of asking its model again. Every write
but the claim is fenced by its holder, so a pass whose claim was taken over can no
longer move the row; a pass that reads on through several batches renews its claim
between them, fenced the same way. The row also keeps a context floor: the entry
sequence at which the owner last turned Profile Memory back on, below which no earlier
turn is ever shown to a model as context. Turning it on restarts every Session of the
owner at once, in one statement that never moves a watermark or a floor back and leaves
a held claim and a stored plan as they are. The Session's entries are this store's own,
which is why the pass reads its conversation here rather than through a second adapter.

The database is also the queue of passes that were lost. A process restart cancels the
passes it had scheduled, and a failed model call leaves its text unread; either way the
Session's watermark stays behind its own text. ``pending_session_memory`` reads exactly
that off the tables, with no record of its own: a Session is pending when its latest entry
is past its watermark, an own turn lies past it, no Run of it is live, its owner's Profile
Memory is on, and it is neither claimed nor backing off. A pass that fails counts the
failure on the row and sets a time before which a sweep leaves the Session alone; one that
moves the watermark clears both.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from dlightrag_memory.consolidation import PlannedWrite

from dlightrag.adapters.postgres.core._migrations import ForeignKeyRequirement, TableRequirement
from dlightrag.engine.answer.memory import (
    PendingSession,
    SessionMemoryPlan,
    SessionMemoryProgress,
    SessionTurn,
)

_CREATE_SESSION_MEMORY_PROGRESS = """
CREATE TABLE IF NOT EXISTS dlightrag_session_memory_progress (
    owner_id          TEXT        NOT NULL,
    session_id        UUID        NOT NULL,
    read_through      BIGINT      NOT NULL DEFAULT 0,
    holder            TEXT,
    claim_expires_at  TIMESTAMPTZ,
    plan_from         BIGINT,
    plan_through      BIGINT,
    plan_writes       JSONB,
    plan_run_id       TEXT,
    context_floor     BIGINT,
    failure_count     INTEGER     NOT NULL DEFAULT 0,
    retry_after       TIMESTAMPTZ,
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id, session_id),
    FOREIGN KEY (owner_id, session_id)
        REFERENCES dlightrag_agent_sessions (owner_id, session_id) ON DELETE CASCADE,
    CONSTRAINT dlightrag_session_memory_progress_claim_check
        CHECK ((holder IS NULL) = (claim_expires_at IS NULL)),
    CONSTRAINT dlightrag_session_memory_progress_plan_check
        CHECK ((plan_writes IS NULL) = (plan_through IS NULL)
               AND (plan_from IS NULL) = (plan_through IS NULL)
               AND (plan_run_id IS NULL) = (plan_through IS NULL)),
    CONSTRAINT dlightrag_session_memory_progress_backoff_check
        CHECK (failure_count >= 0 AND (failure_count > 0) = (retry_after IS NOT NULL))
)
"""

SESSION_MEMORY_PROGRESS_DDL = (_CREATE_SESSION_MEMORY_PROGRESS,)

SESSION_MEMORY_PROGRESS_SCHEMA_TABLE = TableRequirement(
    name="dlightrag_session_memory_progress",
    columns=(
        "owner_id",
        "session_id",
        "read_through",
        "holder",
        "claim_expires_at",
        "plan_from",
        "plan_through",
        "plan_writes",
        "plan_run_id",
        "context_floor",
        "failure_count",
        "retry_after",
        "updated_at",
    ),
    primary_key=("owner_id", "session_id"),
    foreign_keys=(
        ForeignKeyRequirement(
            columns=("owner_id", "session_id"), references="dlightrag_agent_sessions"
        ),
    ),
    checks=(
        "dlightrag_session_memory_progress_claim_check",
        "dlightrag_session_memory_progress_plan_check",
        "dlightrag_session_memory_progress_backoff_check",
    ),
)

# A Session already deleted gets no row: its pass has nothing left to read.
_ENSURE_ROW = """
INSERT INTO dlightrag_session_memory_progress (owner_id, session_id)
SELECT $1, $2
WHERE EXISTS (
    SELECT 1 FROM dlightrag_agent_sessions WHERE owner_id = $1 AND session_id = $2
)
ON CONFLICT (owner_id, session_id) DO NOTHING
"""

_CLAIM = """
UPDATE dlightrag_session_memory_progress
SET holder = $3,
    claim_expires_at = NOW() + make_interval(secs => $4),
    updated_at = NOW()
WHERE owner_id = $1 AND session_id = $2
  AND (holder IS NULL OR claim_expires_at < NOW())
RETURNING read_through, plan_from, plan_through, plan_writes, plan_run_id
"""

# A claim is extended only by the holder it names and only while it is live, so a pass whose
# claim lapsed or was taken over finds out here, before it asks its model for another batch.
_RENEW = """
UPDATE dlightrag_session_memory_progress
SET claim_expires_at = NOW() + make_interval(secs => $4),
    updated_at = NOW()
WHERE owner_id = $1 AND session_id = $2 AND holder = $3 AND claim_expires_at > NOW()
"""

_RELEASE = """
UPDATE dlightrag_session_memory_progress
SET holder = NULL, claim_expires_at = NULL, updated_at = NOW()
WHERE owner_id = $1 AND session_id = $2 AND holder = $3
"""

_STORE_PLAN = """
UPDATE dlightrag_session_memory_progress
SET plan_from = $4, plan_through = $5, plan_writes = $6::jsonb, plan_run_id = $7,
    updated_at = NOW()
WHERE owner_id = $1 AND session_id = $2 AND holder = $3 AND claim_expires_at > NOW()
"""

# The watermark it leaves is the greater of the one asked for and the one the row already held, which
# a restart may have moved past it while the pass was away; the pass reads on from what it is told.
# A pass that moves the watermark has made progress, so whatever it failed at before is forgiven.
_ADVANCE = """
UPDATE dlightrag_session_memory_progress
SET read_through = GREATEST(read_through, $4),
    plan_from = NULL, plan_through = NULL, plan_writes = NULL, plan_run_id = NULL,
    failure_count = 0, retry_after = NULL,
    updated_at = NOW()
WHERE owner_id = $1 AND session_id = $2 AND holder = $3
RETURNING read_through
"""

# A pass that failed leaves its Session unread, and a sweep must not hand it straight back: each
# failure doubles the wait before the next, from ``$4`` seconds up to ``$5``. The right-hand sides
# read the row as it was, so the first failure waits ``$4`` itself. The exponent is capped far past
# where the ceiling binds, so the arithmetic cannot overflow however long a Session keeps failing.
_BACK_OFF = """
UPDATE dlightrag_session_memory_progress
SET failure_count = failure_count + 1,
    retry_after = NOW() + make_interval(
        secs => LEAST($4::float8 * power(2.0::float8, LEAST(failure_count, 20)::float8), $5::float8)
    ),
    updated_at = NOW()
WHERE owner_id = $1 AND session_id = $2 AND holder = $3
"""

# The owner has turned Profile Memory back on: every one of their Sessions starts afresh from its
# latest entry, so that nothing written while it was off is read, nor shown as context, ever after.
# A Session with no progress row gets one; a row keeps its claim and its stored plan, and neither
# the watermark nor the floor moves back. A Child's Session has no pass of its own, so no row.
_RESTART = """
INSERT INTO dlightrag_session_memory_progress (owner_id, session_id, read_through, context_floor)
SELECT agent_session.owner_id, agent_session.session_id,
       agent_session.last_sequence, agent_session.last_sequence
FROM dlightrag_agent_sessions AS agent_session
WHERE agent_session.owner_id = $1
  AND agent_session.last_sequence > 0
  AND NOT EXISTS (
      SELECT 1 FROM dlightrag_answer_child_sessions AS child
      WHERE child.owner_id = agent_session.owner_id
        AND child.child_session_id = agent_session.session_id
  )
ON CONFLICT (owner_id, session_id) DO UPDATE
SET read_through = GREATEST(dlightrag_session_memory_progress.read_through,
                            EXCLUDED.read_through),
    context_floor = GREATEST(COALESCE(dlightrag_session_memory_progress.context_floor, 0),
                             EXCLUDED.context_floor),
    updated_at = NOW()
"""

# Only the Session's own conversation: its entries are the main Session's, never a
# Child's, which runs under a session id of its own. A Child's notification is the
# Child's output written into this Session as input, not the owner's words. An
# assistant entry that only called tools has no text, and a notification is not read
# at all; reading past either is free, and fetching them would let a window of them
# hold the watermark in place. The turns a batch judges and the turns before it that the
# model is shown as context are the same conversation, so both reads take exactly these.
_OWN_TURNS = """
  AND entry_type IN ('user_message', 'assistant_message')
  AND payload_json->>'content' ~ '[^[:space:]]'
  AND payload_json->>'origin' IS DISTINCT FROM 'child_notification'
"""

_TURNS_AFTER = f"""
SELECT sequence, entry_type, payload_json->>'content' AS content
FROM dlightrag_agent_session_entries
WHERE owner_id = $1 AND session_id = $2 AND sequence > $3
{_OWN_TURNS}
ORDER BY sequence
LIMIT $4
"""  # noqa: S608 - interpolates only the trusted _OWN_TURNS constant

# The last ``$4`` rounds before the point: the Session's ``$4`` most recent user turns before it
# and every own turn from the earliest of them on, so each user turn comes with the assistant turns
# that follow it. The user turns are found newest first, walking the (owner_id, session_id,
# sequence) key backwards with a limit, so the walk stops at them however long the Session is; the
# turns from there are one forward range of the same key, already in order. An assistant turn
# before the earliest of those user turns belongs to no round and is not returned; with no user
# turn before the point there is none either.
# Nothing at or below the Session's context floor is returned: those entries were written before
# the owner last turned Profile Memory back on, and no model is shown them. The floor is one
# primary-key read, taken once; none (no row, or a null) limits nothing.
_TURNS_BEFORE = f"""
WITH floor AS (
    SELECT COALESCE(
        (SELECT context_floor FROM dlightrag_session_memory_progress
         WHERE owner_id = $1 AND session_id = $2), 0) AS sequence
), rounds AS (
    SELECT sequence
    FROM dlightrag_agent_session_entries
    WHERE owner_id = $1 AND session_id = $2 AND sequence < $3
      AND sequence > (SELECT sequence FROM floor)
      AND entry_type = 'user_message'
      {_OWN_TURNS}
    ORDER BY sequence DESC
    LIMIT $4
)
SELECT sequence, entry_type, payload_json->>'content' AS content
FROM dlightrag_agent_session_entries
WHERE owner_id = $1 AND session_id = $2 AND sequence < $3
  AND sequence >= (SELECT MIN(sequence) FROM rounds)
{_OWN_TURNS}
ORDER BY sequence
"""  # noqa: S608 - interpolates only the trusted _OWN_TURNS constant

# The Sessions a sweep hands back to a pass. Every condition is read off the tables, because a
# restart that lost a pass left nothing to remember it by. A Session is pending when:
#   - its latest entry is past its watermark (none yet is a watermark of 0), and an own turn, one a
#     pass would read, lies past it: a tail of tool results alone has nothing for a pass to read, and
#     a pass that read nothing would not move the watermark, so it would be pending for ever;
#   - it is a main Session with a succeeded Answer Run (a Child's Session has no routing row), and the
#     newest of those Runs is the one a pass of it is attributed to;
#   - none of its Runs is live: a live Run schedules its own pass when it succeeds;
#   - its owner's Profile Memory is not off (a row of settings says so; none means on): a pass moves
#     nothing while it is off, so such a Session would be handed back at every sweep;
#   - no pass holds it, and it is not backing off after failures.
# The work is staged so it stays cheap as the tables grow. The first stage is one pass over the Session
# counters against their progress rows, narrow rows with no entries among them, and it keeps only the
# Sessions that are behind (the materialized fence is what makes the planner finish it before it
# probes anything). Everything after it is one index probe per such Session and no more: the routing
# index by Session finds its Runs, the Runs' and the entries' primary keys answer the rest. The
# newest conversations come first.
_SWEEP = f"""
WITH behind AS MATERIALIZED (
    SELECT agent_session.owner_id, agent_session.session_id, agent_session.updated_at,
           COALESCE(progress.read_through, 0) AS read_through
    FROM dlightrag_agent_sessions AS agent_session
    LEFT JOIN dlightrag_session_memory_progress AS progress
           ON progress.owner_id = agent_session.owner_id
          AND progress.session_id = agent_session.session_id
    WHERE agent_session.last_sequence > COALESCE(progress.read_through, 0)
      AND (progress.holder IS NULL OR progress.claim_expires_at < NOW())
      AND (progress.retry_after IS NULL OR progress.retry_after <= NOW())
      AND NOT EXISTS (
          SELECT 1 FROM dlightrag_answer_memory_settings AS settings
          WHERE settings.owner_id = agent_session.owner_id AND NOT settings.enabled
      )
)
SELECT behind.owner_id, behind.session_id::text AS session_id, latest.run_id::text AS run_id
FROM behind
CROSS JOIN LATERAL (
    SELECT run.run_id
    FROM dlightrag_answer_run_routing AS routing
    JOIN dlightrag_runs AS run
      ON run.owner_id = routing.owner_id AND run.run_id = routing.run_id
    WHERE routing.owner_id = behind.owner_id AND routing.agent_session_id = behind.session_id
      AND run.run_kind = 'answer' AND run.status = 'succeeded'
    ORDER BY run.finished_at DESC, run.run_id DESC
    LIMIT 1
) AS latest
LEFT JOIN LATERAL (
    SELECT 1 AS live
    FROM dlightrag_answer_run_routing AS routing
    JOIN dlightrag_runs AS run
      ON run.owner_id = routing.owner_id AND run.run_id = routing.run_id
    WHERE routing.owner_id = behind.owner_id AND routing.agent_session_id = behind.session_id
      AND run.status IN ('queued', 'running')
    LIMIT 1
) AS live_run ON TRUE
CROSS JOIN LATERAL (
    SELECT 1 AS unread
    FROM dlightrag_agent_session_entries
    WHERE owner_id = behind.owner_id AND session_id = behind.session_id
      AND sequence > behind.read_through
    {_OWN_TURNS}
    LIMIT 1
) AS unread_text
WHERE live_run.live IS NULL
ORDER BY behind.updated_at DESC, behind.session_id
LIMIT $1
"""  # noqa: S608 - interpolates only the trusted _OWN_TURNS constant


class SessionMemoryRunStoreMixin:
    """Session Memory Pass progress and conversation reads on PGRunStore."""

    _run_read: Callable[..., Awaitable[Any]]
    _run_write: Callable[..., Awaitable[Any]]

    async def claim_session_memory(
        self, *, owner_id: str, session_id: str, holder: str, lease_seconds: float
    ) -> SessionMemoryProgress | None:
        """Take the Session's pass for a while, or None when another holds it."""

        async def operation(conn: Any) -> SessionMemoryProgress | None:
            async with conn.transaction():
                await conn.execute(_ENSURE_ROW, owner_id, uuid.UUID(session_id))
                row = await conn.fetchrow(
                    _CLAIM, owner_id, uuid.UUID(session_id), holder, float(lease_seconds)
                )
            return None if row is None else _progress(row)

        return await self._run_write(operation)

    async def renew_session_memory(
        self, *, owner_id: str, session_id: str, holder: str, lease_seconds: float
    ) -> bool:
        """Extend the holder's live claim; False once it lapsed or another took it over."""

        async def operation(conn: Any) -> bool:
            status = await conn.execute(
                _RENEW, owner_id, uuid.UUID(session_id), holder, float(lease_seconds)
            )
            return _updated(status)

        return await self._run_write(operation)

    async def release_session_memory(self, *, owner_id: str, session_id: str, holder: str) -> None:
        async def operation(conn: Any) -> None:
            await conn.execute(_RELEASE, owner_id, uuid.UUID(session_id), holder)

        await self._run_write(operation)

    async def store_session_memory_plan(
        self, *, owner_id: str, session_id: str, holder: str, plan: SessionMemoryPlan
    ) -> bool:
        """Keep a plan before any of it applies; False once the claim is gone."""
        writes = json.dumps([write.to_json() for write in plan.writes], ensure_ascii=False)

        async def operation(conn: Any) -> bool:
            status = await conn.execute(
                _STORE_PLAN,
                owner_id,
                uuid.UUID(session_id),
                holder,
                plan.read_from,
                plan.read_through,
                writes,
                plan.run_id,
            )
            return _updated(status)

        return await self._run_write(operation)

    async def advance_session_memory(
        self, *, owner_id: str, session_id: str, holder: str, through: int
    ) -> int | None:
        """Move the watermark and clear an applied plan; None once the claim is gone.

        Answers where the watermark stands, which is past ``through`` if a restart moved it there.
        """

        async def operation(conn: Any) -> int | None:
            value = await conn.fetchval(_ADVANCE, owner_id, uuid.UUID(session_id), holder, through)
            return None if value is None else int(value)

        return await self._run_write(operation)

    async def back_off_session_memory(
        self,
        *,
        owner_id: str,
        session_id: str,
        holder: str,
        first_seconds: float,
        max_seconds: float,
    ) -> None:
        """Count a failed pass on the Session's row and set when a sweep may offer it again.

        The wait is ``first_seconds`` after the first failure and doubles with each, to
        ``max_seconds``; fenced by the holder, so a pass whose claim was taken over records nothing.
        """

        async def operation(conn: Any) -> None:
            await conn.execute(
                _BACK_OFF,
                owner_id,
                uuid.UUID(session_id),
                holder,
                float(first_seconds),
                float(max_seconds),
            )

        await self._run_write(operation)

    async def pending_session_memory(self, *, limit: int) -> tuple[PendingSession, ...]:
        """At most ``limit`` Sessions whose pass was lost, newest first, each with its Run.

        A Session is pending when it is behind its own unread text and nothing is about to read
        it: see ``_SWEEP`` for each condition.
        """

        async def operation(conn: Any) -> tuple[PendingSession, ...]:
            rows = await conn.fetch(_SWEEP, int(limit))
            return tuple(
                PendingSession(
                    owner_id=str(row["owner_id"]),
                    session_id=str(row["session_id"]),
                    run_id=str(row["run_id"]),
                )
                for row in rows
            )

        return await self._run_read(operation)

    async def restart_session_memory(self, *, owner_id: str) -> None:
        """Start every one of the owner's Sessions afresh from its latest entry.

        The owner has turned Profile Memory back on. Each Session's watermark and context floor move
        to its latest entry sequence in one statement and never back; a held claim and a stored
        plan are left alone, so a pass in flight reads on from where the watermark now stands.
        """

        async def operation(conn: Any) -> None:
            await conn.execute(_RESTART, owner_id)

        await self._run_write(operation)

    async def session_turns_after(
        self, *, owner_id: str, session_id: str, after: int, limit: int
    ) -> tuple[SessionTurn, ...]:
        """The Session's own user and assistant turns with text, past ``after``."""

        async def operation(conn: Any) -> tuple[SessionTurn, ...]:
            rows = await conn.fetch(_TURNS_AFTER, owner_id, uuid.UUID(session_id), after, limit)
            return tuple(_turn(row) for row in rows)

        return await self._run_read(operation)

    async def session_turns_before(
        self, *, owner_id: str, session_id: str, before: int, rounds: int
    ) -> tuple[SessionTurn, ...]:
        """The Session's own turns with text in its last ``rounds`` rounds before ``before``.

        A round is a user turn and the assistant turns that follow it; oldest first.
        """

        async def operation(conn: Any) -> tuple[SessionTurn, ...]:
            rows = await conn.fetch(_TURNS_BEFORE, owner_id, uuid.UUID(session_id), before, rounds)
            return tuple(_turn(row) for row in rows)

        return await self._run_read(operation)


def _turn(row: Any) -> SessionTurn:
    return SessionTurn(
        sequence=int(row["sequence"]),
        role="user" if row["entry_type"] == "user_message" else "assistant",
        text=str(row["content"]),
    )


def _progress(row: Any) -> SessionMemoryProgress:
    plan = None
    if row["plan_writes"] is not None:
        raw = row["plan_writes"]
        writes = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(writes, list):
            raise ValueError("a stored Session Memory plan is a list of writes")
        plan = SessionMemoryPlan(
            read_from=int(row["plan_from"]),
            read_through=int(row["plan_through"]),
            writes=tuple(PlannedWrite.from_json(write) for write in writes),
            run_id=row["plan_run_id"],
        )
    return SessionMemoryProgress(read_through=int(row["read_through"]), plan=plan)


def _updated(status: Any) -> bool:
    return str(status).rsplit(" ", 1)[-1] != "0"


__all__ = [
    "SESSION_MEMORY_PROGRESS_DDL",
    "SESSION_MEMORY_PROGRESS_SCHEMA_TABLE",
    "SessionMemoryRunStoreMixin",
]
