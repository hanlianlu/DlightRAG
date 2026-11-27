# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Schedules and their Fires in PostgreSQL (ADR 0103).

The database is the queue. A due Schedule is a row whose ``next_fire_at`` has passed, claimed
``FOR UPDATE SKIP LOCKED`` by whichever process looks first; in the same transaction its Fires
are written under a unique ``(schedule_id, slot)`` and ``next_fire_at`` moves on, so a slot
fires at most once in any number of processes. A Fire's Run is linked by id, with no foreign key:
a Run is pruned by its own retention and a Fire keeps what it says. Rows carry ``owner_id``, so
export and erasure find them without a list.

This module owns the DDL, the schema declaration the ``runs`` scope verifies, and the adapter
the Application's Schedule service and look call.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any

from dlightrag.adapters.postgres.core._migrations import (
    ForeignKeyRequirement,
    IndexRequirement,
    TableRequirement,
)
from dlightrag.adapters.postgres.core._operations import ConnectionPool, PostgresOperationRunner
from dlightrag.application.access.principal import UserContext
from dlightrag.application.schedules.cadence import Cadence, CadenceError
from dlightrag.application.schedules.errors import ScheduleLimitError
from dlightrag.application.schedules.model import (
    STREAK_STATES,
    ChangedBy,
    DuePlan,
    Fire,
    FireState,
    Schedule,
    ScheduleMode,
    ScheduleState,
    Target,
    failure_streak_reached,
)
from dlightrag.application.schedules.store import Claim, DueFire
from dlightrag.engine.ai.providers.base import (
    provider_input_tokens,
    provider_output_tokens,
    provider_total_tokens,
    usage_counters,
    usage_to_dict,
)

logger = logging.getLogger(__name__)

_CREATE_SCHEDULES = """
CREATE TABLE IF NOT EXISTS dlightrag_schedules (
    owner_id            TEXT        NOT NULL,
    schedule_id         UUID        NOT NULL,
    name                TEXT        NOT NULL,
    prompt              TEXT        NOT NULL,
    mode                TEXT        NOT NULL,
    workspaces          TEXT[]      NOT NULL,
    target              TEXT        NOT NULL,
    conversation_id     UUID,
    zone                TEXT        NOT NULL,
    cron                TEXT,
    run_at              TIMESTAMPTZ,
    daily_ceiling       BIGINT      NOT NULL,
    goal_max_rounds     BIGINT,
    state               TEXT        NOT NULL,
    pause_reason        TEXT,
    next_fire_at        TIMESTAMPTZ,
    principal_user_id   TEXT        NOT NULL,
    principal_auth_mode TEXT        NOT NULL,
    principal_claims    JSONB       NOT NULL DEFAULT '{}'::jsonb,
    principal_stamped_at TIMESTAMPTZ NOT NULL,
    resumed_at          TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    changed_by          TEXT        NOT NULL DEFAULT 'owner',
    PRIMARY KEY (owner_id, schedule_id),
    CONSTRAINT dlightrag_schedules_mode_check CHECK (mode IN ('research', 'fast')),
    CONSTRAINT dlightrag_schedules_target_check
        CHECK (target IN ('new_conversation', 'same_conversation')),
    CONSTRAINT dlightrag_schedules_state_check
        CHECK (state IN ('active', 'paused', 'completed')),
    CONSTRAINT dlightrag_schedules_cadence_check CHECK ((cron IS NULL) <> (run_at IS NULL)),
    CONSTRAINT dlightrag_schedules_name_check CHECK (char_length(name) BETWEEN 1 AND 120),
    CONSTRAINT dlightrag_schedules_prompt_check CHECK (char_length(prompt) BETWEEN 1 AND 20000),
    CONSTRAINT dlightrag_schedules_workspaces_check
        CHECK (COALESCE(array_length(workspaces, 1), 0) >= 1),
    CONSTRAINT dlightrag_schedules_ceiling_check CHECK (daily_ceiling >= 1),
    CONSTRAINT dlightrag_schedules_goal_check
        CHECK (goal_max_rounds IS NULL OR (goal_max_rounds >= 1 AND mode = 'research')),
    CONSTRAINT dlightrag_schedules_waiting_check
        CHECK ((state <> 'active' OR next_fire_at IS NOT NULL OR run_at IS NOT NULL)
               AND (state <> 'completed' OR next_fire_at IS NULL)),
    CONSTRAINT dlightrag_schedules_pause_check
        CHECK ((state = 'paused') OR pause_reason IS NULL),
    CONSTRAINT dlightrag_schedules_changed_by_check CHECK (changed_by IN ('owner', 'agent'))
)
"""

_CREATE_FIRES = """
CREATE TABLE IF NOT EXISTS dlightrag_schedule_fires (
    owner_id        TEXT        NOT NULL,
    schedule_id     UUID        NOT NULL,
    fire_id         UUID        NOT NULL,
    slot            TIMESTAMPTZ NOT NULL,
    state           TEXT        NOT NULL,
    reason          TEXT,
    run_id          UUID,
    conversation_id UUID,
    missed_count    INTEGER,
    last_slot       TIMESTAMPTZ,
    manual          BOOLEAN     NOT NULL DEFAULT FALSE,
    last_error      TEXT,
    claimed_at      TIMESTAMPTZ NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id, fire_id),
    CONSTRAINT dlightrag_schedule_fires_slot_key UNIQUE (schedule_id, slot),
    FOREIGN KEY (owner_id, schedule_id)
        REFERENCES dlightrag_schedules (owner_id, schedule_id) ON DELETE CASCADE,
    CONSTRAINT dlightrag_schedule_fires_state_check
        CHECK (state IN ('pending', 'started', 'succeeded', 'failed', 'cancelled', 'refused',
                         'skipped_busy', 'skipped_limit', 'missed')),
    CONSTRAINT dlightrag_schedule_fires_run_check
        CHECK (state NOT IN ('started', 'succeeded') OR run_id IS NOT NULL),
    CONSTRAINT dlightrag_schedule_fires_missed_check
        CHECK (missed_count IS NULL OR (state = 'missed' AND missed_count >= 1))
)
"""

# The identity the Agent's Schedules act under: the caller of the owner's latest human turn, one row
# per owner, kept for a day. It is never written into a Run's own input.
_CREATE_OWNER_PRINCIPALS = """
CREATE TABLE IF NOT EXISTS dlightrag_owner_principals (
    owner_id   TEXT        NOT NULL,
    user_id    TEXT        NOT NULL,
    auth_mode  TEXT        NOT NULL,
    claims     JSONB       NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id)
)
"""

_SCHEDULES_ACTIVE_INDEX = IndexRequirement(
    # The look's claim: the active Schedules whose time has come, earliest first.
    "idx_dlightrag_schedules_due",
    "dlightrag_schedules",
    "(next_fire_at) WHERE state = 'active'",
)
_FIRES_RUN_INDEX = IndexRequirement(
    # A settled Run finds its Fire, and the busy check finds a Schedule's latest started one.
    "idx_dlightrag_schedule_fires_run",
    "dlightrag_schedule_fires",
    "(owner_id, run_id) WHERE run_id IS NOT NULL",
)
_FIRES_OPEN_INDEX = IndexRequirement(
    # The look's two sweeps: pending Fires to retry and started ones whose settlement was lost.
    "idx_dlightrag_schedule_fires_open",
    "dlightrag_schedule_fires",
    "(state, updated_at) WHERE state IN ('pending', 'started')",
)
#: Declared here, created and verified with the Run scope's other indexes.
SCHEDULE_INDEXES = (_SCHEDULES_ACTIVE_INDEX, _FIRES_RUN_INDEX, _FIRES_OPEN_INDEX)

SCHEDULES_DDL = (_CREATE_SCHEDULES, _CREATE_FIRES, _CREATE_OWNER_PRINCIPALS)

SCHEDULES_SCHEMA_TABLES = (
    TableRequirement(
        name="dlightrag_schedules",
        columns=(
            "owner_id",
            "schedule_id",
            "name",
            "prompt",
            "mode",
            "workspaces",
            "target",
            "conversation_id",
            "zone",
            "cron",
            "run_at",
            "daily_ceiling",
            "goal_max_rounds",
            "state",
            "pause_reason",
            "next_fire_at",
            "principal_user_id",
            "principal_auth_mode",
            "principal_claims",
            "principal_stamped_at",
            "resumed_at",
            "created_at",
            "updated_at",
            "changed_by",
        ),
        primary_key=("owner_id", "schedule_id"),
        checks=(
            "dlightrag_schedules_mode_check",
            "dlightrag_schedules_target_check",
            "dlightrag_schedules_state_check",
            "dlightrag_schedules_cadence_check",
            "dlightrag_schedules_name_check",
            "dlightrag_schedules_prompt_check",
            "dlightrag_schedules_workspaces_check",
            "dlightrag_schedules_ceiling_check",
            "dlightrag_schedules_goal_check",
            "dlightrag_schedules_waiting_check",
            "dlightrag_schedules_pause_check",
            "dlightrag_schedules_changed_by_check",
        ),
    ),
    TableRequirement(
        name="dlightrag_schedule_fires",
        columns=(
            "owner_id",
            "schedule_id",
            "fire_id",
            "slot",
            "state",
            "reason",
            "run_id",
            "conversation_id",
            "missed_count",
            "last_slot",
            "manual",
            "last_error",
            "claimed_at",
            "created_at",
            "updated_at",
        ),
        primary_key=("owner_id", "fire_id"),
        unique=(("schedule_id", "slot"),),
        foreign_keys=(
            ForeignKeyRequirement(
                columns=("owner_id", "schedule_id"), references="dlightrag_schedules"
            ),
        ),
        checks=(
            "dlightrag_schedule_fires_state_check",
            "dlightrag_schedule_fires_run_check",
            "dlightrag_schedule_fires_missed_check",
        ),
    ),
    TableRequirement(
        name="dlightrag_owner_principals",
        columns=("owner_id", "user_id", "auth_mode", "claims", "updated_at"),
        primary_key=("owner_id",),
    ),
)

#: One owner's active Schedules are counted and added to under this lock, so two requests cannot
#: both take the last place.
_LOCK_OWNER = "SELECT pg_advisory_xact_lock(hashtextextended('dlightrag_schedules:' || $1, 0))"

_COUNT_ACTIVE = """
SELECT count(*) FROM dlightrag_schedules WHERE owner_id = $1 AND state = 'active'
"""

_INSERT_SCHEDULE = """
INSERT INTO dlightrag_schedules (
    owner_id, schedule_id, name, prompt, mode, workspaces, target, conversation_id,
    zone, cron, run_at, daily_ceiling, goal_max_rounds, state, pause_reason, next_fire_at,
    principal_user_id, principal_auth_mode, principal_claims, principal_stamped_at,
    created_at, updated_at, changed_by
) VALUES (
    $1, $2::uuid, $3, $4, $5, $6, $7, $8::uuid,
    $9, $10, $11, $12, $13, $14, $15, $16,
    $17, $18, $19::jsonb, $20, $21, $21, $22
)
RETURNING *
"""

_SELECT_ONE = "SELECT * FROM dlightrag_schedules WHERE owner_id = $1 AND schedule_id = $2::uuid"

#: Keeping an identity also drops every other owner's record older than a day, which nothing reads
#: any more (``_SELECT_PRINCIPAL``): the table holds the day's latest turns and no more.
_UPSERT_PRINCIPAL = """
WITH expired AS (
    DELETE FROM dlightrag_owner_principals
    WHERE owner_id <> $1 AND updated_at <= NOW() - INTERVAL '1 day'
)
INSERT INTO dlightrag_owner_principals (owner_id, user_id, auth_mode, claims)
VALUES ($1, $2, $3, $4::jsonb)
ON CONFLICT (owner_id) DO UPDATE
SET user_id = EXCLUDED.user_id, auth_mode = EXCLUDED.auth_mode, claims = EXCLUDED.claims,
    updated_at = NOW()
"""

#: A record older than a day is not the identity of a turn someone just sent, so the Agent acts
#: under none: a person's latest turn is what the Agent's Schedules are made on behalf of.
_SELECT_PRINCIPAL = """
SELECT user_id, auth_mode, claims FROM dlightrag_owner_principals
WHERE owner_id = $1 AND updated_at > NOW() - INTERVAL '1 day'
"""

_LIST = """
SELECT * FROM dlightrag_schedules WHERE owner_id = $1 ORDER BY created_at, schedule_id
"""

_EDIT = """
UPDATE dlightrag_schedules
SET name = $3, prompt = $4, mode = $5, workspaces = $6, daily_ceiling = $7,
    principal_user_id = $8, principal_auth_mode = $9, principal_claims = $10::jsonb,
    zone = $11, cron = $12, run_at = $13, goal_max_rounds = $14,
    next_fire_at = CASE WHEN $15 THEN $16 ELSE next_fire_at END,
    updated_at = $17, principal_stamped_at = $18, changed_by = $19
WHERE owner_id = $1 AND schedule_id = $2::uuid AND state <> 'completed'
RETURNING *
"""

_PAUSE = """
UPDATE dlightrag_schedules
SET state = 'paused', pause_reason = $3, updated_at = NOW()
WHERE owner_id = $1 AND schedule_id = $2::uuid AND state = 'active'
RETURNING *
"""

_RESUME = """
UPDATE dlightrag_schedules
SET state = 'active', pause_reason = NULL,
    principal_user_id = $3, principal_auth_mode = $4, principal_claims = $5::jsonb,
    principal_stamped_at = NOW(), resumed_at = NOW(), updated_at = NOW(), changed_by = $6
WHERE owner_id = $1 AND schedule_id = $2::uuid AND state = 'paused'
RETURNING *
"""

_DELETE = """
DELETE FROM dlightrag_schedules WHERE owner_id = $1 AND schedule_id = $2::uuid
RETURNING schedule_id
"""

#: The one clock the Schedules read: the database's, as it is when asked rather than when the
#: transaction began, so every process that looks agrees on what is due.
_NOW = "SELECT clock_timestamp()"

_CLAIM_DUE = """
SELECT * FROM dlightrag_schedules
WHERE state = 'active' AND next_fire_at <= $1
ORDER BY next_fire_at
LIMIT $2
FOR UPDATE SKIP LOCKED
"""

_INSERT_FIRE = """
INSERT INTO dlightrag_schedule_fires (
    owner_id, schedule_id, fire_id, slot, state, reason, missed_count, last_slot,
    claimed_at, created_at, updated_at, manual
) VALUES ($1, $2::uuid, $3::uuid, $4, $5, $6, $7, $8, $9, $9, $9, $10)
ON CONFLICT (schedule_id, slot) DO NOTHING
RETURNING *
"""

_ADVANCE = """
UPDATE dlightrag_schedules
SET next_fire_at = $3::timestamptz, updated_at = $4
WHERE owner_id = $1 AND schedule_id = $2::uuid
RETURNING *
"""

#: The newest ``$3`` Fires stay. Older ones are the history this Schedule no longer keeps.
_TRIM_FIRES = """
DELETE FROM dlightrag_schedule_fires
WHERE schedule_id = $1::uuid
  AND fire_id NOT IN (
      SELECT fire_id FROM dlightrag_schedule_fires
      WHERE schedule_id = $1::uuid
      ORDER BY slot DESC
      LIMIT $2
  )
"""

_CLAIM_PENDING = """
UPDATE dlightrag_schedule_fires
SET claimed_at = $1
WHERE fire_id IN (
    SELECT fire_id FROM dlightrag_schedule_fires
    WHERE state = 'pending' AND claimed_at < $2
    ORDER BY slot
    LIMIT $3
    FOR UPDATE SKIP LOCKED
)
RETURNING *
"""

_SCHEDULES_OF_FIRES = """
SELECT * FROM dlightrag_schedules
WHERE (owner_id, schedule_id) IN (
    SELECT owner_id, schedule_id FROM unnest($1::text[], $2::uuid[]) AS fire(owner_id, schedule_id)
)
"""

_FINISH_FIRE = """
UPDATE dlightrag_schedule_fires
SET state = $3, reason = $4, run_id = COALESCE($5::uuid, run_id),
    conversation_id = COALESCE($6::uuid, conversation_id), updated_at = NOW()
WHERE owner_id = $1 AND fire_id = $2::uuid AND state = ANY($7::text[])
RETURNING *
"""

_KEEP_CONVERSATION = """
UPDATE dlightrag_schedules
SET conversation_id = $3::uuid
WHERE owner_id = $1 AND schedule_id = $2::uuid
  AND target = 'same_conversation' AND conversation_id IS NULL
"""

#: A Fire's end locks its Schedule first and the Fire second, the order a delete takes them in
#: (the cascade), so the two cannot wait for each other.
_FIRE_SCHEDULE = """
SELECT schedule_id FROM dlightrag_schedule_fires WHERE owner_id = $1 AND fire_id = $2::uuid
"""

_LOCK_SCHEDULE = """
SELECT state, resumed_at, run_at, next_fire_at FROM dlightrag_schedules
WHERE owner_id = $1 AND schedule_id = $2::uuid
FOR UPDATE
"""

_COMPLETE = """
UPDATE dlightrag_schedules SET state = 'completed', updated_at = NOW()
WHERE owner_id = $1 AND schedule_id = $2::uuid AND state = 'active'
"""

#: The Fires that count toward a streak, newest first, since the Schedule last resumed. A Fire
#: missed after an error that stopped it counts as a failure; a missed gap or a skip does not.
_RECENT_OUTCOMES = """
SELECT CASE WHEN state = 'missed' THEN 'failed' ELSE state END AS state
FROM dlightrag_schedule_fires
WHERE owner_id = $1 AND schedule_id = $2::uuid
  AND created_at >= COALESCE($3::timestamptz, '-infinity'::timestamptz)
  AND (state = ANY($4::text[]) OR (state = 'missed' AND last_error IS NOT NULL))
ORDER BY slot DESC
LIMIT $5
"""

_RECORD_ERROR = """
UPDATE dlightrag_schedule_fires SET last_error = $3
WHERE owner_id = $1 AND fire_id = $2::uuid AND state = 'pending'
"""

_TOUCH_FIRE = """
UPDATE dlightrag_schedule_fires SET updated_at = NOW()
WHERE owner_id = $1 AND fire_id = $2::uuid AND state = 'started'
"""

_PENDING_BEFORE = """
SELECT * FROM dlightrag_schedule_fires
WHERE owner_id = $1 AND schedule_id = $2::uuid AND state = 'pending'
  AND slot < $3 AND fire_id <> $4::uuid
ORDER BY slot
LIMIT 1
"""

_OPEN_FIRE = """
SELECT 1 FROM dlightrag_schedule_fires
WHERE owner_id = $1 AND schedule_id = $2::uuid AND state IN ('pending', 'started')
LIMIT 1
"""

_FIRE_FOR_RUN = """
SELECT * FROM dlightrag_schedule_fires WHERE owner_id = $1 AND run_id = $2::uuid
"""

_PREVIOUS_FIRE = """
SELECT * FROM dlightrag_schedule_fires
WHERE owner_id = $1 AND schedule_id = $2::uuid AND run_id IS NOT NULL AND slot < $3
ORDER BY slot DESC
LIMIT 1
"""

_UNSETTLED_FIRES = """
SELECT * FROM dlightrag_schedule_fires
WHERE state = 'started' AND updated_at < $1
ORDER BY updated_at
LIMIT $2
"""

_LIST_FIRES = """
SELECT fire.* FROM dlightrag_schedule_fires AS fire
WHERE fire.owner_id = $1 AND fire.schedule_id = $2::uuid
ORDER BY fire.slot DESC
LIMIT $3
"""

#: The Runs a Schedule's Fires spent tokens in since a moment: each Fire's Run, and the goal rounds
#: the Session went on to run after it. A round counts once, whichever Fire's Session it is in.
_SPENT_RUNS = """
WITH fired AS (
    SELECT fire.run_id, run.created_at, routing.agent_session_id
    FROM dlightrag_schedule_fires AS fire
    JOIN dlightrag_runs AS run
      ON run.owner_id = fire.owner_id AND run.run_id = fire.run_id
    LEFT JOIN dlightrag_answer_run_routing AS routing
      ON routing.owner_id = run.owner_id AND routing.run_id = run.run_id
    WHERE fire.owner_id = $1 AND fire.schedule_id = $2::uuid AND fire.run_id IS NOT NULL
)
SELECT run.run_id, run.result_json -> 'usage' AS usage
FROM dlightrag_runs AS run
WHERE run.owner_id = $1 AND run.created_at >= $3
  AND (
      run.run_id IN (SELECT run_id FROM fired)
      OR (
          COALESCE(run.accepted_input_json, run.prepared_input_json) ->> 'continuation_kind'
              = 'goal_round'
          AND EXISTS (
              SELECT 1
              FROM dlightrag_answer_run_routing AS routing
              JOIN fired
                ON fired.agent_session_id = routing.agent_session_id
              WHERE routing.owner_id = run.owner_id AND routing.run_id = run.run_id
                AND run.created_at > fired.created_at
          )
      )
  )
"""

#: What a Run that left no usage record of its own (one that failed or was cancelled) spent, read
#: from the model calls it did make: every assistant message of the Run and of its Children states
#: the usage of its call.
_RUN_CALL_USAGE = """
SELECT run_id, payload_json -> 'usage' AS usage
FROM dlightrag_agent_session_entries
WHERE owner_id = $1 AND run_id = ANY($2::uuid[]) AND entry_type = 'assistant_message'
  AND payload_json ? 'usage'
"""


def _usage_tokens(usage: Any) -> int | None:
    """The tokens one usage record states, in whatever dialect its provider speaks; None if none.

    A Run's record holds the counters of its agent, its Children and the two added up; the sum is
    the Inspector's total. A provider that states no total (Anthropic's native counters) is read
    as its prompt tokens, cache reads and writes included, plus its output tokens.
    """
    if isinstance(usage, str):
        usage = json.loads(usage)
    if not isinstance(usage, Mapping):
        return None
    inclusive = usage.get("inclusive_usage_details")
    counters = usage_to_dict(inclusive) if isinstance(inclusive, Mapping) else usage_counters(usage)
    if not counters:
        return None
    total = provider_total_tokens(counters)
    if total is None:
        total = (provider_input_tokens(counters) or 0) + (provider_output_tokens(counters) or 0)
    return total


def _principal_from_row(row: Any, claims_column: str, prefix: str = "") -> UserContext:
    claims = row[claims_column]
    return UserContext(
        user_id=str(row[f"{prefix}user_id"]),
        auth_mode=str(row[f"{prefix}auth_mode"]),
        claims=json.loads(claims) if isinstance(claims, str) else dict(claims),
    )


def _schedule_from_row(row: Any) -> Schedule:
    return Schedule(
        schedule_id=str(row["schedule_id"]),
        owner_id=str(row["owner_id"]),
        name=str(row["name"]),
        prompt=str(row["prompt"]),
        mode=_mode(str(row["mode"])),
        workspaces=tuple(str(item) for item in row["workspaces"]),
        target=Target(str(row["target"])),
        conversation_id=None if row["conversation_id"] is None else str(row["conversation_id"]),
        cadence=Cadence.load(zone_name=str(row["zone"]), cron=row["cron"], at=row["run_at"]),
        max_tokens_per_day=int(row["daily_ceiling"]),
        goal_max_rounds=None if row["goal_max_rounds"] is None else int(row["goal_max_rounds"]),
        state=ScheduleState(str(row["state"])),
        pause_reason=None if row["pause_reason"] is None else str(row["pause_reason"]),
        next_fire_at=row["next_fire_at"],
        principal=_principal_from_row(row, "principal_claims", "principal_"),
        principal_stamped_at=row["principal_stamped_at"],
        resumed_at=row["resumed_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        changed_by="agent" if row["changed_by"] == "agent" else "owner",
    )


def _mode(value: str) -> ScheduleMode:
    return "fast" if value == "fast" else "research"


def _fire_from_row(row: Any) -> Fire:
    return Fire(
        fire_id=str(row["fire_id"]),
        schedule_id=str(row["schedule_id"]),
        owner_id=str(row["owner_id"]),
        slot=row["slot"],
        state=FireState(str(row["state"])),
        reason=None if row["reason"] is None else str(row["reason"]),
        run_id=None if row["run_id"] is None else str(row["run_id"]),
        conversation_id=None if row["conversation_id"] is None else str(row["conversation_id"]),
        missed_count=None if row["missed_count"] is None else int(row["missed_count"]),
        last_slot=row["last_slot"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        manual=bool(row["manual"]),
        last_error=None if row["last_error"] is None else str(row["last_error"]),
    )


def _claims_json(principal: UserContext) -> str:
    return json.dumps(principal.claims, default=str, sort_keys=True)


class PGScheduleStore(PostgresOperationRunner):
    """The Schedule service's and the look's durable storage."""

    def __init__(self, *, pool: ConnectionPool | None = None) -> None:
        super().__init__(pool=pool)

    async def now(self) -> datetime:
        async def operation(conn: Any) -> datetime:
            return await conn.fetchval(_NOW)

        return await self._run(operation)

    async def create(self, schedule: Schedule, *, max_active: int) -> Schedule:
        async def operation(conn: Any) -> Schedule:
            async with conn.transaction():
                await conn.execute(_LOCK_OWNER, schedule.owner_id)
                if int(await conn.fetchval(_COUNT_ACTIVE, schedule.owner_id)) >= max_active:
                    raise ScheduleLimitError(max_active)
                row = await conn.fetchrow(
                    _INSERT_SCHEDULE,
                    schedule.owner_id,
                    schedule.schedule_id,
                    schedule.name,
                    schedule.prompt,
                    schedule.mode,
                    list(schedule.workspaces),
                    str(schedule.target),
                    schedule.conversation_id,
                    schedule.cadence.zone_name,
                    schedule.cadence.cron,
                    schedule.cadence.at,
                    schedule.max_tokens_per_day,
                    schedule.goal_max_rounds,
                    str(schedule.state),
                    schedule.pause_reason,
                    schedule.next_fire_at,
                    schedule.principal.user_id,
                    schedule.principal.auth_mode,
                    _claims_json(schedule.principal),
                    schedule.principal_stamped_at,
                    schedule.created_at,
                    schedule.changed_by,
                )
                return _schedule_from_row(row)

        return await self._run_once(operation)

    async def remember_principal(self, owner_id: str, principal: UserContext) -> None:
        async def operation(conn: Any) -> None:
            await conn.execute(
                _UPSERT_PRINCIPAL,
                owner_id,
                principal.user_id,
                principal.auth_mode,
                _claims_json(principal),
            )

        await self._run(operation)

    async def principal_of(self, owner_id: str) -> UserContext | None:
        async def operation(conn: Any) -> UserContext | None:
            row = await conn.fetchrow(_SELECT_PRINCIPAL, owner_id)
            return None if row is None else _principal_from_row(row, "claims")

        return await self._run(operation)

    async def get(self, owner_id: str, schedule_id: str) -> Schedule | None:
        async def operation(conn: Any) -> Schedule | None:
            row = await conn.fetchrow(_SELECT_ONE, owner_id, schedule_id)
            return None if row is None else _schedule_from_row(row)

        return await self._run(operation)

    async def list(self, owner_id: str) -> tuple[Schedule, ...]:
        async def operation(conn: Any) -> tuple[Schedule, ...]:
            return tuple(_schedule_from_row(row) for row in await conn.fetch(_LIST, owner_id))

        return await self._run(operation)

    async def edit(self, schedule: Schedule, *, reschedule: bool) -> Schedule | None:
        async def operation(conn: Any) -> Schedule | None:
            row = await conn.fetchrow(
                _EDIT,
                schedule.owner_id,
                schedule.schedule_id,
                schedule.name,
                schedule.prompt,
                schedule.mode,
                list(schedule.workspaces),
                schedule.max_tokens_per_day,
                schedule.principal.user_id,
                schedule.principal.auth_mode,
                _claims_json(schedule.principal),
                schedule.cadence.zone_name,
                schedule.cadence.cron,
                schedule.cadence.at,
                schedule.goal_max_rounds,
                reschedule,
                schedule.next_fire_at,
                schedule.updated_at,
                schedule.principal_stamped_at,
                schedule.changed_by,
            )
            return None if row is None else _schedule_from_row(row)

        return await self._run_once(operation)

    async def pause(
        self, owner_id: str, schedule_id: str, *, reason: str | None
    ) -> Schedule | None:
        async def operation(conn: Any) -> Schedule | None:
            row = await conn.fetchrow(_PAUSE, owner_id, schedule_id, reason)
            return None if row is None else _schedule_from_row(row)

        return await self._run_once(operation)

    async def resume(
        self,
        owner_id: str,
        schedule_id: str,
        *,
        principal: UserContext,
        max_active: int,
        changed_by: ChangedBy = "owner",
    ) -> Schedule | None:
        async def operation(conn: Any) -> Schedule | None:
            async with conn.transaction():
                await conn.execute(_LOCK_OWNER, owner_id)
                if int(await conn.fetchval(_COUNT_ACTIVE, owner_id)) >= max_active:
                    raise ScheduleLimitError(max_active)
                row = await conn.fetchrow(
                    _RESUME,
                    owner_id,
                    schedule_id,
                    principal.user_id,
                    principal.auth_mode,
                    _claims_json(principal),
                    changed_by,
                )
                return None if row is None else _schedule_from_row(row)

        return await self._run_once(operation)

    async def delete(self, owner_id: str, schedule_id: str) -> bool:
        async def operation(conn: Any) -> bool:
            return await conn.fetchrow(_DELETE, owner_id, schedule_id) is not None

        return await self._run_once(operation)

    async def claim_due(
        self,
        now: datetime,
        *,
        limit: int,
        fires_kept: int,
        plan: Callable[[Schedule], DuePlan],
    ) -> Claim:
        async def operation(conn: Any) -> Claim:
            due: list[DueFire] = []
            async with conn.transaction():
                rows = await conn.fetch(_CLAIM_DUE, now, limit)
                for row in rows:
                    try:
                        schedule = _schedule_from_row(row)
                    except ValueError as error:
                        # A row that no longer loads must not hold every other Schedule.
                        logger.warning("Schedule %s cannot be read", row["schedule_id"])
                        await conn.execute(
                            _PAUSE,
                            str(row["owner_id"]),
                            str(row["schedule_id"]),
                            f"it can no longer be read: {error}",
                        )
                        continue
                    try:
                        decision = plan(schedule)
                    except CadenceError as error:
                        # A cadence that can no longer be read must not hold every other Schedule.
                        logger.warning("Schedule %s cannot be planned", schedule.schedule_id)
                        await conn.execute(
                            _PAUSE,
                            schedule.owner_id,
                            schedule.schedule_id,
                            f"its cadence can no longer be evaluated: {error}",
                        )
                        continue
                    pending: list[Fire] = []
                    for draft in decision.fires:
                        fire_row = await conn.fetchrow(
                            _INSERT_FIRE,
                            schedule.owner_id,
                            schedule.schedule_id,
                            str(uuid.uuid4()),
                            draft.slot,
                            str(draft.state),
                            draft.reason,
                            draft.missed_count,
                            draft.last_slot,
                            now,
                            False,
                        )
                        if fire_row is not None and draft.state is FireState.PENDING:
                            pending.append(_fire_from_row(fire_row))
                    advanced = await conn.fetchrow(
                        _ADVANCE,
                        schedule.owner_id,
                        schedule.schedule_id,
                        decision.next_fire_at,
                        now,
                    )
                    await conn.execute(_TRIM_FIRES, schedule.schedule_id, fires_kept)
                    if decision.next_fire_at is None and not pending:
                        # A single time that started nothing waits for its owner to name another.
                        await conn.execute(
                            _PAUSE,
                            schedule.owner_id,
                            schedule.schedule_id,
                            decision.fires[-1].reason or "its time passed without a start",
                        )
                    moved = _schedule_from_row(advanced)
                    due.extend(DueFire(moved, fire) for fire in pending)
            return Claim(len(rows), tuple(due))

        return await self._run_once(operation)

    async def claim_pending(
        self, now: datetime, *, lease: timedelta, limit: int
    ) -> tuple[DueFire, ...]:
        async def operation(conn: Any) -> tuple[DueFire, ...]:
            async with conn.transaction():
                fires = [
                    _fire_from_row(row)
                    for row in await conn.fetch(_CLAIM_PENDING, now, now - lease, limit)
                ]
                if not fires:
                    return ()
                schedules: dict[tuple[str, str], Schedule] = {}
                for row in await conn.fetch(
                    _SCHEDULES_OF_FIRES,
                    [fire.owner_id for fire in fires],
                    [fire.schedule_id for fire in fires],
                ):
                    try:
                        schedule = _schedule_from_row(row)
                    except ValueError:
                        logger.warning("Schedule %s cannot be read", row["schedule_id"])
                        continue
                    schedules[(schedule.owner_id, schedule.schedule_id)] = schedule
            return tuple(
                DueFire(schedules[(fire.owner_id, fire.schedule_id)], fire)
                for fire in fires
                if (fire.owner_id, fire.schedule_id) in schedules
            )

        return await self._run_once(operation)

    async def insert_manual_fire(
        self, schedule: Schedule, slot: datetime, *, fires_kept: int
    ) -> DueFire | None:
        async def operation(conn: Any) -> DueFire | None:
            async with conn.transaction():
                row = await conn.fetchrow(
                    _INSERT_FIRE,
                    schedule.owner_id,
                    schedule.schedule_id,
                    str(uuid.uuid4()),
                    slot,
                    str(FireState.PENDING),
                    None,
                    None,
                    None,
                    slot,
                    True,
                )
                if row is None:
                    return None
                await conn.execute(_TRIM_FIRES, schedule.schedule_id, fires_kept)
                return DueFire(schedule, _fire_from_row(row))

        return await self._run_once(operation)

    async def has_open_fire(self, owner_id: str, schedule_id: str) -> bool:
        async def operation(conn: Any) -> bool:
            return await conn.fetchrow(_OPEN_FIRE, owner_id, schedule_id) is not None

        return await self._run(operation)

    async def earlier_pending_fire(
        self, owner_id: str, schedule_id: str, before: datetime, *, exclude: str
    ) -> Fire | None:
        async def operation(conn: Any) -> Fire | None:
            row = await conn.fetchrow(_PENDING_BEFORE, owner_id, schedule_id, before, exclude)
            return None if row is None else _fire_from_row(row)

        return await self._run(operation)

    async def record_error(self, owner_id: str, fire_id: str, error: str) -> None:
        async def operation(conn: Any) -> None:
            await conn.execute(_RECORD_ERROR, owner_id, fire_id, error)

        await self._run_once(operation)

    async def touch_fire(self, owner_id: str, fire_id: str) -> None:
        async def operation(conn: Any) -> None:
            await conn.execute(_TOUCH_FIRE, owner_id, fire_id)

        await self._run_once(operation)

    async def finish_fire(
        self,
        owner_id: str,
        fire_id: str,
        *,
        expected: Sequence[FireState],
        state: FireState,
        reason: str | None = None,
        run_id: str | None = None,
        conversation_id: str | None = None,
        pause_reason: str | None = None,
        failure_streak: int,
    ) -> Fire | None:
        async def operation(conn: Any) -> Fire | None:
            async with conn.transaction():
                head = await conn.fetchrow(_FIRE_SCHEDULE, owner_id, fire_id)
                if head is None:
                    return None
                locked = await conn.fetchrow(_LOCK_SCHEDULE, owner_id, str(head["schedule_id"]))
                if locked is None:
                    return None
                row = await conn.fetchrow(
                    _FINISH_FIRE,
                    owner_id,
                    fire_id,
                    str(state),
                    reason,
                    run_id,
                    conversation_id,
                    [str(item) for item in expected],
                )
                if row is None:
                    return None
                fire = _fire_from_row(row)
                if conversation_id is not None:
                    await conn.execute(
                        _KEEP_CONVERSATION, owner_id, fire.schedule_id, conversation_id
                    )
                active = str(locked["state"]) == ScheduleState.ACTIVE
                if (
                    active
                    and locked["run_at"] is not None
                    and locked["next_fire_at"] is None
                    and not fire.manual
                    and FireState.PENDING in expected
                ):
                    # A single time's own Fire was decided: it is done only if a Run started, and
                    # otherwise waits, paused, for its owner to name another time.
                    if state is FireState.STARTED:
                        await conn.execute(_COMPLETE, owner_id, fire.schedule_id)
                    else:
                        await conn.execute(
                            _PAUSE, owner_id, fire.schedule_id, fire.reason or str(state)
                        )
                    active = False
                if active and _ended_badly(fire):
                    await self._pause_for_failure(
                        conn,
                        fire,
                        locked["resumed_at"],
                        pause_reason=pause_reason,
                        failure_streak=failure_streak,
                    )
                return fire

        return await self._run_once(operation)

    async def _pause_for_failure(
        self,
        conn: Any,
        fire: Fire,
        resumed_at: datetime | None,
        *,
        pause_reason: str | None,
        failure_streak: int,
    ) -> None:
        """Pause the Fire's active Schedule when this bad ending calls for it. Its row is locked."""
        reason = pause_reason
        if reason is None:
            outcomes = [
                FireState(str(row["state"]))
                for row in await conn.fetch(
                    _RECENT_OUTCOMES,
                    fire.owner_id,
                    fire.schedule_id,
                    resumed_at,
                    [str(item) for item in STREAK_STATES],
                    failure_streak,
                )
            ]
            if not failure_streak_reached(outcomes, failure_streak):
                return
            reason = f"the last {failure_streak} fires failed or were refused; the latest: " + (
                fire.reason or str(fire.state)
            )
        await conn.execute(_PAUSE, fire.owner_id, fire.schedule_id, reason)

    async def fire_for_run(self, owner_id: str, run_id: str) -> Fire | None:
        async def operation(conn: Any) -> Fire | None:
            row = await conn.fetchrow(_FIRE_FOR_RUN, owner_id, run_id)
            return None if row is None else _fire_from_row(row)

        return await self._run(operation)

    async def previous_fire(self, owner_id: str, schedule_id: str, before: datetime) -> Fire | None:
        async def operation(conn: Any) -> Fire | None:
            row = await conn.fetchrow(_PREVIOUS_FIRE, owner_id, schedule_id, before)
            return None if row is None else _fire_from_row(row)

        return await self._run(operation)

    async def unsettled_fires(self, *, older_than: datetime, limit: int) -> tuple[Fire, ...]:
        async def operation(conn: Any) -> tuple[Fire, ...]:
            rows = await conn.fetch(_UNSETTLED_FIRES, older_than, limit)
            return tuple(_fire_from_row(row) for row in rows)

        return await self._run(operation)

    async def list_fires(
        self, owner_id: str, schedule_id: str, *, limit: int
    ) -> tuple[Fire, ...] | None:
        async def operation(conn: Any) -> tuple[Fire, ...] | None:
            if await conn.fetchrow(_SELECT_ONE, owner_id, schedule_id) is None:
                return None
            rows = await conn.fetch(_LIST_FIRES, owner_id, schedule_id, limit)
            return tuple(_fire_from_row(row) for row in rows)

        return await self._run(operation)

    async def spent_tokens(self, owner_id: str, schedule_id: str, *, since: datetime) -> int:
        async def operation(conn: Any) -> int:
            total = 0
            unstated: list[Any] = []
            for row in await conn.fetch(_SPENT_RUNS, owner_id, schedule_id, since):
                tokens = _usage_tokens(row["usage"])
                if tokens is None:
                    unstated.append(row["run_id"])
                else:
                    total += tokens
            if unstated:
                for row in await conn.fetch(_RUN_CALL_USAGE, owner_id, unstated):
                    total += _usage_tokens(row["usage"]) or 0
            return total

        return await self._run(operation)


def _ended_badly(fire: Fire) -> bool:
    """Whether this ending counts against the Schedule: a failure, a refusal, or an error that ran out the time."""
    return fire.state in (FireState.FAILED, FireState.REFUSED) or (
        fire.state is FireState.MISSED and fire.last_error is not None
    )


__all__ = [
    "SCHEDULES_DDL",
    "SCHEDULES_SCHEMA_TABLES",
    "SCHEDULE_INDEXES",
    "PGScheduleStore",
]
