# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Child Session, control, and guidance persistence mixed into PGRunStore."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import asyncpg

from dlightrag.application.answer_runs import ChildRosterPageRequest, ChildRosterRowPage
from dlightrag.engine.agent.session.ids import OperationId
from dlightrag.engine.runtime.cancellation import cancellation_notify_key
from dlightrag.engine.runtime.policy import RUN_LEASE_SECONDS
from dlightrag.engine.runtime.records import parse_run_id

logger = logging.getLogger(__name__)

_RUN_ACTIVITY_CHANNEL = "dlightrag_run_activity"
_MAX_PENDING_CHILD_CONTROLS = 100
_MAX_PENDING_CHILD_GUIDANCE = 8
_GUIDANCE_HINT_POLL_SECONDS = 1.0

_HOLD_RUN_LEASE = """
SELECT 1
FROM dlightrag_runs
WHERE owner_id = $1 AND run_id = $2
  AND lease_owner = $3 AND fencing_epoch = $4
  AND status = 'running' AND lease_expires_at > NOW()
FOR UPDATE
"""

_UPSERT_CHILD_SESSION = """
INSERT INTO dlightrag_answer_child_sessions (
    owner_id, run_id, child_session_id, parent_session_id, parent_call_id,
    parent_intent_id, status, objective, context_mode, model_role, tools_json,
    depth, context_snapshot_json, plan_json, budget_json, host_state_json
)
VALUES (
    $1, $2, $3, $4, $5, $6, 'running', $7, $8, $9, $10,
    $11, $12::jsonb, $13::jsonb, $14::jsonb, $15::jsonb
)
ON CONFLICT (owner_id, run_id, child_session_id) DO UPDATE
SET parent_intent_id = COALESCE(
        dlightrag_answer_child_sessions.parent_intent_id,
        EXCLUDED.parent_intent_id
    ),
    objective = COALESCE(dlightrag_answer_child_sessions.objective, EXCLUDED.objective),
    context_mode = COALESCE(
        dlightrag_answer_child_sessions.context_mode, EXCLUDED.context_mode
    ),
    model_role = COALESCE(dlightrag_answer_child_sessions.model_role, EXCLUDED.model_role),
    tools_json = COALESCE(dlightrag_answer_child_sessions.tools_json, EXCLUDED.tools_json),
    depth = CASE
        WHEN dlightrag_answer_child_sessions.context_snapshot_json = '{}'::jsonb
             AND EXCLUDED.context_snapshot_json <> '{}'::jsonb THEN EXCLUDED.depth
        ELSE dlightrag_answer_child_sessions.depth
    END,
    context_snapshot_json = CASE
        WHEN dlightrag_answer_child_sessions.context_snapshot_json = '{}'::jsonb
             AND EXCLUDED.context_snapshot_json <> '{}'::jsonb
            THEN EXCLUDED.context_snapshot_json
        ELSE dlightrag_answer_child_sessions.context_snapshot_json
    END,
    plan_json = COALESCE(dlightrag_answer_child_sessions.plan_json, EXCLUDED.plan_json),
    budget_json = COALESCE(dlightrag_answer_child_sessions.budget_json, EXCLUDED.budget_json),
    host_state_json = (
        dlightrag_answer_child_sessions.host_state_json || EXCLUDED.host_state_json
    ),
    updated_at = NOW()
"""

_CLAIM_CHILD_SESSION = """
UPDATE dlightrag_answer_child_sessions
SET lease_owner = $4,
    lease_expires_at = NOW() + ($5 * INTERVAL '1 second'),
    fencing_epoch = fencing_epoch + 1,
    updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2 AND child_session_id = $3
  AND status = 'running'
  AND (lease_expires_at IS NULL OR lease_expires_at < NOW() OR lease_owner = $4)
RETURNING fencing_epoch
"""

_REQUEST_CHILD_CANCELLATION = """
UPDATE dlightrag_answer_child_sessions
SET cancel_requested_at = COALESCE(cancel_requested_at, NOW()),
    updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2 AND child_session_id = $3
  AND status = 'running'
RETURNING 1
"""

_LOCK_CHILD_SESSION = """
SELECT child.*, operation.operation_sequence, operation.operation_id,
       operation.idempotency_key AS operation_key,
       operation.content AS operation_content,
       operation.origin AS operation_origin,
       operation.status AS operation_status,
       operation.cancellation_origin
FROM dlightrag_answer_child_sessions AS child
LEFT JOIN LATERAL (
    SELECT *
    FROM dlightrag_answer_child_operations AS operation
    WHERE operation.owner_id = child.owner_id
      AND operation.run_id = child.run_id
      AND operation.child_session_id = child.child_session_id
    ORDER BY operation.operation_sequence DESC
    LIMIT 1
) AS operation ON TRUE
WHERE child.owner_id = $1 AND child.run_id = $2 AND child.child_session_id = $3
FOR UPDATE OF child
"""

_SELECT_CHILD_OPERATION_BY_KEY = """
SELECT operation_sequence, operation_id, idempotency_key, request_fingerprint,
       content, origin, status, cancellation_origin, summary, usage_json,
       outcome_json, created_at, updated_at
FROM dlightrag_answer_child_operations
WHERE owner_id = $1 AND run_id = $2 AND child_session_id = $3
  AND idempotency_key = $4
"""

_INSERT_CHILD_OPERATION = """
INSERT INTO dlightrag_answer_child_operations (
    owner_id, run_id, child_session_id, operation_sequence, operation_id,
    idempotency_key, request_fingerprint, content, origin, status
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 'running')
ON CONFLICT (owner_id, run_id, child_session_id, idempotency_key) DO NOTHING
"""

_ACTIVATE_CHILD_OPERATION = """
UPDATE dlightrag_answer_child_sessions
SET status = 'running', cancel_requested_at = NULL, summary = NULL,
    lease_owner = NULL, lease_expires_at = NULL, updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2 AND child_session_id = $3
"""

_CANCEL_CHILD_OPERATION = """
UPDATE dlightrag_answer_child_operations
SET cancellation_origin = COALESCE(cancellation_origin, $5), updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2 AND child_session_id = $3
  AND operation_id = $4 AND status = 'running'
"""

_RETIRE_CHILD_GUIDANCE = """
UPDATE dlightrag_answer_child_guidance
SET status = 'cancelled', updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2 AND child_session_id = $3
  AND status = 'pending'
"""

_RELEASE_CHILD_SESSION_LEASES = """
UPDATE dlightrag_answer_child_sessions
SET lease_owner = NULL,
    lease_expires_at = NULL,
    updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2 AND lease_owner = $3
  AND status = 'running'
"""

_RENEW_CHILD_SESSION_LEASE = """
UPDATE dlightrag_answer_child_sessions
SET lease_expires_at = NOW() + ($6 * INTERVAL '1 second'),
    updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2 AND child_session_id = $3
  AND lease_owner = $4 AND fencing_epoch = $5
  AND status = 'running' AND lease_expires_at > NOW()
RETURNING 1
"""

_SELECT_CHILD_SESSION = """
SELECT child.child_session_id, child.parent_session_id, child.parent_call_id,
       child.status, child.cancel_requested_at, child.summary, child.parent_intent_id,
       child.objective, child.context_mode, child.model_role, child.tools_json, child.usage_json,
       child.depth, child.context_snapshot_json, child.plan_json, child.budget_json,
       child.host_state_json, child.lease_owner, child.lease_expires_at,
       child.fencing_epoch, child.created_at, child.updated_at,
       operation.operation_sequence, operation.operation_id,
       operation.idempotency_key AS operation_key,
       operation.content AS operation_content,
       operation.origin AS operation_origin,
       operation.status AS operation_status,
       operation.cancellation_origin,
       (SELECT jsonb_agg(history.usage_json ORDER BY history.operation_sequence)
        FROM dlightrag_answer_child_operations AS history
        WHERE history.owner_id = child.owner_id AND history.run_id = child.run_id
          AND history.child_session_id = child.child_session_id
          AND history.usage_json IS NOT NULL) AS operation_usage_jsons
FROM dlightrag_answer_child_sessions AS child
LEFT JOIN LATERAL (
    SELECT * FROM dlightrag_answer_child_operations AS operation
    WHERE operation.owner_id = child.owner_id AND operation.run_id = child.run_id
      AND operation.child_session_id = child.child_session_id
    ORDER BY operation.operation_sequence DESC LIMIT 1
) AS operation ON TRUE
WHERE child.owner_id = $1 AND child.run_id = $2 AND child.child_session_id = $3
"""

_SELECT_CHILD_SESSIONS = """
SELECT child.child_session_id, child.parent_session_id, child.parent_call_id,
       child.parent_intent_id, child.status, child.cancel_requested_at, child.summary,
       child.objective, child.context_mode, child.model_role, child.tools_json, child.usage_json,
       child.depth, child.context_snapshot_json, child.plan_json, child.budget_json,
       child.host_state_json, child.lease_owner, child.lease_expires_at,
       child.fencing_epoch, child.created_at, child.updated_at,
       operation.operation_sequence, operation.operation_id,
       operation.idempotency_key AS operation_key,
       operation.content AS operation_content,
       operation.origin AS operation_origin,
       operation.status AS operation_status,
       operation.cancellation_origin,
       (SELECT jsonb_agg(history.usage_json ORDER BY history.operation_sequence)
        FROM dlightrag_answer_child_operations AS history
        WHERE history.owner_id = child.owner_id AND history.run_id = child.run_id
          AND history.child_session_id = child.child_session_id
          AND history.usage_json IS NOT NULL) AS operation_usage_jsons
FROM dlightrag_answer_child_sessions AS child
LEFT JOIN LATERAL (
    SELECT * FROM dlightrag_answer_child_operations AS operation
    WHERE operation.owner_id = child.owner_id AND operation.run_id = child.run_id
      AND operation.child_session_id = child.child_session_id
    ORDER BY operation.operation_sequence DESC LIMIT 1
) AS operation ON TRUE
WHERE child.owner_id = $1 AND child.run_id = $2
ORDER BY child.created_at, child.child_session_id
"""

_CHILD_ROSTER_COLUMNS = """
child_session_id, parent_session_id, parent_call_id, parent_intent_id,
status, cancel_requested_at, summary, objective, context_mode, model_role, tools_json, usage_json,
depth, context_snapshot_json, plan_json, budget_json, host_state_json,
lease_owner, lease_expires_at, fencing_epoch, created_at, updated_at
"""

_SELECT_CHILD_SESSIONS_FIRST_PAGE = f"""
SELECT child.*, operation.operation_sequence, operation.operation_id,
       operation.idempotency_key AS operation_key,
       operation.content AS operation_content, operation.origin AS operation_origin,
       operation.status AS operation_status, operation.cancellation_origin
FROM (
    SELECT {_CHILD_ROSTER_COLUMNS}
    FROM dlightrag_answer_child_sessions
    WHERE owner_id = $1 AND run_id = $2
    ORDER BY created_at DESC, child_session_id DESC
    LIMIT $3
) AS child
LEFT JOIN LATERAL (
    SELECT * FROM dlightrag_answer_child_operations AS operation
    WHERE operation.owner_id = $1 AND operation.run_id = $2
      AND operation.child_session_id = child.child_session_id
    ORDER BY operation.operation_sequence DESC LIMIT 1
) AS operation ON TRUE
ORDER BY child.created_at DESC, child.child_session_id DESC
"""  # noqa: S608 - interpolates only the trusted column constant

_SELECT_CHILD_SESSIONS_AFTER = f"""
SELECT child.*, operation.operation_sequence, operation.operation_id,
       operation.idempotency_key AS operation_key,
       operation.content AS operation_content, operation.origin AS operation_origin,
       operation.status AS operation_status, operation.cancellation_origin
FROM (
    SELECT {_CHILD_ROSTER_COLUMNS}
    FROM dlightrag_answer_child_sessions
    WHERE owner_id = $1 AND run_id = $2
      AND (created_at < $3::timestamptz
           OR (created_at = $3::timestamptz AND child_session_id < $4::uuid))
    ORDER BY created_at DESC, child_session_id DESC
    LIMIT $5
) AS child
LEFT JOIN LATERAL (
    SELECT * FROM dlightrag_answer_child_operations AS operation
    WHERE operation.owner_id = $1 AND operation.run_id = $2
      AND operation.child_session_id = child.child_session_id
    ORDER BY operation.operation_sequence DESC LIMIT 1
) AS operation ON TRUE
ORDER BY child.created_at DESC, child.child_session_id DESC
"""  # noqa: S608 - interpolates only the trusted column constant

_SELECT_AGENT_TRANSCRIPT = """
WITH RECURSIVE authorized AS (
    SELECT agent_session_id, agent_lane_id
    FROM (
        SELECT agent_session_id, agent_lane_id
        FROM dlightrag_answer_run_routing
        WHERE owner_id = $1 AND run_id = $2 AND agent_session_id = $3
        UNION ALL
        SELECT child_session_id, 'main'
        FROM dlightrag_answer_child_sessions
        WHERE owner_id = $1 AND run_id = $2 AND child_session_id = $3
    ) AS bound
), ancestry AS (
    SELECT e.entry_id, e.parent_entry_id, e.sequence, e.entry_type, e.payload_json
    FROM authorized AS a
    JOIN dlightrag_agent_session_entries AS e
      ON e.owner_id = $1 AND e.session_id = a.agent_session_id
    JOIN dlightrag_agent_session_registers AS r
      ON r.owner_id = e.owner_id AND r.session_id = e.session_id
     AND r.register_kind = 'lane_head' AND r.register_key = a.agent_lane_id
     AND e.entry_id = NULLIF(r.payload_json->>'entry_id', '')::uuid
    UNION ALL
    SELECT parent.entry_id, parent.parent_entry_id, parent.sequence,
           parent.entry_type, parent.payload_json
    FROM dlightrag_agent_session_entries AS parent
    JOIN ancestry AS child ON child.parent_entry_id = parent.entry_id
    WHERE parent.owner_id = $1 AND parent.session_id = $3
)
SELECT entry_type, payload_json
FROM ancestry
WHERE entry_type IN ('user_message', 'assistant_message', 'tool_result', 'control_message')
ORDER BY sequence DESC
LIMIT $4
"""

_LOCK_CONTROL_RUN = """
SELECT r.status, r.lease_owner, r.lease_expires_at, r.fencing_epoch,
       rt.requested_mode, rt.resolved_mode
FROM dlightrag_runs AS r
JOIN dlightrag_answer_run_routing AS rt
  ON rt.owner_id = r.owner_id AND rt.run_id = r.run_id
WHERE r.owner_id = $1 AND r.run_id = $2
FOR UPDATE OF r
"""

_NEXT_CONTROL_SEQUENCE = """
SELECT COALESCE(MAX(control_sequence), 0) + 1
FROM dlightrag_agent_controls
WHERE owner_id = $1 AND run_id = $2
"""

_INSERT_CONTROL = """
INSERT INTO dlightrag_agent_controls (
    owner_id, run_id, control_sequence, kind, content, target_session_id,
    target_operation_id, origin, submission_key, request_fingerprint
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
ON CONFLICT DO NOTHING
"""

_SELECT_CONTROL_BY_SUBMISSION = """
SELECT control_sequence, kind, content, target_session_id, target_operation_id,
       origin, submission_key, request_fingerprint, consumed_at, created_at
FROM dlightrag_agent_controls
WHERE owner_id = $1 AND run_id = $2 AND target_session_id = $3
  AND submission_key = $4
"""

_SELECT_PENDING_PARENT_CONTROLS = """
SELECT control_sequence, kind, content, origin, created_at
FROM dlightrag_agent_controls
WHERE owner_id = $1 AND run_id = $2 AND target_session_id IS NULL
  AND consumed_at IS NULL
ORDER BY control_sequence
FOR UPDATE
"""

_SELECT_PENDING_CHILD_CONTROLS = """
SELECT control_sequence, kind, content, origin, created_at
FROM dlightrag_agent_controls
WHERE owner_id = $1 AND run_id = $2 AND target_session_id = $3
  AND target_operation_id = $4 AND consumed_at IS NULL
ORDER BY control_sequence
FOR UPDATE
"""

_CONSUME_PARENT_CONTROLS = """
UPDATE dlightrag_agent_controls
SET consumed_at = NOW()
WHERE owner_id = $1 AND run_id = $2 AND target_session_id IS NULL
  AND control_sequence = ANY($3::bigint[]) AND consumed_at IS NULL
"""

_CONSUME_CHILD_CONTROLS = """
UPDATE dlightrag_agent_controls
SET consumed_at = NOW()
WHERE owner_id = $1 AND run_id = $2 AND target_session_id = $3
  AND target_operation_id = $4
  AND control_sequence = ANY($5::bigint[]) AND consumed_at IS NULL
"""

_COUNT_PENDING_CHILD_CONTROLS = """
SELECT count(*)::int
FROM dlightrag_agent_controls
WHERE owner_id = $1 AND run_id = $2 AND target_session_id = $3
  AND target_operation_id = $4 AND consumed_at IS NULL
"""

_COUNT_PENDING_CHILD_GUIDANCE = """
SELECT count(*)::int
FROM dlightrag_answer_child_guidance
WHERE owner_id = $1 AND run_id = $2 AND child_session_id = $3
  AND child_operation_id = $4 AND status = 'pending' AND expires_at > NOW()
"""

_SELECT_OPERATION_STATE = """
SELECT payload_json->>'state_type'
FROM dlightrag_agent_session_registers
WHERE owner_id = $1 AND session_id = $2 AND register_kind = 'operation_state'
  AND register_key = $3
"""

_INSERT_GUIDANCE = """
INSERT INTO dlightrag_answer_child_guidance (
    owner_id, run_id, request_id, child_session_id, child_operation_id,
    parent_session_id, question, expires_at
)
VALUES ($1, $2, $3, $4, $5, $6, $7, NOW() + ($8 * INTERVAL '1 second'))
ON CONFLICT (owner_id, run_id, request_id) DO NOTHING
"""

_SELECT_GUIDANCE = """
SELECT request_id, child_session_id, child_operation_id, parent_session_id,
       question, status, reply, reply_origin, reply_submission_key,
       reply_fingerprint, expires_at, replied_at, created_at, updated_at
FROM dlightrag_answer_child_guidance
WHERE owner_id = $1 AND run_id = $2 AND request_id = $3
"""

_SELECT_PENDING_GUIDANCE = """
SELECT request_id, child_session_id, child_operation_id, parent_session_id,
       question, status, expires_at, created_at
FROM dlightrag_answer_child_guidance
WHERE owner_id = $1 AND run_id = $2 AND parent_session_id = $3
  AND status = 'pending' AND expires_at > NOW()
ORDER BY created_at, request_id
"""

# Host cancellation receipts are not model-consumed conversation controls.
# Accepted cancellation is observable in Child status and the parent intervention.
_SELECT_CHILD_CONTROLS = """
SELECT control_sequence, kind, content, origin, consumed_at, created_at,
       target_operation_id
FROM dlightrag_agent_controls
WHERE owner_id = $1 AND run_id = $2 AND target_session_id = $3
  AND kind <> 'cancel'
ORDER BY control_sequence DESC
LIMIT $4
"""

_SELECT_CHILD_GUIDANCE = """
SELECT request_id, child_session_id, child_operation_id, parent_session_id,
       question, status, reply, reply_origin, expires_at, replied_at,
       created_at, updated_at
FROM dlightrag_answer_child_guidance
WHERE owner_id = $1 AND run_id = $2 AND child_session_id = $3
ORDER BY created_at DESC, request_id DESC
LIMIT $4
"""

_EXPIRE_GUIDANCE = """
UPDATE dlightrag_answer_child_guidance
SET status = 'expired', updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2 AND request_id = $3
  AND child_session_id = $4 AND child_operation_id = $5
  AND status = 'pending' AND expires_at <= NOW()
RETURNING 1
"""

_REPLY_GUIDANCE = """
UPDATE dlightrag_answer_child_guidance
SET status = 'replied', reply = $4, reply_origin = $5,
    reply_submission_key = $6, reply_fingerprint = $7,
    replied_at = NOW(), updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2 AND request_id = $3
  AND status = 'pending' AND expires_at > NOW()
RETURNING 1
"""

_BIND_CHILD_PARENT_INTENT = """
UPDATE dlightrag_answer_child_sessions
SET parent_intent_id = $4, updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2 AND child_session_id = $3
  AND parent_intent_id IS NULL
"""

_FINISH_CHILD_SESSION = """
UPDATE dlightrag_answer_child_sessions
SET status = $4,
    summary = $5,
    usage_json = $6,
    host_state_json = jsonb_set(
        jsonb_set(host_state_json, '{terminal_outcome}', $7::jsonb, true),
        '{operation_outcomes}',
        COALESCE(host_state_json->'operation_outcomes', '{}'::jsonb)
            || jsonb_build_object($8::text, $7::jsonb),
        true
    ),
    lease_owner = NULL,
    lease_expires_at = NULL,
    updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2 AND child_session_id = $3
  AND status = 'running'
"""

_FINISH_CHILD_OPERATION = """
UPDATE dlightrag_answer_child_operations
SET status = $5, summary = $6, usage_json = $7, outcome_json = $8::jsonb,
    updated_at = NOW()
WHERE owner_id = $1 AND run_id = $2 AND child_session_id = $3
  AND operation_id = $4 AND status = 'running'
"""


def _require_owner(owner_id: str) -> str:
    owner = str(owner_id).strip()
    if not owner:
        raise ValueError("owner_id cannot be empty")
    return owner


async def _mirror_child_intervention(
    conn: Any,
    *,
    owner_id: str,
    run_id: uuid.UUID,
    action: str,
    identity: str,
    submission_key: str,
    content: str,
) -> None:
    """Mirror one accepted user intervention into the parent-only inbox."""
    mirror_key = f"child-intervention:{action}:{identity}:{submission_key}"
    sequence = int(await conn.fetchval(_NEXT_CONTROL_SEQUENCE, owner_id, run_id))
    text = f"User {action} for child {identity}: {content}".strip()
    fingerprint = hashlib.sha256(text.encode()).hexdigest()
    await conn.execute(
        _INSERT_CONTROL,
        owner_id,
        run_id,
        sequence,
        "steer",
        text[:20_000],
        None,
        None,
        "user",
        mirror_key[:200],
        fingerprint,
    )


def _parent_origin_holds_run(
    row: Mapping[str, Any] | Any,
    *,
    worker_id: str | None,
    fencing_epoch: int | None,
) -> bool:
    lease_expires_at = row["lease_expires_at"]
    return bool(
        worker_id
        and fencing_epoch is not None
        and str(row["status"]) == "running"
        and str(row["lease_owner"] or "") == worker_id
        and int(row["fencing_epoch"]) == fencing_epoch
        and lease_expires_at is not None
        and lease_expires_at > datetime.now(UTC)
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _guidance_row(row: Any) -> dict[str, Any]:
    return {
        "request_id": str(row["request_id"]),
        "child_session_id": str(row["child_session_id"]),
        "child_operation_id": str(row["child_operation_id"]),
        "parent_session_id": str(row["parent_session_id"]),
        "question": str(row["question"]),
        "status": str(row["status"]),
        "reply": row["reply"] if "reply" in row else None,
        "reply_origin": row["reply_origin"] if "reply_origin" in row else None,
        "expires_at": row["expires_at"],
        "replied_at": row["replied_at"] if "replied_at" in row else None,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"] if "updated_at" in row else row["created_at"],
    }


def _child_roster_row(row: Any) -> dict[str, Any]:
    return {
        "child_session_id": str(row["child_session_id"]),
        "parent_session_id": str(row["parent_session_id"]),
        "parent_call_id": str(row["parent_call_id"]),
        "parent_intent_id": (
            str(row["parent_intent_id"]) if row["parent_intent_id"] is not None else None
        ),
        "status": str(row["status"]),
        "cancel_requested_at": row["cancel_requested_at"],
        "summary": row["summary"],
        "objective": row["objective"],
        "context": row["context_mode"],
        "model_role": row["model_role"],
        "tools": _json_value(row["tools_json"]),
        "usage": _json_value(row["usage_json"]),
        "depth": int(row["depth"]),
        "context_snapshot": _json_value(row["context_snapshot_json"]),
        "plan": _json_value(row["plan_json"]),
        "budget": _json_value(row["budget_json"]),
        "host_state": _json_value(row["host_state_json"]),
        "fencing_epoch": int(row["fencing_epoch"]),
        **(
            {
                "operation_sequence": int(row["operation_sequence"]),
                "operation_id": str(row["operation_id"]),
                "operation_key": row["operation_key"],
                "operation_input": row["operation_content"],
                "operation_origin": row["operation_origin"],
                "operation_status": row["operation_status"],
                "cancellation_origin": row["cancellation_origin"],
                **(
                    {"operation_usage": _json_value(row["operation_usage_jsons"] or [])}
                    if "operation_usage_jsons" in row
                    else {}
                ),
            }
            if "operation_sequence" in row and row["operation_sequence"] is not None
            else {}
        ),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


class ChildRunStoreMixin:
    """Child Session, control, and guidance operations on PGRunStore."""

    _operation_pool: Any
    _run_read: Callable[..., Awaitable[Any]]
    _run_write: Callable[..., Awaitable[Any]]

    async def upsert_child_session(
        self,
        *,
        owner_id: str,
        run_id: str,
        child_session_id: str,
        parent_session_id: str,
        parent_call_id: str,
        worker_id: str,
        fencing_epoch: int,
        parent_intent_id: str | None = None,
        objective: str | None = None,
        context_mode: str | None = None,
        model_role: str | None = None,
        tools: Sequence[str] | None = None,
        depth: int = 1,
        context_snapshot: Mapping[str, Any] | None = None,
        plan: Mapping[str, Any] | None = None,
        budget: Mapping[str, Any] | None = None,
        host_state: Mapping[str, Any] | None = None,
    ) -> bool:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        child_uuid = parse_run_id(child_session_id)
        parent_uuid = parse_run_id(parent_session_id)
        intent_uuid = parse_run_id(parent_intent_id) if parent_intent_id is not None else None
        if run_uuid is None or child_uuid is None or parent_uuid is None:
            raise ValueError("child session ids must be canonical UUIDs")
        if parent_intent_id is not None and intent_uuid is None:
            raise ValueError("parent intent id must be a canonical UUID")

        async def _operation(conn: Any) -> bool:
            async with conn.transaction():
                held = await conn.fetchval(
                    _HOLD_RUN_LEASE, owner, run_uuid, worker_id, fencing_epoch
                )
                if held is None:
                    return False
                await conn.execute(
                    _UPSERT_CHILD_SESSION,
                    owner,
                    run_uuid,
                    child_uuid,
                    parent_uuid,
                    parent_call_id,
                    intent_uuid,
                    objective,
                    context_mode,
                    model_role,
                    json.dumps(list(tools)) if tools is not None else None,
                    depth,
                    json.dumps(dict(context_snapshot or {}), ensure_ascii=False),
                    json.dumps(dict(plan), ensure_ascii=False) if plan is not None else None,
                    json.dumps(dict(budget), ensure_ascii=False) if budget is not None else None,
                    json.dumps(dict(host_state or {}), ensure_ascii=False),
                )
                if (
                    objective
                    and plan is not None
                    and (context_snapshot or {}).get("parent_entry_id")
                ):
                    operation_key = f"child-session:{child_session_id}"
                    operation_id = uuid.UUID(
                        OperationId.deterministic(idempotency_key=operation_key).value
                    )
                    fingerprint = hashlib.sha256(
                        f"{operation_key}\0{objective}".encode()
                    ).hexdigest()
                    await conn.execute(
                        _INSERT_CHILD_OPERATION,
                        owner,
                        run_uuid,
                        child_uuid,
                        1,
                        operation_id,
                        operation_key,
                        fingerprint,
                        objective,
                        "parent",
                    )
                return True

        return await self._run_write(_operation)

    async def claim_child_session(
        self,
        *,
        owner_id: str,
        run_id: str,
        child_session_id: str,
        worker_id: str,
        fencing_epoch: int,
    ) -> int | None:
        """Acquire the Child's independent lease under the live parent run claim."""
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        child_uuid = parse_run_id(child_session_id)
        if run_uuid is None or child_uuid is None:
            raise ValueError("child session ids must be canonical UUIDs")

        async def _operation(conn: Any) -> int | None:
            async with conn.transaction():
                held = await conn.fetchval(
                    _HOLD_RUN_LEASE, owner, run_uuid, worker_id, fencing_epoch
                )
                if held is None:
                    return None
                value = await conn.fetchval(
                    _CLAIM_CHILD_SESSION,
                    owner,
                    run_uuid,
                    child_uuid,
                    worker_id,
                    RUN_LEASE_SECONDS,
                )
                return int(value) if value is not None else None

        return await self._run_write(_operation)

    async def request_child_cancellation(
        self,
        *,
        owner_id: str,
        run_id: str,
        child_session_id: str,
        worker_id: str,
        fencing_epoch: int,
        cancellation_origin: str = "parent",
    ) -> bool:
        """Persist cancellation before the active Child writer closes its Operation."""
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        child_uuid = parse_run_id(child_session_id)
        if run_uuid is None or child_uuid is None:
            raise ValueError("child session ids must be canonical UUIDs")
        if cancellation_origin not in {"parent", "user", "run"}:
            raise ValueError("invalid Child cancellation origin")

        async def _operation(conn: Any) -> bool:
            async with conn.transaction():
                held = await conn.fetchval(
                    _HOLD_RUN_LEASE, owner, run_uuid, worker_id, fencing_epoch
                )
                if held is None:
                    return False
                child = await conn.fetchrow(_LOCK_CHILD_SESSION, owner, run_uuid, child_uuid)
                if child is None or str(child["status"]) != "running":
                    return False
                requested = await conn.fetchval(
                    _REQUEST_CHILD_CANCELLATION,
                    owner,
                    run_uuid,
                    child_uuid,
                )
                operation_id = child["operation_id"]
                if operation_id is not None:
                    await conn.execute(
                        _CANCEL_CHILD_OPERATION,
                        owner,
                        run_uuid,
                        child_uuid,
                        operation_id,
                        cancellation_origin,
                    )
                await conn.execute(_RETIRE_CHILD_GUIDANCE, owner, run_uuid, child_uuid)
                await conn.execute(
                    "SELECT pg_notify('dlightrag_run_activity', $1)",
                    cancellation_notify_key(owner_id=owner, run_id=str(run_uuid)),
                )
                return requested is not None

        return await self._run_write(_operation)

    async def release_child_sessions(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
    ) -> bool:
        """Release this worker's Child leases without changing accepted lifecycle."""
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return False

        async def _operation(conn: Any) -> bool:
            async with conn.transaction():
                held = await conn.fetchval(
                    _HOLD_RUN_LEASE, owner, run_uuid, worker_id, fencing_epoch
                )
                if held is None:
                    return False
                await conn.execute(
                    _RELEASE_CHILD_SESSION_LEASES,
                    owner,
                    run_uuid,
                    worker_id,
                )
                return True

        return await self._run_write(_operation)

    async def heartbeat_child_session(
        self,
        *,
        owner_id: str,
        run_id: str,
        child_session_id: str,
        worker_id: str,
        fencing_epoch: int,
        child_fencing_epoch: int,
    ) -> bool:
        """Renew one unexpired Child lease under its live parent run claim."""
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        child_uuid = parse_run_id(child_session_id)
        if run_uuid is None or child_uuid is None:
            raise ValueError("child session ids must be canonical UUIDs")

        async def _operation(conn: Any) -> bool:
            async with conn.transaction():
                held = await conn.fetchval(
                    _HOLD_RUN_LEASE, owner, run_uuid, worker_id, fencing_epoch
                )
                if held is None:
                    return False
                renewed = await conn.fetchval(
                    _RENEW_CHILD_SESSION_LEASE,
                    owner,
                    run_uuid,
                    child_uuid,
                    worker_id,
                    child_fencing_epoch,
                    RUN_LEASE_SECONDS,
                )
                return renewed is not None

        return await self._run_write(_operation)

    async def load_child_session(
        self, *, owner_id: str, run_id: str, child_session_id: str
    ) -> dict[str, Any] | None:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        child_uuid = parse_run_id(child_session_id)
        if run_uuid is None or child_uuid is None:
            return None

        async def _operation(conn: Any) -> dict[str, Any] | None:
            row = await conn.fetchrow(_SELECT_CHILD_SESSION, owner, run_uuid, child_uuid)
            if row is None:
                return None
            return {
                "child_session_id": str(row["child_session_id"]),
                "status": str(row["status"]),
                "cancel_requested_at": row["cancel_requested_at"],
                "summary": row["summary"],
                "parent_intent_id": (
                    str(row["parent_intent_id"]) if row["parent_intent_id"] is not None else None
                ),
                "objective": row["objective"],
                "context": row["context_mode"],
                "model_role": row["model_role"],
                "tools": _json_value(row["tools_json"]),
                "usage": _json_value(row["usage_json"]),
                "depth": int(row["depth"]),
                "context_snapshot": _json_value(row["context_snapshot_json"]),
                "plan": _json_value(row["plan_json"]),
                "budget": _json_value(row["budget_json"]),
                "host_state": _json_value(row["host_state_json"]),
                "fencing_epoch": int(row["fencing_epoch"]),
                "operation_sequence": (
                    int(row["operation_sequence"])
                    if row["operation_sequence"] is not None
                    else None
                ),
                "operation_id": (
                    str(row["operation_id"]) if row["operation_id"] is not None else None
                ),
                "operation_key": row["operation_key"],
                "operation_input": row["operation_content"],
                "operation_origin": row["operation_origin"],
                "operation_status": row["operation_status"],
                "cancellation_origin": row["cancellation_origin"],
                "operation_usage": _json_value(row["operation_usage_jsons"] or []),
            }

        return await self._run_read(_operation)

    async def list_child_sessions(
        self, *, owner_id: str, run_id: str
    ) -> tuple[dict[str, Any], ...]:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return ()

        async def _operation(conn: Any) -> tuple[dict[str, Any], ...]:
            rows = await conn.fetch(_SELECT_CHILD_SESSIONS, owner, run_uuid)
            return tuple(_child_roster_row(row) for row in rows)

        return await self._run_read(_operation)

    async def list_child_sessions_page(
        self,
        *,
        owner_id: str,
        run_id: str,
        page: ChildRosterPageRequest,
    ) -> ChildRosterRowPage:
        """Return one physical limit+1 newest-first keyset roster page."""
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        validated = ChildRosterPageRequest(limit=page.limit, cursor=page.cursor)
        cursor = validated.cursor
        if run_uuid is None:
            return ChildRosterRowPage(children=(), has_more=False, fetched_rows=0)
        if cursor is not None and cursor.run_id != run_uuid:
            raise ValueError("child-roster cursor belongs to another run")
        fetch_limit = validated.limit + 1

        async def _operation(conn: Any) -> ChildRosterRowPage:
            if cursor is None:
                rows = await conn.fetch(
                    _SELECT_CHILD_SESSIONS_FIRST_PAGE,
                    owner,
                    run_uuid,
                    fetch_limit,
                )
            else:
                rows = await conn.fetch(
                    _SELECT_CHILD_SESSIONS_AFTER,
                    owner,
                    run_uuid,
                    cursor.created_at,
                    cursor.child_session_id,
                    fetch_limit,
                )
            fetched_rows = len(rows)
            return ChildRosterRowPage(
                children=tuple(_child_roster_row(row) for row in rows[: validated.limit]),
                has_more=fetched_rows > validated.limit,
                fetched_rows=fetched_rows,
            )

        return await self._run_read(_operation)

    async def list_child_controls(
        self,
        *,
        owner_id: str,
        run_id: str,
        child_session_id: str,
        limit: int = 20,
    ) -> tuple[dict[str, Any], ...]:
        """Return newest-first targeted Child controls, including queued and consumed."""
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        child_uuid = parse_run_id(child_session_id)
        if run_uuid is None or child_uuid is None:
            return ()

        async def _operation(conn: Any) -> tuple[dict[str, Any], ...]:
            rows = await conn.fetch(
                _SELECT_CHILD_CONTROLS,
                owner,
                run_uuid,
                child_uuid,
                max(1, min(int(limit), 100)),
            )
            return tuple(
                {
                    "control_sequence": int(row["control_sequence"]),
                    "kind": str(row["kind"]),
                    "content": str(row["content"]),
                    "origin": str(row["origin"]),
                    "consumed": row["consumed_at"] is not None,
                    "consumed_at": row["consumed_at"],
                    "created_at": row["created_at"],
                    "operation_id": (
                        str(row["target_operation_id"])
                        if row["target_operation_id"] is not None
                        else None
                    ),
                }
                for row in rows
            )

        return await self._run_read(_operation)

    async def list_child_guidance(
        self,
        *,
        owner_id: str,
        run_id: str,
        child_session_id: str,
        limit: int = 20,
    ) -> tuple[dict[str, Any], ...]:
        """Return newest-first Child questions without reply fingerprints."""
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        child_uuid = parse_run_id(child_session_id)
        if run_uuid is None or child_uuid is None:
            return ()

        async def _operation(conn: Any) -> tuple[dict[str, Any], ...]:
            rows = await conn.fetch(
                _SELECT_CHILD_GUIDANCE,
                owner,
                run_uuid,
                child_uuid,
                max(1, min(int(limit), 100)),
            )
            return tuple(_guidance_row(row) for row in rows)

        return await self._run_read(_operation)

    async def enqueue_child_control(
        self,
        *,
        owner_id: str,
        run_id: str,
        child_session_id: str,
        content: str,
        submission_key: str,
        origin: str = "user",
        parent_session_id: str | None = None,
        worker_id: str | None = None,
        fencing_epoch: int | None = None,
    ) -> dict[str, Any] | bool:
        """Queue an idempotent steer for only the Child's current Operation."""
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        child_uuid = parse_run_id(child_session_id)
        parent_uuid = parse_run_id(parent_session_id) if parent_session_id is not None else None
        text = content.strip()
        key = submission_key.strip()
        if run_uuid is None or child_uuid is None:
            raise ValueError("child control ids must be canonical UUIDs")
        if parent_session_id is not None and parent_uuid is None:
            raise ValueError("parent session id must be a canonical UUID")
        if not key or len(key) > 200:
            raise ValueError("submission_key must contain between 1 and 200 characters")
        if not text or len(text) > 20_000:
            raise ValueError("control content must contain between 1 and 20000 characters")
        if origin not in {"user", "parent"}:
            raise ValueError("invalid Child control origin")
        fingerprint = hashlib.sha256(
            f"steer\0{child_session_id}\0{text}\0{origin}".encode()
        ).hexdigest()

        async def _operation(conn: Any) -> dict[str, Any] | bool:
            async with conn.transaction():
                run = await conn.fetchrow(_LOCK_CONTROL_RUN, owner, run_uuid)
                if run is None:
                    return {"outcome": "unknown_child"}
                if origin == "parent" and not _parent_origin_holds_run(
                    run, worker_id=worker_id, fencing_epoch=fencing_epoch
                ):
                    return False
                child = await conn.fetchrow(_LOCK_CHILD_SESSION, owner, run_uuid, child_uuid)
                if child is None or (
                    parent_uuid is not None and str(child["parent_session_id"]) != str(parent_uuid)
                ):
                    return {"outcome": "unknown_child"}
                replay = await conn.fetchrow(
                    _SELECT_CONTROL_BY_SUBMISSION,
                    owner,
                    run_uuid,
                    child_uuid,
                    key,
                )
                if replay is not None:
                    if str(replay["request_fingerprint"]) != fingerprint:
                        return {"outcome": "idempotency_conflict"}
                    return {
                        "outcome": "consumed" if replay["consumed_at"] else "queued",
                        "control_sequence": int(replay["control_sequence"]),
                        "operation_id": str(replay["target_operation_id"]),
                        "consumed_at": replay["consumed_at"],
                    }
                if (
                    str(run["status"]) not in {"queued", "running"}
                    or str(child["status"]) != "running"
                    or str(child["operation_status"] or "") != "running"
                    or child["operation_id"] is None
                ):
                    return {"outcome": "terminal_child"}
                operation_state = await conn.fetchval(
                    _SELECT_OPERATION_STATE,
                    owner,
                    child_uuid,
                    str(child["operation_id"]),
                )
                if operation_state in {"completed", "failed", "cancelled"}:
                    return {"outcome": "terminal_child"}
                pending = int(
                    await conn.fetchval(
                        _COUNT_PENDING_CHILD_CONTROLS,
                        owner,
                        run_uuid,
                        child_uuid,
                        child["operation_id"],
                    )
                )
                if pending >= _MAX_PENDING_CHILD_CONTROLS:
                    return {"outcome": "queue_full"}
                sequence = int(await conn.fetchval(_NEXT_CONTROL_SEQUENCE, owner, run_uuid))
                await conn.execute(
                    _INSERT_CONTROL,
                    owner,
                    run_uuid,
                    sequence,
                    "steer",
                    text,
                    child_uuid,
                    child["operation_id"],
                    origin,
                    key,
                    fingerprint,
                )
                if origin == "user":
                    await _mirror_child_intervention(
                        conn,
                        owner_id=owner,
                        run_id=run_uuid,
                        action="steered",
                        identity=child_session_id,
                        submission_key=key,
                        content=text,
                    )
                await conn.execute(
                    "SELECT pg_notify('dlightrag_run_activity', $1)",
                    cancellation_notify_key(owner_id=owner, run_id=str(run_uuid)),
                )
                return {
                    "outcome": "queued",
                    "control_sequence": sequence,
                    "operation_id": str(child["operation_id"]),
                    "consumed_at": None,
                }

        return await self._run_write(_operation)

    async def continue_child_session(
        self,
        *,
        owner_id: str,
        run_id: str,
        child_session_id: str,
        content: str,
        submission_key: str,
        origin: str = "user",
        parent_session_id: str | None = None,
        reauthorize_user_cancelled: bool = False,
        worker_id: str | None = None,
        fencing_epoch: int | None = None,
    ) -> dict[str, Any] | bool:
        """Accept one explicit new Operation without changing the Child's pinned plan."""
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        child_uuid = parse_run_id(child_session_id)
        parent_uuid = parse_run_id(parent_session_id) if parent_session_id is not None else None
        text = content.strip()
        key = submission_key.strip()
        if run_uuid is None or child_uuid is None:
            raise ValueError("child continuation ids must be canonical UUIDs")
        if parent_session_id is not None and parent_uuid is None:
            raise ValueError("parent session id must be a canonical UUID")
        if not key or len(key) > 200:
            raise ValueError("submission_key must contain between 1 and 200 characters")
        if not text or len(text) > 20_000:
            raise ValueError("continuation content must contain between 1 and 20000 characters")
        if origin not in {"user", "parent"}:
            raise ValueError("invalid Child continuation origin")
        fingerprint = hashlib.sha256(
            f"continue\0{child_session_id}\0{text}\0{origin}".encode()
        ).hexdigest()
        operation_id = uuid.UUID(OperationId.deterministic(idempotency_key=key).value)

        async def _operation(conn: Any) -> dict[str, Any] | bool:
            async with conn.transaction():
                run = await conn.fetchrow(_LOCK_CONTROL_RUN, owner, run_uuid)
                if run is None:
                    return {"outcome": "unknown_child"}
                if origin == "parent" and not _parent_origin_holds_run(
                    run, worker_id=worker_id, fencing_epoch=fencing_epoch
                ):
                    return False
                child = await conn.fetchrow(_LOCK_CHILD_SESSION, owner, run_uuid, child_uuid)
                if child is None or (
                    parent_uuid is not None and str(child["parent_session_id"]) != str(parent_uuid)
                ):
                    return {"outcome": "unknown_child"}
                replay = await conn.fetchrow(
                    _SELECT_CHILD_OPERATION_BY_KEY,
                    owner,
                    run_uuid,
                    child_uuid,
                    key,
                )
                if replay is not None:
                    if str(replay["request_fingerprint"]) != fingerprint:
                        return {"outcome": "idempotency_conflict"}
                    return {
                        "outcome": "accepted",
                        "operation_id": str(replay["operation_id"]),
                        "operation_sequence": int(replay["operation_sequence"]),
                        "status": str(replay["status"]),
                    }
                if str(run["status"]) not in {"queued", "running"}:
                    return {"outcome": "run_terminal"}
                if str(child["status"]) == "running":
                    return {"outcome": "child_running"}
                if str(child["cancellation_origin"] or "") == "user" and not (
                    origin == "user" and reauthorize_user_cancelled
                ):
                    return {"outcome": "reauthorization_required"}
                if str(child["status"]) not in {"succeeded", "failed", "cancelled"}:
                    return {"outcome": "unknown_outcome"}
                sequence = int(child["operation_sequence"] or 0) + 1
                await conn.execute(
                    _INSERT_CHILD_OPERATION,
                    owner,
                    run_uuid,
                    child_uuid,
                    sequence,
                    operation_id,
                    key,
                    fingerprint,
                    text,
                    origin,
                )
                await conn.execute(_ACTIVATE_CHILD_OPERATION, owner, run_uuid, child_uuid)
                if origin == "user":
                    await _mirror_child_intervention(
                        conn,
                        owner_id=owner,
                        run_id=run_uuid,
                        action="continued",
                        identity=child_session_id,
                        submission_key=key,
                        content=text,
                    )
                await conn.execute(
                    "SELECT pg_notify('dlightrag_run_activity', $1)",
                    cancellation_notify_key(owner_id=owner, run_id=str(run_uuid)),
                )
                return {
                    "outcome": "accepted",
                    "operation_id": str(operation_id),
                    "operation_sequence": sequence,
                    "status": "running",
                }

        return await self._run_write(_operation)

    async def cancel_child_session_by_owner(
        self,
        *,
        owner_id: str,
        run_id: str,
        child_session_id: str,
        submission_key: str,
        parent_session_id: str | None = None,
    ) -> dict[str, Any]:
        """Persist an owner cancellation without reviving terminal Child work."""
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        child_uuid = parse_run_id(child_session_id)
        parent_uuid = parse_run_id(parent_session_id) if parent_session_id is not None else None
        if run_uuid is None or child_uuid is None:
            raise ValueError("child cancellation ids must be canonical UUIDs")

        if parent_session_id is not None and parent_uuid is None:
            raise ValueError("parent session id must be a canonical UUID")
        key = submission_key.strip()
        if not key or len(key) > 200:
            raise ValueError("submission_key must contain between 1 and 200 characters")
        fingerprint = hashlib.sha256(f"cancel\0{child_session_id}\0user".encode()).hexdigest()

        async def _operation(conn: Any) -> dict[str, Any]:
            async with conn.transaction():
                run = await conn.fetchrow(_LOCK_CONTROL_RUN, owner, run_uuid)
                if run is None:
                    return {"outcome": "unknown_child"}
                child = await conn.fetchrow(_LOCK_CHILD_SESSION, owner, run_uuid, child_uuid)
                if child is None or (
                    parent_uuid is not None and str(child["parent_session_id"]) != str(parent_uuid)
                ):
                    return {"outcome": "unknown_child"}
                replay = await conn.fetchrow(
                    _SELECT_CONTROL_BY_SUBMISSION,
                    owner,
                    run_uuid,
                    child_uuid,
                    key,
                )
                if replay is not None:
                    if str(replay["request_fingerprint"]) != fingerprint:
                        return {"outcome": "idempotency_conflict"}
                    receipt = json.loads(str(replay["content"]))
                    if not isinstance(receipt, dict) or receipt.get("operation_id") != str(
                        replay["target_operation_id"]
                    ):
                        raise ValueError("Invalid persisted Child cancellation receipt")
                    return receipt
                operation_id = child["operation_id"]
                if operation_id is None:
                    return {"outcome": "unknown_outcome"}
                outcome = (
                    "run_terminal"
                    if str(run["status"]) not in {"queued", "running"}
                    else "terminal_child"
                    if str(child["status"]) != "running"
                    else "cancellation_requested"
                )
                sequence = int(await conn.fetchval(_NEXT_CONTROL_SEQUENCE, owner, run_uuid))
                receipt = {
                    "outcome": outcome,
                    "operation_id": str(operation_id),
                    "control_sequence": sequence,
                    "status": str(child["status"]),
                }
                # A cancellation is handled by the host, not queued to the
                # model. Its ControlMessage content is the immutable receipt,
                # including a terminal rejection whose response may be lost.
                await conn.execute(
                    _INSERT_CONTROL,
                    owner,
                    run_uuid,
                    sequence,
                    "cancel",
                    json.dumps(receipt),
                    child_uuid,
                    operation_id,
                    "user",
                    key,
                    fingerprint,
                )
                await conn.execute(
                    _CONSUME_CHILD_CONTROLS, owner, run_uuid, child_uuid, operation_id, [sequence]
                )
                if outcome != "cancellation_requested":
                    return receipt
                await conn.execute(_REQUEST_CHILD_CANCELLATION, owner, run_uuid, child_uuid)
                await conn.execute(
                    _CANCEL_CHILD_OPERATION,
                    owner,
                    run_uuid,
                    child_uuid,
                    operation_id,
                    "user",
                )
                await conn.execute(_RETIRE_CHILD_GUIDANCE, owner, run_uuid, child_uuid)
                await _mirror_child_intervention(
                    conn,
                    owner_id=owner,
                    run_id=run_uuid,
                    action="cancelled",
                    identity=child_session_id,
                    submission_key=key,
                    content="cancellation requested",
                )
                await conn.execute(
                    "SELECT pg_notify('dlightrag_run_activity', $1)",
                    cancellation_notify_key(owner_id=owner, run_id=str(run_uuid)),
                )
                return receipt

        return await self._run_write(_operation)

    async def reply_child_guidance(
        self,
        *,
        owner_id: str,
        run_id: str,
        request_id: str,
        content: str,
        submission_key: str,
        origin: str = "user",
        parent_session_id: str | None = None,
        worker_id: str | None = None,
        fencing_epoch: int | None = None,
    ) -> dict[str, Any] | bool:
        """Correlate one idempotent parent/user reply with a pending Child ask."""
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        request_uuid = parse_run_id(request_id)
        parent_uuid = parse_run_id(parent_session_id) if parent_session_id is not None else None
        text = content.strip()
        key = submission_key.strip()
        if run_uuid is None or request_uuid is None:
            raise ValueError("guidance ids must be canonical UUIDs")
        if not key or len(key) > 200 or not text or len(text) > 20_000:
            raise ValueError("reply and submission key must be non-empty and bounded")
        if origin not in {"user", "parent"}:
            raise ValueError("invalid guidance reply origin")
        fingerprint = hashlib.sha256(f"reply\0{request_id}\0{text}\0{origin}".encode()).hexdigest()

        async def _operation(conn: Any) -> dict[str, Any] | bool:
            async with conn.transaction():
                run = await conn.fetchrow(_LOCK_CONTROL_RUN, owner, run_uuid)
                if run is None:
                    return {"outcome": "unknown_request"}
                if origin == "parent" and not _parent_origin_holds_run(
                    run, worker_id=worker_id, fencing_epoch=fencing_epoch
                ):
                    return False
                guidance = await conn.fetchrow(
                    _SELECT_GUIDANCE + " FOR UPDATE", owner, run_uuid, request_uuid
                )
                if guidance is None or (
                    parent_uuid is not None
                    and str(guidance["parent_session_id"]) != str(parent_uuid)
                ):
                    return {"outcome": "unknown_request"}
                child_id = str(guidance["child_session_id"])
                if str(guidance["status"]) == "replied":
                    if (
                        str(guidance["reply_submission_key"] or "") == key
                        and str(guidance["reply_fingerprint"] or "") == fingerprint
                    ):
                        return {
                            "outcome": "replied",
                            "request_id": request_id,
                            "child_session_id": child_id,
                        }
                    return {
                        "outcome": "already_replied",
                        "request_id": request_id,
                        "child_session_id": child_id,
                    }
                if str(guidance["status"]) != "pending":
                    return {
                        "outcome": str(guidance["status"]),
                        "request_id": request_id,
                        "child_session_id": child_id,
                    }
                updated = await conn.fetchval(
                    _REPLY_GUIDANCE,
                    owner,
                    run_uuid,
                    request_uuid,
                    text,
                    origin,
                    key,
                    fingerprint,
                )
                if updated is None:
                    await conn.execute(
                        "UPDATE dlightrag_answer_child_guidance SET status = 'expired', "
                        "updated_at = NOW() WHERE owner_id = $1 AND run_id = $2 "
                        "AND request_id = $3 AND status = 'pending'",
                        owner,
                        run_uuid,
                        request_uuid,
                    )
                    return {
                        "outcome": "expired",
                        "request_id": request_id,
                        "child_session_id": child_id,
                    }
                if origin == "user":
                    await _mirror_child_intervention(
                        conn,
                        owner_id=owner,
                        run_id=run_uuid,
                        action="replied",
                        identity=str(guidance["child_session_id"]),
                        submission_key=key,
                        content=f"request {request_id}: {text}",
                    )
                await conn.execute(
                    "SELECT pg_notify('dlightrag_run_activity', $1)",
                    cancellation_notify_key(owner_id=owner, run_id=str(run_uuid)),
                )
                return {
                    "outcome": "replied",
                    "request_id": request_id,
                    "child_session_id": child_id,
                }

        return await self._run_write(_operation)

    async def create_child_guidance(
        self,
        *,
        owner_id: str,
        run_id: str,
        request_id: str,
        child_session_id: str,
        child_operation_id: str,
        parent_session_id: str,
        question: str,
        expires_after_seconds: int,
        worker_id: str,
        fencing_epoch: int,
        child_fencing_epoch: int,
    ) -> dict[str, Any] | None:
        """Persist one fenced Child ask before the Child parks."""
        owner = _require_owner(owner_id)
        ids = tuple(
            parse_run_id(value)
            for value in (
                run_id,
                request_id,
                child_session_id,
                child_operation_id,
                parent_session_id,
            )
        )
        if any(value is None for value in ids):
            raise ValueError("guidance ids must be canonical UUIDs")
        run_uuid, request_uuid, child_uuid, operation_uuid, parent_uuid = ids
        text = question.strip()
        if not text or len(text) > 20_000:
            raise ValueError("guidance question must contain between 1 and 20000 characters")
        if not 1 <= expires_after_seconds <= 86_400:
            raise ValueError("guidance expiry must be between 1 and 86400 seconds")

        async def _operation(conn: Any) -> dict[str, Any] | None:
            async with conn.transaction():
                held = await conn.fetchval(
                    _HOLD_RUN_LEASE, owner, run_uuid, worker_id, fencing_epoch
                )
                if held is None:
                    return None
                child = await conn.fetchrow(_LOCK_CHILD_SESSION, owner, run_uuid, child_uuid)
                if (
                    child is None
                    or str(child["lease_owner"] or "") != worker_id
                    or int(child["fencing_epoch"]) != child_fencing_epoch
                    or str(child["operation_id"] or "") != str(operation_uuid)
                    or str(child["parent_session_id"]) != str(parent_uuid)
                    or str(child["status"]) != "running"
                ):
                    return None
                pending = int(
                    await conn.fetchval(
                        _COUNT_PENDING_CHILD_GUIDANCE,
                        owner,
                        run_uuid,
                        child_uuid,
                        operation_uuid,
                    )
                )
                if pending >= _MAX_PENDING_CHILD_GUIDANCE:
                    return {
                        "request_id": request_id,
                        "child_session_id": child_session_id,
                        "child_operation_id": child_operation_id,
                        "parent_session_id": parent_session_id,
                        "question": text,
                        "status": "queue_full",
                    }
                await conn.execute(
                    _INSERT_GUIDANCE,
                    owner,
                    run_uuid,
                    request_uuid,
                    child_uuid,
                    operation_uuid,
                    parent_uuid,
                    text,
                    expires_after_seconds,
                )
                stored = await conn.fetchrow(_SELECT_GUIDANCE, owner, run_uuid, request_uuid)
                if stored is None or (
                    str(stored["child_session_id"]) != str(child_uuid)
                    or str(stored["child_operation_id"]) != str(operation_uuid)
                    or str(stored["question"]) != text
                ):
                    raise ValueError("guidance request id was reused with different input")
                await conn.execute(
                    "SELECT pg_notify('dlightrag_run_activity', $1)",
                    cancellation_notify_key(owner_id=owner, run_id=str(run_uuid)),
                )
                return _guidance_row(stored)

        return await self._run_write(_operation)

    async def load_child_guidance(
        self, *, owner_id: str, run_id: str, request_id: str
    ) -> dict[str, Any] | None:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        request_uuid = parse_run_id(request_id)
        if run_uuid is None or request_uuid is None:
            return None

        async def _operation(conn: Any) -> dict[str, Any] | None:
            row = await conn.fetchrow(_SELECT_GUIDANCE, owner, run_uuid, request_uuid)
            return _guidance_row(row) if row is not None else None

        return await self._run_read(_operation)

    async def wait_for_child_guidance(
        self,
        *,
        owner_id: str,
        run_id: str,
        request_id: str,
        timeout_seconds: float,
    ) -> dict[str, Any] | None:
        """Wait on a PostgreSQL wake hint, then re-read authoritative guidance."""
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        request_uuid = parse_run_id(request_id)
        if run_uuid is None or request_uuid is None:
            return None
        wake_key = cancellation_notify_key(owner_id=owner, run_id=str(run_uuid))
        bounded_timeout = max(0.0, min(float(timeout_seconds), 86_400.0))
        row = await self.load_child_guidance(
            owner_id=owner_id, run_id=run_id, request_id=request_id
        )
        if row is None or str(row["status"]) != "pending":
            return row
        remaining = max(0.0, (row["expires_at"] - datetime.now(UTC)).total_seconds())
        deadline = datetime.now(UTC).timestamp() + min(bounded_timeout, remaining)
        connection = await asyncpg.connect(**self._notify_connect_kwargs())
        queue: asyncio.Queue[None] = asyncio.Queue(maxsize=1)

        def _notified(_conn: object, _pid: object, channel: str, payload: str) -> None:
            if channel == _RUN_ACTIVITY_CHANNEL and payload == wake_key and queue.empty():
                queue.put_nowait(None)

        try:
            await connection.add_listener(_RUN_ACTIVITY_CHANNEL, _notified)
            while True:
                left = deadline - datetime.now(UTC).timestamp()
                if left <= 0:
                    return await self.load_child_guidance(
                        owner_id=owner_id, run_id=run_id, request_id=request_id
                    )
                try:
                    await asyncio.wait_for(
                        queue.get(), timeout=min(left, _GUIDANCE_HINT_POLL_SECONDS)
                    )
                except TimeoutError:
                    pass
                current = await self.load_child_guidance(
                    owner_id=owner_id, run_id=run_id, request_id=request_id
                )
                if current is None or str(current["status"]) != "pending":
                    return current
        finally:
            try:
                await connection.remove_listener(_RUN_ACTIVITY_CHANNEL, _notified)
            except Exception:
                logger.debug("Guidance listener removal failed", exc_info=True)
            await connection.close()

    def _notify_connect_kwargs(self) -> dict[str, Any]:
        """Return dedicated LISTEN connection kwargs that are not the domain pool."""
        pool = self._operation_pool
        if pool is not None:
            kwargs = getattr(pool, "_connect_kwargs", None)
            if isinstance(kwargs, Mapping) and kwargs:
                return dict(kwargs)
        from dlightrag.application.config import get_config

        return dict(get_config().pg_connection_kwargs())

    async def expire_child_guidance(
        self,
        *,
        owner_id: str,
        run_id: str,
        request_id: str,
        child_session_id: str,
        child_operation_id: str,
        worker_id: str,
        fencing_epoch: int,
        child_fencing_epoch: int,
    ) -> bool:
        owner = _require_owner(owner_id)
        ids = tuple(
            parse_run_id(value)
            for value in (run_id, request_id, child_session_id, child_operation_id)
        )
        if any(value is None for value in ids):
            return False
        run_uuid, request_uuid, child_uuid, operation_uuid = ids

        async def _operation(conn: Any) -> bool:
            async with conn.transaction():
                held = await conn.fetchval(
                    _HOLD_RUN_LEASE, owner, run_uuid, worker_id, fencing_epoch
                )
                if held is None:
                    return False
                child = await conn.fetchrow(_LOCK_CHILD_SESSION, owner, run_uuid, child_uuid)
                if (
                    child is None
                    or str(child["lease_owner"] or "") != worker_id
                    or int(child["fencing_epoch"]) != child_fencing_epoch
                    or str(child["operation_id"] or "") != str(operation_uuid)
                ):
                    return False
                return (
                    await conn.fetchval(
                        _EXPIRE_GUIDANCE,
                        owner,
                        run_uuid,
                        request_uuid,
                        child_uuid,
                        operation_uuid,
                    )
                    is not None
                )

        return await self._run_write(_operation)

    async def list_pending_child_guidance(
        self,
        *,
        owner_id: str,
        run_id: str,
        parent_session_id: str,
    ) -> tuple[dict[str, Any], ...]:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        parent_uuid = parse_run_id(parent_session_id)
        if run_uuid is None or parent_uuid is None:
            return ()

        async def _operation(conn: Any) -> tuple[dict[str, Any], ...]:
            rows = await conn.fetch(_SELECT_PENDING_GUIDANCE, owner, run_uuid, parent_uuid)
            return tuple(_guidance_row(row) for row in rows)

        return await self._run_read(_operation)

    async def finish_child_session(
        self,
        *,
        owner_id: str,
        run_id: str,
        child_session_id: str,
        status: str,
        summary: str,
        outcome: Mapping[str, Any],
        worker_id: str,
        fencing_epoch: int,
        usage: Mapping[str, int] | None = None,
        child_fencing_epoch: int | None = None,
    ) -> bool:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        child_uuid = parse_run_id(child_session_id)
        if run_uuid is None or child_uuid is None:
            raise ValueError("child session ids must be canonical UUIDs")

        async def _operation(conn: Any) -> bool:
            async with conn.transaction():
                held = await conn.fetchval(
                    _HOLD_RUN_LEASE, owner, run_uuid, worker_id, fencing_epoch
                )
                if held is None:
                    return False
                child = await conn.fetchrow(_LOCK_CHILD_SESSION, owner, run_uuid, child_uuid)
                if child is None or str(child["status"]) != "running":
                    return False
                if child_fencing_epoch is not None and (
                    str(child["lease_owner"] or "") != worker_id
                    or int(child["fencing_epoch"]) != child_fencing_epoch
                ):
                    return False
                outcome_payload = dict(outcome)
                operation_id = child["operation_id"] or parse_run_id(
                    str(outcome_payload.get("operation_id") or "")
                )
                if operation_id is None:
                    operation_id = uuid.UUID(
                        OperationId.deterministic(
                            idempotency_key=f"legacy-child:{child_session_id}"
                        ).value
                    )
                supplied_operation_id = str(outcome_payload.get("operation_id") or "")
                if (
                    child["operation_id"] is not None
                    and supplied_operation_id
                    and supplied_operation_id != str(child["operation_id"])
                ):
                    return False
                outcome_payload["operation_id"] = str(operation_id)
                outcome_json = json.dumps(outcome_payload, ensure_ascii=False, sort_keys=True)
                tag = await conn.execute(
                    _FINISH_CHILD_SESSION,
                    owner,
                    run_uuid,
                    child_uuid,
                    status,
                    summary,
                    json.dumps(dict(usage)) if usage is not None else None,
                    outcome_json,
                    str(operation_id),
                )
                if str(tag).endswith(" 0"):
                    return False
                if child["operation_id"] is not None:
                    await conn.execute(
                        _FINISH_CHILD_OPERATION,
                        owner,
                        run_uuid,
                        child_uuid,
                        operation_id,
                        status,
                        summary,
                        json.dumps(dict(usage)) if usage is not None else None,
                        outcome_json,
                    )
                await conn.execute(_RETIRE_CHILD_GUIDANCE, owner, run_uuid, child_uuid)
                await conn.execute(
                    "SELECT pg_notify('dlightrag_run_activity', $1)",
                    cancellation_notify_key(owner_id=owner, run_id=str(run_uuid)),
                )
                return True

        return await self._run_write(_operation)

    async def bind_child_parent_intent(
        self,
        *,
        owner_id: str,
        run_id: str,
        child_session_id: str,
        parent_intent_id: str,
        worker_id: str,
        fencing_epoch: int,
    ) -> bool:
        owner = _require_owner(owner_id)
        run_uuid = parse_run_id(run_id)
        child_uuid = parse_run_id(child_session_id)
        intent_uuid = parse_run_id(parent_intent_id)
        if run_uuid is None or child_uuid is None or intent_uuid is None:
            raise ValueError("child session ids must be canonical UUIDs")

        async def _operation(conn: Any) -> bool:
            async with conn.transaction():
                held = await conn.fetchval(
                    _HOLD_RUN_LEASE, owner, run_uuid, worker_id, fencing_epoch
                )
                if held is None:
                    return False
                tag = await conn.execute(
                    _BIND_CHILD_PARENT_INTENT, owner, run_uuid, child_uuid, intent_uuid
                )
                return not str(tag).endswith(" 0")

        return await self._run_write(_operation)
