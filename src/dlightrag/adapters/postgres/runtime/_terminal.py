# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Private cancellation-aware PostgreSQL terminal transition primitive."""

import json
from collections.abc import Mapping
from typing import Any, Literal

from dlightrag.engine.runtime.records import TerminalOutcome

type TerminalStatus = Literal["succeeded", "failed", "cancelled"]

# One fenced terminal transition that also appends the run's single terminal
# event. Prepared input is cleared on every terminal transition.
_FINISH_RUN_SQL = """
WITH bumped AS (
    UPDATE dlightrag_runs
    SET status = $5::text,
        stop_reason = $6::text,
        result_json = $7::jsonb,
        error_kind = $8::text,
        error_message = $9::text,
        phase = NULL,
        prepared_input_json = NULL,
        lease_owner = NULL,
        lease_expires_at = NULL,
        finished_at = NOW(),
        purge_after = NOW() + make_interval(secs => retention_seconds::double precision),
        updated_at = NOW(),
        next_event_sequence = next_event_sequence + 1
    WHERE owner_id = $1 AND run_id = $2
      AND lease_owner = $3 AND fencing_epoch = $4
      AND status = 'running' AND lease_expires_at > NOW()
      AND (NOT $12::boolean OR cancel_requested_at IS NULL)
      AND (
          $5::text <> 'succeeded'
          OR NOT EXISTS (
              SELECT 1
              FROM dlightrag_answer_child_sessions AS child
              WHERE child.owner_id = $1 AND child.run_id = $2
                AND child.status = 'running'
          )
      )
    RETURNING next_event_sequence - 1 AS event_sequence
), cancelled_children AS (
    UPDATE dlightrag_answer_child_sessions AS child
    SET status = 'cancelled',
        cancel_requested_at = COALESCE(child.cancel_requested_at, NOW()),
        summary = 'Child session cancelled because its parent Run terminated.',
        usage_json = NULL,
        host_state_json = jsonb_set(
            child.host_state_json,
            '{terminal_outcome}',
            jsonb_build_object(
                'status', 'cancelled',
                'summary', 'Child session cancelled because its parent Run terminated.',
                'handles', jsonb_build_array(),
                'usage', jsonb_build_object(),
                'child_session_id', child.child_session_id::text,
                'operation_id', COALESCE((
                    SELECT operation.operation_id::text
                    FROM dlightrag_answer_child_operations AS operation
                    WHERE operation.owner_id = child.owner_id
                      AND operation.run_id = child.run_id
                      AND operation.child_session_id = child.child_session_id
                    ORDER BY operation.operation_sequence DESC
                    LIMIT 1
                ), ''),
                'evidence_state', NULL
            ),
            true
        ),
        lease_owner = NULL,
        lease_expires_at = NULL,
        updated_at = NOW()
    WHERE child.owner_id = $1 AND child.run_id = $2
      AND child.status = 'running'
      AND EXISTS (SELECT 1 FROM bumped)
    RETURNING child.owner_id, child.run_id, child.child_session_id
), cancelled_operations AS (
    UPDATE dlightrag_answer_child_operations AS operation
    SET status = 'cancelled', cancellation_origin = 'run', updated_at = NOW()
    WHERE (operation.owner_id, operation.run_id, operation.child_session_id) IN (
        SELECT owner_id, run_id, child_session_id FROM cancelled_children
    )
      AND operation.status = 'running'
    RETURNING operation.child_session_id
), retired_guidance AS (
    UPDATE dlightrag_answer_child_guidance AS guidance
    SET status = 'cancelled', updated_at = NOW()
    WHERE guidance.owner_id = $1 AND guidance.run_id = $2
      AND guidance.status = 'pending'
      AND EXISTS (SELECT 1 FROM bumped)
    RETURNING guidance.request_id
), inserted AS (
    INSERT INTO dlightrag_run_events (
        owner_id, run_id, event_sequence, event_type, payload
    )
    SELECT $1, $2, event_sequence, $10::text, $11::jsonb FROM bumped
    RETURNING event_sequence
)
SELECT event_sequence FROM inserted
"""

_SELECT_CANCELLATION = """
SELECT cancel_requested_at IS NOT NULL
FROM dlightrag_runs
WHERE owner_id = $1 AND run_id = $2
"""


async def finish_fenced_run(
    conn: Any,
    *,
    owner_id: str,
    run_id: Any,
    lease_owner: str,
    fencing_epoch: int,
    status: TerminalStatus,
    stop_reason: str | None,
    result: Mapping[str, object] | None,
    error_kind: str | None,
    error_message: str | None,
    event_type: str,
    payload: Mapping[str, Any],
    withhold_on_cancel: bool,
    cancel_requested: bool | None = None,
) -> TerminalOutcome:
    """Commit one terminal, allowing an earlier cancellation to win.

    ``cancel_requested`` may be supplied by a caller that already holds the run
    row. A caller without that lock leaves it unknown; after any withheld
    terminal outcome, this helper re-reads cancellation and attempts the same
    fenced primitive for the exact cancelled terminal.
    """

    # READ COMMITTED snapshots are taken before a statement waits on a lock.
    # Acquire the Run row first so the child predicate sees a continuation that
    # committed while we waited. All callers own an enclosing transaction.
    await conn.fetchval(
        "SELECT 1 FROM dlightrag_runs WHERE owner_id = $1 AND run_id = $2 FOR UPDATE",
        owner_id,
        run_id,
    )

    async def commit(
        terminal_status: TerminalStatus,
        *,
        terminal_stop_reason: str | None,
        terminal_result: Mapping[str, object] | None,
        terminal_error_kind: str | None,
        terminal_error_message: str | None,
        terminal_event_type: str,
        terminal_payload: Mapping[str, Any],
        withhold: bool,
    ) -> TerminalOutcome:
        sequence = await conn.fetchval(
            _FINISH_RUN_SQL,
            owner_id,
            run_id,
            lease_owner,
            fencing_epoch,
            terminal_status,
            terminal_stop_reason,
            json.dumps(terminal_result, ensure_ascii=False)
            if terminal_result is not None
            else None,
            terminal_error_kind,
            terminal_error_message,
            terminal_event_type,
            json.dumps(dict(terminal_payload), ensure_ascii=False),
            withhold,
        )
        if sequence is None:
            return TerminalOutcome(committed=False, status=None, event_sequence=None)
        return TerminalOutcome(
            committed=True,
            status=terminal_status,
            event_sequence=int(sequence),
        )

    if withhold_on_cancel and cancel_requested is True:
        return await commit(
            "cancelled",
            terminal_stop_reason=None,
            terminal_result=None,
            terminal_error_kind=None,
            terminal_error_message=None,
            terminal_event_type="done",
            terminal_payload={"status": "cancelled"},
            withhold=False,
        )

    outcome = await commit(
        status,
        terminal_stop_reason=stop_reason,
        terminal_result=result,
        terminal_error_kind=error_kind,
        terminal_error_message=error_message,
        terminal_event_type=event_type,
        terminal_payload=payload,
        withhold=withhold_on_cancel,
    )
    if outcome.committed or not withhold_on_cancel:
        return outcome

    if cancel_requested is None:
        cancel_requested = bool(await conn.fetchval(_SELECT_CANCELLATION, owner_id, run_id))
    if not cancel_requested:
        return outcome
    return await commit(
        "cancelled",
        terminal_stop_reason=None,
        terminal_result=None,
        terminal_error_kind=None,
        terminal_error_message=None,
        terminal_event_type="done",
        terminal_payload={"status": "cancelled"},
        withhold=False,
    )


__all__ = ["TerminalStatus", "finish_fenced_run"]
