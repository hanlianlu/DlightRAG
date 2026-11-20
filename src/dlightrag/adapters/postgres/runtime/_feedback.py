# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Answer feedback persistence mixed into PGRunStore."""

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from dlightrag.engine.answer.feedback import MAIN_AGENT, AnswerFeedback, Judgement
from dlightrag.engine.runtime.records import parse_run_id

# Keeps one judgement, and only of a message there is to judge: an assistant message the Run itself
# wrote, in the Session of the agent the judgement names - the Run's main Session for `main`, or
# one of the Run's own Children. The check is part of the statement, so a message that does not
# exist is never kept and nothing can change between the check and the write. The Run's row is
# held while the judgement is written: a Run deleted at that moment is waited for and then reads as
# unknown, instead of failing the foreign key. A judgement already kept is replaced and keeps the
# time it was first given: that column is the training record's, and nothing here reads it back.
# $8 is the Child's id, and NULL for the Run's own agent.
_UPSERT_FEEDBACK = """
INSERT INTO dlightrag_answer_feedback (owner_id, run_id, agent, sequence, value, reasons, note)
SELECT run.owner_id, run.run_id, $3::text, $4::bigint, $5::text, $6::text[], $7::text
FROM dlightrag_runs AS run
WHERE run.owner_id = $1 AND run.run_id = $2
  AND EXISTS (
      SELECT 1
      FROM dlightrag_agent_session_entries AS entry
      WHERE entry.owner_id = run.owner_id AND entry.run_id = run.run_id
        AND entry.sequence = $4 AND entry.entry_type = 'assistant_message'
        AND entry.session_id IN (
            SELECT routing.agent_session_id
            FROM dlightrag_answer_run_routing AS routing
            WHERE routing.owner_id = run.owner_id AND routing.run_id = run.run_id
              AND $8::uuid IS NULL
            UNION ALL
            SELECT child.child_session_id
            FROM dlightrag_answer_child_sessions AS child
            WHERE child.owner_id = run.owner_id AND child.run_id = run.run_id
              AND child.child_session_id = $8::uuid
        )
  )
FOR KEY SHARE OF run
ON CONFLICT (owner_id, run_id, agent, sequence) DO UPDATE
SET value = EXCLUDED.value,
    reasons = EXCLUDED.reasons,
    note = EXCLUDED.note,
    updated_at = NOW()
RETURNING agent, sequence, value, reasons, note, updated_at
"""

_DELETE_FEEDBACK = """
DELETE FROM dlightrag_answer_feedback
WHERE owner_id = $1 AND run_id = $2 AND agent = $3 AND sequence = $4
"""

# The Run's own agent first, then each Child grouped, oldest message first within an agent: the
# order the transcript search answers in. Messages of different Sessions are numbered apart, so
# their sequences are not comparable with one another.
_SELECT_FEEDBACK = """
SELECT agent, sequence, value, reasons, note, updated_at
FROM dlightrag_answer_feedback
WHERE owner_id = $1 AND run_id = $2
ORDER BY agent <> 'main', agent, sequence
"""


def _feedback_row(row: Any) -> AnswerFeedback:
    return AnswerFeedback(
        agent=str(row["agent"]),
        sequence=int(row["sequence"]),
        value=str(row["value"]),
        reasons=tuple(str(reason) for reason in row["reasons"]),
        note=row["note"],
        updated_at=row["updated_at"],
    )


def answer_message_column(*, alias: str, entries: str, owner: str, run: str, agent: str) -> str:
    """A select-list column stating what one agent answered and how its owner judged it.

    An agent's answer is the message it ended on: its last assistant message, when that carries
    no tool calls and says something. The column is ``ROW(sequence, value, reasons, note)`` for
    that message, the last three NULL while its owner has not judged it, and NULL where the agent
    has not answered; ``read_answer_message`` reads it. Only the last assistant message is read.
    Looking further back for an earlier answer reads every message of an agent that is still
    working, which a roster of running Children would pay for on every read.

    ``entries`` is the condition that makes ``entry`` the agent's own Session entries. The caller
    states it, because only the caller knows which index fits: the Run's entries for the main
    agent, whose Session holds every Run of its conversation, and the Session itself for a Child,
    whose Session holds nothing but its Run's. ``owner``, ``run`` and ``agent`` name the
    judgement's key, each as an expression the statement can evaluate.
    """
    return rf"""(
    SELECT ROW(answer.sequence, feedback.value, feedback.reasons, feedback.note)
    FROM (
        SELECT latest.sequence
        FROM (
            SELECT entry.sequence, entry.payload_json
            FROM dlightrag_agent_session_entries AS entry
            WHERE {entries} AND entry.entry_type = 'assistant_message'
            ORDER BY entry.sequence DESC
            LIMIT 1
        ) AS latest
        WHERE latest.payload_json->'tool_calls' = '[]'::jsonb
          AND latest.payload_json->>'content' ~ '\S'
    ) AS answer
    LEFT JOIN dlightrag_answer_feedback AS feedback
      ON feedback.owner_id = {owner} AND feedback.run_id = {run}
     AND feedback.agent = {agent} AND feedback.sequence = answer.sequence
) AS {alias}"""  # noqa: S608 - interpolates only the SQL its callers write


def read_answer_message(column: Sequence[Any] | None) -> tuple[int | None, Judgement | None]:
    """The sequence of the answer message ``answer_message_column`` stated, and the judgement of it.

    Both are None where the agent has not answered, and the judgement is None where its owner has
    not judged the answer. A judgement is always read whole: a vote with its reasons and its note.
    """
    if column is None:
        return None, None
    sequence, value, reasons, note = column
    judgement = None if value is None else Judgement(str(value), tuple(map(str, reasons)), note)
    return int(sequence), judgement


class FeedbackRunStoreMixin:
    """Answer feedback operations on PGRunStore."""

    _run_read: Callable[..., Awaitable[Any]]
    _run_write: Callable[..., Awaitable[Any]]

    async def put_answer_feedback(
        self,
        *,
        owner_id: str,
        run_id: str,
        agent: str,
        sequence: int,
        value: str,
        reasons: Sequence[str],
        note: str | None,
    ) -> AnswerFeedback | None:
        """Keep one judgement of one message, replacing an earlier one; None if it has no message.

        None says the Run, the agent, or the message is not there to judge: unknown, another
        owner's, or written by another Run. An agent is `main` or a Child Session's id in its
        canonical spelling, because the spelling is part of the key and any other names no message.
        """
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return None
        child_uuid = None
        if agent != MAIN_AGENT:
            child_uuid = parse_run_id(agent)
            if child_uuid is None or str(child_uuid) != agent:
                return None

        async def _operation(conn: Any) -> AnswerFeedback | None:
            row = await conn.fetchrow(
                _UPSERT_FEEDBACK,
                owner_id,
                run_uuid,
                agent,
                sequence,
                value,
                list(reasons),
                note,
                child_uuid,
            )
            return None if row is None else _feedback_row(row)

        return await self._run_write(_operation)

    async def delete_answer_feedback(
        self, *, owner_id: str, run_id: str, agent: str, sequence: int
    ) -> None:
        """Take one judgement back. One never given is no error: it is gone either way."""
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return

        async def _operation(conn: Any) -> None:
            await conn.execute(_DELETE_FEEDBACK, owner_id, run_uuid, agent, sequence)

        await self._run_write(_operation)

    async def list_answer_feedback(
        self, *, owner_id: str, run_id: str
    ) -> tuple[AnswerFeedback, ...]:
        """Every judgement this owner gave the messages of one Run: the Run's own agent first,
        then each Child, and within an agent the oldest message first."""
        run_uuid = parse_run_id(run_id)
        if run_uuid is None:
            return ()

        async def _operation(conn: Any) -> tuple[AnswerFeedback, ...]:
            rows = await conn.fetch(_SELECT_FEEDBACK, owner_id, run_uuid)
            return tuple(_feedback_row(row) for row in rows)

        return await self._run_read(_operation)
