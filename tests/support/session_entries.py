# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Session entries as a Run's writer states them, for tests that read what an agent ended on.

The payloads are the ones the Engine writes: an assistant message with its text and its tool
calls, a tool result, a question. A reader that looks inside an entry meets what it meets in
production, and each entry names the Run that wrote it, as the writer does.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from typing import Any

#: An entry's type and its payload.
Entry = tuple[str, dict[str, Any]]


def question(text: str = "a question") -> Entry:
    return "user_message", {"content": text}


def tool_use(text: str = "") -> Entry:
    """An assistant message that calls a tool: the agent has more to do."""
    return "assistant_message", {
        "content": text,
        "reasoning": "",
        "stop_reason": "tool_use",
        "tool_calls": [{"id": "call-1", "name": "search", "arguments": {}, "argument_error": None}],
    }


def tool_result() -> Entry:
    return "tool_result", {
        "tool_name": "search",
        "call_id": "call-1",
        "outcome": "succeeded",
        "content": [{"type": "text", "text": "a result"}],
    }


def answer(text: str = "The answer.") -> Entry:
    """An assistant message that calls no tool: the agent has said what it had to say."""
    return "assistant_message", {
        "content": text,
        "reasoning": "",
        "stop_reason": "stop",
        "tool_calls": [],
    }


async def write_entries(
    pool: Any, *, owner: str, session: str, run_id: str, entries: Sequence[Entry]
) -> list[int]:
    """Append ``entries`` to ``session`` as ``run_id`` wrote them, and return each one's sequence.

    The sequences continue the Session's own numbering, so a conversation's Session can be written
    Run after Run, and a Child's Session from its first entry.
    """
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute(
            "INSERT INTO dlightrag_agent_sessions (owner_id, session_id, lease_run_id, "
            "fencing_epoch) VALUES ($1, $2, $3, 1) ON CONFLICT DO NOTHING",
            owner,
            uuid.UUID(session),
            uuid.UUID(run_id),
        )
        last = await conn.fetchval(
            "SELECT COALESCE(max(sequence), 0) FROM dlightrag_agent_session_entries "
            "WHERE owner_id = $1 AND session_id = $2",
            owner,
            uuid.UUID(session),
        )
        sequences = [int(last) + number for number in range(1, len(entries) + 1)]
        await conn.executemany(
            "INSERT INTO dlightrag_agent_session_entries (owner_id, session_id, sequence, "
            "entry_id, entry_type, schema_version, timestamp, payload_json, search_text, run_id) "
            "VALUES ($1, $2, $3, $4, $5, 2, NOW(), $6::jsonb, '', $7)",
            [
                (
                    owner,
                    uuid.UUID(session),
                    sequence,
                    uuid.uuid4(),
                    entry_type,
                    json.dumps(payload),
                    uuid.UUID(run_id),
                )
                for sequence, (entry_type, payload) in zip(sequences, entries, strict=True)
            ],
        )
    return sequences
