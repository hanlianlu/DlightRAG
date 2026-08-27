# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Fold canonical session entries into one derived working context.

Durable Research rebuilds this projection from the selected session head before
every provider call. In-process callers without a Repository may append exchanges
to the same bounded projection directly.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

from dlightrag.engine.agent.session.entries import (
    AssistantMessageEntry,
    CompactionEntry,
    ControlMessageEntry,
    SessionEntry,
    ToolResultMessageEntry,
    UserMessageEntry,
)
from dlightrag.engine.agent.session.projection import render_compaction_summary
from dlightrag.engine.agent.tool_content import tool_content_message_fields
from dlightrag.engine.ai.messages import ToolCall
from dlightrag.engine.ai.tokens import estimate_messages_tokens


class PriorTurns:
    """Earlier caller turns plus a bounded continuation for omitted pairs."""

    def __init__(
        self,
        messages: list[dict[str, Any]] | None = None,
        *,
        episodic_summary: str = "",
    ) -> None:
        self._messages = list(messages or [])
        self._episodic_summary = episodic_summary.strip()

    def __len__(self) -> int:
        return len(self._messages)

    @property
    def messages(self) -> list[dict[str, Any]]:
        return list(self._messages)

    @property
    def episodic_summary(self) -> str:
        return self._episodic_summary


def fold_tool_call(call: ToolCall) -> dict[str, Any]:
    """Project one provider-neutral tool call to its model-message shape."""
    message: dict[str, Any] = {
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": json.dumps(
                call.arguments,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    }
    if call.thought_signature is not None:
        message["thought_signature"] = call.thought_signature
    return message


def fold_assistant_message(entry: AssistantMessageEntry) -> dict[str, Any]:
    """Project one complete assistant entry to its model-message shape.

    ``tool_calls`` is omitted when empty: OpenAI-compatible endpoints reject
    empty tool-call arrays, and no provider requires them.
    """
    message: dict[str, Any] = {
        "role": "assistant",
        "content": entry.content,
    }
    if entry.tool_calls:
        message["tool_calls"] = [fold_tool_call(call) for call in entry.tool_calls]
    if entry.provider_state is not None:
        message["provider_state"] = entry.provider_state
    return message


def fold_tool_message(entry: ToolResultMessageEntry) -> dict[str, Any]:
    """Project one effect result entry to its model-message shape."""
    return {
        "role": "tool",
        "tool_call_id": entry.result.call_id,
        "name": entry.result.tool_name,
        **tool_content_message_fields(entry.result.parts),
        "is_error": entry.result.outcome != "succeeded",
    }


def fold_entries(entries: Sequence[SessionEntry]) -> list[dict[str, Any]]:
    """Fold ordered non-projection entries into model-context messages.

    Compaction entries are audit facts, not chronological messages. The active
    projection is materialized once by ``project_session_messages`` before its
    retained suffix.
    """
    messages: list[dict[str, Any]] = []
    for entry in entries:
        if isinstance(entry, UserMessageEntry):
            messages.append({"role": "user", "content": entry.content})
        elif isinstance(entry, AssistantMessageEntry):
            messages.append(fold_assistant_message(entry))
        elif isinstance(entry, ToolResultMessageEntry):
            messages.append(fold_tool_message(entry))
        elif isinstance(entry, ControlMessageEntry):
            messages.append({"role": "user", "content": entry.content})
        elif isinstance(entry, CompactionEntry):
            continue
    return messages


def project_session_messages(
    entries: Sequence[SessionEntry],
    projection: object | None,
) -> list[dict[str, Any]]:
    """Materialize one active summary before its retained non-compaction suffix."""
    if projection is None:
        return fold_entries(entries)
    from dlightrag.engine.agent.session.projection import (
        ContextProjection,
        projection_source_digest,
    )

    if not isinstance(projection, ContextProjection):
        raise TypeError("active projection must be a ContextProjection")
    branch_entries = [entry for entry in entries if not isinstance(entry, CompactionEntry)]
    if projection.covered_through_entry_id is not None:
        covered_index = next(
            (
                index
                for index, entry in enumerate(branch_entries)
                if entry.entry_id == projection.covered_through_entry_id
            ),
            None,
        )
        if covered_index is None:
            raise ValueError("active projection does not belong to this branch")
        digest = projection_source_digest(
            [entry.entry_id for entry in branch_entries[: covered_index + 1]]
        )
        if digest != projection.source_digest:
            raise ValueError("active projection source branch digest changed")
        if projection.first_retained_entry_id is None:
            retained = branch_entries[covered_index + 1 :]
        else:
            retained_index = next(
                (
                    index
                    for index, entry in enumerate(branch_entries)
                    if entry.entry_id == projection.first_retained_entry_id
                ),
                None,
            )
            if retained_index is None or retained_index <= covered_index:
                raise ValueError("active projection retained Head is not on this branch")
            retained = branch_entries[retained_index:]
    else:
        retained = [
            entry
            for entry in branch_entries
            if entry.sequence >= projection.first_retained_sequence
        ]
    messages: list[dict[str, Any]] = []
    if projection.summary is not None:
        messages.append(
            {
                "role": "user",
                "content": render_compaction_summary(projection.summary),
            }
        )
    messages.extend(fold_entries(retained))
    return messages


def exchange_starts(entries: Sequence[SessionEntry]) -> tuple[int, ...]:
    """Return entry indexes that start a complete assistant/tool exchange.

    An exchange starts at an assistant entry that carries tool calls and ends
    after the effect-result entries that answer those calls. Validation-result
    entries without an intent still belong to their preceding exchange.
    """
    starts: list[int] = []
    open_calls = 0
    for index, entry in enumerate(entries):
        if isinstance(entry, AssistantMessageEntry):
            if entry.tool_calls:
                if open_calls == 0:
                    starts.append(index)
                open_calls += len(entry.tool_calls)
        elif isinstance(entry, ToolResultMessageEntry):
            open_calls = max(0, open_calls - 1)
    return tuple(starts)


def select_compaction_boundary(
    entries: Sequence[SessionEntry],
    *,
    retained_tail_tokens: int,
) -> int:
    """Return the first retained entry index targeting the tail budget.

    Walks exchanges newest-first and keeps whole exchanges: an assistant's tool
    calls are never split from their results. When the budget cannot keep any
    complete exchange, only the single newest exchange is retained, and when the
    tail already fits the budget the boundary is the first entry.
    """
    if retained_tail_tokens < 0:
        raise ValueError("retained_tail_tokens cannot be negative")
    starts = exchange_starts(entries)
    if not starts:
        return 0
    boundaries = (*starts, len(entries))
    retained_start = starts[-1]
    newest = entries[starts[-1] :]
    remaining = retained_tail_tokens - estimate_messages_tokens(fold_entries(newest))
    if remaining < 0:
        return starts[-1]
    for position in reversed(range(len(starts) - 1)):
        exchange = entries[boundaries[position] : boundaries[position + 1]]
        remaining -= estimate_messages_tokens(fold_entries(exchange))
        if remaining < 0:
            break
        retained_start = starts[position]
    return retained_start


def _without_reasoning(message: dict[str, Any]) -> dict[str, Any]:
    if message.get("role") != "assistant":
        return message
    reduced = {key: value for key, value in message.items() if key != "provider_state"}
    calls = cast(list[dict[str, Any]], reduced.get("tool_calls") or [])
    if calls:
        reduced["tool_calls"] = [
            {key: value for key, value in call.items() if key != "thought_signature"}
            for call in calls
        ]
    return reduced


class WorkingContextProjection:
    """Every assistant/tool exchange one session produced, newest first to replay.

    Provider-native reasoning is what makes an exchange expensive and is valid
    only as an unmodified replay, so the policy-sized recent tail carries it and
    older exchanges keep just the call and its result: a later turn still sees
    which angle was spent without paying for the thinking behind it.

    In durable Research it is only a projection cache rebuilt from the active
    session graph before each provider call. In-process callers may append to it
    directly because they have no durable Session Repository.
    """

    def __init__(self, *, retained_tail_tokens: int) -> None:
        if retained_tail_tokens < 0:
            raise ValueError("retained_tail_tokens cannot be negative")
        self._retained_tail_tokens = retained_tail_tokens
        self._exchanges: list[list[dict[str, Any]]] = []

    @property
    def retained_tail_tokens(self) -> int:
        return self._retained_tail_tokens

    def record(self, exchange: list[dict[str, Any]]) -> None:
        self._exchanges.append(exchange)

    def canonical_json(self) -> dict[str, Any]:
        """Return every exchange, provider-native state included, in order."""
        return {"exchanges": [[dict(message) for message in ex] for ex in self._exchanges]}

    @classmethod
    def from_canonical_json(
        cls,
        state: Mapping[str, Any],
        *,
        retained_tail_tokens: int,
    ) -> WorkingContextProjection:
        """Rebuild a derived working projection from canonical exchanges."""
        exchanges = state.get("exchanges")
        if not isinstance(exchanges, Sequence):
            raise ValueError("working projection state has no exchanges")
        projection = cls(retained_tail_tokens=retained_tail_tokens)
        projection._exchanges = [
            [dict(cast(Mapping[str, Any], message)) for message in cast(Sequence[Any], exchange)]
            for exchange in exchanges
        ]
        return projection

    def messages(self) -> list[dict[str, Any]]:
        if not self._exchanges:
            return []
        newest = len(self._exchanges) - 1
        replay_from = newest
        budget = self._retained_tail_tokens - estimate_messages_tokens(self._exchanges[newest])
        for index in reversed(range(newest)):
            budget -= estimate_messages_tokens(self._exchanges[index])
            if budget < 0:
                break
            replay_from = index
        messages: list[dict[str, Any]] = []
        for index, exchange in enumerate(self._exchanges):
            if index >= replay_from:
                messages.extend(exchange)
            else:
                messages.extend(_without_reasoning(message) for message in exchange)
        return messages


__all__ = [
    "PriorTurns",
    "WorkingContextProjection",
    "exchange_starts",
    "fold_assistant_message",
    "fold_entries",
    "fold_tool_call",
    "fold_tool_message",
    "project_session_messages",
    "select_compaction_boundary",
]
