# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Fold session entries into model context, plus working-memory episode replay.

One pure fold reconstructs the active model context for live execution and for
replay; derived model messages are never persisted separately. The episode
record keeps the pre-journal working exchanges for the M2-era orchestrator and
replays them with the same retained-tail policy the old in-memory episode used,
so live behavior does not change until the journal-based orchestration lands.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

from dlightrag_ai.messages import ToolCall
from dlightrag_ai.tokens import (
    estimate_messages_tokens,
    estimate_tokens,
    truncate_to_estimated_tokens,
)

from dlightrag_agent.session.entries import (
    AssistantMessageEntry,
    CompactionEntry,
    ContextInjectionEntry,
    EffectResultEntry,
    ProfileFactEntry,
    SessionEntry,
    SessionTerminalEntry,
    UserMessageEntry,
)
from dlightrag_agent.session.projection import render_compaction_summary


class PriorTurns:
    """Earlier caller-supplied turns, replayed once per request."""

    def __init__(self, messages: list[dict[str, Any]] | None = None) -> None:
        self._messages = list(messages or [])

    def __len__(self) -> int:
        return len(self._messages)

    @property
    def messages(self) -> list[dict[str, Any]]:
        return list(self._messages)


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
    """Project one complete assistant entry to its model-message shape."""
    message: dict[str, Any] = {
        "role": "assistant",
        "content": entry.content,
        "tool_calls": [fold_tool_call(call) for call in entry.tool_calls],
    }
    if entry.provider_state is not None:
        message["provider_state"] = entry.provider_state
    return message


def fold_tool_message(entry: EffectResultEntry) -> dict[str, Any]:
    """Project one effect result entry to its model-message shape."""
    return {
        "role": "tool",
        "tool_call_id": entry.result.call_id,
        "name": entry.result.tool_name,
        "content": entry.result.content,
        "is_error": entry.result.outcome != "succeeded",
    }


def fold_entries(entries: Sequence[SessionEntry]) -> list[dict[str, Any]]:
    """Fold ordered journal entries into the active model-context messages.

    Accounting entries (profile facts, session terminal) produce no model
    message. A compaction entry renders its typed summary deterministically at
    its sequence position, so live and replay folds are byte-for-byte equal.
    """
    messages: list[dict[str, Any]] = []
    for entry in entries:
        if isinstance(entry, UserMessageEntry):
            messages.append({"role": "user", "content": entry.content})
        elif isinstance(entry, AssistantMessageEntry):
            messages.append(fold_assistant_message(entry))
        elif isinstance(entry, EffectResultEntry):
            messages.append(fold_tool_message(entry))
        elif isinstance(entry, ContextInjectionEntry):
            messages.append({"role": "user", "content": entry.content})
        elif isinstance(entry, CompactionEntry):
            messages.append(
                {
                    "role": "user",
                    "content": render_compaction_summary(entry.summary),
                }
            )
        elif isinstance(entry, (ProfileFactEntry, SessionTerminalEntry)):
            continue
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
        elif isinstance(entry, EffectResultEntry):
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


def head_tail_text(text: str, *, head_tokens: int, tail_tokens: int) -> str:
    """Bound one individually oversized body to a head and a tail.

    Durable evidence/resource/journal handles are appended by the caller after
    this cut, so a truncated body still keeps its retrievable provenance.
    """
    if head_tokens <= 0 or tail_tokens <= 0:
        raise ValueError("head and tail token budgets must be positive")
    if estimate_tokens(text) <= head_tokens + tail_tokens:
        return text
    head = truncate_to_estimated_tokens(text, head_tokens)
    tail_cut = _tail_prefix_within(text, tail_tokens)
    if not head.strip() and not tail_cut.strip():
        return ""
    if head and tail_cut:
        return f"{head}\n…\n{tail_cut}"
    return head or tail_cut


def _tail_prefix_within(text: str, tail_tokens: int) -> str:
    """Return the longest suffix of text within the estimator token budget."""
    if tail_tokens <= 0:
        return ""
    low = 0
    high = min(len(text), tail_tokens * 4 + 16)
    while low < high:
        midpoint = (low + high + 1) // 2
        suffix = text[-midpoint:]
        if estimate_tokens(suffix) <= tail_tokens:
            low = midpoint
        else:
            high = midpoint - 1
    return text[-low:] if low else ""


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


class SessionEpisode:
    """Every assistant/tool exchange one session produced, newest first to replay.

    Provider-native reasoning is what makes an exchange expensive and is valid
    only as an unmodified replay, so the policy-sized recent tail carries it and
    older exchanges keep just the call and its result: a later turn still sees
    which angle was spent without paying for the thinking behind it.

    The episode is pre-journal working memory; its canonical codec carries the
    same exchanges until journal-based orchestration replaces it.
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
    ) -> SessionEpisode:
        """Rebuild an episode from its canonical exchanges."""
        exchanges = state.get("exchanges")
        if not isinstance(exchanges, Sequence):
            raise ValueError("episode state has no exchanges")
        episode = cls(retained_tail_tokens=retained_tail_tokens)
        episode._exchanges = [
            [dict(cast(Mapping[str, Any], message)) for message in cast(Sequence[Any], exchange)]
            for exchange in exchanges
        ]
        return episode

    @property
    def last_exchange(self) -> list[dict[str, Any]]:
        return list(self._exchanges[-1]) if self._exchanges else []

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
    "SessionEpisode",
    "exchange_starts",
    "fold_assistant_message",
    "fold_entries",
    "fold_tool_call",
    "fold_tool_message",
    "head_tail_text",
    "select_compaction_boundary",
]
