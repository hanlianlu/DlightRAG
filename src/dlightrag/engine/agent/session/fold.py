# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Fold canonical session entries into one derived working context.

Durable Research rebuilds this projection from the selected session head before
every provider call. In-process callers without a Repository may append exchanges
to the same bounded projection directly.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Any, cast

from dlightrag.engine.agent.session.entries import (
    AssistantMessageEntry,
    CompactionEntry,
    ControlMessageEntry,
    SessionEntry,
    ToolResultMessageEntry,
    UserMessageEntry,
)
from dlightrag.engine.agent.session.ids import EntryId
from dlightrag.engine.agent.session.projection import ContextProjection, render_compaction_summary
from dlightrag.engine.agent.tool_content import tool_content_message_fields
from dlightrag.engine.ai.messages import tool_call_message as fold_tool_call
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
        message["provider_state"] = _sorted_keys(entry.provider_state)
    return message


def _sorted_keys(value: Any) -> Any:
    """The same value with every mapping's keys sorted and every list kept in order.

    A Session read back from PostgreSQL returns provider state, such as reasoning
    details or thinking blocks, in jsonb's key order rather than the order the Run
    that made the turn still holds, so a replayed turn is the same bytes either way,
    as ``tool_call_message`` makes a call's arguments.
    """
    if isinstance(value, Mapping):
        return {key: _sorted_keys(value[key]) for key in sorted(value)}
    if isinstance(value, list | tuple):
        return [_sorted_keys(item) for item in value]
    return value


def fold_tool_message(entry: ToolResultMessageEntry) -> dict[str, Any]:
    """Project one effect result entry to its model-message shape."""
    return {
        "role": "tool",
        "tool_call_id": entry.result.call_id,
        "name": entry.result.tool_name,
        **tool_content_message_fields(entry.result.parts),
        "is_error": entry.result.outcome != "succeeded",
    }


def fold_entries(
    entries: Sequence[SessionEntry],
    *,
    included_incomplete_host_user_entry_ids: Collection[EntryId] = (),
) -> list[dict[str, Any]]:
    """Fold ordered non-projection entries into model-context messages.

    Compaction entries are audit facts, not chronological messages. The active
    projection is materialized once by ``project_session_messages`` before its
    retained suffix. Fast Host user entries carry an ``acceptance_id``; only a
    matching Assistant makes that turn model history. A Fast fold names the exact
    unanswered turns it keeps anyway: a failed or cancelled turn it continues, and
    the current reserved User Entry, which it removes again before serializing the
    separately supplied query.
    """
    completed_host_turns = {
        entry.acceptance_id
        for entry in entries
        if isinstance(entry, AssistantMessageEntry) and entry.acceptance_id is not None
    }
    messages: list[dict[str, Any]] = []
    for entry in entries:
        if isinstance(entry, UserMessageEntry):
            if (
                entry.acceptance_id is not None
                and entry.acceptance_id not in completed_host_turns
                and entry.entry_id not in included_incomplete_host_user_entry_ids
            ):
                continue
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
    *,
    included_incomplete_host_user_entry_ids: Collection[EntryId] = (),
    re_readable_handles: bool = True,
) -> list[dict[str, Any]]:
    """Materialize one active summary before its retained non-compaction suffix."""
    retained = retained_session_entries(entries, projection)
    messages: list[dict[str, Any]] = []
    if isinstance(projection, ContextProjection) and projection.summary is not None:
        messages.append(
            {
                "role": "user",
                "content": render_compaction_summary(
                    projection.summary, re_readable_handles=re_readable_handles
                ),
            }
        )
    messages.extend(
        fold_entries(
            retained,
            included_incomplete_host_user_entry_ids=included_incomplete_host_user_entry_ids,
        )
    )
    return messages


def conversation_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """What was asked and what was answered, without the work in between.

    A Research turn's projection also carries each tool call and result, and the
    provider state behind its answer. A call that only continues the conversation,
    as routing and Fast do, is handed none of it: hundreds of kilobytes it does not
    need, and unfinished work a model may take up instead of its own task.

    The images its tools viewed are what the answer saw, so they stay: as durable
    attachments of the latest user message before them — the question, or a steer,
    the turn was answering — which the Run hydrates and a provider shows, each under
    its own name, as images of that message. Content stays as written, so a reader of
    words alone sees no difference.
    """
    conversation: list[dict[str, Any]] = []
    viewed: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "tool":
            viewed.extend(dict(attachment) for attachment in message.get("attachments") or ())
            continue
        if role not in {"user", "assistant"} or message.get("tool_calls"):
            continue
        _attach_viewed(conversation, viewed)
        if message.get("content"):
            conversation.append({"role": role, "content": message["content"]})
    _attach_viewed(conversation, viewed)
    return conversation


def _attach_viewed(conversation: list[dict[str, Any]], viewed: list[dict[str, Any]]) -> None:
    """Move the images a turn's tools viewed onto the user message that turn answered.

    A history that starts after its question, which compaction never leaves, has no
    message to carry them, and they stay out rather than arrive in an empty turn.
    """
    question = next(
        (message for message in reversed(conversation) if message["role"] == "user"), None
    )
    if question is not None and viewed:
        question.setdefault("attachments", []).extend(viewed)
    viewed.clear()


def retained_session_entries(
    entries: Sequence[SessionEntry], projection: object | None
) -> Sequence[SessionEntry]:
    """Select the exact immutable suffix used by the model-facing projection."""
    if projection is None:
        return entries
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
    return retained


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


def host_turn_starts(entries: Sequence[SessionEntry]) -> tuple[int, ...]:
    """Return direct Host conversation-turn starts for Fast compaction."""
    return tuple(
        index
        for index, entry in enumerate(entries)
        if isinstance(entry, UserMessageEntry | ControlMessageEntry)
    )


def select_compaction_boundary(
    entries: Sequence[SessionEntry],
    *,
    retained_tail_tokens: int,
    starts: Sequence[int] | None = None,
) -> int:
    """Return the first retained entry index targeting the tail budget.

    Walks exchanges newest-first and keeps whole exchanges: an assistant's tool
    calls are never split from their results. When the budget cannot keep any
    complete exchange, only the single newest exchange is retained, and when the
    tail already fits the budget the boundary is the first entry.
    """
    if retained_tail_tokens < 0:
        raise ValueError("retained_tail_tokens cannot be negative")
    resolved_starts = tuple(starts) if starts is not None else exchange_starts(entries)
    if not resolved_starts:
        return 0
    boundaries = (*resolved_starts, len(entries))
    retained_start = resolved_starts[-1]
    newest = entries[resolved_starts[-1] :]
    remaining = retained_tail_tokens - estimate_messages_tokens(fold_entries(newest))
    if remaining < 0:
        return resolved_starts[-1]
    for position in reversed(range(len(resolved_starts) - 1)):
        exchange = entries[boundaries[position] : boundaries[position + 1]]
        remaining -= estimate_messages_tokens(fold_entries(exchange))
        if remaining < 0:
            break
        retained_start = resolved_starts[position]
    return retained_start


class WorkingContextProjection:
    """Every assistant/tool exchange one session produced, in order to replay.

    An exchange travels whole, provider-native state included. Reasoning is
    valid only as an unmodified replay: DeepSeek requires every previous turn's
    ``reasoning_content`` back on any request that carries tools and rejects a
    partial history with HTTP 400, and Gemini signs tool calls that stop
    verifying once filtered. Bounding the request is the compaction boundary's
    job, where whole exchanges are replaced by one summary; stripping state from
    an exchange that is still replayed produces a history no provider ever sent.

    In durable Research it is only a projection cache rebuilt from the active
    session graph before each provider call. In-process callers may append to it
    directly because they have no durable Session Repository.

    After a compaction the projection opens with the summary that replaced the
    covered prefix, kept apart from the exchanges so a reader knows where that
    prefix ended.
    """

    def __init__(self) -> None:
        self._summary: dict[str, Any] | None = None
        self._exchanges: list[list[dict[str, Any]]] = []

    @property
    def summary(self) -> dict[str, Any] | None:
        """The compaction summary this projection opens with, if one is active."""
        return self._summary

    def summarize(self, message: dict[str, Any]) -> None:
        """Open the projection with the summary that replaced its covered prefix."""
        self._summary = message

    def record(self, exchange: list[dict[str, Any]]) -> None:
        self._exchanges.append(exchange)

    def canonical_json(self) -> dict[str, Any]:
        """Return the summary and every exchange, provider-native state included."""
        state: dict[str, Any] = {
            "exchanges": [[dict(message) for message in ex] for ex in self._exchanges]
        }
        if self._summary is not None:
            state["summary"] = dict(self._summary)
        return state

    @classmethod
    def from_canonical_json(cls, state: Mapping[str, Any]) -> WorkingContextProjection:
        """Rebuild a derived working projection from canonical exchanges."""
        exchanges = state.get("exchanges")
        if not isinstance(exchanges, Sequence):
            raise ValueError("working projection state has no exchanges")
        projection = cls()
        summary = state.get("summary")
        if isinstance(summary, Mapping):
            projection._summary = dict(cast(Mapping[str, Any], summary))
        projection._exchanges = [
            [dict(cast(Mapping[str, Any], message)) for message in cast(Sequence[Any], exchange)]
            for exchange in exchanges
        ]
        return projection

    def messages(self) -> list[dict[str, Any]]:
        """Return the summary and every exchange verbatim, provider-native state included."""
        messages: list[dict[str, Any]] = [] if self._summary is None else [self._summary]
        for exchange in self._exchanges:
            messages.extend(exchange)
        return messages


__all__ = [
    "PriorTurns",
    "WorkingContextProjection",
    "exchange_starts",
    "fold_assistant_message",
    "fold_entries",
    "fold_tool_call",
    "fold_tool_message",
    "host_turn_starts",
    "project_session_messages",
    "select_compaction_boundary",
]
