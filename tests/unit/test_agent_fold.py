# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Semantic Entry fold and complete exchange boundaries."""

import json
from datetime import UTC, datetime

from dlightrag.engine.agent.session.effects import ToolResultEntry
from dlightrag.engine.agent.session.entries import (
    AssistantMessageEntry,
    CompactionEntry,
    ControlMessageEntry,
    ToolResultMessageEntry,
    UserMessageEntry,
)
from dlightrag.engine.agent.session.fold import (
    WorkingContextProjection,
    conversation_messages,
    exchange_starts,
    fold_assistant_message,
    fold_entries,
    host_turn_starts,
    project_session_messages,
    select_compaction_boundary,
)
from dlightrag.engine.agent.session.ids import EntryId, IntentId, ProjectionId, SessionId
from dlightrag.engine.agent.session.projection import ContextProjection, projection_source_digest
from dlightrag.engine.ai.messages import ToolCall


def _now():
    return datetime.now(UTC)


def _result(session_id: SessionId, call_id: str, source: int) -> ToolResultMessageEntry:
    return ToolResultMessageEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        timestamp=_now(),
        result=ToolResultEntry.text(
            tool_name="lookup", call_id=call_id, outcome="succeeded", text=f"result {source}"
        ),
        intent_id=IntentId.new(),
        source_index=source,
        contract_version=1,
        input_schema_digest="a" * 64,
        replay_policy="never",
        attempt_id=None,
        effective_input_digest="b" * 64,
    )


def test_a_replayed_turn_carries_its_provider_state_in_one_key_order() -> None:
    """A Run holds provider state in the provider's order, a reload in jsonb's order."""
    session_id, entry_id = SessionId.new(), EntryId.new()

    def turn(provider_state: dict[str, object]) -> AssistantMessageEntry:
        return AssistantMessageEntry(
            entry_id=entry_id,
            session_id=session_id,
            timestamp=datetime(2026, 9, 30, tzinfo=UTC),
            content="answer",
            stop_reason="stop",
            provider_state=provider_state,
        )

    in_memory = turn(
        {"reasoning_details": [{"type": "reasoning.text", "text": "t", "format": "x"}], "a": 1}
    )
    reloaded = turn(
        {"a": 1, "reasoning_details": [{"text": "t", "type": "reasoning.text", "format": "x"}]}
    )

    assert json.dumps(fold_assistant_message(in_memory)) == json.dumps(
        fold_assistant_message(reloaded)
    )


def test_conversation_messages_leave_out_the_work_between_turns() -> None:
    image_question = [
        {"type": "text", "text": "and this one?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
    ]
    messages = [
        {"role": "user", "content": "find me a video"},
        {
            "role": "assistant",
            "content": "Searching.",
            "tool_calls": [{"id": "call-1", "type": "function"}],
            "provider_state": {"opaque": 1},
        },
        {"role": "tool", "tool_call_id": "call-1", "name": "search_web", "content": "x" * 50_000},
        {"role": "assistant", "content": "Here are three videos.", "provider_state": {"opaque": 2}},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": image_question},
    ]

    assert conversation_messages(messages) == [
        {"role": "user", "content": "find me a video"},
        {"role": "assistant", "content": "Here are three videos."},
        {"role": "user", "content": image_question},
    ]


def test_conversation_messages_keep_what_a_turn_viewed_on_the_question_it_answered() -> None:
    """Once tool work left Fast's history, a Fast follow-up after a Research turn sent
    none of the pixels that turn viewed. They are what the earlier answer saw, so they
    stay, as attachments of the question it answered; the question's words do not
    change."""
    page = {"resource_id": "res-1", "media_type": "image/png", "content_digest": "a" * 64}
    overview = {**page, "resource_id": "res-2"}

    def view(call_id: str, attachment: dict[str, str]) -> list[dict[str, object]]:
        return [
            {"role": "assistant", "content": "", "tool_calls": [{"id": call_id}]},
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": "a page",
                "attachments": [attachment],
            },
        ]

    messages = [
        {"role": "user", "content": "what does page 3 show?"},
        *view("call-1", page),
        *view("call-2", overview),
        {"role": "assistant", "content": "A rising revenue chart."},
        {"role": "user", "content": "and page 4?"},
        *view("call-3", page),
    ]

    conversation = conversation_messages(messages)

    assert conversation == [
        {"role": "user", "content": "what does page 3 show?", "attachments": [page, overview]},
        {"role": "assistant", "content": "A rising revenue chart."},
        # A turn that stopped before its answer still keeps what it viewed.
        {"role": "user", "content": "and page 4?", "attachments": [page]},
    ]
    # The Run hydrates the view's own copies, not the projection it was read from.
    assert conversation[0]["attachments"][0] is not page


def test_viewed_images_join_the_latest_user_message_and_never_an_empty_one() -> None:
    page = {"resource_id": "res-1", "media_type": "image/png", "content_digest": "a" * 64}
    tool = {"role": "tool", "tool_call_id": "call-1", "content": "a page", "attachments": [page]}

    # Viewed after an answer's words, the images still join the message it answered.
    assert conversation_messages(
        [
            {"role": "user", "content": "what does page 3 show?"},
            {"role": "assistant", "content": "Looking."},
            tool,
        ]
    ) == [
        {"role": "user", "content": "what does page 3 show?", "attachments": [page]},
        {"role": "assistant", "content": "Looking."},
    ]
    # With no user message before them, no empty turn is made to carry them.
    assert conversation_messages([tool, {"role": "assistant", "content": "Done."}]) == [
        {"role": "assistant", "content": "Done."}
    ]


def test_fold_projects_only_conversation_semantics_in_source_order() -> None:
    session_id = SessionId.new()
    entries = (
        UserMessageEntry(
            entry_id=EntryId.new(), session_id=session_id, timestamp=_now(), content="question"
        ),
        AssistantMessageEntry(
            entry_id=EntryId.new(),
            session_id=session_id,
            timestamp=_now(),
            content="",
            stop_reason="tool_use",
            tool_calls=(ToolCall("c1", "lookup", {"value": "x"}),),
        ),
        _result(session_id, "c1", 0),
        ControlMessageEntry(
            entry_id=EntryId.new(),
            session_id=session_id,
            timestamp=_now(),
            control_id="s1",
            content="correct",
        ),
        CompactionEntry(
            entry_id=EntryId.new(),
            session_id=session_id,
            timestamp=_now(),
            projection_id=ProjectionId.new(),
            summary=None,
            covered_through_sequence=0,
            first_retained_sequence=1,
        ),
    )
    messages = fold_entries(entries)
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "tool",
        "user",
    ]
    assert messages[2]["tool_call_id"] == "c1"
    assert messages[3]["content"] == "correct"


def test_fold_omits_incomplete_fast_host_users_from_authoritative_history() -> None:
    session_id = SessionId.new()
    succeeded = UserMessageEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        timestamp=_now(),
        content="successful question",
        acceptance_id="run-succeeded",
    )
    answer = AssistantMessageEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        timestamp=_now(),
        content="successful answer",
        stop_reason="stop",
        acceptance_id="run-succeeded",
    )
    failed = UserMessageEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        timestamp=_now(),
        content="failed question",
        acceptance_id="run-failed",
    )
    current = UserMessageEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        timestamp=_now(),
        content="current question",
        acceptance_id="run-current",
    )
    entries = (succeeded, answer, failed, current)

    assert [message["content"] for message in fold_entries(entries)] == [
        "successful question",
        "successful answer",
    ]
    assert [
        message["content"]
        for message in fold_entries(
            entries,
            included_incomplete_host_user_entry_id=current.entry_id,
        )
    ] == ["successful question", "successful answer", "current question"]


def test_exchange_boundaries_never_split_assistant_from_ordered_results() -> None:
    session_id = SessionId.new()
    entries = (
        UserMessageEntry(
            entry_id=EntryId.new(), session_id=session_id, timestamp=_now(), content="question"
        ),
        AssistantMessageEntry(
            entry_id=EntryId.new(),
            session_id=session_id,
            timestamp=_now(),
            content="",
            stop_reason="tool_use",
            tool_calls=(
                ToolCall("c1", "lookup", {"value": "x"}),
                ToolCall("c2", "lookup", {"value": "y"}),
            ),
        ),
        _result(session_id, "c1", 0),
        _result(session_id, "c2", 1),
    )
    assert exchange_starts(entries) == (1,)
    assert select_compaction_boundary(entries, retained_tail_tokens=0) == 1


def test_direct_host_turns_are_complete_compaction_exchanges() -> None:
    session_id = SessionId.new()
    entries = (
        UserMessageEntry(
            entry_id=EntryId.new(), session_id=session_id, timestamp=_now(), content="old"
        ),
        AssistantMessageEntry(
            entry_id=EntryId.new(),
            session_id=session_id,
            timestamp=_now(),
            content="answer",
            stop_reason="stop",
        ),
        UserMessageEntry(
            entry_id=EntryId.new(), session_id=session_id, timestamp=_now(), content="current"
        ),
    )

    starts = host_turn_starts(entries)
    assert starts == (0, 2)
    assert select_compaction_boundary(entries, retained_tail_tokens=0, starts=starts) == 2


def test_projection_is_bound_to_physical_branch_entry_identity() -> None:
    session_id = SessionId.new()
    user = UserMessageEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        timestamp=_now(),
        sequence=1,
        content="old",
    )
    assistant = AssistantMessageEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        timestamp=_now(),
        sequence=2,
        content="new",
        stop_reason="stop",
    )
    projection = ContextProjection(
        projection_id=ProjectionId.new(),
        covered_through_sequence=1,
        first_retained_sequence=2,
        covered_through_entry_id=user.entry_id,
        first_retained_entry_id=assistant.entry_id,
        source_digest=projection_source_digest([user.entry_id]),
        summary='{"goal":"summary"}',
    )
    messages = project_session_messages((user, assistant), projection)
    assert messages[-1]["content"] == "new"


def test_working_projection_replays_signed_state_without_filtering() -> None:
    """Every exchange is replayed verbatim, however much history precedes it.

    DeepSeek requires every previous turn's ``reasoning_content`` back on a
    request that carries tools and rejects a partial history with HTTP 400, and
    Gemini signs its tool calls. A projection that drops either produces a
    history no provider ever sent, so bounding is the compaction boundary's job.
    """
    projection = WorkingContextProjection()
    for index in range(4):
        projection.record(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"c{index}",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                            "thought_signature": f"sig-{index}",
                        }
                    ],
                    "provider_state": {"reasoning_content": f"thinking {index}"},
                },
                {"role": "tool", "tool_call_id": f"c{index}", "content": "result"},
            ]
        )

    assistants = [
        message for message in projection.messages() if message.get("role") == "assistant"
    ]
    assert [message["provider_state"] for message in assistants] == [
        {"reasoning_content": f"thinking {index}"} for index in range(4)
    ]
    assert [
        call["thought_signature"] for message in assistants for call in message["tool_calls"]
    ] == [f"sig-{index}" for index in range(4)]


def test_working_projection_round_trips_signed_state_through_canonical_json() -> None:
    state = {
        "exchanges": [
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "c1", "thought_signature": "sig"}],
                    "provider_state": {"reasoning_content": "thinking"},
                }
            ]
        ]
    }
    projection = WorkingContextProjection.from_canonical_json(state)

    assert projection.messages() == state["exchanges"][0]
    assert projection.canonical_json() == state
