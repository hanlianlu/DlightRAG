# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable completion semantics for the browser answer SSE stream."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dlightrag.core.answer_turn import PreparedAnswerTurn
from dlightrag.storage.web_conversations import CommitTurnResult
from dlightrag.web.answer_events import stream_answer_events
from dlightrag.web.conversations import PreparedWebConversation


async def _tokens():
    yield "Complete answer"


async def _collect(*, service: AsyncMock, result: CommitTurnResult | None = None):
    if result is not None:
        service.commit_answer.return_value = result
    manager = SimpleNamespace(
        config=SimpleNamespace(answer_stream_idle_timeout=30, workspace="default"),
        _aanswer_stream_prepared=AsyncMock(return_value=({"chunks": []}, _tokens())),
    )
    prepared = PreparedWebConversation(
        principal_id="a" * 64,
        conversation_id="11111111-1111-4111-8111-111111111111",
        content_revision=2,
        text_history=(),
    )
    events = [
        event
        async for event in stream_answer_events(
            manager=manager,
            cfg=SimpleNamespace(
                citations=SimpleNamespace(highlights=SimpleNamespace(enabled=False))
            ),
            turn=PreparedAnswerTurn(current_query="hello", retrieval_query="hello"),
            workspaces=["default"],
            workspace="default",
            conversation_service=service,
            prepared_conversation=prepared,
            validated_images=(),
        )
    ]
    return events


async def test_successful_stream_commits_once_before_done() -> None:
    service = AsyncMock()
    service.commit_answer.return_value = CommitTurnResult(
        saved=True, reason=None, summary=None, turn_id="turn-id"
    )

    events = await _collect(service=service)

    service.commit_answer.assert_awaited_once()
    done = next(event for event in events if "event: done" in event)
    assert '"conversation_saved": true' in done


async def test_revision_conflict_is_visible_and_not_appended() -> None:
    service = AsyncMock()
    events = await _collect(
        service=service,
        result=CommitTurnResult(
            saved=False, reason="conversation_changed", summary=None, turn_id=None
        ),
    )

    done = next(event for event in events if "event: done" in event)
    assert '"conversation_saved": false' in done
    assert '"conversation_save_reason": "conversation_changed"' in done


async def test_model_stream_failure_does_not_commit_partial_turn() -> None:
    async def failing_tokens():
        yield "partial"
        raise RuntimeError("provider failed")

    service = AsyncMock()
    manager = SimpleNamespace(
        config=SimpleNamespace(answer_stream_idle_timeout=30, workspace="default"),
        _aanswer_stream_prepared=AsyncMock(return_value=({"chunks": []}, failing_tokens())),
    )
    prepared = PreparedWebConversation(
        principal_id="a" * 64,
        conversation_id="11111111-1111-4111-8111-111111111111",
        content_revision=2,
        text_history=(),
    )

    events = [
        event
        async for event in stream_answer_events(
            manager=manager,
            cfg=SimpleNamespace(),
            turn=PreparedAnswerTurn(current_query="hello", retrieval_query="hello"),
            workspaces=["default"],
            workspace="default",
            conversation_service=service,
            prepared_conversation=prepared,
            validated_images=(),
        )
    ]

    assert any("event: error" in event for event in events)
    assert not any("event: done" in event for event in events)
    service.commit_answer.assert_not_awaited()


async def test_cancellation_propagates_without_committing() -> None:
    async def cancelled_tokens():
        yield "partial"
        raise asyncio.CancelledError

    service = AsyncMock()
    manager = SimpleNamespace(
        config=SimpleNamespace(answer_stream_idle_timeout=30, workspace="default"),
        _aanswer_stream_prepared=AsyncMock(return_value=({"chunks": []}, cancelled_tokens())),
    )
    prepared = PreparedWebConversation(
        principal_id="a" * 64,
        conversation_id="11111111-1111-4111-8111-111111111111",
        content_revision=2,
        text_history=(),
    )

    async def consume() -> None:
        async for _event in stream_answer_events(
            manager=manager,
            cfg=SimpleNamespace(),
            turn=PreparedAnswerTurn(current_query="hello", retrieval_query="hello"),
            workspaces=["default"],
            workspace="default",
            conversation_service=service,
            prepared_conversation=prepared,
            validated_images=(),
        ):
            pass

    with pytest.raises(asyncio.CancelledError):
        await consume()
    service.commit_answer.assert_not_awaited()
