# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable completion semantics for the browser answer SSE stream."""

import asyncio
import datetime
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dlightrag.core.answer_turn import PreparedAnswerTurn
from dlightrag.storage.web_conversations import CommitTurnResult
from dlightrag.utils.images import ValidatedWebImage
from dlightrag.web.answer_events import stream_answer_events
from dlightrag.web.conversations import PreparedWebConversation, WebConversationUnavailableError


def _record_observations(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    captured: dict[str, object] = {"updates": []}

    class RecordingHandle:
        def update(self, **kwargs) -> None:
            updates = captured["updates"]
            assert isinstance(updates, list)
            updates.append(kwargs)

    @asynccontextmanager
    async def fake_observation(_name: str, **kwargs):
        captured["start"] = kwargs
        yield RecordingHandle()

    monkeypatch.setattr("dlightrag.web.answer_events.trace_observation", fake_observation)
    return captured


async def _tokens():
    yield "Complete answer"


async def _collect(
    *,
    service: AsyncMock,
    result: CommitTurnResult | None = None,
    validated_images: tuple[ValidatedWebImage, ...] = (),
):
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
            validated_images=validated_images,
            submission_id="22222222-2222-4222-8222-222222222222",
        )
    ]
    return events


async def test_successful_stream_commits_once_before_done() -> None:
    service = AsyncMock()
    now = datetime.datetime(2026, 7, 13, tzinfo=datetime.UTC)
    service.commit_answer.return_value = CommitTurnResult(
        saved=True,
        reason=None,
        summary={
            "conversation_id": "11111111-1111-4111-8111-111111111111",
            "title": "Hello",
            "content_revision": 3,
            "created_at": now,
            "updated_at": now,
        },
        turn_id="turn-id",
        current_image_ids=("durable-image",),
    )

    events = await _collect(service=service)

    service.commit_answer.assert_awaited_once()
    done = next(event for event in events if "event: done" in event)
    assert '"conversation_saved": true' in done


async def test_revision_conflict_is_visible_and_not_appended() -> None:
    service = AsyncMock()
    image = ValidatedWebImage(
        image_id="ephemeral-image",
        ordinal=1,
        mime_type="image/png",
        image_bytes=b"png",
        data_uri="data:image/png;base64,cG5n",
        content_sha256="digest",
    )
    events = await _collect(
        service=service,
        result=CommitTurnResult(
            saved=False, reason="conversation_changed", summary=None, turn_id=None
        ),
        validated_images=(image,),
    )

    done = next(event for event in events if "event: done" in event)
    assert '"conversation_saved": false' in done
    assert '"conversation_save_reason": "conversation_changed"' in done
    assert "ephemeral-image" not in done
    assert '"current_image_ids": []' in done


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
            submission_id="22222222-2222-4222-8222-222222222222",
        )
    ]

    assert any("event: error" in event for event in events)
    assert not any("event: done" in event for event in events)
    service.commit_answer.assert_not_awaited()


async def test_cancellation_propagates_without_committing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _record_observations(monkeypatch)

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
            submission_id="22222222-2222-4222-8222-222222222222",
        ):
            pass

    with pytest.raises(asyncio.CancelledError):
        await consume()
    service.commit_answer.assert_not_awaited()
    updates = captured["updates"]
    assert isinstance(updates, list)
    assert updates[-1]["metadata"] == {
        "conversation_saved": False,
        "conversation_save_reason": "cancelled",
    }


async def test_failure_records_terminal_unsaved_observation_without_exception_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _record_observations(monkeypatch)

    async def failing_tokens():
        yield "partial"
        raise RuntimeError("secret provider detail")

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

    _events = [
        event
        async for event in stream_answer_events(
            manager=manager,
            cfg=SimpleNamespace(),
            turn=PreparedAnswerTurn(
                current_query="private prompt", retrieval_query="private prompt"
            ),
            workspaces=["default"],
            workspace="default",
            conversation_service=service,
            prepared_conversation=prepared,
            validated_images=(),
            submission_id="22222222-2222-4222-8222-222222222222",
        )
    ]

    serialized = json.dumps(captured)
    assert "private prompt" not in serialized
    assert "secret provider detail" not in serialized
    assert "11111111-1111-4111-8111-111111111111" not in serialized
    start = captured["start"]
    assert isinstance(start, dict)
    metadata = start["metadata"]
    assert len(metadata["principal_hash"]) == 64
    assert len(metadata["conversation_hash"]) == 64
    assert metadata["history_turns_loaded"] == 0
    assert '"conversation_saved": false' in serialized
    assert "answer_failed" in serialized
    assert '"level": "ERROR"' in serialized


@pytest.mark.parametrize(
    ("result", "expected_saved", "expected_reason"),
    (
        (
            CommitTurnResult(True, None, None, "turn", current_image_ids=("stored",)),
            True,
            None,
        ),
        (
            CommitTurnResult(False, "conversation_changed", None, None),
            False,
            "conversation_changed",
        ),
    ),
)
async def test_completed_stream_records_terminal_save_outcome(
    monkeypatch: pytest.MonkeyPatch,
    result: CommitTurnResult,
    expected_saved: bool,
    expected_reason: str | None,
) -> None:
    captured = _record_observations(monkeypatch)
    service = AsyncMock()

    await _collect(service=service, result=result)

    updates = captured["updates"]
    assert isinstance(updates, list)
    terminal = updates[-1]["metadata"]
    assert terminal["conversation_saved"] is expected_saved
    assert terminal["conversation_save_reason"] == expected_reason


async def test_storage_failure_records_unsaved_and_exposes_no_image_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _record_observations(monkeypatch)
    service = AsyncMock()
    service.commit_answer.side_effect = WebConversationUnavailableError
    image = ValidatedWebImage(
        image_id="ephemeral-image",
        ordinal=1,
        mime_type="image/png",
        image_bytes=b"png",
        data_uri="data:image/png;base64,cG5n",
        content_sha256="digest",
    )

    events = await _collect(service=service, validated_images=(image,))

    done = next(event for event in events if "event: done" in event)
    assert '"current_image_ids": []' in done
    assert "ephemeral-image" not in done
    updates = captured["updates"]
    assert isinstance(updates, list)
    assert updates[-1]["metadata"] == {
        "conversation_saved": False,
        "conversation_save_reason": "storage_unavailable",
    }


async def test_cancellation_during_commit_does_not_cancel_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _record_observations(monkeypatch)
    started = asyncio.Event()
    release = asyncio.Event()
    commit_cancelled = False

    async def commit_answer(*_args, **_kwargs):
        nonlocal commit_cancelled
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            commit_cancelled = True
            raise
        return CommitTurnResult(True, None, None, "turn")

    service = AsyncMock()
    service.commit_answer.side_effect = commit_answer
    consume = asyncio.create_task(_collect(service=service))
    await started.wait()

    consume.cancel()
    await asyncio.sleep(0)
    assert commit_cancelled is False
    assert consume.done() is False

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await consume
    updates = captured["updates"]
    assert isinstance(updates, list)
    assert updates[-1]["metadata"] == {
        "conversation_saved": True,
        "conversation_save_reason": None,
    }


async def test_cancelled_client_records_unknown_post_commit_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _record_observations(monkeypatch)
    started = asyncio.Event()
    release = asyncio.Event()

    async def commit_answer(*_args, **_kwargs):
        started.set()
        await release.wait()
        return CommitTurnResult(False, "commit_outcome_unknown", None, None)

    service = AsyncMock()
    service.commit_answer.side_effect = commit_answer
    consume = asyncio.create_task(_collect(service=service))
    await started.wait()
    consume.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await consume
    updates = captured["updates"]
    assert isinstance(updates, list)
    assert updates[-1]["metadata"] == {
        "conversation_saved": False,
        "conversation_save_reason": "commit_outcome_unknown",
    }


async def test_saving_heartbeat_keeps_persistence_wait_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dlightrag.web.answer_events._PERSISTENCE_HEARTBEAT_SECONDS", 0.001)
    service = AsyncMock()

    async def commit_answer(*_args, **_kwargs):
        await asyncio.sleep(0.005)
        return CommitTurnResult(True, None, None, "turn")

    service.commit_answer.side_effect = commit_answer

    events = await _collect(service=service)

    assert any('"phase": "saving"' in event for event in events)


async def test_generator_close_after_saving_heartbeat_finishes_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dlightrag.web.answer_events._PERSISTENCE_HEARTBEAT_SECONDS", 0.001)
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()
    commit_cancelled = False

    async def commit_answer(*_args, **_kwargs):
        nonlocal commit_cancelled
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            commit_cancelled = True
            raise
        finished.set()
        return CommitTurnResult(True, None, None, "turn")

    service = AsyncMock()
    service.commit_answer.side_effect = commit_answer
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
    stream = stream_answer_events(
        manager=manager,
        cfg=SimpleNamespace(citations=SimpleNamespace(highlights=SimpleNamespace(enabled=False))),
        turn=PreparedAnswerTurn(current_query="hello", retrieval_query="hello"),
        workspaces=["default"],
        workspace="default",
        conversation_service=service,
        prepared_conversation=prepared,
        validated_images=(),
        submission_id="22222222-2222-4222-8222-222222222222",
    )

    while True:
        event = await anext(stream)
        if '"phase": "saving"' in event:
            break
    await started.wait()
    close_task = asyncio.create_task(stream.aclose())
    await asyncio.sleep(0)

    assert close_task.done() is False
    assert commit_cancelled is False
    release.set()
    await close_task
    assert finished.is_set()
