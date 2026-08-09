# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable completion semantics for the browser answer SSE stream."""

import asyncio
import datetime
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, Mock

import pytest

from dlightrag.core.answer.capability import AnswerImageCapability
from dlightrag.core.answer.errors import (
    CURRENT_DOCUMENT_PARSE_FAILED,
    CURRENT_IMAGES_UNSUPPORTED,
    AnswerImageError,
    AnswerInputError,
    CurrentDocumentParseError,
    CurrentImagePayloadError,
)
from dlightrag.storage.web_conversations import CommitTurnResult
from dlightrag.web.answer_events import stream_answer_events
from dlightrag.web.attachment_models import ValidatedWebAttachment
from dlightrag.web.conversations import PreparedWebConversation, WebConversationUnavailableError

if TYPE_CHECKING:
    from dlightrag.core.servicemanager import RAGServiceManager

_CONVERSATION_ID = "11111111-1111-4111-8111-111111111111"
_SUBMISSION_ID = "22222222-2222-4222-8222-222222222222"
_PRINCIPAL = "a" * 64


def _fake_manager(**attrs: Any) -> RAGServiceManager:
    attrs.setdefault("answer_image_capability", None)
    return cast("RAGServiceManager", SimpleNamespace(**attrs))


def _make_service() -> AsyncMock:
    """A conversation service whose sync ``build_answer_resources`` returns a list."""
    service = AsyncMock()
    service.build_answer_resources = Mock(return_value=[])
    return service


def _prepared(**overrides: Any) -> PreparedWebConversation:
    defaults: dict[str, Any] = {
        "principal_id": _PRINCIPAL,
        "conversation_id": _CONVERSATION_ID,
        "content_revision": 2,
        "text_history": (),
    }
    defaults.update(overrides)
    return PreparedWebConversation(**defaults)


def _image_attachment() -> ValidatedWebAttachment:
    return ValidatedWebAttachment(
        attachment_id="ephemeral-image",
        ordinal=1,
        filename="chart.png",
        mime_type="image/png",
        suffix=".png",
        attachment_bytes=b"png",
        content_sha256="digest",
        kind="image",
    )


def _record_observations(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    from contextlib import asynccontextmanager

    captured: dict[str, object] = {"updates": []}

    class RecordingHandle:
        def update(self, **kwargs) -> None:
            updates = captured["updates"]
            assert isinstance(updates, list)
            updates.append(kwargs)

    @asynccontextmanager
    async def fake_observation(name: str, **kwargs):
        captured["name"] = name
        captured["start"] = kwargs
        yield RecordingHandle()

    monkeypatch.setattr("dlightrag.web.answer_events.trace_observation", fake_observation)
    return captured


class _TracedStream:
    """Async token iterator exposing a trace, like the real AnswerStream."""

    def __init__(self, tokens: list[str], trace: dict[str, Any]) -> None:
        self._tokens = list(tokens)
        self.trace = trace
        self.answer = "".join(self._tokens)

    def __aiter__(self) -> AsyncIterator[str]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[str]:
        for token in self._tokens:
            yield token


def _metadata_updates(captured: dict[str, object]) -> list[dict[str, Any]]:
    updates = captured["updates"]
    assert isinstance(updates, list)
    result: list[dict[str, Any]] = []
    for update in updates:
        if not isinstance(update, dict):
            continue
        metadata = update.get("metadata")
        if isinstance(metadata, dict):
            result.append(metadata)
    return result


async def _tokens(tokens: tuple[str, ...] = ("Complete answer",)):
    for token in tokens:
        yield token


async def _collect(
    *,
    service: AsyncMock,
    result: CommitTurnResult | None = None,
    validated_attachments: tuple[ValidatedWebAttachment, ...] = (),
    contexts: dict[str, Any] | None = None,
    tokens: tuple[str, ...] = ("Complete answer",),
    token_iter: Any = None,
    manager: Any = None,
):
    if result is not None:
        service.commit_answer.return_value = result
    if manager is None:
        manager = _fake_manager(
            config=SimpleNamespace(answer_stream_idle_timeout=30, workspace="default"),
            _aanswer_stream_prepared=AsyncMock(
                return_value=(contexts or {"chunks": []}, token_iter or _tokens(tokens))
            ),
        )
    events = [
        event
        async for event in stream_answer_events(
            manager=manager,
            cfg=SimpleNamespace(
                citations=SimpleNamespace(highlights=SimpleNamespace(enabled=False))
            ),
            query="hello",
            workspaces=["default"],
            workspace="default",
            conversation_service=service,
            prepared_conversation=_prepared(),
            validated_attachments=validated_attachments,
            submission_id=_SUBMISSION_ID,
        )
    ]
    return events


async def _collect_stream_error(
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> tuple[list[str], dict[str, object], AsyncMock, AsyncMock]:
    captured = _record_observations(monkeypatch)
    service = _make_service()
    answer_stream_prepared = AsyncMock(side_effect=error)
    manager = _fake_manager(
        config=SimpleNamespace(answer_stream_idle_timeout=30, workspace="default"),
        answer_image_capability=None,
        _aanswer_stream_prepared=answer_stream_prepared,
    )
    events = [
        event
        async for event in stream_answer_events(
            manager=manager,
            cfg=SimpleNamespace(),
            query="summarize this",
            workspaces=["default"],
            workspace="default",
            conversation_service=service,
            prepared_conversation=_prepared(),
            validated_attachments=(),
            submission_id=_SUBMISSION_ID,
        )
    ]
    return events, captured, service, answer_stream_prepared


def test_answer_input_errors_expose_an_explicit_public_message() -> None:
    error = AnswerInputError("Safe message", error_kind="TEST_INPUT_ERROR")

    assert error.public_message == "Safe message"
    assert str(error) == "Safe message"


def test_current_document_parse_error_builds_its_public_contract_from_filename() -> None:
    error = CurrentDocumentParseError("broken.docx")

    assert error.public_message == (
        "Could not read broken.docx. Check that the document is valid and "
        "the document parser is available."
    )
    assert str(error) == error.public_message
    assert error.error_kind == CURRENT_DOCUMENT_PARSE_FAILED


async def test_successful_stream_commits_once_before_done() -> None:
    service = _make_service()
    now = datetime.datetime(2026, 7, 13, tzinfo=datetime.UTC)
    service.commit_answer.return_value = CommitTurnResult(
        saved=True,
        reason=None,
        summary={
            "conversation_id": _CONVERSATION_ID,
            "title": "Hello",
            "content_revision": 3,
            "created_at": now,
            "updated_at": now,
        },
        turn_id="turn-id",
        current_attachment_ids=("durable-attachment",),
    )

    events = await _collect(service=service, tokens=("Complete answer",))

    service.commit_answer.assert_awaited_once()
    answer_sources = service.commit_answer.await_args.kwargs["answer_sources"]
    assert answer_sources["answer_images"] == []
    assert "composer_" not in json.dumps(answer_sources)
    done = next(event for event in events if "event: done" in event)
    assert '"conversation_saved": true' in done
    assert "durable-attachment" in done


async def test_committed_submission_replays_without_regenerating() -> None:
    now = datetime.datetime(2026, 7, 13, tzinfo=datetime.UTC)
    service = _make_service()
    manager = _fake_manager(
        config=SimpleNamespace(answer_stream_idle_timeout=30, workspace="default"),
        _aanswer_stream_prepared=AsyncMock(side_effect=AssertionError("must not regenerate")),
    )
    committed = CommitTurnResult(
        saved=True,
        reason=None,
        summary={
            "conversation_id": _CONVERSATION_ID,
            "title": "Prior",
            "content_revision": 3,
            "created_at": now,
            "updated_at": now,
        },
        turn_id="turn-id",
        current_attachment_ids=("prior-attachment",),
        assistant_text="Prior answer",
        answer_sources={"sources": [], "answer_images": []},
        replayed=True,
    )

    events = [
        event
        async for event in stream_answer_events(
            manager=manager,
            cfg=SimpleNamespace(),
            query="hello",
            workspaces=["default"],
            workspace="default",
            conversation_service=service,
            prepared_conversation=_prepared(committed_submission=committed),
            validated_attachments=(),
            submission_id=_SUBMISSION_ID,
        )
    ]

    done = next(event for event in events if "event: done" in event)
    assert '"conversation_saved": true' in done
    assert "Prior answer" in done
    assert "prior-attachment" in done
    service.commit_answer.assert_not_awaited()


async def test_revision_conflict_is_visible_and_not_appended() -> None:
    service = _make_service()

    events = await _collect(
        service=service,
        result=CommitTurnResult(
            saved=False, reason="conversation_changed", summary=None, turn_id=None
        ),
        validated_attachments=(_image_attachment(),),
    )

    done = next(event for event in events if "event: done" in event)
    assert '"conversation_saved": false' in done
    assert '"conversation_save_reason": "conversation_changed"' in done
    assert "ephemeral-image" not in done
    assert '"current_attachment_ids": []' in done


async def test_model_stream_failure_does_not_commit_partial_turn() -> None:
    async def failing_tokens():
        yield "partial"
        raise RuntimeError("provider failed")

    service = _make_service()
    manager = _fake_manager(
        config=SimpleNamespace(answer_stream_idle_timeout=30, workspace="default"),
        _aanswer_stream_prepared=AsyncMock(return_value=({"chunks": []}, failing_tokens())),
    )

    events = await _collect(service=service, manager=manager)

    assert any("event: error" in event for event in events)
    assert not any("event: done" in event for event in events)
    service.commit_answer.assert_not_awaited()


async def test_transport_and_capability_metrics_reach_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _record_observations(monkeypatch)
    now = datetime.datetime(2026, 7, 13, tzinfo=datetime.UTC)
    service = _make_service()
    service.commit_answer.return_value = CommitTurnResult(
        saved=True,
        reason=None,
        summary={
            "conversation_id": _CONVERSATION_ID,
            "title": "Hi",
            "content_revision": 3,
            "created_at": now,
            "updated_at": now,
        },
        turn_id="turn-id",
    )
    trace = {
        "answer_images_current": 1,
        "answer_images_rag": 2,
        "answer_images_total": 3,
        "answer_image_budget_used_bytes": 4096,
    }
    manager = _fake_manager(
        config=SimpleNamespace(answer_stream_idle_timeout=30, workspace="default"),
        answer_image_capability=AnswerImageCapability(
            status="supported",
            configured_ceiling=8,
            effective_max_images=6,
            provider="test",
            base_url=None,
            model="m",
            failure_kind=None,
        ),
        _aanswer_stream_prepared=AsyncMock(
            return_value=({"chunks": []}, _TracedStream(["answer"], trace))
        ),
    )

    await _collect(service=service, manager=manager)

    capability = [
        metadata
        for metadata in _metadata_updates(captured)
        if "answer_image_capability_status" in metadata
    ]
    assert capability, "capability metrics were not emitted"
    caps = capability[-1]
    assert caps["answer_image_capability_status"] == "supported"
    assert caps["answer_image_configured_ceiling"] == 8
    assert caps["answer_image_effective_limit"] == 6

    # Streaming shares the non-streaming span name, so one Langfuse view covers both.
    assert captured["name"] == "answer_pipeline"
    start = captured["start"]
    assert isinstance(start, dict)
    assert start["metadata"]["stream"] is True
    updates = captured["updates"]
    assert isinstance(updates, list)
    outputs = [update["output"] for update in updates if "output" in update]
    assert outputs == [{"answer_len": 6, "source_count": 0, "context_chunk_count": 0}]

    transport = [
        metadata for metadata in _metadata_updates(captured) if "answer_images_total" in metadata
    ]
    assert transport, "transport metrics were not emitted"
    metrics = transport[-1]
    assert metrics["answer_images_total"] == 3
    assert metrics["answer_images_current"] == 1
    assert metrics["answer_images_rag"] == 2
    assert metrics["answer_image_bytes_total"] == 4096


async def test_current_image_payload_error_maps_to_limit_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events, captured, service, _prepared_mock = await _collect_stream_error(
        monkeypatch, CurrentImagePayloadError("2 current-turn images exceed 1")
    )

    error_event = next(event for event in events if "event: error" in event)
    error_payload = json.loads(error_event.split("data: ", 1)[1])
    assert error_payload == {
        "message": "2 current-turn images exceed 1",
        "error_kind": "CURRENT_IMAGE_LIMIT_EXCEEDED",
    }
    assert not any("event: token" in event for event in events)
    error_kinds = [
        metadata["error_kind"]
        for metadata in _metadata_updates(captured)
        if "error_kind" in metadata
    ]
    assert error_kinds == ["CURRENT_IMAGE_LIMIT_EXCEEDED"]
    service.commit_answer.assert_not_awaited()


async def test_current_document_parse_error_emits_safe_typed_error_before_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parse_error = CurrentDocumentParseError("broken.docx")
    parse_error.__cause__ = RuntimeError("raw parser exception: customer secret")
    events, captured, service, _prepared_mock = await _collect_stream_error(
        monkeypatch, parse_error
    )

    error_event = next(event for event in events if "event: error" in event)
    error_payload = json.loads(error_event.split("data: ", 1)[1])
    assert error_payload == {
        "message": parse_error.public_message,
        "error_kind": CURRENT_DOCUMENT_PARSE_FAILED,
    }
    assert "customer secret" not in "".join(events)
    error_kinds = [
        metadata["error_kind"]
        for metadata in _metadata_updates(captured)
        if "error_kind" in metadata
    ]
    assert error_kinds == [CURRENT_DOCUMENT_PARSE_FAILED]
    service.commit_answer.assert_not_awaited()


async def test_answer_image_error_emits_its_safe_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = AnswerImageError(
        "Images are unavailable for this answer model.",
        error_kind=CURRENT_IMAGES_UNSUPPORTED,
    )
    events, _captured, _service, _prepared_mock = await _collect_stream_error(monkeypatch, error)

    error_event = next(event for event in events if "event: error" in event)
    assert json.loads(error_event.split("data: ", 1)[1]) == {
        "message": error.public_message,
        "error_kind": CURRENT_IMAGES_UNSUPPORTED,
    }


async def test_cancellation_propagates_without_committing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _record_observations(monkeypatch)

    async def cancelled_tokens():
        yield "partial"
        raise asyncio.CancelledError

    service = _make_service()
    manager = _fake_manager(
        config=SimpleNamespace(answer_stream_idle_timeout=30, workspace="default"),
        _aanswer_stream_prepared=AsyncMock(return_value=({"chunks": []}, cancelled_tokens())),
    )

    async def consume() -> None:
        async for _event in stream_answer_events(
            manager=manager,
            cfg=SimpleNamespace(),
            query="hello",
            workspaces=["default"],
            workspace="default",
            conversation_service=service,
            prepared_conversation=_prepared(),
            validated_attachments=(),
            submission_id=_SUBMISSION_ID,
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


async def _run_failing_stream(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    captured = _record_observations(monkeypatch)

    async def failing_tokens():
        yield "partial"
        raise RuntimeError("secret provider detail")

    service = _make_service()
    manager = _fake_manager(
        config=SimpleNamespace(answer_stream_idle_timeout=30, workspace="default"),
        _aanswer_stream_prepared=AsyncMock(return_value=({"chunks": []}, failing_tokens())),
    )

    _events = [
        event
        async for event in stream_answer_events(
            manager=manager,
            cfg=SimpleNamespace(),
            query="private prompt",
            workspaces=["default"],
            workspace="default",
            conversation_service=service,
            prepared_conversation=_prepared(),
            validated_attachments=(),
            submission_id=_SUBMISSION_ID,
        )
    ]
    return captured


async def test_failure_records_error_detail_and_raw_ids_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = await _run_failing_stream(monkeypatch)

    serialized = json.dumps(captured)
    start = captured["start"]
    assert isinstance(start, dict)
    # Full traceability by default: query, raw error text, and raw IDs are captured.
    assert start["input"] == {"query": "private prompt"}
    assert start["metadata"]["principal_id"] == _PRINCIPAL
    assert start["metadata"]["conversation_id"] == _CONVERSATION_ID
    assert "secret provider detail" in serialized
    assert '"conversation_saved": false' in serialized
    assert "answer_failed" in serialized
    assert '"level": "ERROR"' in serialized


async def test_privacy_mode_redacts_error_text_and_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dlightrag.observability._trace_sensitive", False)
    captured = await _run_failing_stream(monkeypatch)

    serialized = json.dumps(captured)
    # Privacy mode: generic error text and hashed IDs; raw values must not leak.
    assert "secret provider detail" not in serialized
    assert _CONVERSATION_ID not in serialized
    assert "answer_stream_failed" in serialized
    start = captured["start"]
    assert isinstance(start, dict)
    metadata = start["metadata"]
    assert len(metadata["principal_hash"]) == 64
    assert len(metadata["conversation_hash"]) == 64
    assert metadata["history_turns_loaded"] == 0


@pytest.mark.parametrize(
    ("result", "expected_saved", "expected_reason"),
    (
        (
            CommitTurnResult(True, None, None, "turn", current_attachment_ids=("stored",)),
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
    service = _make_service()

    await _collect(service=service, result=result)

    updates = captured["updates"]
    assert isinstance(updates, list)
    terminal = updates[-1]["metadata"]
    assert terminal["conversation_saved"] is expected_saved
    assert terminal["conversation_save_reason"] == expected_reason


async def test_storage_failure_records_unsaved_and_exposes_no_attachment_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _record_observations(monkeypatch)
    service = _make_service()
    service.commit_answer.side_effect = WebConversationUnavailableError

    events = await _collect(service=service, validated_attachments=(_image_attachment(),))

    done = next(event for event in events if "event: done" in event)
    assert '"current_attachment_ids": []' in done
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

    service = _make_service()
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

    service = _make_service()
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
    service = _make_service()

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

    service = _make_service()
    service.commit_answer.side_effect = commit_answer
    manager = _fake_manager(
        config=SimpleNamespace(answer_stream_idle_timeout=30, workspace="default"),
        _aanswer_stream_prepared=AsyncMock(return_value=({"chunks": []}, _tokens())),
    )
    stream = stream_answer_events(
        manager=manager,
        cfg=SimpleNamespace(citations=SimpleNamespace(highlights=SimpleNamespace(enabled=False))),
        query="hello",
        workspaces=["default"],
        workspace="default",
        conversation_service=service,
        prepared_conversation=_prepared(),
        validated_attachments=(),
        submission_id=_SUBMISSION_ID,
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


async def test_attachments_thread_into_commit_and_surface_attachment_ids() -> None:
    from dlightrag.web.attachment_models import validate_web_attachments

    (document,) = validate_web_attachments(
        [("notes.md", "text/markdown", b"# Termination clause")],
        max_attachments=6,
        image_max_bytes=15 * 1024 * 1024,
    )
    service = _make_service()
    now = datetime.datetime(2026, 7, 13, tzinfo=datetime.UTC)
    service.commit_answer.return_value = CommitTurnResult(
        saved=True,
        reason=None,
        summary={
            "conversation_id": _CONVERSATION_ID,
            "title": "Hello",
            "content_revision": 3,
            "created_at": now,
            "updated_at": now,
        },
        turn_id="turn-id",
        current_attachment_ids=("durable-doc",),
    )

    events = await _collect(service=service, validated_attachments=(document,))

    service.build_answer_resources.assert_called_once()
    assert service.build_answer_resources.call_args.args[1] == (document,)
    service.commit_answer.assert_awaited_once()
    assert service.commit_answer.await_args.kwargs["attachments"] == (document,)
    done = next(event for event in events if "event: done" in event)
    assert "durable-doc" in done
