# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable Answer executor ownership and failure behavior."""

import asyncio
import io
import uuid
from collections.abc import Mapping
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from dlightrag_agent.session.fold import PriorTurns
from dlightrag_ai.capacity import ModelProfile
from dlightrag_ai.fingerprints import ModelFingerprint
from dlightrag_ai.scheduler import ModelScheduler
from dlightrag_ai.telemetry import NOOP_TELEMETRY
from PIL import Image

from dlightrag.answer.capabilities import AnswerCapabilities
from dlightrag.answer.capability import AnswerImageCapability
from dlightrag.answer.errors import CurrentDocumentParseError
from dlightrag.answer.executor import (
    AnswerExecutor,
    AnswerExecutorSettings,
    AnswerResourceResolver,
    AnswerResourceSettings,
    OrchestratorRun,
    _close_execution_resources,
)
from dlightrag.answer.highlights import SemanticHighlightSettings
from dlightrag.answer.runs.execution import (
    AnswerRunInput,
    AnswerRunRequest,
    AttachmentReference,
    LinkReference,
    PinnedModelProfile,
    build_current_answer_resources,
    in_memory_attachment_loader,
)
from dlightrag.runtime import RunExecutionError, RunSession, artifact_digest


def _executor() -> AnswerExecutor:
    return AnswerExecutor(
        store=MagicMock(),
        pool=MagicMock(),
        retrieve=AsyncMock(),
        models=MagicMock(),
        capabilities=MagicMock(),
        resources=MagicMock(),
        settings=AnswerExecutorSettings(
            default_top_k=10,
            default_chunk_top_k=20,
            semantic_highlights=SemanticHighlightSettings(
                enabled=True,
                timeout=10.0,
                max_concurrency=8,
                batch_size=8,
                max_input_chars=4096,
                cache_size=500,
            ),
        ),
        telemetry=NOOP_TELEMETRY,
    )


def _resource_resolver() -> AnswerResourceResolver:
    capabilities = MagicMock()
    capabilities.refresh_answer = AsyncMock(
        return_value=AnswerCapabilities(
            answer=AnswerImageCapability(
                status="supported",
                configured_ceiling=3,
                effective_max_images=3,
                provider="test",
                base_url=None,
                model="test-model",
                failure_kind=None,
            ),
            vlm_status="unknown",
        )
    )
    return AnswerResourceResolver(
        settings=AnswerResourceSettings(
            max_attachments=6,
            max_attachment_bytes=10_000_000,
            max_total_attachment_bytes=20_000_000,
            image_max_bytes=5_000_000,
            image_max_pixels=4_000_000,
        ),
        models=MagicMock(),
        capabilities=capabilities,
    )


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), "white").save(buffer, format="PNG")
    return buffer.getvalue()


async def test_child_model_calls_inherit_run_scheduler_ownership() -> None:
    scheduler = ModelScheduler(max_concurrency=1)
    first_started = asyncio.Event()
    second_queued = asyncio.Event()
    release_first = asyncio.Event()
    order: list[str] = []

    async def operation(label: str, *, block: bool = False) -> str:
        order.append(label)
        if block:
            first_started.set()
            await release_first.wait()
        return label

    async def execute(session: Any) -> Mapping[str, Any]:
        if session.run_id == "run-a":
            first = asyncio.create_task(scheduler.run(lambda: operation("a1", block=True)))
            await first_started.wait()
            second = asyncio.create_task(scheduler.run(lambda: operation("a2")))
            await asyncio.sleep(0)
            second_queued.set()
            await asyncio.gather(first, second)
            return {"run": "a"}
        await scheduler.run(lambda: operation("b1"))
        return {"run": "b"}

    executor = _executor()
    executor._execute = execute  # type: ignore[method-assign]
    run_a = asyncio.create_task(
        executor.execute(cast(RunSession, MagicMock(owner_id="owner", run_id="run-a")))
    )
    await second_queued.wait()
    run_b = asyncio.create_task(
        executor.execute(cast(RunSession, MagicMock(owner_id="owner", run_id="run-b")))
    )
    await asyncio.sleep(0)
    release_first.set()

    assert await asyncio.gather(run_a, run_b) == [{"run": "a"}, {"run": "b"}]
    assert order == ["a1", "b1", "a2"]


async def test_actionable_answer_errors_keep_their_public_message() -> None:
    executor = _executor()
    executor._execute = AsyncMock(  # type: ignore[method-assign]
        side_effect=CurrentDocumentParseError("report.pdf")
    )

    with pytest.raises(RunExecutionError) as raised:
        await executor.execute(cast(RunSession, MagicMock()))

    assert raised.value.kind == "CURRENT_DOCUMENT_PARSE_FAILED"
    assert "report.pdf" in raised.value.public_message


async def test_unknown_errors_map_to_generic_public_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    executor = _executor()
    executor._execute = AsyncMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("postgres://user:secret@host/db")
    )
    session = MagicMock(owner_id="owner", run_id="run-correlated")

    with pytest.raises(RunExecutionError) as raised:
        await executor.execute(cast(RunSession, session))

    assert raised.value.kind == "ANSWER_STREAM_FAILED"
    assert raised.value.public_message == "Answer run failed."
    assert "Answer run run-correlated execution failed" in caplog.text
    assert "postgres://user:secret@host/db" in caplog.text


async def test_url_current_image_is_pinned_once_for_durable_replay() -> None:
    resolver = _resource_resolver()
    image_bytes = _png_bytes()
    inline_bytes = b"notes"
    resolver.materialize_link_image = AsyncMock(return_value=image_bytes)  # type: ignore[method-assign]
    request = AnswerRunRequest(
        query="inspect",
        links=(
            LinkReference(
                url="https://example.com/chart.png",
                filename="chart.png",
                ordinal=0,
                mime_type="image/png",
            ),
        ),
        attachments=(
            AttachmentReference(
                digest=artifact_digest(inline_bytes),
                filename="notes.txt",
                mime_type="text/plain",
                ordinal=0,
            ),
        ),
    )

    pinned, artifacts = await resolver.pin_current_image_links(request, (inline_bytes,))

    assert pinned.links == ()
    assert [item.filename for item in pinned.attachments] == ["chart.png", "notes.txt"]
    assert [item.ordinal for item in pinned.attachments] == [0, 1]
    assert artifacts == [image_bytes, inline_bytes]
    resources = await build_current_answer_resources(
        links=pinned.links,
        attachments=pinned.attachments,
        attachment_loaders=[
            in_memory_attachment_loader(image_bytes),
            in_memory_attachment_loader(inline_bytes),
        ],
    )
    resolver.materialize_link_image = AsyncMock(  # type: ignore[method-assign]
        side_effect=AssertionError("durable replay must not refetch the URL")
    )
    images, _remaining, _image_resources = await resolver.prepare_current_images(resources)

    assert len(images) == 1
    resolver.materialize_link_image.assert_not_awaited()  # type: ignore[attr-defined]


async def test_unavailable_url_image_is_durably_demoted_to_an_ordinary_link() -> None:
    resolver = _resource_resolver()
    materialize = AsyncMock(return_value=None)
    resolver.materialize_link_image = materialize  # type: ignore[method-assign]
    request = AnswerRunRequest(
        query="inspect",
        links=(
            LinkReference(
                url="https://example.com/chart.png",
                filename="chart.png",
                ordinal=0,
                mime_type="image/png",
            ),
        ),
    )

    pinned, artifacts = await resolver.pin_current_image_links(request, ())
    resources = await build_current_answer_resources(
        links=pinned.links,
        attachments=pinned.attachments,
        attachment_loaders=(),
    )
    images, _remaining, _image_resources = await resolver.prepare_current_images(resources)

    assert artifacts == []
    assert pinned.links[0].mime_type is None
    assert images == []
    materialize.assert_awaited_once()


async def test_stream_close_failure_does_not_skip_registry_close(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Stream:
        def __aiter__(self):
            return self

        async def __anext__(self) -> str:
            raise StopAsyncIteration

        async def aclose(self) -> None:
            raise RuntimeError("stream close failed")

    registry = MagicMock(aclose=AsyncMock())

    await _close_execution_resources(Stream(), registry)

    registry.aclose.assert_awaited_once()
    assert "Failed to close Answer stream" in caplog.text


async def test_research_run_seeds_the_pinned_session_journal() -> None:
    profile = ModelProfile(context_window_tokens=10_000, supports_tools=True)
    request = AnswerRunInput(
        query="resume",
        workspaces=("default",),
        pinned_models=tuple(
            PinnedModelProfile(
                role=role,
                fingerprint=ModelFingerprint("openai", f"test-{role}", None),
                profile=profile,
            )
            for role in ("extract", "keyword", "query", "vlm")
        ),
        context_policy_revision="m1-v1",
        model_catalog_revision="test",
        idempotency_fingerprint="request-hash",
        session_id=str(uuid.uuid7()),
    )
    prepared = MagicMock(tools=[], evidence=MagicMock(ledger_state_json=lambda: "{}"))
    orchestrator = MagicMock(resolved_mode="research")
    orchestrator.prepare_run.return_value = prepared
    orchestrator.staged_artifacts.return_value = ()
    orchestrator.answer_stream = AsyncMock(
        return_value=({"chunks": [], "entities": [], "relationships": []}, None)
    )
    registry = MagicMock(aclose=AsyncMock())
    executor = _executor()
    store = cast(Any, executor._store)
    store.load_routing = AsyncMock(
        return_value=MagicMock(
            resolved_mode="research",
            research_session_id=request.session_id,
            requested_mode="research",
            valid_modes=("fast", "research"),
        )
    )
    store.resolve = AsyncMock(return_value="research")
    executor.prepare_orchestrated_run = AsyncMock(  # type: ignore[method-assign]
        return_value=OrchestratorRun(
            orchestrator=orchestrator,
            image_descriptions=[],
            query_images=None,
            history=PriorTurns(),
            current_image_count=0,
            workspaces=["default"],
            registry=registry,
        )
    )

    from dlightrag_agent.session.ids import SessionId
    from dlightrag_agent.session.memory import InMemoryAgentSessionStore

    journal = InMemoryAgentSessionStore()
    session = MagicMock(
        owner_id="owner",
        run_id="run-1",
        prepared_input=request.as_request(),
        enter_phase=AsyncMock(),
        flush_tokens=AsyncMock(),
        emit_token=AsyncMock(),
        execution=MagicMock(session_store=journal, progress_store=MagicMock()),
    )

    result = await executor.execute(cast(RunSession, session))

    assert result["answer"] == ""
    snapshot = await journal.load(SessionId(request.session_id))
    assert snapshot.version == 1
    kinds = {entry.__class__.__name__ for entry in snapshot.entries}
    assert "ProfileFactEntry" in kinds
    assert "UserMessageEntry" not in kinds  # no history turns
    orchestrator.answer_stream.assert_awaited_once()


async def test_resumed_research_recovers_the_episode_from_the_folded_journal() -> None:
    profile = ModelProfile(context_window_tokens=10_000, supports_tools=True)
    request = AnswerRunInput(
        query="resume",
        workspaces=("default",),
        pinned_models=tuple(
            PinnedModelProfile(
                role=role,
                fingerprint=ModelFingerprint("openai", f"test-{role}", None),
                profile=profile,
            )
            for role in ("extract", "keyword", "query", "vlm")
        ),
        context_policy_revision="m1-v1",
        model_catalog_revision="test",
        idempotency_fingerprint="request-hash",
        session_id=str(uuid.uuid7()),
    )
    from datetime import UTC, datetime

    from dlightrag_agent.session.entries import (
        AssistantMessageEntry,
        UserMessageEntry,
    )
    from dlightrag_agent.session.ids import EntryId, SessionId
    from dlightrag_agent.session.memory import InMemoryAgentSessionStore

    session_id = SessionId(request.session_id)
    now = datetime.now(UTC)
    journal = InMemoryAgentSessionStore()
    await journal.append(
        session_id=session_id,
        expected_version=0,
        entries=[
            UserMessageEntry(
                entry_id=EntryId.new(), session_id=session_id, timestamp=now, content="q"
            ),
            AssistantMessageEntry(
                entry_id=EntryId.new(),
                session_id=session_id,
                timestamp=now,
                content="done",
                stop_reason="stop",
            ),
        ],
    )
    recovered: list[str] = []
    prepared = MagicMock(
        tools=[],
        evidence=MagicMock(ledger_state_json=lambda: "{}"),
        episode=MagicMock(
            record=MagicMock(side_effect=lambda exchange: recovered.append("record"))
        ),
    )
    orchestrator = MagicMock(resolved_mode="research")
    orchestrator.prepare_run.return_value = prepared
    orchestrator.staged_artifacts.return_value = ()
    orchestrator.answer_stream = AsyncMock(
        return_value=({"chunks": [], "entities": [], "relationships": []}, None)
    )
    orchestrator.recover_from_fold = AsyncMock(
        side_effect=lambda run, snapshot: recovered.append("recovered")
    )
    registry = MagicMock(aclose=AsyncMock())
    executor = _executor()
    store = cast(Any, executor._store)
    store.load_routing = AsyncMock(
        return_value=MagicMock(
            resolved_mode="research",
            research_session_id=request.session_id,
            requested_mode="research",
            valid_modes=("fast", "research"),
        )
    )
    store.resolve = AsyncMock(return_value="research")
    executor.prepare_orchestrated_run = AsyncMock(  # type: ignore[method-assign]
        return_value=OrchestratorRun(
            orchestrator=orchestrator,
            image_descriptions=[],
            query_images=None,
            history=PriorTurns(),
            current_image_count=0,
            workspaces=["default"],
            registry=registry,
        )
    )
    session = MagicMock(
        owner_id="owner",
        run_id="run-1",
        prepared_input=request.as_request(),
        enter_phase=AsyncMock(),
        flush_tokens=AsyncMock(),
        emit_token=AsyncMock(),
        execution=MagicMock(session_store=journal, progress_store=MagicMock()),
    )

    result = await executor.execute(cast(RunSession, session))

    assert result["answer"] == ""
    assert "recovered" in recovered
    orchestrator.answer_stream.assert_awaited_once()
