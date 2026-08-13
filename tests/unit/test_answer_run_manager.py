# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Manager-owned durable Answer run creation, execution, status, and cancellation."""

import asyncio
import base64
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from dlightrag.citations.streaming import AnswerStream
from dlightrag.core.agent.orchestrator import AnswerOrchestrator
from dlightrag.core.answer.synthesizer import AnswerSynthesizer
from dlightrag.core.answer_runs.coordinator import RunSession
from dlightrag.core.answer_runs.execution import AnswerRunInput, AttachmentReference
from dlightrag.core.memory.conversation import PriorTurns
from dlightrag.core.retrieval.protocols import RetrievalResult
from dlightrag.core.servicemanager import (
    RAGServiceManager,
    RAGServiceUnavailableError,
    _OrchestratorRun,
)
from dlightrag.storage.answer_runs import (
    ClaimedRun,
    PendingArtifact,
    RunCheckpoint,
    artifact_digest,
)

_OWNER = "owner-alpha"
_VISUAL_BYTES = b"\x89PNG\r\n\x1a\nfake-corpus-visual"
_VISUAL_B64 = base64.b64encode(_VISUAL_BYTES).decode("ascii")


class _RecordingStore:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.cancelled: list[str] = []

    async def create_run(
        self,
        *,
        owner_id: str,
        request: Mapping[str, Any],
        idempotency_key: str | None = None,
        artifacts: Sequence[PendingArtifact] = (),
        references: Sequence[Any] = (),
    ) -> Any:
        self.created.append(
            {
                "owner_id": owner_id,
                "request": dict(request),
                "idempotency_key": idempotency_key,
                "artifacts": list(artifacts),
                "references": list(references),
            }
        )
        return MagicMock(replayed=False)

    async def get_run(self, *, owner_id: str, run_id: str) -> Any:
        return MagicMock(run_id=run_id, owner_id=owner_id, status="queued")

    async def request_cancellation(self, *, owner_id: str, run_id: str) -> Any:
        self.cancelled.append(run_id)
        return MagicMock(outcome="pending")

    async def load_artifact(self, *, owner_id: str, digest: str) -> bytes | None:
        return b"attachment-bytes"

    async def list_run_artifacts(self, *, owner_id: str, run_id: str) -> tuple[Any, ...]:
        return ()

    async def claim_next(self, *, worker_id: str) -> Any:
        await asyncio.sleep(0.05)
        return None

    async def sweep_once(self) -> Any:
        return None


class _Session:
    """The fenced surface the manager's executor is allowed to touch."""

    def __init__(self, *, request: Mapping[str, Any], checkpoint: RunCheckpoint | None) -> None:
        self.owner_id = _OWNER
        self.run_id = "run-1"
        self.request = request
        self.checkpoint = checkpoint
        self.completed_turns = checkpoint.completed_turns if checkpoint else 0
        self.phases: list[str] = []
        self.tokens: list[str] = []
        self._pending: list[str] = []

    async def check_cancelled(self) -> None:
        return None

    async def enter_phase(self, phase: str) -> None:
        self.phases.append(phase)

    async def emit_token(self, text: str) -> None:
        self._pending.append(text)

    async def flush_tokens(self) -> None:
        if self._pending:
            self.tokens.append("".join(self._pending))
            self._pending.clear()

    async def commit_checkpoint(self, envelope: Mapping[str, Any]) -> None:
        self.completed_turns += 1


def _manager(store: _RecordingStore) -> RAGServiceManager:
    config = MagicMock()
    config.max_async = 2
    manager = RAGServiceManager.__new__(RAGServiceManager)
    manager._config = config
    manager._closed = False
    manager._answer_run_store = cast(Any, store)
    manager._answer_coordinator = None
    manager._answer_store_lock = asyncio.Lock()
    manager._answer_runtime_lock = asyncio.Lock()
    return manager


def _bare_manager() -> RAGServiceManager:
    manager = _manager(_RecordingStore())
    manager._answer_run_store = None
    return manager


def _install_store_class(
    monkeypatch: pytest.MonkeyPatch,
    *,
    on_initialize: Callable[[], Awaitable[None]] | None = None,
) -> list[Any]:
    """Count how many stores the manager builds while callers race to start it."""
    created: list[Any] = []

    class _Store:
        def __init__(self) -> None:
            self.initializations = 0
            created.append(self)

        async def initialize(self, *, validate_only: bool = False) -> None:
            self.initializations += 1
            if on_initialize is not None:
                await on_initialize()
                return
            await asyncio.sleep(0)
            await asyncio.sleep(0)

        async def claim_next(self, *, worker_id: str) -> Any:
            await asyncio.sleep(0.05)
            return None

        async def sweep_once(self) -> Any:
            return None

    monkeypatch.setattr("dlightrag.storage.answer_runs.PGAnswerRunStore", _Store)
    return created


def _runtime_tasks() -> list[str]:
    names: list[str] = []
    for task in asyncio.all_tasks():
        code = getattr(task.get_coro(), "cr_code", None)
        name = str(getattr(code, "co_qualname", ""))
        if name.startswith("AnswerRunCoordinator."):
            names.append(name)
    return sorted(names)


_ONE_RUNTIME = [
    "AnswerRunCoordinator._maintain_forever",
    "AnswerRunCoordinator._schedule_forever",
    "AnswerRunCoordinator._sweep_forever",
]


class _Synthesizer:
    async def generate_stream(
        self,
        query: str,
        contexts: Any,
        *,
        conversation_history: PriorTurns | None = None,
    ) -> tuple[Any, AsyncIterator[str]]:
        async def _stream() -> AsyncIterator[str]:
            yield "hello "
            yield "world"

        return contexts, AnswerStream(_stream())


async def _retrieve(query: str) -> RetrievalResult:
    return RetrievalResult(
        contexts={
            "chunks": [
                {
                    "chunk_id": "c1",
                    "content": "evidence",
                    "reference_id": "1",
                    "file_path": "book.pdf",
                    "_workspace": "default",
                    "metadata": {
                        "source_type": "corpus",
                        "source_uri": "corpus://book.pdf",
                        "source_download_locator": "corpus://book.pdf",
                        "title": "book.pdf",
                    },
                }
            ],
            "entities": [],
            "relationships": [],
        },
        trace={"retrieval": "ok"},
    )


class TestRunCreation:
    async def test_normalized_request_and_attachment_blobs_are_stored(self) -> None:
        store = _RecordingStore()
        manager = _manager(store)
        content = b"attachment-bytes"
        request = AnswerRunInput(
            query="why",
            workspaces=("default",),
            attachments=(
                AttachmentReference(
                    digest=artifact_digest(content),
                    filename="a.txt",
                    mime_type="text/plain",
                    ordinal=0,
                ),
            ),
        )

        await manager.astart_answer_run(
            owner_id=_OWNER,
            request=request,
            idempotency_key="key-1",
            attachment_bytes=[content],
        )
        await _close_runtime(manager)

        created = store.created[0]
        assert created["owner_id"] == _OWNER
        assert created["idempotency_key"] == "key-1"
        assert created["request"]["query"] == "why"
        assert created["request"]["workspaces"] == ["default"]
        assert created["request"]["attachments"][0]["digest"] == artifact_digest(content)
        assert created["artifacts"][0].content == content
        assert created["references"][0].reference_kind == "current_attachment"

    async def test_status_and_cancellation_are_owner_scoped(self) -> None:
        store = _RecordingStore()
        manager = _manager(store)

        run = await manager.aget_answer_run(owner_id=_OWNER, run_id="run-1")
        outcome = await manager.acancel_answer_run(owner_id=_OWNER, run_id="run-1")

        assert run is not None and run.owner_id == _OWNER
        assert outcome.outcome == "pending"
        assert store.cancelled == ["run-1"]

    async def test_accepting_a_run_starts_the_one_local_runtime(self) -> None:
        store = _RecordingStore()
        manager = _manager(store)

        try:
            await manager.astart_answer_run(
                owner_id=_OWNER, request=AnswerRunInput(query="why", workspaces=("default",))
            )
            assert manager._answer_coordinator is not None
            assert _runtime_tasks() == _ONE_RUNTIME
        finally:
            await _close_runtime(manager)


class TestRuntimeStartup:
    """Concurrent callers share one store, one coordinator, and one task pair."""

    async def test_simultaneous_starts_share_one_store_and_one_coordinator(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        created = _install_store_class(monkeypatch)
        manager = _bare_manager()

        try:
            await asyncio.gather(*(manager.astart_answer_runtime() for _ in range(6)))
            assert len(created) == 1
            assert created[0].initializations == 1
            assert manager._answer_run_store is created[0]
            assert _runtime_tasks() == _ONE_RUNTIME
        finally:
            await _close_runtime(manager)

        assert _runtime_tasks() == []

    async def test_simultaneous_subscribers_start_one_runtime(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        created = _install_store_class(monkeypatch)
        manager = _bare_manager()

        generators = await asyncio.gather(
            *(manager.asubscribe_answer_run(owner_id=_OWNER, run_id="run-1") for _ in range(6))
        )
        try:
            assert len(created) == 1
            assert _runtime_tasks() == _ONE_RUNTIME
        finally:
            for generator in generators:
                await generator.aclose()
            await _close_runtime(manager)

    async def test_a_closed_manager_never_recreates_the_runtime(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        created = _install_store_class(monkeypatch)
        manager = _bare_manager()
        manager._closed = True

        with pytest.raises(RAGServiceUnavailableError):
            await manager.astart_answer_runtime()

        assert manager._answer_coordinator is None
        assert created == []
        assert _runtime_tasks() == []

    async def test_closing_mid_start_leaves_no_orphaned_worker(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        initializing = asyncio.Event()
        release = asyncio.Event()

        async def _hold() -> None:
            initializing.set()
            await release.wait()

        created = _install_store_class(monkeypatch, on_initialize=_hold)
        manager = _bare_manager()

        start = asyncio.ensure_future(manager.astart_answer_runtime())
        await initializing.wait()
        manager._closed = True
        release.set()

        with pytest.raises(RAGServiceUnavailableError):
            await start

        assert len(created) == 1
        assert manager._answer_coordinator is None
        assert _runtime_tasks() == []


class TestRunExecution:
    async def test_executes_immutable_input_and_returns_the_canonical_result(self) -> None:
        store = _RecordingStore()
        manager = _manager(store)
        orchestrator = AnswerOrchestrator(
            synthesizer=cast(AnswerSynthesizer, _Synthesizer()),
            retrieve_knowledge_base=_retrieve,
        )
        prepared: list[Any] = []

        async def _prepare(turn: Any, **kwargs: Any) -> _OrchestratorRun:
            prepared.append((turn, kwargs))
            return _OrchestratorRun(
                orchestrator=orchestrator,
                image_descriptions=[],
                query_images=None,
                history=PriorTurns(),
                current_image_count=0,
                ws_list=["default"],
                registry=None,
            )

        manager._prepare_orchestrated_run = _prepare  # type: ignore[method-assign]
        request = AnswerRunInput(query="why", workspaces=("default",)).as_request()
        session = _Session(request=request, checkpoint=None)

        result = await manager._execute_answer_run(cast(RunSession, session))

        assert session.tokens == ["hello world"]
        assert session.phases == ["planning", "searching", "generating"]
        assert result["answer"] == "hello world"
        assert result["trace"]["query_image_description_count"] == 0
        assert "sources" in result and "contexts" in result
        turn, kwargs = prepared[0]
        assert turn.current_query == "why"
        assert kwargs["workspaces"] == ["default"]

    async def test_attachments_are_replayed_as_lazy_readers_over_stored_bytes(self) -> None:
        store = _RecordingStore()
        manager = _manager(store)
        request = AnswerRunInput(
            query="why",
            attachments=(
                AttachmentReference(
                    digest=artifact_digest(b"attachment-bytes"),
                    filename="a.txt",
                    mime_type="text/plain",
                    ordinal=0,
                ),
            ),
        )

        resources = await manager._answer_run_resources(
            request, owner_id=_OWNER, store=cast(Any, store)
        )

        assert resources is not None
        assert resources[0].filename == "a.txt"
        assert resources[0].content is None
        assert resources[0].loader is not None
        assert await resources[0].loader() == b"attachment-bytes"

    async def test_image_attachments_are_materialized_for_current_image_admission(
        self,
    ) -> None:
        store = _RecordingStore()
        manager = _manager(store)
        request = AnswerRunInput(
            query="why",
            attachments=(
                AttachmentReference(
                    digest=artifact_digest(b"attachment-bytes"),
                    filename="chart.png",
                    mime_type="image/png",
                    ordinal=0,
                ),
            ),
        )

        resources = await manager._answer_run_resources(
            request, owner_id=_OWNER, store=cast(Any, store)
        )

        assert resources is not None
        # A lazy reader would arrive after current-image admission has run.
        assert resources[0].content == b"attachment-bytes"
        assert resources[0].loader is None

    async def test_the_canonical_result_holds_no_raw_context_payloads(self) -> None:
        store = _RecordingStore()
        manager = _manager(store)
        orchestrator = AnswerOrchestrator(
            synthesizer=cast(AnswerSynthesizer, _CitingSynthesizer()),
            retrieve_knowledge_base=_retrieve_visual,
        )

        async def _prepare(turn: Any, **kwargs: Any) -> _OrchestratorRun:
            return _OrchestratorRun(
                orchestrator=orchestrator,
                image_descriptions=[],
                query_images=None,
                history=PriorTurns(),
                current_image_count=0,
                ws_list=["default"],
                registry=None,
            )

        manager._prepare_orchestrated_run = _prepare  # type: ignore[method-assign]
        request = AnswerRunInput(query="why", workspaces=("default",)).as_request()

        result = await manager._execute_answer_run(
            cast(RunSession, _Session(request=request, checkpoint=None))
        )

        chunk = result["contexts"]["chunks"][0]
        assert "image_data" not in chunk
        assert "_evidence_key" not in chunk
        assert "source_uri" not in chunk["metadata"]
        assert "source_download_locator" not in chunk["metadata"]
        assert _VISUAL_B64 not in json.dumps(result)
        assert "data:image" not in json.dumps(result)
        assert result["sources"][0]["source_uri"] == "corpus://book.pdf"
        assert result["answer_images"][0]["chunk_id"] == "c1"
        assert result["trace"]["retrieval"] == "ok"


class _CitingSynthesizer:
    async def generate_stream(
        self,
        query: str,
        contexts: Any,
        *,
        conversation_history: PriorTurns | None = None,
    ) -> tuple[Any, AsyncIterator[str]]:
        async def _stream() -> AsyncIterator[str]:
            yield "the drawing shows it [1]"

        return contexts, AnswerStream(_stream())


async def _retrieve_visual(query: str) -> RetrievalResult:
    return RetrievalResult(
        contexts={
            "chunks": [
                {
                    "chunk_id": "c1",
                    "content": "evidence",
                    "reference_id": "1",
                    "file_path": "book.pdf",
                    "_workspace": "default",
                    "_evidence_key": "search_knowledge_base:c1",
                    "image_data": _VISUAL_B64,
                    "metadata": {
                        "source_type": "corpus",
                        "source_uri": "corpus://book.pdf",
                        "source_download_locator": "/srv/private/book.pdf",
                        "title": "book.pdf",
                    },
                }
            ],
            "entities": [],
            "relationships": [],
        },
        trace={"retrieval": "ok"},
    )


async def _close_runtime(manager: RAGServiceManager) -> None:
    coordinator = manager._answer_coordinator
    manager._answer_coordinator = None
    if coordinator is not None:
        await coordinator.aclose()


def test_claimed_checkpoint_is_offered_to_the_executor() -> None:
    """A claim carries the envelope the executor must restore before its first call."""
    checkpoint = RunCheckpoint(version=1, completed_turns=2, state={"episode": {}})
    claimed = ClaimedRun(run=MagicMock(completed_turns=2), checkpoint=checkpoint)
    assert claimed.checkpoint is not None
    assert claimed.checkpoint.completed_turns == claimed.run.completed_turns
