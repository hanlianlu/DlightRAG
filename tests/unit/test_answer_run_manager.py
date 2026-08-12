# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Manager-owned durable Answer run creation, execution, status, and cancellation."""

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, cast
from unittest.mock import MagicMock

from dlightrag.citations.streaming import AnswerStream
from dlightrag.core.agent.orchestrator import AnswerOrchestrator
from dlightrag.core.answer.synthesizer import AnswerSynthesizer
from dlightrag.core.answer_runs.coordinator import RunSession
from dlightrag.core.answer_runs.execution import AnswerRunInput, AttachmentReference
from dlightrag.core.memory.conversation import PriorTurns
from dlightrag.core.retrieval.protocols import RetrievalResult
from dlightrag.core.servicemanager import RAGServiceManager, _OrchestratorRun
from dlightrag.storage.answer_runs import (
    ClaimedRun,
    PendingArtifact,
    RunCheckpoint,
    artifact_digest,
)

_OWNER = "owner-alpha"


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
    manager._answer_run_store = cast(Any, store)
    manager._answer_coordinator = None
    return manager


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
        assert session.phases == ["searching", "generating"]
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

        resources = manager._answer_run_resources(request, owner_id=_OWNER, store=cast(Any, store))

        assert resources is not None
        assert resources[0].filename == "a.txt"
        assert resources[0].content is None
        assert resources[0].loader is not None
        assert await resources[0].loader() == b"attachment-bytes"


def test_claimed_checkpoint_is_offered_to_the_executor() -> None:
    """A claim carries the envelope the executor must restore before its first call."""
    checkpoint = RunCheckpoint(version=1, completed_turns=2, state={"episode": {}})
    claimed = ClaimedRun(run=MagicMock(completed_turns=2), checkpoint=checkpoint)
    assert claimed.checkpoint is not None
    assert claimed.checkpoint.completed_turns == claimed.run.completed_turns
