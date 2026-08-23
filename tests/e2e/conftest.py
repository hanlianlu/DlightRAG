# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for E2E Playwright tests.

Starts a real FastAPI server on a random port with a mocked Application so the
browser can exercise the full HTML/JS/CSS pipeline without
needing PostgreSQL, LLM, or embedding backends.

Usage (opt-in, requires Playwright)::

    pytest tests/e2e/ -m e2e
"""

import base64
import socket
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Generator, Mapping
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from playwright.sync_api import Browser, Page, sync_playwright

from dlightrag.ai.capacity import CONTEXT_POLICY_REVISION, ModelProfile
from dlightrag.ai.catalog import MODEL_CATALOG_REVISION
from dlightrag.ai.fingerprints import ModelFingerprint
from dlightrag.ai.media import MODEL_IMAGE_MAX_PIXELS
from dlightrag.ai.settings import (
    MODEL_ROLE_NAMES,
    EmbeddingSettings,
    ModelCapacityOverrideSettings,
    ModelRoleSettings,
    ModelSettings,
)
from dlightrag.answer.capabilities import AnswerCapabilities
from dlightrag.answer.capability import AnswerImageCapability
from dlightrag.answer.runs.execution import (
    AnswerRunInput,
    AttachmentReference,
    PinnedModelProfile,
)
from dlightrag.api.server import create_app
from dlightrag.config import DlightragConfig, set_config
from dlightrag.runtime import AnswerRunEvent, AnswerRunRecord
from dlightrag.services.answers import AnswerInputArtifact
from dlightrag.web.conversation_models import ConversationHistory, ConversationSummary, LinkedTurn
from dlightrag.web.conversations import WebAnswerSubmission, project_conversation_turn
from tests.config_helpers import mutate_config

MOCK_WORKSPACES = [
    {"workspace": "default", "display_name": "Default", "embedding_model": "voyage-multimodal-3.5"},
    {
        "workspace": "research",
        "display_name": "Research",
        "embedding_model": "voyage-multimodal-3.5",
    },
]

MOCK_WORKSPACE_LIST = ["default", "research"]

ANSWER_TEXT = "DlightRAG is a multimodal RAG system."
_ANSWER_TOKENS = ("DlightRAG is a ", "multimodal RAG system.")
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)
_E2E_PROFILE = ModelProfile(
    context_window_tokens=1_000_000,
    max_output_tokens=128_000,
    supports_images=True,
    supports_tools=True,
    supports_reasoning=True,
)


def _run_request(
    *,
    query: str,
    workspaces: Any,
    attachments: Any,
    idempotency_fingerprint: str,
) -> dict[str, Any]:
    return AnswerRunInput(
        query=query,
        workspaces=tuple(workspaces),
        attachments=tuple(
            AttachmentReference(
                digest=attachment.content_sha256,
                filename=attachment.filename,
                mime_type=attachment.mime_type,
                ordinal=attachment.ordinal,
                byte_size=attachment.byte_size,
            )
            for attachment in attachments
        ),
        pinned_models=tuple(
            PinnedModelProfile(
                role=role,
                fingerprint=ModelFingerprint("openai", f"e2e-{role}", None),
                profile=_E2E_PROFILE,
            )
            for role in MODEL_ROLE_NAMES
        ),
        context_policy_revision=CONTEXT_POLICY_REVISION,
        model_catalog_revision=MODEL_CATALOG_REVISION,
        idempotency_fingerprint=idempotency_fingerprint,
    ).as_request()


def _run_record(
    run_id: str,
    request: dict[str, Any],
    *,
    status: str,
    result: dict[str, Any] | None = None,
    cancel_requested: bool = False,
) -> AnswerRunRecord:
    now = datetime.now(UTC)
    terminal = status in ("succeeded", "failed", "cancelled")
    return AnswerRunRecord(
        owner_id="e2e",
        run_id=run_id,
        idempotency_key=None,
        prepared_input=request,
        status=status,  # type: ignore[arg-type]
        phase=None,
        stop_reason=None,
        cancel_requested_at=now if cancel_requested else None,
        lease_owner=None,
        lease_expires_at=None,
        fencing_epoch=0,
        durable_progress_version=0,
        last_reclaim_progress_version=0,
        reclaims_without_progress=0,
        next_event_sequence=1,
        events_trimmed_at=None,
        result=result,
        error_kind=None,
        error_message=None,
        created_at=now,
        updated_at=now,
        started_at=None,
        finished_at=now if terminal else None,
    )


class E2EConversationService:
    """Resettable in-memory Web conversation service for browser-only tests.

    Mirrors the durable contract: one submission becomes one run plus its linked
    turn, and every read projects that turn from the run's recorded state.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._conversations: dict[str, dict[str, Any]] = {}
            self._runs: dict[str, dict[str, Any]] = {}

    async def initialize(self) -> None:
        return None

    @staticmethod
    def _summary(value: dict[str, Any]) -> ConversationSummary:
        return ConversationSummary(
            conversation_id=value["conversation_id"],
            title=value["title"],
            created_at=value["created_at"],
            updated_at=value["updated_at"],
        )

    async def create(self, _user: Any) -> ConversationSummary:
        now = datetime.now(UTC)
        value = {
            "conversation_id": str(uuid4()),
            "title": None,
            "created_at": now,
            "updated_at": now,
            "turns": [],
        }
        with self._lock:
            self._conversations[value["conversation_id"]] = value
        return self._summary(value)

    async def list(self, _user: Any) -> list[ConversationSummary]:
        with self._lock:
            values = sorted(
                self._conversations.values(),
                key=lambda value: (value["updated_at"], value["conversation_id"]),
                reverse=True,
            )
            return [self._summary(value) for value in values]

    async def history(
        self, _user: Any, conversation_id: str, **_: Any
    ) -> ConversationHistory | None:
        with self._lock:
            value = self._conversations.get(conversation_id)
            if value is None:
                return None
            return ConversationHistory(
                conversation=self._summary(value),
                turns=[project_conversation_turn(turn) for turn in value["turns"]],
            )

    async def rename(
        self, _user: Any, conversation_id: str, title: str
    ) -> ConversationSummary | None:
        with self._lock:
            value = self._conversations.get(conversation_id)
            if value is None:
                return None
            value["title"] = title
            value["updated_at"] = datetime.now(UTC)
            return self._summary(value)

    async def delete(self, _user: Any, conversation_id: str) -> bool:
        with self._lock:
            value = self._conversations.pop(conversation_id, None)
            for turn in (value or {}).get("turns", []):
                self._runs.pop(turn.run.run_id, None)
            return value is not None

    async def delete_all(self, _user: Any) -> int:
        with self._lock:
            count = len(self._conversations)
            self._conversations.clear()
            self._runs.clear()
            return count

    async def start_answer(
        self,
        _user: Any,
        *,
        conversation_id: str | None,
        submission_id: str,
        query: str,
        workspaces: Any,
        attachments: Any = (),
        mode: str | None = None,
    ) -> WebAnswerSubmission | None:
        with self._lock:
            if conversation_id is None:
                for candidate in self._conversations.values():
                    for turn in candidate["turns"]:
                        if turn.submission_id == submission_id:
                            return WebAnswerSubmission(
                                run=turn.run,
                                turn_id=turn.turn_id,
                                turn_number=turn.turn_number,
                                conversation=self._summary(candidate),
                            )
                now = datetime.now(UTC)
                conversation_id = str(uuid4())
                self._conversations[conversation_id] = {
                    "conversation_id": conversation_id,
                    "title": None,
                    "created_at": now,
                    "updated_at": now,
                    "turns": [],
                }
            value = self._conversations.get(conversation_id)
            if value is None:
                return None
            for turn in value["turns"]:
                if turn.submission_id == submission_id:
                    return WebAnswerSubmission(
                        run=turn.run,
                        turn_id=turn.turn_id,
                        turn_number=turn.turn_number,
                        conversation=self._summary(value),
                    )
            run_id = str(uuid4())
            requested_mode = mode or "auto"
            request = _run_request(
                query=query,
                workspaces=workspaces,
                attachments=attachments,
                idempotency_fingerprint=submission_id,
            )
            turn = LinkedTurn(
                turn_id=str(uuid4()),
                turn_number=len(value["turns"]) + 1,
                submission_id=submission_id,
                created_at=datetime.now(UTC),
                run=_run_record(run_id, request, status="queued"),
            )
            value["turns"].append(turn)
            value["title"] = value["title"] or " ".join(query.split())[:120]
            value["updated_at"] = datetime.now(UTC)
            self._runs[run_id] = {
                "conversation_id": conversation_id,
                "requested_mode": requested_mode,
                "bytes": {
                    attachment.ordinal: (attachment.attachment_bytes, attachment.mime_type)
                    for attachment in attachments
                },
            }
            return WebAnswerSubmission(
                run=turn.run,
                turn_id=turn.turn_id,
                turn_number=turn.turn_number,
                conversation=self._summary(value),
            )

    async def turn_for_run(self, _user: Any, run_id: str) -> LinkedTurn | None:
        with self._lock:
            return self._find_turn(run_id)

    def _find_turn(self, run_id: str) -> LinkedTurn | None:
        entry = self._runs.get(run_id)
        if entry is None:
            return None
        value = self._conversations.get(entry["conversation_id"])
        if value is None:
            return None
        return next((turn for turn in value["turns"] if turn.run.run_id == run_id), None)

    def finish_run(self, run_id: str, *, status: str = "succeeded") -> None:
        """Record the terminal state a worker would have committed."""
        with self._lock:
            entry = self._runs.get(run_id)
            if entry is None:
                return
            value = self._conversations[entry["conversation_id"]]
            for index, turn in enumerate(value["turns"]):
                if turn.run.run_id != run_id:
                    continue
                result = (
                    {
                        "answer": ANSWER_TEXT,
                        "contexts": {"chunks": []},
                        "sources": [],
                        "answer_images": [],
                        "trace": {},
                        "image_descriptions": [],
                    }
                    if status == "succeeded"
                    else None
                )
                value["turns"][index] = LinkedTurn(
                    turn_id=turn.turn_id,
                    turn_number=turn.turn_number,
                    submission_id=turn.submission_id,
                    created_at=turn.created_at,
                    run=_run_record(run_id, turn.run.prepared_input, status=status, result=result),
                )

    async def attachment(self, _user: Any, run_id: str, ordinal: int) -> Any:
        with self._lock:
            entry = self._runs.get(run_id)
            if entry is None or ordinal not in entry["bytes"]:
                return None
            turn = self._find_turn(run_id)
            if turn is None:
                return None
            content, _mime = entry["bytes"][ordinal]
        reference = next(
            item
            for item in _request_attachments(turn.run.prepared_input or {})
            if item["ordinal"] == ordinal
        )
        return AnswerInputArtifact(
            reference_kind="current_attachment",
            ordinal=ordinal,
            filename=reference["filename"],
            mime_type=reference["mime_type"],
            digest=reference["digest"],
            content=content,
        )

    async def thumbnail(self, user: Any, run_id: str, ordinal: int) -> tuple[bytes, str] | None:
        stored = await self.attachment(user, run_id, ordinal)
        if stored is None or not stored.mime_type.lower().startswith("image/"):
            return None
        return stored.content, stored.mime_type

    def seed_image_history(self, *, turn_count: int) -> str:
        """Create one long image conversation for browser loading probes."""
        conversation_id = str(uuid4())
        now = datetime.now(UTC)
        value: dict[str, Any] = {
            "conversation_id": conversation_id,
            "title": "Image history",
            "created_at": now,
            "updated_at": now,
            "turns": [],
        }
        for index in range(1, turn_count + 1):
            run_id = str(uuid4())
            submission_id = str(uuid4())
            request = _run_request(
                query=f"Image question {index}",
                workspaces=["default"],
                attachments=(
                    SimpleNamespace(
                        content_sha256=f"{index:064d}",
                        filename="chart.png",
                        mime_type="image/png",
                        ordinal=1,
                        byte_size=len(_PNG),
                    ),
                ),
                idempotency_fingerprint=submission_id,
            )
            value["turns"].append(
                LinkedTurn(
                    turn_id=str(uuid4()),
                    turn_number=index,
                    submission_id=submission_id,
                    created_at=now,
                    run=_run_record(
                        run_id,
                        request,
                        status="succeeded",
                        result={
                            "answer": f"Image answer {index}",
                            "contexts": {"chunks": []},
                            "sources": [],
                            "answer_images": [],
                            "trace": {},
                            "image_descriptions": [],
                        },
                    ),
                )
            )
            self._runs[run_id] = {
                "conversation_id": conversation_id,
                "bytes": {1: (_PNG, "image/png")},
            }
        with self._lock:
            self._conversations[conversation_id] = value
        return conversation_id


def _request_attachments(request: Mapping[str, Any]) -> list[dict[str, Any]]:
    return list(request.get("attachments") or [])


def _event(sequence: int, event_type: str, payload: dict[str, Any]) -> AnswerRunEvent:
    return AnswerRunEvent(
        sequence=sequence,
        event_type=event_type,  # type: ignore[arg-type]
        payload=payload,
        created_at=datetime.now(UTC),
    )


def _free_port() -> int:
    """Return an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def e2e_conversation_service() -> E2EConversationService:
    return E2EConversationService()


@pytest.fixture(scope="session")
def e2e_base_url(
    e2e_conversation_service: E2EConversationService,
) -> Generator[str, Any]:
    """Start one real FastAPI server for the E2E session on a random port."""
    working_directory = tempfile.TemporaryDirectory(prefix="dlightrag-e2e-")
    application_config = DlightragConfig(
        deployment={"working_dir": working_directory.name},
        models={
            "chat": ModelRoleSettings(default=ModelSettings(model="gpt-5.4-mini", api_key="test")),
            "capacity_overrides": [
                ModelCapacityOverrideSettings(
                    provider="openai",
                    model="gpt-5.4-mini",
                    context_window_tokens=400_000,
                    max_output_tokens=128_000,
                    supports_images=True,
                    supports_tools=True,
                    supports_reasoning=True,
                )
            ],
            "embedding": EmbeddingSettings(
                provider="voyage",
                model="voyage-multimodal-3.5",
                api_key="test",
                startup_probe=False,
            ),
        },
    )
    mutate_config(application_config, "answer.generation.image_max_pixels", MODEL_IMAGE_MAX_PIXELS)
    mutate_config(application_config, "answer.generation.max_attachments", 6)
    mutate_config(application_config, "answer.generation.max_attachment_bytes", 100 * 1024 * 1024)
    mutate_config(
        application_config,
        "answer.generation.max_total_attachment_bytes",
        128 * 1024 * 1024,
    )
    set_config(application_config)
    application_double = AsyncMock()
    application_double.config = application_config

    def _events(*, owner_id: str, run_id: str, after_sequence: int = 0) -> Any:
        """Replay this run's durable events from the caller's cursor.

        The browser is the only consumer, so the whole log is short and
        deterministic: two phases, the answer tokens, then the terminal event.
        Resuming after a sequence never repeats an earlier frame.
        """
        del owner_id

        async def _iterate() -> Any:
            log: list[AnswerRunEvent] = [
                _event(1, "progress", {"phase": "planning"}),
                _event(2, "progress", {"phase": "generating"}),
                *(
                    _event(3 + index, "token", {"text": token})
                    for index, token in enumerate(_ANSWER_TOKENS)
                ),
            ]
            e2e_conversation_service.finish_run(run_id)
            turn = await e2e_conversation_service.turn_for_run(None, run_id)
            result = dict(turn.run.result or {}) if turn is not None else {}
            log.append(
                _event(
                    3 + len(_ANSWER_TOKENS),
                    "done",
                    {"status": "succeeded", "result": result},
                )
            )
            for event in log:
                if event.sequence > after_sequence:
                    yield event

        return _iterate()

    answer_image_capability = AnswerImageCapability(
        status="supported",
        configured_ceiling=3,
        effective_max_images=3,
        provider="test",
        base_url=None,
        model="test-model",
        failure_kind=None,
    )
    application_double.answers = SimpleNamespace(
        subscribe=MagicMock(side_effect=_events),
        cancel=AsyncMock(),
        capabilities=AsyncMock(
            return_value=AnswerCapabilities(
                answer=answer_image_capability,
                vlm_status="unknown",
            )
        ),
    )
    application_double.corpora.list_workspaces.return_value = MOCK_WORKSPACE_LIST
    application_double.corpora.alist_workspace_records.return_value = MOCK_WORKSPACES
    application_double.corpora.list_ingested_files.return_value = []
    application_double.corpora.get_pipeline_status.return_value = {
        "busy": False,
        "pending_enqueues": 0,
    }
    application_double.corpora.file_panel_snapshot.return_value = {
        "files": [],
        "pipeline_status": {"busy": False, "pending_enqueues": 0},
    }
    application_double.corpora.delete_files.return_value = {"deleted_count": 0}
    application_double.web_conversations = e2e_conversation_service

    port = _free_port()
    import uvicorn

    with patch(
        "dlightrag.api.server.Application.acreate",
        AsyncMock(return_value=application_double),
    ):
        app = create_app(include_web_app=True)
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        server = uvicorn.Server(config)
        t = threading.Thread(target=server.run, daemon=True)
        t.start()
        base_url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"{base_url}/web/", timeout=0.25):
                    break
            except OSError, urllib.error.URLError:
                time.sleep(0.05)
        else:
            server.should_exit = True
            t.join(timeout=3)
            raise RuntimeError("E2E server did not become ready")
        yield base_url
        server.should_exit = True
        t.join(timeout=3)
    working_directory.cleanup()


@pytest.fixture(scope="session")
def browser() -> Generator[Browser, Any]:
    """Session-scoped browser — reuse across tests for speed."""
    with sync_playwright() as pw:
        b = pw.chromium.launch(headless=True)
        try:
            yield b
        finally:
            b.close()


@pytest.fixture
def page(
    browser: Browser,
    e2e_base_url: str,
    e2e_conversation_service: E2EConversationService,
) -> Generator[Page, Any]:
    """Fresh page per test, already pointed at the running server."""
    e2e_conversation_service.reset()
    context = browser.new_context(base_url=e2e_base_url)
    page_obj = context.new_page()
    page_obj.set_default_timeout(10000)
    yield page_obj
    context.close()
