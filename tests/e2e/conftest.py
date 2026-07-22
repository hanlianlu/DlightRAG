# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for E2E Playwright tests.

Starts a real FastAPI server on a random port with a mocked RAG service
manager so the browser can exercise the full HTML/JS/CSS pipeline without
needing PostgreSQL, LLM, or embedding backends.

Usage (opt-in, requires Playwright)::

    pytest tests/e2e/ -m e2e
"""

import base64
import socket
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from playwright.sync_api import Browser, Page, sync_playwright

from dlightrag.api.server import create_app
from dlightrag.citations.schemas import SourceReferencePayload
from dlightrag.storage.web_conversations import CommitTurnResult, StoredConversationImage
from dlightrag.web.conversation_models import (
    ConversationHistory,
    ConversationImageReference,
    ConversationSummary,
    ConversationTurn,
)
from dlightrag.web.conversations import PreparedWebConversation
from dlightrag.web.safe_html import safe_answer_done

MOCK_WORKSPACES = [
    {"workspace": "default", "display_name": "Default", "embedding_model": "voyage-multimodal-3.5"},
    {
        "workspace": "research",
        "display_name": "Research",
        "embedding_model": "voyage-multimodal-3.5",
    },
]

MOCK_WORKSPACE_LIST = ["default", "research"]


class E2EConversationService:
    """Resettable in-memory Web conversation service for browser-only tests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._conversations: dict[str, dict[str, Any]] = {}

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
            "content_revision": 0,
            "created_at": now,
            "updated_at": now,
            "turns": [],
            "images": {},
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

    async def history(self, _user: Any, conversation_id: str) -> ConversationHistory | None:
        with self._lock:
            value = self._conversations.get(conversation_id)
            if value is None:
                return None
            return ConversationHistory(
                conversation=self._summary(value),
                turns=list(value["turns"]),
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
            return self._conversations.pop(conversation_id, None) is not None

    async def prepare_answer(
        self,
        _user: Any,
        conversation_id: str,
        submission_id: str | None = None,
    ) -> PreparedWebConversation | None:
        del submission_id
        with self._lock:
            value = self._conversations.get(conversation_id)
            if value is None:
                return None
            text_history: list[dict[str, Any]] = []
            for turn in value["turns"]:
                text_history.extend(
                    (
                        {"role": "user", "content": turn.user_text},
                        {"role": "assistant", "content": turn.assistant_text},
                    )
                )
            return PreparedWebConversation(
                principal_id="e2e",
                conversation_id=conversation_id,
                content_revision=value["content_revision"],
                text_history=tuple(text_history),
            )

    async def image(
        self, _user: Any, conversation_id: str, image_id: str
    ) -> StoredConversationImage | None:
        with self._lock:
            value = self._conversations.get(conversation_id)
            if value is None:
                return None
            return value["images"].get(image_id)

    async def thumbnail(
        self, _user: Any, conversation_id: str, image_id: str
    ) -> StoredConversationImage | None:
        return await self.image(_user, conversation_id, image_id)

    def seed_image_history(self, *, turn_count: int) -> str:
        """Create one long image conversation for browser loading probes."""
        conversation_id = str(uuid4())
        now = datetime.now(UTC)
        png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/"
            "x8AAwMCAO+/p9sAAAAASUVORK5CYII="
        )
        value: dict[str, Any] = {
            "conversation_id": conversation_id,
            "title": "Image history",
            "content_revision": turn_count,
            "created_at": now,
            "updated_at": now,
            "turns": [],
            "images": {},
        }
        for index in range(1, turn_count + 1):
            image_id = str(uuid4())
            original_url = f"/web/conversations/{conversation_id}/images/{image_id}"
            value["images"][image_id] = StoredConversationImage(
                image_id=image_id,
                mime_type="image/png",
                image_bytes=png,
            )
            value["turns"].append(
                ConversationTurn(
                    turn_id=str(uuid4()),
                    turn_number=index,
                    user_text=f"Image question {index}",
                    assistant_text=f"Image answer {index}",
                    user_images=[
                        ConversationImageReference(
                            image_id=image_id,
                            ordinal=1,
                            mime_type="image/png",
                            url=original_url,
                            thumbnail_url=original_url + "/thumbnail",
                            label=f"Turn {index}, image 1",
                        )
                    ],
                    answer_sources={},
                    answer_html=safe_answer_done(
                        answer=f"Image answer {index}",
                        sources=[],
                        answer_images=[],
                    ),
                    queried_workspaces=["default"],
                    created_at=now,
                )
            )
        with self._lock:
            self._conversations[conversation_id] = value
        return conversation_id

    async def commit_answer(
        self,
        prepared: PreparedWebConversation,
        *,
        submission_id: str,
        user_text: str,
        assistant_text: str,
        answer_sources: dict[str, Any],
        queried_workspaces: list[str],
        images: tuple[Any, ...],
        image_descriptions: dict[str, str],
    ) -> CommitTurnResult:
        del submission_id
        with self._lock:
            value = self._conversations.get(prepared.conversation_id)
            if value is None or value["content_revision"] != prepared.content_revision:
                return CommitTurnResult(False, "conversation_changed", None, None)
            now = datetime.now(UTC)
            turn_number = len(value["turns"]) + 1
            turn_id = str(uuid4())
            image_references: list[ConversationImageReference] = []
            for image in images:
                value["images"][image.image_id] = StoredConversationImage(
                    image_id=image.image_id,
                    mime_type=image.mime_type,
                    image_bytes=image.image_bytes,
                )
                image_references.append(
                    ConversationImageReference(
                        image_id=image.image_id,
                        ordinal=image.ordinal,
                        mime_type=image.mime_type,
                        url=(
                            f"/web/conversations/{prepared.conversation_id}/images/{image.image_id}"
                        ),
                        thumbnail_url=(
                            f"/web/conversations/{prepared.conversation_id}/images/"
                            f"{image.image_id}/thumbnail"
                        ),
                        label=f"Turn {turn_number}, image {image.ordinal}",
                    )
                )
            source_values = answer_sources.get("sources", [])
            sources = [SourceReferencePayload.model_validate(item) for item in source_values]
            answer_images = answer_sources.get("answer_images", [])
            value["turns"].append(
                ConversationTurn(
                    turn_id=turn_id,
                    turn_number=turn_number,
                    user_text=user_text,
                    assistant_text=assistant_text,
                    user_images=image_references,
                    answer_sources=answer_sources,
                    answer_html=safe_answer_done(
                        answer=assistant_text,
                        sources=sources,
                        answer_images=answer_images,
                    ),
                    queried_workspaces=queried_workspaces,
                    created_at=now,
                )
            )
            value["content_revision"] += 1
            value["title"] = value["title"] or " ".join(user_text.split())[:120]
            value["updated_at"] = now
            summary = {
                "conversation_id": prepared.conversation_id,
                "title": value["title"],
                "content_revision": value["content_revision"],
                "created_at": value["created_at"],
                "updated_at": value["updated_at"],
            }
            return CommitTurnResult(
                saved=True,
                reason=None,
                summary=summary,
                turn_id=turn_id,
                current_image_ids=tuple(image.image_id for image in images),
                assistant_text=assistant_text,
                answer_sources=answer_sources,
                image_descriptions=image_descriptions,
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
    manager = AsyncMock()
    manager.config = SimpleNamespace(
        workspace="default",
        input_dir_path=Path("."),
        answer_stream_idle_timeout=120.0,
        citations=SimpleNamespace(highlights=SimpleNamespace(enabled=False)),
        embedding=SimpleNamespace(model="voyage-multimodal-3.5"),
        query_images=SimpleNamespace(
            max_current_images=3,
            max_upload_bytes=15 * 1024 * 1024,
        ),
    )

    async def _token_stream() -> Any:
        yield "DlightRAG is a "
        yield "multimodal RAG system."

    async def _mock_answer_stream(*__: Any, **___: Any) -> Any:
        return ({}, _token_stream())

    manager._aanswer_stream_prepared.side_effect = _mock_answer_stream
    manager.answer_image_capability = SimpleNamespace(
        status="supported",
        effective_max_images=3,
    )
    manager.alist_workspaces.return_value = MOCK_WORKSPACE_LIST
    manager.alist_workspace_records.return_value = MOCK_WORKSPACES
    manager.alist_ingested_files.return_value = []
    manager.aget_pipeline_status.return_value = {"busy": False, "pending_enqueues": 0}
    manager.aget_file_panel_snapshot.return_value = {
        "files": [],
        "pipeline_status": {"busy": False, "pending_enqueues": 0},
    }
    manager.aingest.return_value = {"job_id": "e2e-test-job", "file_count": 1}
    manager.adelete_files.return_value = {"deleted_count": 0}

    port = _free_port()
    import uvicorn

    with patch("dlightrag.api.server.RAGServiceManager.acreate", AsyncMock(return_value=manager)):
        app = create_app(include_web=True)
        app.state.web_conversation_service = e2e_conversation_service
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
