# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for E2E Playwright tests.

Starts a real FastAPI server on a random port with a mocked RAG service
manager so the browser can exercise the full HTML/JS/CSS pipeline without
needing PostgreSQL, LLM, or embedding backends.

Usage (opt-in, requires Playwright)::

    pytest tests/e2e/ -m e2e
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock

import pytest
from playwright.sync_api import Browser, Page, Playwright

from dlightrag.api.server import create_app

MOCK_WORKSPACES = [
    {"workspace": "default", "display_name": "Default", "embedding_model": "voyage-multimodal-3.5"}
]

MOCK_WORKSPACE_LIST = [{"workspace": "default", "display_name": "Default"}]


def _free_port() -> int:
    """Return an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def e2e_base_url() -> Generator[str, Any, None]:
    """Start a real FastAPI server with a mocked manager on a random port."""
    app = create_app(include_web=True)

    manager = AsyncMock()
    manager.aretrieve.return_value = {"chunks": [], "sources": []}
    manager.aanswer.return_value = {
        "answer": "DlightRAG is a multimodal RAG system.",
        "sources": [],
    }

    # SSE streaming: yield token → preview → done → meta
    async def _mock_answer_stream(**__: Any) -> Any:
        yield {"type": "progress", "phase": "generating"}
        yield {"type": "token", "data": "DlightRAG is a "}
        yield {
            "type": "preview",
            "data": "<p>DlightRAG is a <strong>multimodal RAG</strong></p>",
        }
        yield {
            "type": "done",
            "answer": "DlightRAG is a multimodal RAG system.",
            "html": '<div id="answer-content"><p>DlightRAG is a <strong>multimodal RAG</strong> system.</p></div>',
            "current_image_ids": [],
        }
        yield {"type": "meta", "history_kept": 0}

    manager.aanswer_stream.side_effect = lambda **kw: _mock_answer_stream(**kw)
    manager.list_workspaces.return_value = MOCK_WORKSPACE_LIST
    manager.aingest.return_value = {"job_id": "e2e-test-job", "file_count": 1}
    manager.adelete_files.return_value = {"deleted_count": 0}
    manager.config.max_upload_bytes = 100 * 1024 * 1024

    app.state.manager = manager

    port = _free_port()
    import uvicorn

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    time.sleep(0.5)  # let uvicorn bind
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    t.join(timeout=3)


@pytest.fixture(scope="module")
def browser(playwright: Playwright) -> Generator[Browser, Any, None]:
    """Module-scoped browser — reuse across tests for speed."""
    b = playwright.chromium.launch(headless=True)
    yield b
    b.close()


@pytest.fixture
def page(browser: Browser, e2e_base_url: str) -> Generator[Page, Any, None]:
    """Fresh page per test, already pointed at the running server."""
    context = browser.new_context(base_url=e2e_base_url)
    page_obj = context.new_page()
    page_obj.set_default_timeout(10000)
    yield page_obj
    context.close()
