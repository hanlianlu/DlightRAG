# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for E2E Playwright tests.

Starts a real FastAPI server on a random port with a mocked RAG service
manager so the browser can exercise the full HTML/JS/CSS pipeline without
needing PostgreSQL, LLM, or embedding backends.

Usage (opt-in, requires Playwright)::

    pytest tests/e2e/ -m e2e
"""

import socket
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from playwright.sync_api import Browser, Page, sync_playwright

from dlightrag.api.server import create_app

MOCK_WORKSPACES = [
    {"workspace": "default", "display_name": "Default", "embedding_model": "voyage-multimodal-3.5"}
]

MOCK_WORKSPACE_LIST = ["default"]


def _free_port() -> int:
    """Return an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def e2e_base_url() -> Generator[str, Any]:
    """Start a real FastAPI server with a mocked manager on a random port."""
    manager = AsyncMock()
    manager.config = SimpleNamespace(
        workspace="default",
        input_dir_path=Path("."),
        citations=SimpleNamespace(highlights=SimpleNamespace(enabled=False)),
        embedding=SimpleNamespace(model="voyage-multimodal-3.5"),
    )

    async def _token_stream() -> Any:
        yield "DlightRAG is a "
        yield "multimodal RAG system."

    async def _mock_answer_stream(*__: Any, **___: Any) -> Any:
        return ({}, _token_stream())

    manager.aanswer_stream.side_effect = _mock_answer_stream
    manager.aget_checkpoint_history.return_value = []
    manager.list_workspaces.return_value = MOCK_WORKSPACE_LIST
    manager.list_workspace_records.return_value = MOCK_WORKSPACES
    manager.list_ingested_files.return_value = []
    manager.get_pipeline_status.return_value = {"busy": False, "pending_enqueues": 0}
    manager.aingest.return_value = {"job_id": "e2e-test-job", "file_count": 1}
    manager.adelete_files.return_value = {"deleted_count": 0}

    port = _free_port()
    import uvicorn

    with patch("dlightrag.api.server.RAGServiceManager.create", AsyncMock(return_value=manager)):
        app = create_app(include_web=True)
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


@pytest.fixture(scope="module")
def browser() -> Generator[Browser, Any]:
    """Module-scoped browser — reuse across tests for speed."""
    with sync_playwright() as pw:
        b = pw.chromium.launch(headless=True)
        try:
            yield b
        finally:
            b.close()


@pytest.fixture
def page(browser: Browser, e2e_base_url: str) -> Generator[Page, Any]:
    """Fresh page per test, already pointed at the running server."""
    context = browser.new_context(base_url=e2e_base_url)
    page_obj = context.new_page()
    page_obj.set_default_timeout(10000)
    yield page_obj
    context.close()
