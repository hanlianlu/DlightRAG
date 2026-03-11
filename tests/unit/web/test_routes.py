"""Smoke tests for web routes using FastAPI TestClient."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def app():
    from dlightrag.api.server import app as real_app

    mock_manager = MagicMock()
    mock_manager.list_ingested_files = AsyncMock(return_value=["report.pdf", "analysis.xlsx"])
    mock_manager.list_workspaces = AsyncMock(return_value=["default", "finance"])
    real_app.state.manager = mock_manager

    return real_app


@pytest.fixture()
def client(app):
    return TestClient(app)


def test_index_page(client):
    resp = client.get("/web/")
    assert resp.status_code == 200
    assert "DlightRAG" in resp.text
    assert "query-form" in resp.text


def test_file_list(client):
    resp = client.get("/web/files")
    assert resp.status_code == 200
    assert "report.pdf" in resp.text
    assert "analysis.xlsx" in resp.text


def test_workspace_switch(client):
    resp = client.post(
        "/web/workspaces/switch",
        data={"workspace": "finance"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert "dlightrag_workspace=finance" in resp.headers.get("set-cookie", "")
