"""Smoke tests for web routes."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from tests.unit.conftest import answer_capability_view


@pytest.fixture()
def app(test_config):
    from dlightrag.adapters.http.server import create_app

    assert test_config is not None
    real_app = create_app(include_web_app=True)

    mock_application = MagicMock()
    mock_application.config = test_config
    mock_application.corpora.file_panel_snapshot = AsyncMock(
        return_value={
            "files": [
                {"file_path": "/data/report.pdf", "file_name": "report.pdf"},
                {"file_path": "/data/analysis.xlsx", "file_name": "analysis.xlsx"},
            ],
            "pipeline_status": {"busy": False, "pending_enqueues": 0},
        }
    )
    mock_application.corpora.list_workspaces = AsyncMock(return_value=["default", "finance"])
    mock_application.corpora.alist_workspace_records = AsyncMock(
        return_value=[
            {
                "workspace": "default",
                "display_name": "Default",
                "embedding_model": "voyage-multimodal-3.5",
            },
            {
                "workspace": "finance",
                "display_name": "Finance",
                "embedding_model": "voyage-multimodal-3.5",
            },
        ]
    )
    real_app.state.application = mock_application

    capability_view = answer_capability_view()
    mock_application.answers.capabilities = capability_view.read

    return real_app


@pytest.fixture()
async def client(app):
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        follow_redirects=False,
    ) as c:
        yield c


async def test_index_page(client):
    resp = await client.get("/web/")
    assert resp.status_code == 200
    assert "DlightRAG" in resp.text
    assert "<dl-app>" in resp.text
    assert "/static/app/assets/app-" in resp.text


async def test_file_list(client):
    resp = await client.get("/web/api/files")
    assert resp.status_code == 200
    assert [item["file_name"] for item in resp.json()["files"]] == [
        "report.pdf",
        "analysis.xlsx",
    ]
