"""Smoke tests for web routes."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture()
def app():
    from dlightrag.api.server import create_app

    real_app = create_app(include_web=True)

    mock_manager = MagicMock()
    mock_manager.aget_file_panel_snapshot = AsyncMock(
        return_value={
            "files": [
                {"file_path": "/data/report.pdf", "file_name": "report.pdf"},
                {"file_path": "/data/analysis.xlsx", "file_name": "analysis.xlsx"},
            ],
            "pipeline_status": {"busy": False, "pending_enqueues": 0},
        }
    )
    mock_manager.alist_workspaces = AsyncMock(return_value=["default", "finance"])
    mock_manager.alist_workspace_records = AsyncMock(
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
    real_app.state.manager = mock_manager

    mock_manager.answer_image_capability = None

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
    assert "query-form" in resp.text


async def test_file_list(client):
    resp = await client.get("/web/files")
    assert resp.status_code == 200
    assert "report.pdf" in resp.text
    assert "analysis.xlsx" in resp.text
