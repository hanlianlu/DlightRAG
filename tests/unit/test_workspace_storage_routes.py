# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""REST contracts for workspace storage status and promotion fence mapping."""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from dlightrag.adapters.http.server import create_app
from dlightrag.application.config import DlightragConfig, set_config
from dlightrag.application.errors import WorkspaceWriteFencedError
from dlightrag.engine.ai.settings import EmbeddingSettings, ModelRoleSettings, ModelSettings
from tests.support.application_double import application_double


def _embedding_config() -> EmbeddingSettings:
    return EmbeddingSettings(
        provider="voyage",
        model="voyage-multimodal-3.5",
        api_key="test",
        startup_probe=False,
    )


@pytest.fixture()
async def route_client(
    tmp_path: Path,
) -> AsyncIterator[tuple[AsyncClient, Any]]:
    config = DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
        # type: ignore[call-arg]
        deployment={"working_dir": str(tmp_path)},
        models={
            "chat": ModelRoleSettings(default=ModelSettings(model="gpt-5.4-mini", api_key="test")),
            "embedding": _embedding_config(),
        },
    )
    set_config(config)
    application = application_double(config)
    app = create_app()
    app.state.application = application
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        follow_redirects=False,
    ) as client:
        yield client, application


async def test_storage_status_route_returns_admin_facts(
    route_client: tuple[AsyncClient, Any],
) -> None:
    client, application = route_client
    application.corpora.get_workspace_storage_status.return_value = {
        "workspace": "finance",
        "storage_tier": "hot",
        "promotion_state": "none",
        "ingested_docs_total": 10,
        "ingested_chunks_total": 150,
        "promotion_retry_count": 0,
        "promotion_last_error": None,
        "promotion_next_retry_at": None,
        "write_fenced": False,
        "retry_after_seconds": None,
    }

    response = await client.get("/workspaces/finance/storage")

    assert response.status_code == 200
    assert response.json()["storage_tier"] == "hot"
    application.corpora.get_workspace_storage_status.assert_awaited_once_with("finance")


async def test_storage_status_route_404_for_unknown_workspace(
    route_client: tuple[AsyncClient, Any],
) -> None:
    client, application = route_client
    application.corpora.get_workspace_storage_status.return_value = None

    response = await client.get("/workspaces/missing/storage")

    assert response.status_code == 404


async def test_update_metadata_under_fence_maps_to_409_with_retry_after(
    route_client: tuple[AsyncClient, Any],
) -> None:
    client, application = route_client
    application.corpora.update_metadata.side_effect = WorkspaceWriteFencedError(
        workspace="finance", retry_after_seconds=8.0
    )

    response = await client.post(
        "/metadata/doc-1",
        params={"workspace": "finance"},
        json={"metadata": {"team": "core"}},
    )

    assert response.status_code == 409
    assert response.headers["Retry-After"] == "8"
