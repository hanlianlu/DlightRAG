# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the REST-owned source download adapter."""

import logging
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from dlightrag.api.server import create_app
from dlightrag.application.access import AccessDeniedError
from dlightrag.application.config import DlightragConfig, set_config
from dlightrag.application.corpus_admin import (
    LocalDownloadTarget,
    RedirectDownloadTarget,
    SourceDownloadInvalidError,
    SourceDownloadNotFoundError,
    SourceDownloadUnavailableError,
)
from dlightrag.engine.ai.settings import EmbeddingSettings, ModelRoleSettings, ModelSettings


def _embedding_config() -> EmbeddingSettings:
    return EmbeddingSettings(
        provider="voyage",
        model="voyage-multimodal-3.5",
        api_key="test",
        startup_probe=False,
    )


@pytest.fixture()
def tmp_working_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
async def route_client(
    tmp_working_dir: Path,
) -> AsyncIterator[tuple[AsyncClient, AsyncMock]]:
    config = DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
        # type: ignore[call-arg]
        deployment={"working_dir": str(tmp_working_dir)},
        models={
            "chat": ModelRoleSettings(default=ModelSettings(model="gpt-5.4-mini", api_key="test")),
            "embedding": _embedding_config(),
        },
    )
    set_config(config)
    application_double = AsyncMock()
    application_double.config = config
    app = create_app(include_web_app=False)
    app.state.application = application_double
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        follow_redirects=False,
    ) as client:
        yield client, application_double


async def test_local_markdown_download_is_attachment(
    route_client: tuple[AsyncClient, AsyncMock],
    tmp_working_dir: Path,
) -> None:
    client, application_double = route_client
    source = tmp_working_dir / "inputs" / "default" / "notes.md"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("# Notes", encoding="utf-8")
    application_double.corpora.prepare_source_download.return_value = LocalDownloadTarget(
        path=source.resolve(),
        media_type="text/markdown",
        filename="notes.md",
    )

    response = await client.get(
        "/files/raw/doc-notes",
        params={"workspace": "default"},
    )

    assert response.status_code == 200
    assert response.content == b"# Notes"
    assert response.headers["content-type"].startswith("text/markdown")
    assert 'attachment; filename="notes.md"' in response.headers["content-disposition"]
    application_double.corpora.prepare_source_download.assert_awaited_once_with(
        "default", "doc-notes"
    )


async def test_document_download_normalizes_workspace(
    route_client: tuple[AsyncClient, AsyncMock],
) -> None:
    client, application_double = route_client
    application_double.corpora.prepare_source_download.side_effect = SourceDownloadNotFoundError(
        "Source not found"
    )

    response = await client.get(
        "/files/raw/doc-report",
        params={"workspace": "Finance-Team"},
    )

    assert response.status_code == 404
    application_double.corpora.prepare_source_download.assert_awaited_once_with(
        "finance_team", "doc-report"
    )


async def test_remote_download_redirects_to_prepared_target(
    route_client: tuple[AsyncClient, AsyncMock],
) -> None:
    client, application_double = route_client
    application_double.corpora.prepare_source_download.return_value = RedirectDownloadTarget(
        url="https://cdn.example.com/report.pdf?signature=ephemeral"
    )

    response = await client.get(
        "/files/raw/doc-report",
        params={"workspace": "finance"},
    )

    assert response.status_code == 302
    assert response.headers["location"] == (
        "https://cdn.example.com/report.pdf?signature=ephemeral"
    )


@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (SourceDownloadInvalidError("Source download metadata is invalid"), 400),
        (SourceDownloadNotFoundError("Source not found"), 404),
        (SourceDownloadUnavailableError("S3 credentials not configured"), 503),
    ],
)
async def test_source_download_maps_core_errors(
    route_client: tuple[AsyncClient, AsyncMock],
    error: Exception,
    status_code: int,
) -> None:
    client, application_double = route_client
    application_double.corpora.prepare_source_download.side_effect = error

    response = await client.get("/files/raw/doc-report")

    assert response.status_code == status_code
    assert response.json()["detail"] == str(error)


async def test_download_authorization_precedes_metadata_lookup(
    tmp_working_dir: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class DenyFinanceWorkspace:
        async def check(self, user, action, *, workspace=None):
            if workspace == "finance":
                raise AccessDeniedError("denied")

        async def filter_workspaces(self, user, action, workspaces):
            return [workspace for workspace in workspaces if workspace != "finance"]

    config = DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
        # type: ignore[call-arg]
        deployment={"working_dir": str(tmp_working_dir)},
        models={
            "chat": ModelRoleSettings(default=ModelSettings(model="gpt-5.4-mini", api_key="test")),
            "embedding": _embedding_config(),
        },
    )
    set_config(config)
    application_double = AsyncMock()
    application_double.config = config
    with caplog.at_level(logging.INFO, logger="dlightrag.api.routes.files"):
        app = create_app(include_web_app=False)
        app.state.application = application_double
        app.state.access_control = DenyFinanceWorkspace()
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            follow_redirects=False,
        ) as client:
            response = await client.get(
                "/files/raw/doc-payroll",
                params={"workspace": "finance"},
            )

    assert response.status_code == 403
    application_double.corpora.prepare_source_download.assert_not_awaited()
    record = next(
        record
        for record in caplog.records
        if record.message == "source_download_projection_outcome"
    )
    assert getattr(record, "outcome", None) == "unauthorized"
    assert getattr(record, "workspace", None) == "finance"
    assert "doc-payroll" not in caplog.text
