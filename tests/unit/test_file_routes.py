# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for file-serving endpoint — covers each dispatch branch + security."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from dlightrag.api.server import app
from dlightrag.config import DlightragConfig, EmbeddingConfig, ModelConfig


@pytest.fixture()
def tmp_working_dir(tmp_path: Path) -> Path:
    sources = tmp_path / "sources" / "local"
    sources.mkdir(parents=True)
    (sources / "report.pdf").write_bytes(b"%PDF-1.4 test content")
    return tmp_path


@pytest.fixture()
def client(tmp_working_dir: Path) -> TestClient:
    config = DlightragConfig(  # type: ignore[call-arg]
        working_dir=str(tmp_working_dir),
        chat=ModelConfig(model="gpt-4.1-mini", api_key="test"),
        embedding=EmbeddingConfig(api_key="test"),
    )
    with (
        patch("dlightrag.api.server.get_config", return_value=config),
        patch("dlightrag.api.server.RAGServiceManager.create", new_callable=AsyncMock),
    ):
        with TestClient(app) as c:
            yield c


class TestFileEndpoint:
    def test_streams_local_file(self, client: TestClient) -> None:
        """Happy path: local file served with correct content and MIME type."""
        resp = client.get("/api/files/sources/local/report.pdf")
        assert resp.status_code == 200
        assert b"%PDF-1.4 test content" in resp.content
        assert resp.headers["content-type"] == "application/pdf"

    def test_rejects_path_traversal(self, client: TestClient) -> None:
        """Security critical: URL-encoded .. must not escape working_dir."""
        resp = client.get("/api/files/%2e%2e/%2e%2e/%2e%2e/etc/passwd")
        assert resp.status_code == 403

    def test_azure_503_when_unconfigured(self, client: TestClient) -> None:
        """Azure request without connection_string → 503, not 500."""
        resp = client.get("/api/files/azure://container/blob.pdf")
        assert resp.status_code == 503

    def test_snowflake_returns_400(self, client: TestClient) -> None:
        """Snowflake has no downloadable file — explicit 400."""
        resp = client.get("/api/files/snowflake://my_table")
        assert resp.status_code == 400


class TestFileEndpointAzureRedirect:
    @patch(
        "dlightrag.api.server.generate_azure_sas_url",
        return_value="https://acct.blob.core.windows.net/c/b?sig=x",
    )
    def test_azure_302_redirect(self, mock_sas, tmp_working_dir: Path) -> None:
        """Azure blobs get 302 redirect to SAS URL — no data proxied."""
        config = DlightragConfig(  # type: ignore[call-arg]
            working_dir=str(tmp_working_dir),
            chat=ModelConfig(model="gpt-4.1-mini", api_key="test"),
            embedding=EmbeddingConfig(api_key="test"),
            blob_connection_string="DefaultEndpointsProtocol=https;AccountName=acct;AccountKey=dGVzdA==;EndpointSuffix=core.windows.net",
        )
        with (
            patch("dlightrag.api.server.get_config", return_value=config),
            patch("dlightrag.api.server.RAGServiceManager.create", new_callable=AsyncMock),
        ):
            with TestClient(app, follow_redirects=False) as c:
                resp = c.get("/api/files/azure://mycontainer/doc.pdf")
                assert resp.status_code == 302
                assert "blob.core.windows.net" in resp.headers["location"]
