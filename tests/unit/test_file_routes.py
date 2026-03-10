# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for file-serving endpoint — covers each dispatch branch + security."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dlightrag.api.file_routes import create_file_router
from dlightrag.config import DlightragConfig


@pytest.fixture()
def tmp_working_dir(tmp_path: Path) -> Path:
    sources = tmp_path / "sources" / "local"
    sources.mkdir(parents=True)
    (sources / "report.pdf").write_bytes(b"%PDF-1.4 test content")
    return tmp_path


@pytest.fixture()
def client(tmp_working_dir: Path) -> TestClient:
    config = DlightragConfig(openai_api_key="test", working_dir=str(tmp_working_dir))
    app = FastAPI()
    app.include_router(create_file_router(config))
    return TestClient(app)


class TestFileEndpoint:

    def test_streams_local_file(self, client: TestClient) -> None:
        """Happy path: local file served with correct content and MIME type."""
        resp = client.get("/api/files/sources/local/report.pdf")
        assert resp.status_code == 200
        assert b"%PDF-1.4 test content" in resp.content
        assert resp.headers["content-type"] == "application/pdf"

    def test_rejects_path_traversal(self, client: TestClient) -> None:
        """Security critical: URL-encoded .. must not escape working_dir."""
        # Plain "../.." is normalised away by the HTTP stack before reaching the handler.
        # URL-encoded dots (%2e%2e) survive normalisation and exercise our guard.
        resp = client.get("/api/files/%2e%2e/%2e%2e/%2e%2e/etc/passwd")
        assert resp.status_code == 403

    @patch("dlightrag.api.file_routes.generate_azure_sas_url", return_value="https://acct.blob.core.windows.net/c/b?sig=x")
    def test_azure_302_redirect(self, mock_sas, tmp_working_dir: Path) -> None:
        """Azure blobs get 302 redirect to SAS URL — no data proxied."""
        config = DlightragConfig(
            openai_api_key="test",
            working_dir=str(tmp_working_dir),
            blob_connection_string="DefaultEndpointsProtocol=https;AccountName=acct;AccountKey=dGVzdA==;EndpointSuffix=core.windows.net",
        )
        app = FastAPI()
        app.include_router(create_file_router(config))
        c = TestClient(app, follow_redirects=False)
        resp = c.get("/api/files/azure://mycontainer/doc.pdf")
        assert resp.status_code == 302
        assert "blob.core.windows.net" in resp.headers["location"]

    def test_azure_503_when_unconfigured(self, client: TestClient) -> None:
        """Azure request without connection_string → 503, not 500."""
        resp = client.get("/api/files/azure://container/blob.pdf")
        assert resp.status_code == 503

    def test_snowflake_returns_400(self, client: TestClient) -> None:
        """Snowflake has no downloadable file — explicit 400."""
        resp = client.get("/api/files/snowflake://my_table")
        assert resp.status_code == 400
