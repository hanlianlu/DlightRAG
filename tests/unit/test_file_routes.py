# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for file-serving endpoint — covers each dispatch branch + security."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from dlightrag.api.server import create_app
from dlightrag.config import DlightragConfig, EmbeddingConfig, LLMConfig, ModelConfig


def _embedding_config() -> EmbeddingConfig:
    return EmbeddingConfig(
        provider="voyage",
        model="voyage-multimodal-3.5",
        api_key="test",
        startup_probe=False,
    )


@pytest.fixture()
def tmp_working_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
async def client(tmp_working_dir: Path) -> AsyncIterator[AsyncClient]:
    config = DlightragConfig(  # type: ignore[call-arg]
        working_dir=str(tmp_working_dir),
        llm=LLMConfig(default=ModelConfig(model="gpt-4.1-mini", api_key="test")),
        embedding=_embedding_config(),
    )
    with (
        patch("dlightrag.config.get_config", return_value=config),
        patch("dlightrag.api.routes.files.get_config", return_value=config),
        patch("dlightrag.api.server.RAGServiceManager.create", new_callable=AsyncMock),
    ):
        app = create_app(include_web=False)
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            follow_redirects=False,
        ) as c:
            yield c


class TestFileEndpoint:
    async def test_streams_canonical_input_file(
        self, client: AsyncClient, tmp_working_dir: Path
    ) -> None:
        """Primary path: files under input_dir/<workspace>/ stream directly."""
        canonical = tmp_working_dir / "inputs" / "default" / "docs" / "canonical.pdf"
        canonical.parent.mkdir(parents=True)
        canonical.write_bytes(b"%PDF-1.4 canonical input content")

        resp = await client.get("/api/files/default/docs/canonical.pdf")
        assert resp.status_code == 200
        assert b"%PDF-1.4 canonical input content" in resp.content
        assert resp.headers["content-type"] == "application/pdf"

    async def test_does_not_stream_from_working_dir(
        self, client: AsyncClient, tmp_working_dir: Path
    ) -> None:
        """Local file serving is scoped to input_dir, not working_dir."""
        docs = tmp_working_dir / "docs"
        docs.mkdir(parents=True)
        (docs / "report.pdf").write_bytes(b"%PDF-1.4 test content")

        resp = await client.get("/api/files/docs/report.pdf")
        assert resp.status_code == 404

    async def test_rejects_path_traversal(self, client: AsyncClient) -> None:
        """Security critical: URL-encoded .. must not escape input_dir."""
        resp = await client.get("/api/files/%2e%2e/%2e%2e/%2e%2e/etc/passwd")
        assert resp.status_code == 403

    async def test_canonicalizes_workspace_query_for_bare_filename(
        self, client: AsyncClient, tmp_working_dir: Path
    ) -> None:
        canonical = tmp_working_dir / "inputs" / "test_ws" / "report.pdf"
        canonical.parent.mkdir(parents=True)
        canonical.write_bytes(b"%PDF-1.4 workspace query")

        resp = await client.get("/api/files/report.pdf", params={"workspace": "test-ws"})

        assert resp.status_code == 200
        assert b"%PDF-1.4 workspace query" in resp.content

    async def test_rejects_windows_absolute_path(self, client: AsyncClient) -> None:
        resp = await client.get(r"/api/files/C:\Users\me\secret.pdf")
        assert resp.status_code == 403

    async def test_azure_503_when_unconfigured(self, client: AsyncClient) -> None:
        """Azure request without connection_string → 503, not 500."""
        resp = await client.get("/api/files/azure://container/blob.pdf")
        assert resp.status_code == 503

    async def test_s3_returns_501(self, client: AsyncClient) -> None:
        """S3 presigned URL support not yet implemented — explicit 501."""
        resp = await client.get("/api/files/s3://my-bucket/key.pdf")
        assert resp.status_code == 501


class TestFileEndpointAzureRedirect:
    @patch(
        "dlightrag.api.routes.files.generate_azure_sas_url",
        return_value="https://acct.blob.core.windows.net/c/b?sig=x",
    )
    async def test_azure_302_redirect(self, mock_sas, tmp_working_dir: Path) -> None:
        """Azure blobs get 302 redirect to SAS URL — no data proxied."""
        config = DlightragConfig(  # type: ignore[call-arg]
            working_dir=str(tmp_working_dir),
            llm=LLMConfig(default=ModelConfig(model="gpt-4.1-mini", api_key="test")),
            embedding=_embedding_config(),
            blob_connection_string="DefaultEndpointsProtocol=https;AccountName=acct;AccountKey=dGVzdA==;EndpointSuffix=core.windows.net",
        )
        with (
            patch("dlightrag.config.get_config", return_value=config),
            patch("dlightrag.api.routes.files.get_config", return_value=config),
            patch("dlightrag.api.server.RAGServiceManager.create", new_callable=AsyncMock),
        ):
            app = create_app(include_web=False)
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
                follow_redirects=False,
            ) as c:
                resp = await c.get("/api/files/azure://mycontainer/doc.pdf")
                assert resp.status_code == 302
                assert "blob.core.windows.net" in resp.headers["location"]
