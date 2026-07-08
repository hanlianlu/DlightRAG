# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for file-serving endpoint — covers each dispatch branch + security."""

from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from dlightrag.access_control import AccessDeniedError
from dlightrag.api.server import create_app
from dlightrag.config import DlightragConfig, EmbeddingConfig, LLMConfig, ModelConfig, set_config
from dlightrag.sourcing.aws_s3 import S3CredentialsUnavailable


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
        llm=LLMConfig(default=ModelConfig(model="gpt-5.4-mini", api_key="test")),
        embedding=_embedding_config(),
    )
    set_config(config)
    with patch("dlightrag.api.server.RAGServiceManager.create", new_callable=AsyncMock):
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

    async def test_streams_nested_upload_batch_file(
        self, client: AsyncClient, tmp_working_dir: Path
    ) -> None:
        """Bare basename resolves a file staged under __uploads__/<batch>/ (2 levels deep)."""
        staged = (
            tmp_working_dir
            / "inputs"
            / "default"
            / "__uploads__"
            / "b1a2c3d4"
            / "PyCaret 3.0 cheat_sheet.pdf"
        )
        staged.parent.mkdir(parents=True)
        staged.write_bytes(b"%PDF-1.4 nested upload content")

        resp = await client.get("/api/files/default/PyCaret%203.0%20cheat_sheet.pdf")

        assert resp.status_code == 200
        assert b"%PDF-1.4 nested upload content" in resp.content
        assert "attachment" in resp.headers.get("content-disposition", "")
        # Starlette RFC 5987-encodes non-ASCII / spaced filenames (filename*=).
        assert "PyCaret%203.0%20cheat_sheet.pdf" in resp.headers.get("content-disposition", "")

    async def test_streams_retained_remote_source_file(
        self, client: AsyncClient, tmp_working_dir: Path
    ) -> None:
        """Retained remote sources live under __remote_sources__/<type>/ (2 levels deep)."""
        staged = (
            tmp_working_dir
            / "inputs"
            / "default"
            / "__remote_sources__"
            / "s3"
            / "report__a1b2c3d4e5f6.pdf"
        )
        staged.parent.mkdir(parents=True)
        staged.write_bytes(b"%PDF-1.4 retained remote content")

        resp = await client.get("/api/files/default/report__a1b2c3d4e5f6.pdf")

        assert resp.status_code == 200
        assert b"%PDF-1.4 retained remote content" in resp.content

    async def test_streams_deeply_nested_folder_upload(
        self, client: AsyncClient, tmp_working_dir: Path
    ) -> None:
        """Folder uploads nested more than one level deep are still resolvable by basename."""
        staged = tmp_working_dir / "inputs" / "default" / "topic" / "sub" / "deep.pdf"
        staged.parent.mkdir(parents=True)
        staged.write_bytes(b"%PDF-1.4 deep folder content")

        resp = await client.get("/api/files/default/deep.pdf")

        assert resp.status_code == 200
        assert b"%PDF-1.4 deep folder content" in resp.content

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

    async def test_rejects_symlink_escape(self, client: AsyncClient, tmp_working_dir: Path) -> None:
        """Resolved local paths must stay inside the authorized workspace."""
        workspace_dir = tmp_working_dir / "inputs" / "default"
        workspace_dir.mkdir(parents=True)
        outside_dir = tmp_working_dir / "outside"
        outside_dir.mkdir()
        direct_secret = outside_dir / "direct-secret.pdf"
        direct_secret.write_bytes(b"direct secret")
        canonical_secret = outside_dir / "canonical-secret.pdf"
        canonical_secret.write_bytes(b"canonical secret")

        try:
            (workspace_dir / "direct.pdf").symlink_to(direct_secret)
            (workspace_dir / "linked").symlink_to(outside_dir, target_is_directory=True)
        except (NotImplementedError, OSError) as exc:
            pytest.skip(f"symlink creation unavailable: {exc}")

        direct_resp = await client.get("/api/files/default/direct.pdf")
        canonical_resp = await client.get("/api/files/canonical-secret.pdf")

        assert direct_resp.status_code == 404
        assert canonical_resp.status_code == 404
        assert b"direct secret" not in direct_resp.content
        assert b"canonical secret" not in canonical_resp.content

    async def test_canonicalizes_workspace_query_for_bare_filename(
        self, client: AsyncClient, tmp_working_dir: Path
    ) -> None:
        canonical = tmp_working_dir / "inputs" / "test_ws" / "report.pdf"
        canonical.parent.mkdir(parents=True)
        canonical.write_bytes(b"%PDF-1.4 workspace query")

        resp = await client.get("/api/files/report.pdf", params={"workspace": "test-ws"})

        assert resp.status_code == 200
        assert b"%PDF-1.4 workspace query" in resp.content

    async def test_rejects_conflicting_path_and_query_workspaces(
        self, client: AsyncClient, tmp_working_dir: Path
    ) -> None:
        secret = tmp_working_dir / "inputs" / "secret_ws" / "report.pdf"
        secret.parent.mkdir(parents=True)
        secret.write_bytes(b"secret workspace content")

        resp = await client.get("/api/files/secret_ws/report.pdf", params={"workspace": "default"})

        assert resp.status_code == 403
        assert b"secret workspace content" not in resp.content

    async def test_embedded_workspace_controls_download_authorization(
        self, tmp_working_dir: Path
    ) -> None:
        secret = tmp_working_dir / "inputs" / "secret_ws" / "report.pdf"
        secret.parent.mkdir(parents=True)
        secret.write_bytes(b"secret workspace content")

        class DenySecretWorkspace:
            async def check(self, user, action, *, workspace=None):
                if workspace == "secret_ws":
                    raise AccessDeniedError("denied")

            async def filter_workspaces(self, user, action, workspaces):
                return [workspace for workspace in workspaces if workspace != "secret_ws"]

        config = DlightragConfig(  # type: ignore[call-arg]
            working_dir=str(tmp_working_dir),
            llm=LLMConfig(default=ModelConfig(model="gpt-5.4-mini", api_key="test")),
            embedding=_embedding_config(),
        )
        set_config(config)
        with patch("dlightrag.api.server.RAGServiceManager.create", new_callable=AsyncMock):
            app = create_app(include_web=False)
            app.state.access_control = DenySecretWorkspace()
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
                follow_redirects=False,
            ) as c:
                resp = await c.get("/api/files/secret_ws/report.pdf")

        assert resp.status_code == 403
        assert b"secret workspace content" not in resp.content

    async def test_rejects_windows_absolute_path(self, client: AsyncClient) -> None:
        resp = await client.get(r"/api/files/C:\Users\me\secret.pdf")
        assert resp.status_code == 403

    async def test_rejects_windows_path_traversal(self, client: AsyncClient) -> None:
        resp = await client.get("/api/files/default%5C..%5Csecret_ws%5Creport.pdf")
        assert resp.status_code == 403

    async def test_azure_503_when_unconfigured(self, client: AsyncClient) -> None:
        """Azure request without connection_string → 503, not 500."""
        resp = await client.get("/api/files/azure://container/blob.pdf")
        assert resp.status_code == 503

    async def test_s3_503_when_unconfigured(self, client: AsyncClient) -> None:
        """S3 request without usable AWS credentials -> 503, not 500."""
        with patch(
            "dlightrag.api.routes.files.generate_s3_presigned_url",
            new_callable=AsyncMock,
        ) as mock_presign:
            mock_presign.side_effect = S3CredentialsUnavailable("missing credentials")

            resp = await client.get("/api/files/s3://my-bucket/key.pdf")

        assert resp.status_code == 503

    async def test_https_source_redirects_to_original_url(self, client: AsyncClient) -> None:
        resp = await client.get("/api/files/https://api.bynder.com/docs/getting-started")

        assert resp.status_code == 302
        assert resp.headers["location"] == "https://api.bynder.com/docs/getting-started"

    async def test_https_source_redirect_preserves_encoded_query(self, client: AsyncClient) -> None:
        resp = await client.get(
            "/api/files/https://cdn.example.com/report.pdf%3Fsig%3Dx%26download%3D1"
        )

        assert resp.status_code == 302
        assert resp.headers["location"] == "https://cdn.example.com/report.pdf?sig=x&download=1"

    async def test_https_source_rejects_unsafe_target(self, client: AsyncClient) -> None:
        """Credentialed / localhost / private targets are refused, not redirected."""
        for bad in (
            "https://localhost/secret",
            "https://127.0.0.1/secret",
            "https://user:pass@example.com/x",
        ):
            resp = await client.get(f"/api/files/{bad}", follow_redirects=False)
            assert resp.status_code == 400, bad


class TestFileEndpointAzureRedirect:
    @patch(
        "dlightrag.api.routes.files.generate_azure_sas_url",
        return_value="https://acct.blob.core.windows.net/c/b?sig=x",
    )
    async def test_azure_302_redirect(self, mock_sas, tmp_working_dir: Path) -> None:
        """Azure blobs get 302 redirect to SAS URL — no data proxied."""
        config = DlightragConfig(  # type: ignore[call-arg]
            working_dir=str(tmp_working_dir),
            llm=LLMConfig(default=ModelConfig(model="gpt-5.4-mini", api_key="test")),
            embedding=_embedding_config(),
            blob_connection_string="DefaultEndpointsProtocol=https;AccountName=acct;AccountKey=dGVzdA==;EndpointSuffix=core.windows.net",
        )
        set_config(config)
        with patch("dlightrag.api.server.RAGServiceManager.create", new_callable=AsyncMock):
            app = create_app(include_web=False)
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
                follow_redirects=False,
            ) as c:
                resp = await c.get("/api/files/azure://mycontainer/doc.pdf")
                assert resp.status_code == 302
                assert "blob.core.windows.net" in resp.headers["location"]


class TestFileEndpointS3Redirect:
    @patch(
        "dlightrag.api.routes.files.generate_s3_presigned_url",
        new_callable=AsyncMock,
    )
    async def test_s3_302_redirect(self, mock_presign, tmp_working_dir: Path) -> None:
        """S3 objects get 302 redirect to presigned URL — no data proxied."""
        mock_presign.return_value = "https://my-bucket.s3.amazonaws.com/docs/report.pdf?sig=x"
        config = DlightragConfig(  # type: ignore[call-arg]
            working_dir=str(tmp_working_dir),
            llm=LLMConfig(default=ModelConfig(model="gpt-5.4-mini", api_key="test")),
            embedding=_embedding_config(),
        )
        set_config(config)
        with patch("dlightrag.api.server.RAGServiceManager.create", new_callable=AsyncMock):
            app = create_app(include_web=False)
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
                follow_redirects=False,
            ) as c:
                resp = await c.get("/api/files/s3://my-bucket/docs/report.pdf")

        assert resp.status_code == 302
        assert resp.headers["location"].startswith("https://my-bucket.s3.amazonaws.com/")
        mock_presign.assert_awaited_once_with(
            raw_path="s3://my-bucket/docs/report.pdf",
            expiry_seconds=3600,
            region=None,
        )
