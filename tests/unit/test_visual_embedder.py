# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for VisualEmbedder (visual embedding via OpenAI-compatible API)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from dlightrag.unifiedrepresent.embedder import VisualEmbedder, VoyageProvider, httpx_text_embed

DIM = 128


def _make_embedder(**overrides: object) -> VisualEmbedder:
    """Create a VisualEmbedder with sensible defaults."""
    defaults: dict[str, object] = {
        "model": "test-model",
        "base_url": "https://api.example.com/v1",
        "api_key": "sk-test",
        "dim": DIM,
        "batch_size": 4,
    }
    defaults.update(overrides)
    return VisualEmbedder(**defaults)  # type: ignore[arg-type]


def _tiny_image() -> Image.Image:
    """Return a 2x2 red PIL image for testing."""
    return Image.new("RGB", (2, 2), color="red")


def _mock_response(dim: int, n: int = 1) -> MagicMock:
    """Build a mock httpx response returning *n* embeddings of size *dim*."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"data": [{"embedding": [0.1] * dim} for _ in range(n)]}
    return resp


# ---------------------------------------------------------------------------
# TestVisualEmbedderInit
# ---------------------------------------------------------------------------


class TestVisualEmbedderInit:
    def test_strips_trailing_slashes(self) -> None:
        emb = _make_embedder(base_url="https://api.example.com/v1///")
        assert emb.base_url == "https://api.example.com/v1"

    def test_client_has_auth_header(self) -> None:
        emb = _make_embedder(api_key="sk-secret-key")
        assert emb._client.headers["authorization"] == "Bearer sk-secret-key"


# ---------------------------------------------------------------------------
# TestEmbedPages
# ---------------------------------------------------------------------------


class TestEmbedPages:
    async def test_empty_list_returns_zero_rows(self) -> None:
        emb = _make_embedder()
        result = await emb.embed_pages([])
        assert isinstance(result, np.ndarray)
        assert result.shape == (0, DIM)
        assert result.dtype == np.float32

    async def test_single_image(self) -> None:
        emb = _make_embedder()
        emb._client.post = AsyncMock(return_value=_mock_response(DIM, n=1))

        result = await emb.embed_pages([_tiny_image()])

        assert result.shape == (1, DIM)
        assert result.dtype == np.float32

    async def test_concurrency_five_images_batch_size_two(self) -> None:
        emb = _make_embedder(batch_size=2)
        images = [_tiny_image() for _ in range(5)]

        # Each image is embedded individually → 5 HTTP calls (batch_size controls concurrency).
        emb._client.post = AsyncMock(return_value=_mock_response(DIM, n=1))

        result = await emb.embed_pages(images)

        assert emb._client.post.call_count == 5
        assert result.shape == (5, DIM)

    async def test_result_dtype_is_float32(self) -> None:
        emb = _make_embedder()
        emb._client.post = AsyncMock(return_value=_mock_response(DIM, n=1))

        result = await emb.embed_pages([_tiny_image(), _tiny_image()])

        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# TestEmbedTexts
# ---------------------------------------------------------------------------


class TestEmbedTexts:
    async def test_empty_list_returns_zero_rows(self) -> None:
        emb = _make_embedder()
        result = await emb.embed_texts([])
        assert isinstance(result, np.ndarray)
        assert result.shape == (0, DIM)
        assert result.dtype == np.float32

    async def test_batch_of_three(self) -> None:
        emb = _make_embedder()
        emb._client.post = AsyncMock(return_value=_mock_response(DIM, n=3))

        result = await emb.embed_texts(["a", "b", "c"])

        assert result.shape == (3, DIM)
        assert result.dtype == np.float32
        # Single HTTP call with all texts batched
        emb._client.post.assert_awaited_once()

    async def test_sends_encoding_format_float(self) -> None:
        emb = _make_embedder()
        emb._client.post = AsyncMock(return_value=_mock_response(DIM, n=1))

        await emb.embed_texts(["hello"])

        payload = emb._client.post.call_args[1]["json"]
        assert payload["encoding_format"] == "float"
        assert payload["input"] == ["hello"]

    async def test_calls_embeddings_endpoint(self) -> None:
        emb = _make_embedder(base_url="https://api.example.com/v1")
        emb._client.post = AsyncMock(return_value=_mock_response(DIM, n=1))

        await emb.embed_texts(["hello"])

        call_args = emb._client.post.call_args
        assert call_args[0][0] == "https://api.example.com/v1/embeddings"

    async def test_dim_mismatch_raises(self) -> None:
        emb = _make_embedder(dim=DIM)
        emb._client.post = AsyncMock(return_value=_mock_response(DIM + 10, n=1))

        with pytest.raises(ValueError, match="Expected embedding dim"):
            await emb.embed_texts(["query"])


# ---------------------------------------------------------------------------
# TestVoyageProvider
# ---------------------------------------------------------------------------


class TestVoyageProvider:
    def test_image_payload_passes_full_data_uri(self) -> None:
        prov = VoyageProvider()
        data_uri = "data:image/png;base64,AAAA"
        payload = prov.build_image_payload("voyage-multimodal-3", data_uri)
        img_content = payload["inputs"][0]["content"][0]
        assert img_content["type"] == "image_base64"
        assert img_content["image_base64"] == data_uri

    def test_text_payload_nested_structure(self) -> None:
        prov = VoyageProvider()
        payload = prov.build_text_payload("voyage-multimodal-3", ["hello", "world"])
        assert len(payload["inputs"]) == 2
        assert payload["inputs"][0]["content"][0] == {"type": "text", "text": "hello"}
        assert payload["inputs"][1]["content"][0] == {"type": "text", "text": "world"}

    def test_endpoint(self) -> None:
        prov = VoyageProvider()
        assert prov.endpoint == "/multimodalembeddings"


# ---------------------------------------------------------------------------
# TestHttpxTextEmbedVoyage
# ---------------------------------------------------------------------------


class TestHttpxTextEmbedVoyage:
    async def test_uses_voyage_endpoint_and_payload(self) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": [{"embedding": [0.1] * 128}],
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("dlightrag.unifiedrepresent.embedder.httpx.AsyncClient", return_value=mock_client):
            result = await httpx_text_embed(
                texts=["hello"],
                model="voyage-multimodal-3",
                base_url="https://api.voyageai.com/v1",
                api_key="sk-test",
                provider=VoyageProvider(),
            )

        call_args = mock_client.post.call_args
        assert "/multimodalembeddings" in call_args[0][0]
        payload = call_args[1]["json"]
        assert payload["inputs"][0]["content"][0]["type"] == "text"
        assert result.shape == (1, 128)
