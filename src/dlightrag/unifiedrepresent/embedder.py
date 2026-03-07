# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Visual embedding for unified representational RAG.

Embeds page images into visual vectors via OpenAI-compatible multimodal
embedding API (e.g., qwen3-vl-embedding).
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from abc import ABC, abstractmethod

import httpx
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class MultimodalEmbedProvider(ABC):
    """Strategy for different multimodal embedding API protocols."""

    @property
    @abstractmethod
    def endpoint(self) -> str:
        """API endpoint path (e.g., '/embeddings')."""

    @abstractmethod
    def build_image_payload(self, model: str, image_b64: str) -> dict: ...

    @abstractmethod
    def build_text_payload(self, model: str, texts: list[str]) -> dict: ...

    def parse_response(self, response_json: dict) -> list[list[float]]:
        return [item["embedding"] for item in response_json["data"]]


class OpenAICompatProvider(MultimodalEmbedProvider):
    """Qwen-VL, GME, Xinference, Azure Voyage 3.5 — POST /embeddings with data URI."""

    @property
    def endpoint(self) -> str:
        return "/embeddings"

    def build_image_payload(self, model: str, image_b64: str) -> dict:
        return {"model": model, "input": image_b64, "encoding_format": "float"}

    def build_text_payload(self, model: str, texts: list[str]) -> dict:
        return {"model": model, "input": texts, "encoding_format": "float"}


class VoyageProvider(MultimodalEmbedProvider):
    """Voyage Multimodal-3 — POST /multimodalembeddings.

    Voyage multimodal API accepts full data URIs in the image_base64 field
    (e.g., "data:image/png;base64,..."). No output_dimension support on
    the multimodal endpoint (only on text-only Voyage models).
    """

    @property
    def endpoint(self) -> str:
        return "/multimodalembeddings"

    def build_image_payload(self, model: str, image_b64: str) -> dict:
        return {
            "model": model,
            "inputs": [{"content": [{"type": "image_base64", "image_base64": image_b64}]}],
        }

    def build_text_payload(self, model: str, texts: list[str]) -> dict:
        return {
            "model": model,
            "inputs": [{"content": [{"type": "text", "text": t}]} for t in texts],
        }


# -----------------------------------------------------------------------
# Standalone text embedding function (deepcopy-safe via functools.partial)
# -----------------------------------------------------------------------


async def httpx_text_embed(
    texts: list[str],
    model: str = "",
    base_url: str = "",
    api_key: str = "",
    provider: MultimodalEmbedProvider | None = None,
) -> np.ndarray:
    """Embed texts via httpx POST to an embedding endpoint.

    Designed to be used with ``functools.partial`` to bind model/base_url/api_key
    (and optionally provider), then wrapped in LightRAG's ``EmbeddingFunc``.

    When *provider* is ``None`` (the default), uses ``OpenAICompatProvider`` which
    posts to ``/embeddings`` with ``encoding_format: "float"`` — compatible with
    Xinference VL embedding models.  Pass ``VoyageProvider()`` to hit the Voyage
    ``/multimodalembeddings`` endpoint instead.

    A new ``httpx.AsyncClient`` is created per call (same pattern as
    ``openai_embed`` which creates a new ``AsyncOpenAI`` per call).
    """
    if not texts:
        return np.empty((0,), dtype=np.float32)

    prov = provider or OpenAICompatProvider()
    payload = prov.build_text_payload(model, texts)

    async with httpx.AsyncClient(
        timeout=120.0,
        headers={"Authorization": f"Bearer {api_key}"},
        transport=httpx.AsyncHTTPTransport(retries=2),
    ) as client:
        resp = await client.post(
            f"{base_url.rstrip('/')}{prov.endpoint}",
            json=payload,
        )
        resp.raise_for_status()
        return np.array(
            prov.parse_response(resp.json()),
            dtype=np.float32,
        )


class VisualEmbedder:
    """Embed page images (and text queries) via an OpenAI-compatible multimodal embedding API.

    Reuses a single ``httpx.AsyncClient`` for connection pooling across requests.
    Call ``aclose()`` when done to release resources.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        dim: int,
        batch_size: int = 4,
        timeout: float = 120.0,
        provider: MultimodalEmbedProvider | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.dim = dim
        self.batch_size = batch_size
        self.provider = provider or OpenAICompatProvider()
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
            transport=httpx.AsyncHTTPTransport(retries=2),
        )

    async def aclose(self) -> None:
        """Release the underlying HTTP connection pool."""
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def embed_pages(self, images: list[Image.Image]) -> np.ndarray:
        """Embed a list of PIL images into visual vectors.

        Multimodal embedding APIs treat a list of images as a single
        composite input (returning 1 vector), so each image must be
        embedded individually.  ``batch_size`` controls concurrency.

        Returns:
            np.ndarray of shape ``(len(images), dim)`` with dtype float32.
        """
        if not images:
            return np.empty((0, self.dim), dtype=np.float32)

        sem = asyncio.Semaphore(self.batch_size)

        async def _embed_one(img: Image.Image) -> list[float]:
            async with sem:
                image_b64 = self._image_to_b64(img)
                payload = self.provider.build_image_payload(self.model, image_b64)
                resp = await self._client.post(
                    f"{self.base_url}{self.provider.endpoint}",
                    json=payload,
                )
                resp.raise_for_status()
                embeddings = self.provider.parse_response(resp.json())
                vec = embeddings[0]
                if len(vec) != self.dim:
                    raise ValueError(f"Expected embedding dim {self.dim}, got {len(vec)}")
                return vec

        results = await asyncio.gather(*[_embed_one(img) for img in images])
        return np.asarray(results, dtype=np.float32)

    async def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts via the same multimodal embedding API.

        Used as the LightRAG ``embedding_func`` in unified mode so that
        entity/relationship embedding goes through the same httpx client
        (and ``encoding_format: "float"``) as visual embedding — avoiding
        LightRAG's ``openai_embed`` which uses ``encoding_format: "base64"``
        and the ``openai`` Python client, both of which can fail with
        Xinference VL embedding models.

        Returns:
            np.ndarray of shape ``(len(texts), dim)`` with dtype float32.
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        payload = self.provider.build_text_payload(self.model, texts)
        resp = await self._client.post(
            f"{self.base_url}{self.provider.endpoint}",
            json=payload,
        )
        resp.raise_for_status()
        embeddings = self.provider.parse_response(resp.json())
        if len(embeddings) > 0 and len(embeddings[0]) != self.dim:
            raise ValueError(f"Expected embedding dim {self.dim}, got {len(embeddings[0])}")
        return np.asarray(embeddings, dtype=np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _image_to_b64(image: Image.Image) -> str:
        """Convert a PIL Image to a ``data:image/png;base64,...`` URI string."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
