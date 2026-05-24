# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Context-aware text and image embedding over DlightRAG's provider registry."""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

import httpx
from PIL import Image

from dlightrag.models.embedding_inputs import (
    EmbeddingInput,
    ImageEmbeddingInput,
    TextEmbeddingInput,
)
from dlightrag.models.providers.embed_base import EmbedProvider

logger = logging.getLogger(__name__)

EmbeddingContext = Literal["query", "document"]
AsymmetricMode = Literal["auto", "require", "disable"]


def resolve_asymmetric(provider: EmbedProvider, mode: AsymmetricMode) -> bool:
    """Resolve asymmetric config to the active runtime behavior."""
    if mode == "disable":
        return False
    if provider.supports_asymmetric:
        return True
    if mode == "require":
        raise ValueError(f"{provider.__class__.__name__} does not support asymmetric embeddings")
    return False


class MultimodalEmbedder:
    """Embed text and images into one shared multimodal vector space."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        dim: int,
        provider: EmbedProvider,
        asymmetric: AsymmetricMode = "auto",
        batch_size: int = 4,
        timeout: float = 120.0,
    ) -> None:
        if not provider.supports_images:
            raise ValueError(f"{provider.__class__.__name__} does not support image embeddings")
        self.model = model
        self.base_url = base_url.rstrip("/") if base_url else "https://api.openai.com/v1"
        self.dim = dim
        self.provider = provider
        self.asymmetric = resolve_asymmetric(provider, asymmetric)
        self.supports_asymmetric = self.asymmetric
        self.batch_size = batch_size
        self.api_key = api_key
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers=provider.request_headers(api_key),
            transport=httpx.AsyncHTTPTransport(retries=2),
        )

    async def aclose(self) -> None:
        """Release the underlying HTTP connection pool."""
        await self._client.aclose()

    async def embed_texts(
        self, texts: list[str], *, context: EmbeddingContext = "document"
    ) -> list[list[float]]:
        """Embed a batch of text inputs."""
        if not texts:
            return []
        inputs: list[EmbeddingInput] = [TextEmbeddingInput(text=text) for text in texts]
        payload = self.provider.build_payload(
            self.model,
            inputs,
            context=context,
            asymmetric=self.asymmetric,
            output_dimension=self.dim,
        )
        data = await self._post(payload)
        vectors = self.provider.parse_response(data)
        self._validate_vectors(vectors)
        return vectors

    async def embed_images(
        self, images: list[Image.Image], *, context: EmbeddingContext
    ) -> list[list[float]]:
        """Embed images with explicit query/document context."""
        if not images:
            return []

        sem = asyncio.Semaphore(self.batch_size)

        async def one(image: Image.Image) -> list[float]:
            async with sem:
                payload = self._build_image_payload(image, context=context)
                data = await self._post(payload)
                vectors = self.provider.parse_response(data)
                self._validate_vectors(vectors)
                return vectors[0]

        return await asyncio.gather(*(one(image) for image in images))

    async def embed_index_images(self, images: list[Image.Image]) -> list[list[float]]:
        """Embed document/index-side images."""
        return await self.embed_images(images, context="document")

    async def embed_query_images(self, images: list[Image.Image]) -> list[list[float]]:
        """Embed query-side images."""
        return await self.embed_images(images, context="query")

    async def probe_image_embedding(self) -> None:
        """Probe that the configured provider can embed an image."""
        await self.embed_index_images([Image.new("RGB", (1, 1), "white")])

    def build_image_payload_for_test(
        self, image: Image.Image, *, context: EmbeddingContext
    ) -> dict:
        """Expose payload construction to unit tests without HTTP calls."""
        return self._build_image_payload(image, context=context)

    def validate_vectors_for_test(self, vectors: list[list[float]]) -> None:
        """Expose vector validation to unit tests."""
        self._validate_vectors(vectors)

    async def _post(self, payload: dict) -> dict:
        url = f"{self.base_url}{self.provider.endpoint_for_model(self.model)}"
        headers = self.provider.request_headers(self.api_key)
        response = await self._client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

    def _build_image_payload(self, image: Image.Image, *, context: EmbeddingContext) -> dict:
        return self.provider.build_payload(
            self.model,
            [ImageEmbeddingInput.from_pil(image)],
            context=context,
            asymmetric=self.asymmetric,
            output_dimension=self.dim,
        )

    def _validate_vectors(self, vectors: list[list[float]]) -> None:
        if vectors and len(vectors[0]) != self.dim:
            raise ValueError(f"Expected embedding dim {self.dim}, got {len(vectors[0])}")
