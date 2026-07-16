# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Context-aware text and image embedding over DlightRAG's provider registry."""

import logging
import math
from typing import Literal

import httpx
from PIL import Image

from dlightrag.models.embedding_inputs import (
    EmbeddingInput,
    ImageEmbeddingInput,
    MultimodalEmbeddingInput,
    TextEmbeddingInput,
)
from dlightrag.models.providers.embed_base import EmbedProvider
from dlightrag.utils.images import bounded_embedding_image_data_uri

logger = logging.getLogger(__name__)

EmbeddingContext = Literal["query", "document"]
AsymmetricMode = Literal["auto", "require", "disable"]
EmbeddingInputModality = Literal["auto", "text", "multimodal"]
ResolvedEmbeddingInputModality = Literal["text", "multimodal"]


def resolve_asymmetric(provider: EmbedProvider, mode: AsymmetricMode) -> bool:
    """Resolve asymmetric config to the active runtime behavior."""
    if mode == "disable":
        return False
    if provider.supports_asymmetric:
        return True
    if mode == "require":
        raise ValueError(f"{provider.__class__.__name__} does not support asymmetric embeddings")
    return False


def resolve_embedding_input_modality(
    provider: EmbedProvider,
    mode: EmbeddingInputModality,
) -> ResolvedEmbeddingInputModality:
    """Resolve configured input policy against one transport serializer."""
    if mode == "text":
        return "text"
    if mode == "auto":
        return "multimodal" if provider.image_input_capability == "native" else "text"
    if provider.image_input_capability == "unsupported":
        raise ValueError(
            f"{provider.__class__.__name__} cannot satisfy input_modality='multimodal'"
        )
    return "multimodal"


class MultimodalEmbedder:
    """Embed text, and images when the provider supports a shared vector space."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        dim: int,
        provider: EmbedProvider,
        input_modality: EmbeddingInputModality = "auto",
        asymmetric: AsymmetricMode = "auto",
        batch_size: int = 4,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/") if base_url else "https://api.openai.com/v1"
        self.dim = dim
        self.provider = provider
        self.input_modality = resolve_embedding_input_modality(provider, input_modality)
        self.supports_images = self.input_modality == "multimodal"
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
        self._validate_vectors(vectors, expected_count=len(texts))
        return vectors

    def _fused_input(self, description: str, image: Image.Image) -> MultimodalEmbeddingInput:
        """Build one interleaved text+image input for fused embedding."""
        data_uri = bounded_embedding_image_data_uri(image)
        parts: list[TextEmbeddingInput | ImageEmbeddingInput] = []
        text = description.strip()
        if text:
            parts.append(TextEmbeddingInput(text=text))
        parts.append(ImageEmbeddingInput(data_uri=data_uri))
        return MultimodalEmbeddingInput(parts=parts)

    async def embed_index_fused(self, items: list[tuple[str, Image.Image]]) -> list[list[float]]:
        """Embed (description, image) pairs as fused document vectors in one request.

        Any image-capable provider is a unified multimodal model, so each VLM
        description and its image are interleaved into one vector and the visual
        chunk stays reachable by text queries (closes the modality gap). An empty
        description degrades gracefully to an image-only vector.

        All pairs are sent as a single provider request; callers chunk the input
        to respect provider batch limits.
        """
        self._ensure_image_support()
        if not items:
            return []
        inputs: list[EmbeddingInput] = [
            self._fused_input(description, image) for description, image in items
        ]
        payload = self.provider.build_payload(
            self.model,
            inputs,
            context="document",
            asymmetric=self.asymmetric,
            output_dimension=self.dim,
        )
        data = await self._post(payload)
        vectors = self.provider.parse_response(data)
        self._validate_vectors(vectors, expected_count=len(items))
        return vectors

    async def embed_query_images(self, images: list[Image.Image]) -> list[list[float]]:
        """Embed query-side images (image-only) for direct visual retrieval.

        All images are sent as one query-context request. This preserves the raw
        visual signal that the VLM-description path loses; the caller fuses these
        results with the text/BM25/KG legs via RRF, so partial overlap is fine.
        Works for any image-capable provider: the query-image vector matches the
        index in the provider's shared text-image space (cross-modal), whether the
        index vectors are fused or LightRAG's native VLM->text descriptions.
        """
        self._ensure_image_support()
        if not images:
            return []
        inputs: list[EmbeddingInput] = [
            ImageEmbeddingInput(data_uri=bounded_embedding_image_data_uri(image))
            for image in images
        ]
        payload = self.provider.build_payload(
            self.model,
            inputs,
            context="query",
            asymmetric=self.asymmetric,
            output_dimension=self.dim,
        )
        data = await self._post(payload)
        vectors = self.provider.parse_response(data)
        self._validate_vectors(vectors, expected_count=len(images))
        return vectors

    async def probe_image_embedding(self) -> None:
        """Probe that the provider can embed an image (gates the direct-visual leg)."""
        await self.embed_query_images([Image.new("RGB", (1, 1), "white")])

    def build_fused_payload_for_test(
        self, description: str, image: Image.Image, *, context: EmbeddingContext
    ) -> dict:
        """Expose fused payload construction to unit tests without HTTP calls."""
        self._ensure_image_support()
        return self.provider.build_payload(
            self.model,
            [self._fused_input(description, image)],
            context=context,
            asymmetric=self.asymmetric,
            output_dimension=self.dim,
        )

    def validate_vectors_for_test(self, vectors: list[list[float]]) -> None:
        """Expose vector validation to unit tests."""
        self._validate_vectors(vectors)

    async def _post(self, payload: dict) -> dict:
        url = f"{self.base_url}{self.provider.endpoint_for_model(self.model)}"
        headers = self.provider.request_headers(self.api_key)
        response = await self._client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

    def _validate_vectors(
        self,
        vectors: list[list[float]],
        *,
        expected_count: int | None = None,
    ) -> None:
        if expected_count is not None and len(vectors) != expected_count:
            raise ValueError(f"Expected {expected_count} embedding vectors, got {len(vectors)}")
        for index, vector in enumerate(vectors):
            if len(vector) != self.dim:
                raise ValueError(
                    f"Expected embedding dim {self.dim}, got {len(vector)} at index {index}"
                )
            if not all(isinstance(value, int | float) and math.isfinite(value) for value in vector):
                raise ValueError(f"Embedding vector at index {index} contains non-finite values")

    def _ensure_image_support(self) -> None:
        if not self.supports_images:
            raise ValueError(
                f"{self.provider.__class__.__name__} does not support image embeddings"
            )
