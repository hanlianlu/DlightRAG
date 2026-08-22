# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Context-aware text and image embedding execution."""

import asyncio
import logging
import math
from collections.abc import Sequence
from typing import Any

import httpx
from PIL import Image

from dlightrag.ai.contracts import AsymmetricMode, InputModality, ResolvedInputModality
from dlightrag.ai.embedding_inputs import (
    EmbeddingInput,
    ImageEmbeddingInput,
    MultimodalEmbeddingInput,
    TextEmbeddingInput,
)
from dlightrag.ai.fingerprints import ModelFingerprint, model_fingerprint
from dlightrag.ai.media import bounded_embedding_image_data_uri
from dlightrag.ai.providers.embed_base import EmbeddingContext, EmbedProvider
from dlightrag.ai.providers.embed_providers import get_embed_provider
from dlightrag.ai.scheduler import ModelScheduler
from dlightrag.ai.settings import EmbeddingSettings
from dlightrag.ai.telemetry import NOOP_TELEMETRY, Telemetry, telemetry_error_message

logger = logging.getLogger(__name__)


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
    mode: InputModality,
) -> ResolvedInputModality:
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
    """Embed text and images in one provider-owned vector space."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        dim: int,
        provider: EmbedProvider,
        input_modality: InputModality = "auto",
        asymmetric: AsymmetricMode = "auto",
        timeout: float = 120.0,
        fingerprint: ModelFingerprint,
        scheduler: ModelScheduler,
        telemetry: Telemetry = NOOP_TELEMETRY,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/") if base_url else "https://api.openai.com/v1"
        self.dim = dim
        self.provider = provider
        self.input_modality = resolve_embedding_input_modality(provider, input_modality)
        self.supports_images = self.input_modality == "multimodal"
        self.asymmetric = resolve_asymmetric(provider, asymmetric)
        self.supports_asymmetric = self.asymmetric
        self.api_key = api_key
        self.fingerprint = fingerprint
        self._scheduler = scheduler
        self._telemetry = telemetry
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
        return await self._request_vectors(
            payload,
            expected_count=len(texts),
            context=context,
            modality="text",
        )

    async def embed_text(self, text: str) -> list[float]:
        """Embed one text input as a query-side vector."""
        (vector,) = await self.embed_texts([text], context="query")
        return vector

    # Plain-text vector entry points: document and query batches over raw
    # strings, so a host that owns this embedder can hand it to any text
    # vector consumer (e.g. a storage adapter's dense leg) with no wrapper
    # class — the consumer only needs the two methods plus dim.
    async def embed_documents(self, texts: Sequence[str]) -> Sequence[list[float]]:
        return await self.embed_texts(list(texts), context="document")

    async def embed_query(self, text: str) -> list[float]:
        return await self.embed_text(text)

    @property
    def embedding_fingerprint(self) -> str:
        """One canonical identity string for the embedding space.

        The ``TextEmbedder`` port declares ``fingerprint: str``; this property
        is that string form of the structured ``ModelFingerprint`` so storage
        adapters can persist it in a TEXT column.
        """
        endpoint = self.fingerprint.endpoint_fingerprint
        base = f"{self.fingerprint.provider}:{self.fingerprint.model}"
        return f"{base}@{endpoint}" if endpoint else base

    def _fused_input(self, description: str, image: Image.Image) -> MultimodalEmbeddingInput:
        data_uri = bounded_embedding_image_data_uri(image)
        parts: list[TextEmbeddingInput | ImageEmbeddingInput] = []
        text = description.strip()
        if text:
            parts.append(TextEmbeddingInput(text=text))
        parts.append(ImageEmbeddingInput(data_uri=data_uri))
        return MultimodalEmbeddingInput(parts=parts)

    def _build_fused_payload(
        self, items: list[tuple[str, Image.Image]], *, context: EmbeddingContext
    ) -> dict[str, Any]:
        inputs: list[EmbeddingInput] = [
            self._fused_input(description, image) for description, image in items
        ]
        return self.provider.build_payload(
            self.model,
            inputs,
            context=context,
            asymmetric=self.asymmetric,
            output_dimension=self.dim,
        )

    async def embed_index_fused(self, items: list[tuple[str, Image.Image]]) -> list[list[float]]:
        """Embed description-image pairs as fused document vectors."""
        self._ensure_image_support()
        if not items:
            return []
        payload = await asyncio.to_thread(self._build_fused_payload, items, context="document")
        return await self._request_vectors(
            payload,
            expected_count=len(items),
            context="document",
            modality="multimodal",
        )

    async def embed_query_images(self, images: list[Image.Image]) -> list[list[float]]:
        """Embed query-side images in one batched provider request."""
        self._ensure_image_support()
        if not images:
            return []
        payload = await asyncio.to_thread(self._build_query_image_payload, images)
        return await self._request_vectors(
            payload,
            expected_count=len(images),
            context="query",
            modality="image",
        )

    def _build_query_image_payload(self, images: list[Image.Image]) -> dict[str, Any]:
        inputs: list[EmbeddingInput] = [
            ImageEmbeddingInput(data_uri=bounded_embedding_image_data_uri(image))
            for image in images
        ]
        return self.provider.build_payload(
            self.model,
            inputs,
            context="query",
            asymmetric=self.asymmetric,
            output_dimension=self.dim,
        )

    async def probe_image_embedding(self) -> None:
        """Probe that the provider can embed an image."""
        await self.embed_query_images([Image.new("RGB", (1, 1), "white")])

    async def _request_vectors(
        self,
        payload: dict[str, Any],
        *,
        expected_count: int,
        context: EmbeddingContext,
        modality: str,
    ) -> list[list[float]]:
        return await self._scheduler.run(
            lambda: self._execute_request(
                payload,
                expected_count=expected_count,
                context=context,
                modality=modality,
            )
        )

    async def _execute_request(
        self,
        payload: dict[str, Any],
        *,
        expected_count: int,
        context: EmbeddingContext,
        modality: str,
    ) -> list[list[float]]:
        async with self._telemetry.observe(
            f"embed_{self.model}",
            as_type="embedding",
            input={"input_count": expected_count},
            metadata={
                "context": context,
                "modality": modality,
                "provider": self.fingerprint.provider,
                "endpoint_fingerprint": self.fingerprint.endpoint_fingerprint,
            },
            model=self.fingerprint.model,
        ) as observation:
            try:
                data = await self._post(payload)
                vectors = self.provider.parse_response(data)
                self._validate_vectors(vectors, expected_count=expected_count)
            except Exception as exc:
                observation.update(
                    level="ERROR",
                    status_message=telemetry_error_message(self._telemetry, exc),
                )
                raise
            observation.update(output={"embedding_count": len(vectors)})
            return vectors

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
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


def create_embedding_model(
    settings: EmbeddingSettings,
    *,
    scheduler: ModelScheduler,
    telemetry: Telemetry = NOOP_TELEMETRY,
) -> MultimodalEmbedder:
    """Build a closeable embedding model from immutable settings."""
    return MultimodalEmbedder(
        model=settings.model,
        api_key=settings.api_key or "",
        base_url=settings.base_url or "",
        dim=settings.dim,
        provider=get_embed_provider(settings.provider),
        input_modality=settings.input_modality,
        asymmetric=settings.asymmetric,
        timeout=settings.timeout,
        fingerprint=model_fingerprint(settings),
        scheduler=scheduler,
        telemetry=telemetry,
    )


__all__ = [
    "MultimodalEmbedder",
    "create_embedding_model",
    "resolve_asymmetric",
    "resolve_embedding_input_modality",
]
