# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Budgeted, retrying execution for provider-owned embedding protocols."""

from __future__ import annotations

import asyncio
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
from PIL import Image

from dlightrag.engine.ai.contracts import InputModality, ResolvedInputModality
from dlightrag.engine.ai.embedding_inputs import (
    EmbeddingInput,
    ImageEmbeddingInput,
    MultimodalEmbeddingInput,
    TextEmbeddingInput,
)
from dlightrag.engine.ai.fingerprints import ModelFingerprint, model_endpoint_fingerprint
from dlightrag.engine.ai.media import bounded_embedding_image_data_uri
from dlightrag.engine.ai.providers.embed_base import (
    EmbeddingContext,
    EmbedProvider,
    input_image_bytes,
)
from dlightrag.engine.ai.providers.embed_providers import get_embed_provider
from dlightrag.engine.ai.scheduler import ModelScheduler
from dlightrag.engine.ai.settings import EmbeddingSettings
from dlightrag.engine.ai.telemetry import NOOP_TELEMETRY, Telemetry, telemetry_error_message

_RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})
_MAX_RETRIES = 2
_RETRY_BASE_SECONDS = 0.5
_RETRY_JITTER_SECONDS = 0.25


@dataclass(frozen=True, slots=True)
class _EmbeddingRequest:
    payload: dict[str, Any]
    expected_count: int
    estimated_image_bytes: int


@dataclass(frozen=True, slots=True)
class _EmbeddingOutcome:
    vectors: list[list[float]]
    retries: int
    usage: dict[str, int | float]


def resolve_embedding_input_modality(
    provider: EmbedProvider,
    model: str,
    mode: InputModality,
) -> ResolvedInputModality:
    """Resolve local input policy against the selected model's fusion contract."""
    if mode == "text":
        return "text"
    native_multimodal = provider.capabilities(model).native_multimodal
    if mode == "auto":
        return "multimodal" if native_multimodal else "text"
    if not native_multimodal:
        raise ValueError(
            f"{provider.__class__.__name__} model {model!r} cannot satisfy "
            "input_modality='multimodal': native single-vector text+image fusion is required"
        )
    return "multimodal"


class MultimodalEmbedder:
    """Embed canonical text and fused visual chunks in one provider-owned space."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        dim: int,
        provider: EmbedProvider,
        input_modality: InputModality = "auto",
        timeout: float = 120.0,
        fingerprint: ModelFingerprint,
        scheduler: ModelScheduler,
        telemetry: Telemetry = NOOP_TELEMETRY,
    ) -> None:
        if dim < 1:
            raise ValueError("Embedding dimension must be positive")
        self.model = model
        self.provider = provider
        self.base_url = (base_url or provider.default_base_url).rstrip("/")
        if not self.base_url:
            raise ValueError(f"{provider.__class__.__name__} requires an embedding base_url")
        self.request_url = provider.request_url(self.base_url, model)
        self.dim = dim
        self._capabilities = provider.capabilities(model)
        self._capabilities.output_dimension.request_value(dim, model=model)
        self.input_modality = resolve_embedding_input_modality(provider, model, input_modality)
        self.supports_images = self.input_modality == "multimodal"
        self.supports_asymmetric = self._capabilities.asymmetric
        self.api_key = api_key
        self.fingerprint = fingerprint
        self._scheduler = scheduler
        self._telemetry = telemetry
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers=provider.request_headers(api_key, base_url=self.base_url),
            transport=httpx.AsyncHTTPTransport(retries=0),
        )

    async def aclose(self) -> None:
        """Release the underlying HTTP connection pool."""
        await self._client.aclose()

    async def embed_texts(
        self,
        texts: list[str],
        *,
        context: EmbeddingContext = "document",
    ) -> list[list[float]]:
        """Embed text inputs with provider and token-budget batch splitting."""
        inputs: list[EmbeddingInput] = [TextEmbeddingInput(text=text) for text in texts]
        return await self._embed_inputs(inputs, context=context, modality="text")

    async def embed_text(self, text: str) -> list[float]:
        """Embed one text input as a query-side vector."""
        (vector,) = await self.embed_texts([text], context="query")
        return vector

    async def embed_documents(self, texts: Sequence[str]) -> Sequence[list[float]]:
        """Expose the document side of the plain-text embedding port."""
        return await self.embed_texts(list(texts), context="document")

    async def embed_query(self, text: str) -> list[float]:
        """Expose the query side of the plain-text embedding port."""
        return await self.embed_text(text)

    @property
    def embedding_fingerprint(self) -> str:
        """Return the canonical embedding-space identity for storage ports."""
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
        self,
        items: list[tuple[str, Image.Image]],
        *,
        context: EmbeddingContext,
    ) -> dict[str, Any]:
        """Build one provider batch for focused contract tests."""
        inputs: list[EmbeddingInput] = [
            self._fused_input(description, image) for description, image in items
        ]
        return self.provider.build_payload(
            self.model,
            inputs,
            context=context,
            output_dimension=self.dim,
        )

    async def embed_index_fused(
        self,
        items: list[tuple[str, Image.Image]],
    ) -> list[list[float]]:
        """Embed description-image pairs as canonical fused document vectors."""
        self._ensure_image_support()
        inputs: list[EmbeddingInput] = await asyncio.to_thread(
            lambda: [self._fused_input(description, image) for description, image in items]
        )
        return await self._embed_inputs(inputs, context="document", modality="multimodal")

    async def embed_query_images(self, images: list[Image.Image]) -> list[list[float]]:
        """Embed query-side images with provider batch limits."""
        self._ensure_image_support()
        inputs: list[EmbeddingInput] = await asyncio.to_thread(
            lambda: [
                ImageEmbeddingInput(data_uri=bounded_embedding_image_data_uri(image))
                for image in images
            ]
        )
        return await self._embed_inputs(inputs, context="query", modality="image")

    def _build_query_image_payload(self, images: list[Image.Image]) -> dict[str, Any]:
        """Build one provider image batch for focused contract tests."""
        inputs: list[EmbeddingInput] = [
            ImageEmbeddingInput(data_uri=bounded_embedding_image_data_uri(image))
            for image in images
        ]
        return self.provider.build_payload(
            self.model,
            inputs,
            context="query",
            output_dimension=self.dim,
        )

    async def probe_image_embedding(self) -> None:
        """Probe both image-query and native fused-document capabilities."""
        image = Image.new("RGB", (1, 1), "white")
        try:
            await self.embed_query_images([image])
            await self.embed_index_fused([("DlightRAG fusion probe", image)])
        finally:
            image.close()

    async def _embed_inputs(
        self,
        inputs: list[EmbeddingInput],
        *,
        context: EmbeddingContext,
        modality: str,
    ) -> list[list[float]]:
        requests = await asyncio.to_thread(self._plan_requests, inputs, context=context)
        return await self._execute_requests(
            requests,
            expected_count=len(inputs),
            context=context,
            modality=modality,
        )

    def _plan_requests(
        self,
        inputs: list[EmbeddingInput],
        *,
        context: EmbeddingContext,
    ) -> list[_EmbeddingRequest]:
        if not inputs:
            raise ValueError("Embedding request requires at least one input")
        capabilities = self._capabilities
        image_byte_counts = [input_image_bytes(item) for item in inputs]

        batches: list[tuple[list[EmbeddingInput], int]] = []
        current: list[EmbeddingInput] = []
        current_image_bytes = 0
        image_request_limit = capabilities.max_image_bytes_per_request
        for item, image_bytes in zip(inputs, image_byte_counts, strict=True):
            if image_request_limit is not None and image_bytes > image_request_limit:
                raise ValueError(
                    f"One embedding input for model {self.model!r} exceeds the request image-byte "
                    f"limit: estimated {image_bytes}, limit {image_request_limit}"
                )
            image_overflow = (
                current
                and image_request_limit is not None
                and current_image_bytes + image_bytes > image_request_limit
            )
            if current and (len(current) >= capabilities.max_inputs or image_overflow):
                batches.append((current, current_image_bytes))
                current = []
                current_image_bytes = 0
            current.append(item)
            current_image_bytes += image_bytes
        if current:
            batches.append((current, current_image_bytes))

        return [
            _EmbeddingRequest(
                payload=self.provider.build_payload(
                    self.model,
                    batch,
                    context=context,
                    output_dimension=self.dim,
                ),
                expected_count=len(batch),
                estimated_image_bytes=image_bytes,
            )
            for batch, image_bytes in batches
        ]

    async def _execute_requests(
        self,
        requests: list[_EmbeddingRequest],
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
                "request_count": len(requests),
                "inline_image_bytes": sum(item.estimated_image_bytes for item in requests),
            },
            model=self.fingerprint.model,
        ) as observation:
            try:
                raw_outcomes = await asyncio.gather(
                    *(
                        self._scheduler.run(lambda request=request: self._execute_request(request))
                        for request in requests
                    ),
                    return_exceptions=True,
                )
                outcomes: list[_EmbeddingOutcome] = []
                for outcome in raw_outcomes:
                    if isinstance(outcome, BaseException):
                        raise outcome
                    outcomes.append(outcome)
                vectors = [vector for outcome in outcomes for vector in outcome.vectors]
                self._validate_vectors(vectors, expected_count=expected_count)
            except Exception as exc:
                observation.update(
                    level="ERROR",
                    status_message=telemetry_error_message(self._telemetry, exc),
                )
                raise
            usage = _merge_usage(outcome.usage for outcome in outcomes)
            observation.update(
                output={
                    "embedding_count": len(vectors),
                    "request_count": len(requests),
                    "retry_count": sum(outcome.retries for outcome in outcomes),
                    "usage": usage,
                }
            )
            return vectors

    async def _execute_request(self, request: _EmbeddingRequest) -> _EmbeddingOutcome:
        data, retries = await self._post(request.payload)
        vectors = self.provider.parse_response(data, expected_count=request.expected_count)
        self._validate_vectors(vectors, expected_count=request.expected_count)
        return _EmbeddingOutcome(
            vectors=vectors,
            retries=retries,
            usage=self.provider.response_usage(data),
        )

    async def _post(self, payload: dict[str, Any]) -> tuple[Mapping[str, Any], int]:
        for attempt in range(_MAX_RETRIES + 1):
            response: httpx.Response | None = None
            try:
                response = await self._client.post(self.request_url, json=payload)
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, Mapping):
                    raise ValueError("Embedding response JSON must be an object")
                return data, attempt
            except httpx.TransportError:
                if attempt >= _MAX_RETRIES:
                    raise
            except httpx.HTTPStatusError as exc:
                if attempt >= _MAX_RETRIES or not _retryable_status(exc.response.status_code):
                    raise
                response = exc.response
            delay = _retry_delay(response, attempt=attempt)
            await asyncio.sleep(delay)
        raise RuntimeError("Embedding retry loop exhausted")

    def _validate_vectors(
        self,
        vectors: object,
        *,
        expected_count: int | None = None,
    ) -> None:
        if not isinstance(vectors, list):
            raise ValueError("Embedding vectors must be a list")
        if expected_count is not None and len(vectors) != expected_count:
            raise ValueError(f"Expected {expected_count} embedding vectors, got {len(vectors)}")
        for index, vector in enumerate(vectors):
            if not isinstance(vector, list) or len(vector) != self.dim:
                actual = len(vector) if isinstance(vector, list) else "non-list"
                raise ValueError(
                    f"Expected embedding dim {self.dim}, got {actual} at index {index}"
                )
            if not all(
                type(value) in {int, float} and math.isfinite(value)  # noqa: E721
                for value in vector
            ):
                raise ValueError(f"Embedding vector at index {index} contains invalid values")
            norm = math.hypot(*(float(value) for value in vector))
            if not math.isfinite(norm) or norm == 0.0:
                raise ValueError(f"Embedding vector at index {index} has invalid norm")

    def _ensure_image_support(self) -> None:
        if not self.supports_images:
            raise ValueError(
                f"{self.provider.__class__.__name__} model {self.model!r} does not support "
                "native fused image embeddings"
            )


def _retryable_status(status_code: int) -> bool:
    return status_code in _RETRYABLE_STATUS_CODES or 500 <= status_code <= 599


def _retry_delay(response: httpx.Response | None, *, attempt: int) -> float:
    if response is not None:
        value = response.headers.get("Retry-After")
        if value:
            try:
                return max(0.0, float(value))
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(value)
                    if retry_at.tzinfo is None:
                        retry_at = retry_at.replace(tzinfo=UTC)
                    return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())
                except TypeError, ValueError, OverflowError:
                    pass
    return _RETRY_BASE_SECONDS * (2**attempt) + random.uniform(  # noqa: S311
        0,
        _RETRY_JITTER_SECONDS,
    )


def _merge_usage(values: Iterable[dict[str, int | float]]) -> dict[str, int | float]:
    merged: dict[str, int | float] = {}
    for usage in values:
        for key, value in usage.items():
            merged[key] = merged.get(key, 0) + value
    return merged


def create_embedding_model(
    settings: EmbeddingSettings,
    *,
    scheduler: ModelScheduler,
    telemetry: Telemetry = NOOP_TELEMETRY,
) -> MultimodalEmbedder:
    """Build a closeable embedding model from immutable settings."""
    provider = get_embed_provider(settings.provider)
    resolved_base_url = settings.base_url or provider.default_base_url
    return MultimodalEmbedder(
        model=settings.model,
        api_key=settings.api_key or "",
        base_url=resolved_base_url,
        dim=settings.dim,
        provider=provider,
        input_modality=settings.input_modality,
        timeout=settings.timeout,
        fingerprint=model_endpoint_fingerprint(
            settings.provider,
            settings.model,
            resolved_base_url,
        ),
        scheduler=scheduler,
        telemetry=telemetry,
    )


__all__ = [
    "MultimodalEmbedder",
    "create_embedding_model",
    "resolve_embedding_input_modality",
]
