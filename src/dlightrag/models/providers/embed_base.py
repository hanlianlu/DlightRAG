# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Abstract base for multimodal embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from dlightrag.models.embedding_inputs import EmbeddingInput

EmbeddingContext = Literal["query", "document"]
EmbedContext = EmbeddingContext


class EmbedProvider(ABC):
    """Strategy for provider-specific multimodal embedding API protocols."""

    endpoint: str = ""
    supports_images: bool = False
    supports_asymmetric: bool = False
    default_dim: int | None = None
    known_dims: frozenset[int] | None = None

    @abstractmethod
    def build_payload(
        self,
        model: str,
        inputs: list[EmbeddingInput],
        *,
        context: EmbeddingContext,
        asymmetric: bool = False,
        output_dimension: int | None = None,
    ) -> dict:
        """Build request payload for the embedding API."""

    def endpoint_for_model(self, model: str) -> str:
        """Return endpoint path relative to base_url."""
        return self.endpoint.format(model=model)

    def request_headers(self, api_key: str) -> dict[str, str]:
        """Return provider-specific auth headers."""
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}

    @property
    def max_images_per_request(self) -> int:
        """Hard API limit on images per request."""
        return 128

    def parse_response(self, data: dict) -> list[list[float]]:
        """Extract vectors from OpenAI-compatible JSON response."""
        return [item["embedding"] for item in data["data"]]
