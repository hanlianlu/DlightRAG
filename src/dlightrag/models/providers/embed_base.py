# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Abstract base for multimodal embedding providers."""

from abc import ABC, abstractmethod
from typing import Literal

from dlightrag.models.embedding_inputs import EmbeddingInput

EmbeddingContext = Literal["query", "document"]
EmbedContext = EmbeddingContext
ImageInputCapability = Literal["unsupported", "opt_in", "native"]


class EmbedProvider(ABC):
    """Strategy for provider-specific multimodal embedding API protocols."""

    endpoint: str = ""
    image_input_capability: ImageInputCapability = "unsupported"
    supports_asymmetric: bool = False
    # Whether the provider fuses interleaved text+image into a single vector (a
    # "unified multimodal" model). Enables text-retrievable visual chunk vectors.
    supports_fused_multimodal: bool = False

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

    def parse_response(self, data: dict) -> list[list[float]]:
        """Extract vectors from OpenAI-compatible JSON response."""
        return [item["embedding"] for item in data["data"]]
