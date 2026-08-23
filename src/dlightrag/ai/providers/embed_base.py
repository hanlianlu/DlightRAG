# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Provider seam for embedding request and response protocols."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from dlightrag.ai.embedding_inputs import (
    EmbeddingInput,
    ImageEmbeddingInput,
    MultimodalEmbeddingInput,
    TextEmbeddingInput,
)
from dlightrag.ai.tokens import estimate_tokens

EmbeddingContext = Literal["query", "document"]
ImageInputCapability = Literal["unsupported", "native"]


@dataclass(frozen=True, slots=True)
class OutputDimensionPolicy:
    """How one known model treats the configured final vector dimension."""

    send_upstream: bool = False
    fixed: int | None = None
    minimum: int | None = None
    maximum: int | None = None
    allowed: tuple[int, ...] = ()

    def request_value(self, configured: int, *, model: str) -> int | None:
        """Validate the final dimension and return the optional wire value."""
        if self.fixed is not None and configured != self.fixed:
            raise ValueError(
                f"Embedding model {model!r} has fixed dimension {self.fixed}, got {configured}"
            )
        if self.allowed and configured not in self.allowed:
            values = ", ".join(str(value) for value in self.allowed)
            raise ValueError(
                f"Embedding model {model!r} only supports dimensions {values}, got {configured}"
            )
        if self.minimum is not None and configured < self.minimum:
            raise ValueError(
                f"Embedding model {model!r} requires dimension >= {self.minimum}, got {configured}"
            )
        if self.maximum is not None and configured > self.maximum:
            raise ValueError(
                f"Embedding model {model!r} requires dimension <= {self.maximum}, got {configured}"
            )
        return configured if self.send_upstream else None


@dataclass(frozen=True, slots=True)
class EmbedModelCapabilities:
    """Provider-owned facts needed to plan safe embedding requests."""

    image_input: ImageInputCapability = "unsupported"
    fused_inputs: bool = False
    asymmetric: bool = False
    max_inputs: int = 64
    max_tokens_per_input: int | None = None
    max_tokens_per_request: int | None = None
    max_image_bytes_per_request: int | None = None
    token_safety_margin: float = 1.2
    output_dimension: OutputDimensionPolicy = OutputDimensionPolicy()

    @property
    def native_multimodal(self) -> bool:
        """Whether both DlightRAG visual paths can share one canonical vector."""
        return self.image_input == "native" and self.fused_inputs


def input_parts(item: EmbeddingInput) -> list[TextEmbeddingInput | ImageEmbeddingInput]:
    """Flatten one provider-neutral embedding item into ordered parts."""
    if isinstance(item, MultimodalEmbeddingInput):
        return item.parts
    return [item]


def input_text(item: EmbeddingInput) -> str:
    """Join the text-bearing parts used for request budgeting."""
    return "\n".join(
        part.text for part in input_parts(item) if isinstance(part, TextEmbeddingInput)
    )


def input_image_bytes(item: EmbeddingInput) -> int:
    """Estimate decoded bytes carried by inline base64 image parts."""
    total = 0
    for part in input_parts(item):
        if not isinstance(part, ImageEmbeddingInput) or not part.data_uri:
            continue
        encoded = part.data_uri.partition(",")[2]
        if not encoded:
            continue
        padding = len(encoded) - len(encoded.rstrip("="))
        total += max(0, (len(encoded) * 3) // 4 - padding)
    return total


def numeric_usage(
    values: object,
    *,
    prefix: str = "",
) -> dict[str, int | float]:
    """Keep numeric, non-boolean provider usage fields under stable names."""
    if not isinstance(values, Mapping):
        return {}
    return {
        f"{prefix}{key}": value
        for key, value in values.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }


class EmbedProvider(ABC):
    """Adapter for one embedding wire protocol.

    Callers plan generic batch and token limits from :meth:`capabilities`; URL,
    auth, payload, response ordering, and provider usage stay behind this seam.
    """

    default_base_url: str = ""

    def capabilities(self, model: str) -> EmbedModelCapabilities:
        """Return conservative facts for an unknown model by default."""
        del model
        return EmbedModelCapabilities()

    @abstractmethod
    def request_url(self, base_url: str, model: str) -> str:
        """Return the complete request URL owned by this adapter."""

    def request_headers(self, api_key: str, *, base_url: str) -> dict[str, str]:
        """Return provider-specific authentication headers."""
        del base_url
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def build_payload(
        self,
        model: str,
        inputs: list[EmbeddingInput],
        *,
        context: EmbeddingContext,
        output_dimension: int,
    ) -> dict[str, Any]:
        """Validate one batch and serialize it for the provider."""
        capabilities = self.capabilities(model)
        self._validate_inputs(inputs, capabilities=capabilities)
        if len(inputs) > capabilities.max_inputs:
            raise ValueError(
                f"Embedding model {model!r} accepts at most {capabilities.max_inputs} inputs"
            )
        wire_dimension = capabilities.output_dimension.request_value(
            output_dimension,
            model=model,
        )
        return self._build_payload(
            model,
            inputs,
            context=context,
            output_dimension=wire_dimension,
        )

    @abstractmethod
    def _build_payload(
        self,
        model: str,
        inputs: list[EmbeddingInput],
        *,
        context: EmbeddingContext,
        output_dimension: int | None,
    ) -> dict[str, Any]:
        """Serialize an already validated embedding batch."""

    def estimate_input_tokens(self, model: str, item: EmbeddingInput) -> int:
        """Conservatively estimate text tokens for non-OpenAI providers."""
        capabilities = self.capabilities(model)
        raw = estimate_tokens(input_text(item))
        return math.ceil(raw * capabilities.token_safety_margin)

    def parse_response(self, data: Mapping[str, Any], *, expected_count: int) -> list[list[float]]:
        """Extract vectors from an ordered ``data`` response."""
        items = data.get("data")
        if not isinstance(items, list) or len(items) != expected_count:
            actual = len(items) if isinstance(items, list) else "non-list"
            raise ValueError(f"Expected {expected_count} embedding response items, got {actual}")
        vectors: list[list[float]] = []
        for index, item in enumerate(items):
            if not isinstance(item, Mapping) or not isinstance(item.get("embedding"), list):
                raise ValueError(f"Embedding response item {index} has no vector")
            vectors.append(item["embedding"])
        return vectors

    def response_usage(self, data: Mapping[str, Any]) -> dict[str, int | float]:
        """Return numeric provider usage fields for logical-call telemetry."""
        return numeric_usage(data.get("usage"))

    @staticmethod
    def _validate_inputs(
        inputs: list[EmbeddingInput],
        *,
        capabilities: EmbedModelCapabilities,
    ) -> None:
        if not inputs:
            raise ValueError("Embedding request requires at least one input")
        for index, item in enumerate(inputs):
            parts = input_parts(item)
            if not parts:
                raise ValueError(f"Embedding input {index} has no parts")
            has_text = False
            has_image = False
            for part in parts:
                if isinstance(part, TextEmbeddingInput):
                    if not part.text.strip():
                        raise ValueError(f"Embedding text input {index} is empty")
                    has_text = True
                else:
                    part.as_payload_value()
                    has_image = True
            if has_image and capabilities.image_input != "native":
                raise ValueError("Selected embedding model does not support image inputs")
            if has_text and has_image and not capabilities.fused_inputs:
                raise ValueError("Selected embedding model cannot fuse text and image inputs")
