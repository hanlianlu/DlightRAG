# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Provider-neutral embedding input records."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TextEmbeddingInput:
    """Text segment sent to an embedding provider."""

    text: str


@dataclass(frozen=True)
class ImageEmbeddingInput:
    """Image segment sent to an embedding provider."""

    data_uri: str | None = None
    path: str | None = None
    url: str | None = None

    def as_payload_value(self) -> str:
        if self.data_uri:
            return self.data_uri
        if self.url:
            return self.url
        if self.path:
            return self.path
        raise ValueError("ImageEmbeddingInput requires data_uri, url, or path")


@dataclass(frozen=True)
class MultimodalEmbeddingInput:
    """One fused multimodal input made from ordered text/image parts."""

    parts: list[TextEmbeddingInput | ImageEmbeddingInput]


type EmbeddingInput = TextEmbeddingInput | ImageEmbeddingInput | MultimodalEmbeddingInput
