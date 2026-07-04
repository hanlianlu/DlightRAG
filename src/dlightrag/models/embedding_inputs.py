# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Provider-neutral embedding input records."""

import base64
import io
from dataclasses import dataclass

from PIL import Image


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

    @classmethod
    def from_pil(cls, image: Image.Image) -> ImageEmbeddingInput:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return cls(data_uri=f"data:image/png;base64,{encoded}")

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
