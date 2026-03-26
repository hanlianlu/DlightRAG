# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Concrete EmbedProvider implementations."""

from __future__ import annotations

from dlightrag.models.providers.embed_base import EmbedProvider


class OpenAICompatEmbedProvider(EmbedProvider):
    """POST /embeddings — Xinference VL, Qwen-VL, Azure, standard OpenAI."""

    @property
    def endpoint(self) -> str:
        return "/embeddings"

    def build_payload(self, model: str, inputs: list[str]) -> dict:
        return {"model": model, "input": inputs, "encoding_format": "float"}


class VoyageEmbedProvider(EmbedProvider):
    """POST /multimodalembeddings — Voyage multimodal 3/3.5."""

    @property
    def endpoint(self) -> str:
        return "/multimodalembeddings"

    def build_payload(self, model: str, inputs: list[str]) -> dict:
        items = []
        for inp in inputs:
            if inp.startswith("data:image/"):
                items.append({"content": [{"type": "image_base64", "image_base64": inp}]})
            else:
                items.append({"content": [{"type": "text", "text": inp}]})
        return {"model": model, "inputs": items}


class JinaEmbedProvider(EmbedProvider):
    """POST /v1/embeddings — Jina v4."""

    @property
    def endpoint(self) -> str:
        return "/v1/embeddings"

    def build_payload(self, model: str, inputs: list[str]) -> dict:
        return {"model": model, "input": inputs, "encoding_type": "float"}


class DashScopeEmbedProvider(EmbedProvider):
    """POST /api/v1/services/embeddings/... — Qwen via Aliyun DashScope."""

    @property
    def endpoint(self) -> str:
        return "/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"

    def build_payload(self, model: str, inputs: list[str]) -> dict:
        contents = []
        for inp in inputs:
            if inp.startswith("data:image/"):
                contents.append({"image": inp})
            else:
                contents.append({"text": inp})
        return {"model": model, "input": {"contents": contents}}

    def parse_response(self, data: dict) -> list[list[float]]:
        return [item["embedding"] for item in data["output"]["embeddings"]]


class OllamaEmbedProvider(EmbedProvider):
    """POST /api/embed — local Ollama."""

    @property
    def endpoint(self) -> str:
        return "/api/embed"

    def build_payload(self, model: str, inputs: list[str]) -> dict:
        return {"model": model, "input": inputs}

    def parse_response(self, data: dict) -> list[list[float]]:
        return data["embeddings"]


_EMBED_REGISTRY: dict[str, type[EmbedProvider]] = {
    "openai": OpenAICompatEmbedProvider,
    "voyage": VoyageEmbedProvider,
    "jina": JinaEmbedProvider,
    "dashscope": DashScopeEmbedProvider,
    "ollama": OllamaEmbedProvider,
}


def detect_embed_provider(
    model: str,
    provider: str | None = None,
    *,
    base_url: str | None = None,
) -> EmbedProvider:
    """Auto-detect embed provider: explicit > base_url heuristic > model name."""
    if provider is not None:
        cls = _EMBED_REGISTRY.get(provider.lower())
        if cls is None:
            available = ", ".join(sorted(_EMBED_REGISTRY))
            raise ValueError(f"Unknown embed provider {provider!r}. Available: {available}")
        return cls()

    if base_url:
        url_lower = base_url.lower()
        if ":11434" in url_lower:
            return OllamaEmbedProvider()

    name = model.lower()
    if "voyage" in name:
        return VoyageEmbedProvider()
    if "jina" in name:
        return JinaEmbedProvider()

    return OpenAICompatEmbedProvider()
