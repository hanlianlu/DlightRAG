# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Concrete multimodal EmbedProvider implementations."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from dlightrag.models.embedding_inputs import (
    EmbeddingInput,
    ImageEmbeddingInput,
    MultimodalEmbeddingInput,
    TextEmbeddingInput,
)
from dlightrag.models.providers.embed_base import EmbedContext, EmbedProvider

_DATA_URI_RE = re.compile(r"^data:([^;]+);base64,(.+)$", re.DOTALL)


def _parts(item: EmbeddingInput) -> list[TextEmbeddingInput | ImageEmbeddingInput]:
    if isinstance(item, MultimodalEmbeddingInput):
        return item.parts
    return [item]


def _string_value(item: EmbeddingInput) -> str | list[str]:
    values: list[str] = []
    for part in _parts(item):
        if isinstance(part, TextEmbeddingInput):
            values.append(part.text)
        else:
            values.append(part.as_payload_value())
    return values[0] if len(values) == 1 else values


def _image_data_uri_parts(value: str) -> tuple[str, str] | None:
    match = _DATA_URI_RE.match(value)
    if match is None:
        return None
    return match.group(1), match.group(2)


def _to_voyage_content_part(part: TextEmbeddingInput | ImageEmbeddingInput) -> dict[str, str]:
    if isinstance(part, TextEmbeddingInput):
        return {"type": "text", "text": part.text}
    return {"type": "image_base64", "image_base64": part.as_payload_value()}


def _to_voyage_item(item: EmbeddingInput) -> dict[str, list[dict[str, str]]]:
    return {"content": [_to_voyage_content_part(part) for part in _parts(item)]}


def _to_dashscope_item(item: EmbeddingInput) -> dict[str, Any]:
    values: list[dict[str, str]] = []
    for part in _parts(item):
        if isinstance(part, TextEmbeddingInput):
            values.append({"text": part.text})
        else:
            values.append({"image": part.as_payload_value()})
    return values[0] if len(values) == 1 else {"contents": values}


def _to_jina_item(item: EmbeddingInput) -> str | dict[str, str] | list[dict[str, str]]:
    values: list[dict[str, str]] = []
    for part in _parts(item):
        if isinstance(part, TextEmbeddingInput):
            values.append({"text": part.text})
        else:
            values.append({"image": part.as_payload_value()})
    return values[0] if len(values) == 1 else values


def _to_gemini_part(part: TextEmbeddingInput | ImageEmbeddingInput) -> dict[str, Any]:
    if isinstance(part, TextEmbeddingInput):
        return {"text": part.text}

    value = part.as_payload_value()
    parsed = _image_data_uri_parts(value)
    if parsed is not None:
        mime_type, data = parsed
        return {"inline_data": {"mime_type": mime_type, "data": data}}
    return {"file_data": {"file_uri": value}}


def _to_gemini_content(item: EmbeddingInput) -> dict[str, list[dict[str, Any]]]:
    return {"parts": [_to_gemini_part(part) for part in _parts(item)]}


class OpenAICompatEmbedProvider(EmbedProvider):
    """POST /embeddings for OpenAI-compatible embedding servers."""

    endpoint = "/embeddings"
    supports_images = True
    supports_asymmetric = False
    default_dim = None
    known_dims = None

    def build_payload(
        self,
        model: str,
        inputs: list[EmbeddingInput],
        *,
        context: EmbedContext,
        asymmetric: bool = False,
        output_dimension: int | None = None,
    ) -> dict:
        payload: dict[str, Any] = {
            "model": model,
            "input": [_string_value(item) for item in inputs],
            "encoding_format": "float",
        }
        if output_dimension is not None:
            payload["dimensions"] = output_dimension
        return payload


class QwenOpenAICompatEmbedProvider(OpenAICompatEmbedProvider):
    """OpenAI-compatible Qwen3-VL embedding servers, including LM Studio."""

    supports_images = True
    supports_asymmetric = False
    default_dim = 2048


class VoyageEmbedProvider(EmbedProvider):
    """POST /multimodalembeddings for Voyage multimodal 3/3.5."""

    endpoint = "/multimodalembeddings"
    supports_images = True
    supports_asymmetric = True
    default_dim = 1024
    known_dims = frozenset({1024})

    def build_payload(
        self,
        model: str,
        inputs: list[EmbeddingInput],
        *,
        context: EmbedContext,
        asymmetric: bool = False,
        output_dimension: int | None = None,
    ) -> dict:
        payload: dict[str, Any] = {
            "model": model,
            "inputs": [_to_voyage_item(item) for item in inputs],
        }
        if asymmetric:
            payload["input_type"] = context
        if output_dimension is not None:
            payload["output_dimension"] = output_dimension
        return payload


class JinaEmbedProvider(EmbedProvider):
    """POST /v1/embeddings for Jina v4 multimodal embeddings."""

    endpoint = "/v1/embeddings"
    supports_images = True
    supports_asymmetric = True
    default_dim = 2048
    known_dims = None

    def build_payload(
        self,
        model: str,
        inputs: list[EmbeddingInput],
        *,
        context: EmbedContext,
        asymmetric: bool = False,
        output_dimension: int | None = None,
    ) -> dict:
        payload: dict[str, Any] = {
            "model": model,
            "input": [_to_jina_item(item) for item in inputs],
            "encoding_type": "float",
        }
        if asymmetric:
            payload["task"] = "retrieval.query" if context == "query" else "retrieval.passage"
        if output_dimension is not None:
            payload["dimensions"] = output_dimension
        return payload


class DashScopeQwenEmbedProvider(EmbedProvider):
    """DashScope Qwen3-VL multimodal embedding endpoint."""

    endpoint = "/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"
    supports_images = True
    supports_asymmetric = False
    default_dim = 2560
    known_dims = frozenset({2560, 2048, 1536, 1024, 768, 512, 256})

    def build_payload(
        self,
        model: str,
        inputs: list[EmbeddingInput],
        *,
        context: EmbedContext,
        asymmetric: bool = False,
        output_dimension: int | None = None,
    ) -> dict:
        payload: dict[str, Any] = {
            "model": model,
            "input": {"contents": [_to_dashscope_item(item) for item in inputs]},
        }
        if output_dimension is not None:
            payload["parameters"] = {"dimension": output_dimension}
        return payload

    def parse_response(self, data: dict) -> list[list[float]]:
        return [item["embedding"] for item in data["output"]["embeddings"]]


class GeminiEmbedProvider(EmbedProvider):
    """Gemini Embedding 2 multimodal endpoint.

    Gemini Embedding 2 does not accept the older ``task_type`` parameter, so
    DlightRAG treats it as symmetric for LightRAG context injection.
    """

    endpoint = "/models/{model}:embedContent"
    supports_images = True
    supports_asymmetric = False
    default_dim = 3072
    known_dims = frozenset({3072, 1536, 768})

    def build_payload(
        self,
        model: str,
        inputs: list[EmbeddingInput],
        *,
        context: EmbedContext,
        asymmetric: bool = False,
        output_dimension: int | None = None,
    ) -> dict:
        if len(inputs) == 1:
            payload: dict[str, Any] = {"content": _to_gemini_content(inputs[0])}
        else:
            payload = {
                "contents": [_to_gemini_content(item) for item in inputs],
            }
        if output_dimension is not None:
            payload["output_dimensionality"] = output_dimension
        return payload

    def request_headers(self, api_key: str) -> dict[str, str]:
        return {"x-goog-api-key": api_key} if api_key else {}

    def parse_response(self, data: dict) -> list[list[float]]:
        if "embedding" in data:
            return [data["embedding"]["values"]]
        if "embeddings" in data:
            return [
                item["values"] if "values" in item else item["embedding"]["values"]
                for item in data["embeddings"]
            ]
        return super().parse_response(data)


class OllamaEmbedProvider(EmbedProvider):
    """POST /api/embed for text-only local Ollama embeddings."""

    endpoint = "/api/embed"
    supports_images = False
    supports_asymmetric = False

    def build_payload(
        self,
        model: str,
        inputs: list[EmbeddingInput],
        *,
        context: EmbedContext,
        asymmetric: bool = False,
        output_dimension: int | None = None,
    ) -> dict:
        return {"model": model, "input": [_string_value(item) for item in inputs]}

    def request_headers(self, api_key: str) -> dict[str, str]:
        return {}

    def parse_response(self, data: dict) -> list[list[float]]:
        return data["embeddings"]


_EMBED_REGISTRY: dict[str, type[EmbedProvider]] = {
    "voyage": VoyageEmbedProvider,
    "dashscope_qwen": DashScopeQwenEmbedProvider,
    "qwen_openai_compatible": QwenOpenAICompatEmbedProvider,
    "gemini": GeminiEmbedProvider,
    "jina": JinaEmbedProvider,
    "openai_compatible": OpenAICompatEmbedProvider,
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
        parsed = urlparse(base_url)
        host = (parsed.hostname or "").lower()
        port = parsed.port
        if port == 11434:
            return OllamaEmbedProvider()
        if host.endswith(".generativelanguage.googleapis.com") or host == "generativelanguage.googleapis.com":
            return GeminiEmbedProvider()
        if "dashscope" in host or "aliyuncs" in host:
            return DashScopeQwenEmbedProvider()

    name = model.lower()
    if "voyage" in name:
        return VoyageEmbedProvider()
    if "jina" in name:
        return JinaEmbedProvider()
    if "gemini" in name:
        return GeminiEmbedProvider()
    if "qwen3-vl-embedding" in name:
        return DashScopeQwenEmbedProvider()

    return OpenAICompatEmbedProvider()
