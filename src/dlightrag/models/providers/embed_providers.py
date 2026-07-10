# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Concrete multimodal EmbedProvider implementations."""

import re
from typing import Any

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


def _to_dashscope_part(part: TextEmbeddingInput | ImageEmbeddingInput) -> dict[str, str]:
    if isinstance(part, TextEmbeddingInput):
        return {"text": part.text}
    return {"image": part.as_payload_value()}


def _to_jina_image(part: ImageEmbeddingInput) -> dict[str, str]:
    if part.url:
        return {"url": part.url}
    if part.path:
        raise ValueError("Jina image embeddings require a URL or base64 bytes, not a local path")

    value = part.as_payload_value()
    parsed = _image_data_uri_parts(value)
    return {"bytes": parsed[1] if parsed is not None else value}


def _to_jina_item(item: EmbeddingInput) -> str | dict[str, str] | list[dict[str, str]]:
    values: list[dict[str, str]] = []
    for part in _parts(item):
        if isinstance(part, TextEmbeddingInput):
            values.append({"text": part.text})
        else:
            values.append(_to_jina_image(part))
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


class OpenAICompatibleEmbedProvider(EmbedProvider):
    """POST /embeddings for OpenAI-compatible embedding servers."""

    endpoint = "/embeddings"
    image_input_capability = "opt_in"
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
        payload: dict[str, Any] = {
            "model": model,
            "input": [_string_value(item) for item in inputs],
            "encoding_format": "float",
        }
        if output_dimension is not None:
            payload["dimensions"] = output_dimension
        return payload


class VoyageEmbedProvider(EmbedProvider):
    """POST /multimodalembeddings for Voyage multimodal 3/3.5."""

    endpoint = "/multimodalembeddings"
    image_input_capability = "native"
    supports_asymmetric = True

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
    image_input_capability = "native"
    supports_asymmetric = True

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
    image_input_capability = "native"
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
        if any(isinstance(item, MultimodalEmbeddingInput) for item in inputs):
            if len(inputs) != 1 or not isinstance(inputs[0], MultimodalEmbeddingInput):
                raise ValueError("DashScope fused embedding expects exactly one multimodal input")
            payload = {
                "model": model,
                "input": {"contents": [_to_dashscope_part(part) for part in inputs[0].parts]},
            }
            parameters: dict[str, Any] = {"enable_fusion": True}
        else:
            payload = {
                "model": model,
                "input": {"contents": [_to_dashscope_item(item) for item in inputs]},
            }
            parameters = {}
        if output_dimension is not None:
            parameters["dimension"] = output_dimension
        if parameters:
            payload["parameters"] = parameters
        return payload

    def parse_response(self, data: dict) -> list[list[float]]:
        return [item["embedding"] for item in data["output"]["embeddings"]]


class GeminiEmbedProvider(EmbedProvider):
    """Gemini Embedding 2 multimodal endpoint.

    Gemini Embedding 2 does not accept the older ``task_type`` parameter, so
    DlightRAG treats it as symmetric for LightRAG context injection.
    """

    endpoint = "/models/{model}:embedContent"
    image_input_capability = "native"
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
    "gemini": GeminiEmbedProvider,
    "jina": JinaEmbedProvider,
    "openai_compatible": OpenAICompatibleEmbedProvider,
    "ollama": OllamaEmbedProvider,
}


def get_embed_provider(provider: str) -> EmbedProvider:
    """Instantiate an embedding transport serializer by explicit config name."""
    cls = _EMBED_REGISTRY.get(provider)
    if cls is None:
        available = ", ".join(sorted(_EMBED_REGISTRY))
        raise ValueError(f"Unknown embedding provider {provider!r}. Available: {available}")
    return cls()
