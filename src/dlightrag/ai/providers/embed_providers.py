# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Concrete embedding wire-protocol adapters."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import tiktoken

from dlightrag.ai.embedding_inputs import (
    EmbeddingInput,
    ImageEmbeddingInput,
    TextEmbeddingInput,
)
from dlightrag.ai.providers.embed_base import (
    EmbeddingContext,
    EmbedModelCapabilities,
    EmbedProvider,
    OutputDimensionPolicy,
    input_parts,
    input_text,
    numeric_usage,
)

_DATA_URI_RE = re.compile(r"^data:([^;]+);base64,(.+)$", re.DOTALL)


def _endpoint(base_url: str, suffix: str, *, complete_suffix: str | None = None) -> str:
    """Append one endpoint path without duplicating a configured version path."""
    parsed = urlsplit(base_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid embedding base_url: {base_url!r}")
    if parsed.query or parsed.fragment:
        raise ValueError("Embedding base_url must not contain query parameters or fragments")
    path = parsed.path.rstrip("/")
    terminal = complete_suffix or suffix
    if path.endswith(terminal):
        resolved = path
    else:
        resolved = f"{path}{suffix}"
    return urlunsplit((parsed.scheme, parsed.netloc, resolved, "", ""))


def _image_data_uri_parts(value: str) -> tuple[str, str] | None:
    match = _DATA_URI_RE.match(value)
    if match is None:
        return None
    return match.group(1), match.group(2)


def _text_values(inputs: list[EmbeddingInput]) -> list[str]:
    values: list[str] = []
    for item in inputs:
        if not isinstance(item, TextEmbeddingInput):
            raise ValueError("This embedding protocol accepts text inputs only")
        values.append(item.text)
    return values


def _image_values(inputs: list[EmbeddingInput]) -> list[str]:
    values: list[str] = []
    for item in inputs:
        if not isinstance(item, ImageEmbeddingInput):
            raise ValueError("This embedding request requires image-only inputs")
        values.append(item.as_payload_value())
    return values


def _strict_indexed_vectors(
    data: Mapping[str, Any],
    *,
    expected_count: int,
) -> list[list[float]]:
    items = data.get("data")
    if not isinstance(items, list):
        raise ValueError("Embedding response data must be a list")
    ordered: list[list[float] | None] = [None] * expected_count
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("Embedding response item must be an object")
        index = item.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("Embedding response item is missing a valid index")
        if index < 0 or index >= expected_count:
            raise ValueError(f"Embedding response index {index} is out of range")
        if ordered[index] is not None:
            raise ValueError(f"Embedding response index {index} is duplicated")
        vector = item.get("embedding")
        if not isinstance(vector, list):
            raise ValueError(f"Embedding response item {index} has no vector")
        ordered[index] = vector
    missing = [index for index, vector in enumerate(ordered) if vector is None]
    if missing:
        raise ValueError(f"Embedding response indices do not cover the request: missing {missing}")
    return [vector for vector in ordered if vector is not None]


def _to_voyage_content_part(part: TextEmbeddingInput | ImageEmbeddingInput) -> dict[str, str]:
    if isinstance(part, TextEmbeddingInput):
        return {"type": "text", "text": part.text}
    return {"type": "image_base64", "image_base64": part.as_payload_value()}


def _to_voyage_item(item: EmbeddingInput) -> dict[str, list[dict[str, str]]]:
    return {"content": [_to_voyage_content_part(part) for part in input_parts(item)]}


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
    for part in input_parts(item):
        if isinstance(part, TextEmbeddingInput):
            values.append({"text": part.text})
        else:
            values.append(_to_jina_image(part))
    return values[0] if len(values) == 1 else values


def _gemini_text(value: str, context: EmbeddingContext) -> str:
    if context == "query":
        return f"task: search result | query: {value}"
    return f"title: none | text: {value}"


def _to_gemini_part(
    part: TextEmbeddingInput | ImageEmbeddingInput,
    *,
    context: EmbeddingContext,
) -> dict[str, Any]:
    if isinstance(part, TextEmbeddingInput):
        return {"text": _gemini_text(part.text, context)}
    value = part.as_payload_value()
    parsed = _image_data_uri_parts(value)
    if parsed is not None:
        mime_type, data = parsed
        return {"inlineData": {"mimeType": mime_type, "data": data}}
    return {"fileData": {"fileUri": value}}


def _to_gemini_content(
    item: EmbeddingInput,
    *,
    context: EmbeddingContext,
) -> dict[str, list[dict[str, Any]]]:
    return {"parts": [_to_gemini_part(part, context=context) for part in input_parts(item)]}


def _to_cohere_content_part(
    part: TextEmbeddingInput | ImageEmbeddingInput,
) -> dict[str, Any]:
    if isinstance(part, TextEmbeddingInput):
        return {"type": "text", "text": part.text}
    return {
        "type": "image_url",
        "image_url": {"url": part.as_payload_value()},
    }


def _to_cohere_input(item: EmbeddingInput) -> dict[str, list[dict[str, Any]]]:
    return {"content": [_to_cohere_content_part(part) for part in input_parts(item)]}


_OPENAI_DIMENSIONS: dict[str, OutputDimensionPolicy] = {
    "text-embedding-3-large": OutputDimensionPolicy(
        send_upstream=True,
        minimum=1,
        maximum=3072,
    ),
    "text-embedding-3-small": OutputDimensionPolicy(
        send_upstream=True,
        minimum=1,
        maximum=1536,
    ),
    "text-embedding-ada-002": OutputDimensionPolicy(fixed=1536),
}


class OpenAICompatibleEmbedProvider(EmbedProvider):
    """Conservative OpenAI-shaped text embedding protocol."""

    default_base_url = ""

    def capabilities(self, model: str) -> EmbedModelCapabilities:
        del model
        return EmbedModelCapabilities(max_inputs=64)

    def request_url(self, base_url: str, model: str) -> str:
        del model
        return _endpoint(base_url, "/embeddings")

    def _build_payload(
        self,
        model: str,
        inputs: list[EmbeddingInput],
        *,
        context: EmbeddingContext,
        output_dimension: int | None,
    ) -> dict[str, Any]:
        del context, output_dimension
        return {
            "model": model,
            "input": _text_values(inputs),
            "encoding_format": "float",
        }

    def parse_response(self, data: Mapping[str, Any], *, expected_count: int) -> list[list[float]]:
        return _strict_indexed_vectors(data, expected_count=expected_count)


class OpenAIEmbedProvider(OpenAICompatibleEmbedProvider):
    """Official OpenAI/Azure OpenAI v1 text embedding contract."""

    default_base_url = "https://api.openai.com/v1"

    def capabilities(self, model: str) -> EmbedModelCapabilities:
        return EmbedModelCapabilities(
            max_inputs=2048,
            max_tokens_per_input=8192,
            max_tokens_per_request=300_000,
            token_safety_margin=1.0 if model in _OPENAI_DIMENSIONS else 1.2,
            output_dimension=_OPENAI_DIMENSIONS.get(model, OutputDimensionPolicy()),
        )

    def request_headers(self, api_key: str, *, base_url: str) -> dict[str, str]:
        if not api_key:
            return {}
        hostname = (urlsplit(base_url).hostname or "").lower()
        if hostname.endswith(".openai.azure.com"):
            return {"api-key": api_key}
        return {"Authorization": f"Bearer {api_key}"}

    def _build_payload(
        self,
        model: str,
        inputs: list[EmbeddingInput],
        *,
        context: EmbeddingContext,
        output_dimension: int | None,
    ) -> dict[str, Any]:
        payload = super()._build_payload(
            model,
            inputs,
            context=context,
            output_dimension=output_dimension,
        )
        if output_dimension is not None:
            payload["dimensions"] = output_dimension
        return payload

    def estimate_input_tokens(self, model: str, item: EmbeddingInput) -> int:
        if model not in _OPENAI_DIMENSIONS:
            return super().estimate_input_tokens(model, item)
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(input_text(item)))


_VOYAGE_MULTIMODAL = EmbedModelCapabilities(
    image_input="native",
    fused_inputs=True,
    asymmetric=True,
    max_inputs=1000,
    max_tokens_per_input=32_000,
    max_tokens_per_request=320_000,
    output_dimension=OutputDimensionPolicy(
        send_upstream=True,
        allowed=(256, 512, 1024, 2048),
    ),
)


class VoyageEmbedProvider(EmbedProvider):
    """Voyage multimodal embedding protocol."""

    default_base_url = "https://api.voyageai.com/v1"

    def capabilities(self, model: str) -> EmbedModelCapabilities:
        if model == "voyage-multimodal-3.5":
            return _VOYAGE_MULTIMODAL
        return EmbedModelCapabilities(asymmetric=True)

    def request_url(self, base_url: str, model: str) -> str:
        del model
        return _endpoint(base_url, "/multimodalembeddings")

    def _build_payload(
        self,
        model: str,
        inputs: list[EmbeddingInput],
        *,
        context: EmbeddingContext,
        output_dimension: int | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "inputs": [_to_voyage_item(item) for item in inputs],
        }
        if self.capabilities(model).asymmetric:
            payload["input_type"] = context
        if output_dimension is not None:
            payload["output_dimension"] = output_dimension
        return payload


_JINA_V4 = EmbedModelCapabilities(
    image_input="native",
    fused_inputs=True,
    asymmetric=True,
    max_inputs=64,
    max_tokens_per_input=32_768,
    output_dimension=OutputDimensionPolicy(
        send_upstream=True,
        minimum=128,
        maximum=2048,
    ),
)


class JinaEmbedProvider(EmbedProvider):
    """Jina v4 native fused multimodal embedding protocol."""

    default_base_url = "https://api.jina.ai/v1"

    def capabilities(self, model: str) -> EmbedModelCapabilities:
        if model == "jina-embeddings-v4":
            return _JINA_V4
        return EmbedModelCapabilities(asymmetric=True)

    def request_url(self, base_url: str, model: str) -> str:
        del model
        parsed = urlsplit(base_url)
        suffix = "/embeddings" if parsed.path.rstrip("/").endswith("/v1") else "/v1/embeddings"
        return _endpoint(base_url, suffix, complete_suffix="/v1/embeddings")

    def _build_payload(
        self,
        model: str,
        inputs: list[EmbeddingInput],
        *,
        context: EmbeddingContext,
        output_dimension: int | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "input": [_to_jina_item(item) for item in inputs],
            "encoding_type": "float",
        }
        if self.capabilities(model).asymmetric:
            payload["task"] = "retrieval.query" if context == "query" else "retrieval.passage"
        if output_dimension is not None:
            payload["dimensions"] = output_dimension
        return payload


_GEMINI_EMBEDDING_2 = EmbedModelCapabilities(
    image_input="native",
    fused_inputs=True,
    asymmetric=True,
    max_inputs=1,
    max_tokens_per_input=8192,
    output_dimension=OutputDimensionPolicy(
        send_upstream=True,
        minimum=128,
        maximum=3072,
    ),
)


class GeminiEmbedProvider(EmbedProvider):
    """Gemini Embedding 2 single-content multimodal protocol."""

    default_base_url = "https://generativelanguage.googleapis.com/v1beta"

    def capabilities(self, model: str) -> EmbedModelCapabilities:
        if model == "gemini-embedding-2":
            return _GEMINI_EMBEDDING_2
        return EmbedModelCapabilities(asymmetric=True, max_inputs=1)

    def request_url(self, base_url: str, model: str) -> str:
        return _endpoint(base_url, f"/models/{model}:embedContent")

    def request_headers(self, api_key: str, *, base_url: str) -> dict[str, str]:
        del base_url
        return {"x-goog-api-key": api_key} if api_key else {}

    def _build_payload(
        self,
        model: str,
        inputs: list[EmbeddingInput],
        *,
        context: EmbeddingContext,
        output_dimension: int | None,
    ) -> dict[str, Any]:
        if len(inputs) != 1:
            raise ValueError("Gemini Embedding 2 requires exactly one synchronous input")
        payload: dict[str, Any] = {
            "model": f"models/{model}",
            "content": _to_gemini_content(inputs[0], context=context),
        }
        if output_dimension is not None:
            payload["outputDimensionality"] = output_dimension
        return payload

    def parse_response(self, data: Mapping[str, Any], *, expected_count: int) -> list[list[float]]:
        if expected_count != 1:
            raise ValueError("Gemini synchronous embedding response covers exactly one input")
        embedding = data.get("embedding")
        if not isinstance(embedding, Mapping) or not isinstance(embedding.get("values"), list):
            raise ValueError("Gemini embedding response has no vector")
        return [embedding["values"]]

    def response_usage(self, data: Mapping[str, Any]) -> dict[str, int | float]:
        return numeric_usage(data.get("usageMetadata"))


_COHERE_V4 = EmbedModelCapabilities(
    image_input="native",
    fused_inputs=True,
    asymmetric=True,
    max_inputs=96,
    max_tokens_per_input=128_000,
    max_image_bytes_per_request=20_000_000,
    output_dimension=OutputDimensionPolicy(
        send_upstream=True,
        allowed=(256, 512, 1024, 1536),
    ),
)


class CohereEmbedProvider(EmbedProvider):
    """Cohere Embed v4 native v2 protocol."""

    default_base_url = "https://api.cohere.com"

    def capabilities(self, model: str) -> EmbedModelCapabilities:
        if model == "embed-v4.0":
            return _COHERE_V4
        return EmbedModelCapabilities(asymmetric=True, max_inputs=96)

    def request_url(self, base_url: str, model: str) -> str:
        del model
        parsed = urlsplit(base_url)
        suffix = "/embed" if parsed.path.rstrip("/").endswith("/v2") else "/v2/embed"
        return _endpoint(base_url, suffix, complete_suffix="/v2/embed")

    def _build_payload(
        self,
        model: str,
        inputs: list[EmbeddingInput],
        *,
        context: EmbeddingContext,
        output_dimension: int | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "embedding_types": ["float"],
            "truncate": "NONE",
        }
        if self.capabilities(model).asymmetric:
            has_image = any(
                isinstance(part, ImageEmbeddingInput)
                for item in inputs
                for part in input_parts(item)
            )
            payload["input_type"] = (
                "image"
                if has_image
                else ("search_query" if context == "query" else "search_document")
            )
        if all(isinstance(item, TextEmbeddingInput) for item in inputs):
            payload["texts"] = _text_values(inputs)
        elif all(isinstance(item, ImageEmbeddingInput) for item in inputs):
            payload["images"] = _image_values(inputs)
        else:
            payload["inputs"] = [_to_cohere_input(item) for item in inputs]
        if output_dimension is not None:
            payload["output_dimension"] = output_dimension
        return payload

    def parse_response(self, data: Mapping[str, Any], *, expected_count: int) -> list[list[float]]:
        embeddings = data.get("embeddings")
        if isinstance(embeddings, Mapping):
            embeddings = embeddings.get("float")
        if not isinstance(embeddings, list) or len(embeddings) != expected_count:
            actual = len(embeddings) if isinstance(embeddings, list) else "non-list"
            raise ValueError(f"Expected {expected_count} Cohere vectors, got {actual}")
        if not all(isinstance(vector, list) for vector in embeddings):
            raise ValueError("Cohere float embeddings must be lists")
        return embeddings

    def response_usage(self, data: Mapping[str, Any]) -> dict[str, int | float]:
        usage = super().response_usage(data)
        meta = data.get("meta")
        billed = meta.get("billed_units") if isinstance(meta, Mapping) else None
        usage.update(numeric_usage(billed, prefix="billed_"))
        return usage


_AZURE_COHERE_V4_NAMES = frozenset({"cohere-embed-v4", "cohere-embed-4"})
_AZURE_COHERE_HOST_SUFFIXES = (
    ".inference.ai.azure.com",
    ".models.ai.azure.com",
    ".services.ai.azure.com",
)


class AzureCohereEmbedProvider(CohereEmbedProvider):
    """Azure Foundry Cohere Embed v4 deployment protocol."""

    default_base_url = ""

    def capabilities(self, model: str) -> EmbedModelCapabilities:
        if model.casefold() in _AZURE_COHERE_V4_NAMES:
            return _COHERE_V4
        return EmbedModelCapabilities(asymmetric=True, max_inputs=96)

    def request_url(self, base_url: str, model: str) -> str:
        del model
        parsed = urlsplit(base_url.strip())
        hostname = (parsed.hostname or "").lower()
        if parsed.scheme not in {"http", "https"} or not any(
            hostname.endswith(suffix) for suffix in _AZURE_COHERE_HOST_SUFFIXES
        ):
            raise ValueError("Azure Cohere base_url must be an official deployment scoring URI")
        path = parsed.path.rstrip("/")
        if path.endswith("/embed"):
            suffix = ""
        elif path.endswith("/v1"):
            suffix = "/embed"
        else:
            suffix = "/v1/embed"
        return _endpoint(base_url, suffix, complete_suffix="/embed")


_EMBED_REGISTRY: dict[str, type[EmbedProvider]] = {
    "azure_cohere": AzureCohereEmbedProvider,
    "cohere": CohereEmbedProvider,
    "gemini": GeminiEmbedProvider,
    "jina": JinaEmbedProvider,
    "openai": OpenAIEmbedProvider,
    "openai_compatible": OpenAICompatibleEmbedProvider,
    "voyage": VoyageEmbedProvider,
}


def get_embed_provider(provider: str) -> EmbedProvider:
    """Instantiate an embedding adapter by explicit wire-protocol name."""
    cls = _EMBED_REGISTRY.get(provider)
    if cls is None:
        available = ", ".join(sorted(_EMBED_REGISTRY))
        raise ValueError(f"Unknown embedding provider {provider!r}. Available: {available}")
    return cls()
