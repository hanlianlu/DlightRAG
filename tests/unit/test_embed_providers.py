# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Contract tests for embedding URL, auth, payload, and response adapters."""

from typing import Any, cast

import pytest

from dlightrag.ai.embedding_inputs import (
    ImageEmbeddingInput,
    MultimodalEmbeddingInput,
    TextEmbeddingInput,
)
from dlightrag.ai.providers import embed_providers
from dlightrag.ai.providers.embed_base import EmbedProvider
from dlightrag.ai.providers.embed_providers import (
    AzureCohereEmbedProvider,
    CohereEmbedProvider,
    GeminiEmbedProvider,
    JinaEmbedProvider,
    OpenAICompatibleEmbedProvider,
    OpenAIEmbedProvider,
    VoyageEmbedProvider,
)


class TestEmbedProviderABC:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            cast(Any, EmbedProvider)()


def test_openai_official_payload_sends_dimensions_only_for_v3_models() -> None:
    provider = OpenAIEmbedProvider()
    v3 = provider.build_payload(
        "text-embedding-3-large",
        [TextEmbeddingInput(text="hello")],
        context="query",
        output_dimension=1536,
    )
    ada = provider.build_payload(
        "text-embedding-ada-002",
        [TextEmbeddingInput(text="hello")],
        context="query",
        output_dimension=1536,
    )

    assert v3 == {
        "model": "text-embedding-3-large",
        "input": ["hello"],
        "encoding_format": "float",
        "dimensions": 1536,
    }
    assert "dimensions" not in ada


def test_openai_official_limits_match_the_published_request_contract() -> None:
    capabilities = OpenAIEmbedProvider().capabilities("text-embedding-3-large")

    assert capabilities.max_inputs == 2048
    assert capabilities.max_tokens_per_input == 8192
    assert capabilities.max_tokens_per_request == 300_000


def test_openai_official_rejects_wrong_fixed_dimension() -> None:
    with pytest.raises(ValueError, match="fixed dimension 1536"):
        OpenAIEmbedProvider().build_payload(
            "text-embedding-ada-002",
            [TextEmbeddingInput(text="hello")],
            context="document",
            output_dimension=1024,
        )


def test_openai_and_generic_reorder_strict_response_indices() -> None:
    response = {
        "data": [
            {"index": 1, "embedding": [0.3, 0.4]},
            {"index": 0, "embedding": [0.1, 0.2]},
        ]
    }

    assert OpenAIEmbedProvider().parse_response(response, expected_count=2) == [
        [0.1, 0.2],
        [0.3, 0.4],
    ]
    assert OpenAICompatibleEmbedProvider().parse_response(response, expected_count=2) == [
        [0.1, 0.2],
        [0.3, 0.4],
    ]


@pytest.mark.parametrize(
    "data",
    [
        {"data": [{"embedding": [0.1]}]},
        {"data": [{"index": 0, "embedding": [0.1]}, {"index": 0, "embedding": [0.2]}]},
        {"data": [{"index": 2, "embedding": [0.1]}]},
        {"data": [{"index": 1, "embedding": [0.1]}]},
    ],
)
def test_openai_index_contract_rejects_missing_duplicate_or_non_covering_indices(
    data: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="index|indices"):
        OpenAIEmbedProvider().parse_response(data, expected_count=2)


def test_openai_compatible_is_minimal_text_only_without_dimensions() -> None:
    provider = OpenAICompatibleEmbedProvider()
    payload = provider.build_payload(
        "private-embedder",
        [TextEmbeddingInput(text="hello")],
        context="document",
        output_dimension=2048,
    )

    assert payload == {
        "model": "private-embedder",
        "input": ["hello"],
        "encoding_format": "float",
    }
    with pytest.raises(ValueError, match="does not support image"):
        provider.build_payload(
            "private-embedder",
            [ImageEmbeddingInput(data_uri="data:image/png;base64,abc")],
            context="document",
            output_dimension=2048,
        )


def test_openai_urls_and_azure_auth_follow_v1_contract() -> None:
    provider = OpenAIEmbedProvider()

    assert (
        provider.request_url("https://api.openai.com/v1", "text-embedding-3-large")
        == "https://api.openai.com/v1/embeddings"
    )
    assert (
        provider.request_url(
            "https://example.openai.azure.com/openai/v1/embeddings",
            "deployment",
        )
        == "https://example.openai.azure.com/openai/v1/embeddings"
    )
    assert provider.request_headers(
        "secret",
        base_url="https://example.openai.azure.com/openai/v1",
    ) == {"api-key": "secret"}


def test_voyage_fused_payload_always_maps_retrieval_context() -> None:
    payload = VoyageEmbedProvider().build_payload(
        "voyage-multimodal-3.5",
        [
            MultimodalEmbeddingInput(
                parts=[
                    TextEmbeddingInput(text="a bar chart of GDP"),
                    ImageEmbeddingInput(data_uri="data:image/png;base64,abc"),
                ]
            )
        ],
        context="document",
        output_dimension=1024,
    )

    assert payload["input_type"] == "document"
    assert payload["output_dimension"] == 1024
    assert payload["inputs"][0]["content"] == [
        {"type": "text", "text": "a bar chart of GDP"},
        {"type": "image_base64", "image_base64": "data:image/png;base64,abc"},
    ]


def test_gemini_embedding_2_prefixes_text_and_aggregates_one_fused_content() -> None:
    provider = GeminiEmbedProvider()
    payload = provider.build_payload(
        "gemini-embedding-2",
        [
            MultimodalEmbeddingInput(
                parts=[
                    TextEmbeddingInput(text="chart"),
                    ImageEmbeddingInput(data_uri="data:image/png;base64,abc"),
                ]
            )
        ],
        context="document",
        output_dimension=1536,
    )

    assert provider.capabilities("gemini-embedding-2").asymmetric is True
    assert payload["model"] == "models/gemini-embedding-2"
    assert payload["outputDimensionality"] == 1536
    assert payload["content"]["parts"] == [
        {"text": "title: none | text: chart"},
        {"inlineData": {"mimeType": "image/png", "data": "abc"}},
    ]


def test_gemini_parses_vector_and_native_usage_metadata() -> None:
    provider = GeminiEmbedProvider()
    data = {
        "embedding": {"values": [0.1, 0.2]},
        "usageMetadata": {"promptTokenCount": 7, "cached": False},
    }

    assert provider.parse_response(data, expected_count=1) == [[0.1, 0.2]]
    assert provider.response_usage(data) == {"promptTokenCount": 7}


def test_gemini_embedding_2_rejects_multi_input_sync_payload() -> None:
    with pytest.raises(ValueError, match="at most 1 inputs"):
        GeminiEmbedProvider().build_payload(
            "gemini-embedding-2",
            [TextEmbeddingInput(text="one"), TextEmbeddingInput(text="two")],
            context="query",
            output_dimension=3072,
        )


def test_gemini_query_prefix_matches_official_retrieval_instruction() -> None:
    payload = GeminiEmbedProvider().build_payload(
        "gemini-embedding-2",
        [TextEmbeddingInput(text="revenue chart")],
        context="query",
        output_dimension=3072,
    )

    assert payload["content"]["parts"] == [{"text": "task: search result | query: revenue chart"}]


def test_jina_v4_is_native_fused_but_v5_is_conservative_text_only() -> None:
    provider = JinaEmbedProvider()
    assert provider.capabilities("jina-embeddings-v4").native_multimodal is True
    assert provider.capabilities("jina-embeddings-v5-omni-small").native_multimodal is False

    payload = provider.build_payload(
        "jina-embeddings-v4",
        [
            MultimodalEmbeddingInput(
                parts=[
                    TextEmbeddingInput(text="chart"),
                    ImageEmbeddingInput(data_uri="data:image/jpeg;base64,abc"),
                ]
            )
        ],
        context="query",
        output_dimension=2048,
    )
    assert payload["task"] == "retrieval.query"
    assert payload["input"] == [[{"text": "chart"}, {"bytes": "abc"}]]


def test_jina_url_does_not_duplicate_v1() -> None:
    provider = JinaEmbedProvider()
    assert (
        provider.request_url("https://api.jina.ai/v1", "jina-embeddings-v4")
        == "https://api.jina.ai/v1/embeddings"
    )
    assert (
        provider.request_url("https://api.jina.ai", "jina-embeddings-v4")
        == "https://api.jina.ai/v1/embeddings"
    )


def test_cohere_v4_fused_payload_uses_float_none_truncation_and_dimension() -> None:
    payload = CohereEmbedProvider().build_payload(
        "embed-v4.0",
        [
            MultimodalEmbeddingInput(
                parts=[
                    TextEmbeddingInput(text="chart"),
                    ImageEmbeddingInput(data_uri="data:image/png;base64,abc"),
                ]
            )
        ],
        context="document",
        output_dimension=1536,
    )

    assert payload == {
        "model": "embed-v4.0",
        "embedding_types": ["float"],
        "truncate": "NONE",
        "input_type": "image",
        "inputs": [
            {
                "content": [
                    {"type": "text", "text": "chart"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc"},
                    },
                ]
            }
        ],
        "output_dimension": 1536,
    }


def test_cohere_v4_text_query_and_float_response_contract() -> None:
    provider = CohereEmbedProvider()
    payload = provider.build_payload(
        "embed-v4.0",
        [TextEmbeddingInput(text="hello")],
        context="query",
        output_dimension=1024,
    )

    assert payload["texts"] == ["hello"]
    assert payload["input_type"] == "search_query"
    assert provider.parse_response({"embeddings": {"float": [[0.1, 0.2]]}}, expected_count=1) == [
        [0.1, 0.2]
    ]


def test_cohere_v4_limits_match_the_published_request_contract() -> None:
    capabilities = CohereEmbedProvider().capabilities("embed-v4.0")

    assert capabilities.max_inputs == 96
    assert capabilities.max_tokens_per_input == 128_000
    assert capabilities.max_image_bytes_per_request == 20_000_000


def test_public_cohere_url_owns_v2_route() -> None:
    provider = CohereEmbedProvider()
    assert provider.request_url("https://api.cohere.com", "embed-v4.0") == (
        "https://api.cohere.com/v2/embed"
    )
    assert provider.request_url("https://api.cohere.com/v2", "embed-v4.0") == (
        "https://api.cohere.com/v2/embed"
    )


def test_azure_cohere_accepts_full_or_official_root_and_rejects_unknown_roots() -> None:
    provider = AzureCohereEmbedProvider()
    root = "https://deployment.eastus.inference.ai.azure.com"

    assert provider.request_url(root, "Cohere-embed-v4") == f"{root}/v1/embed"
    assert provider.request_url(f"{root}/v1", "Cohere-embed-v4") == f"{root}/v1/embed"
    assert provider.request_url(f"{root}/custom/embed", "Cohere-embed-v4") == (
        f"{root}/custom/embed"
    )
    with pytest.raises(ValueError, match="official deployment"):
        provider.request_url("https://example.com", "Cohere-embed-v4")


@pytest.mark.parametrize(
    ("provider_name", "provider_type"),
    [
        ("azure_cohere", AzureCohereEmbedProvider),
        ("cohere", CohereEmbedProvider),
        ("gemini", GeminiEmbedProvider),
        ("jina", JinaEmbedProvider),
        ("openai", OpenAIEmbedProvider),
        ("openai_compatible", OpenAICompatibleEmbedProvider),
        ("voyage", VoyageEmbedProvider),
    ],
)
def test_get_embed_provider_uses_explicit_registry(
    provider_name: str,
    provider_type: type[EmbedProvider],
) -> None:
    assert isinstance(embed_providers.get_embed_provider(provider_name), provider_type)


def test_get_embed_provider_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown embedding provider 'ollama'"):
        embed_providers.get_embed_provider("ollama")
