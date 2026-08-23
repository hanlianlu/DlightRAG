# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for context-aware multimodal embedding."""

import asyncio
import io
import threading
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from dlightrag.ai.contracts import InputModality, ResolvedInputModality
from dlightrag.ai.embedding import (
    MultimodalEmbedder as _MultimodalEmbedder,
)
from dlightrag.ai.embedding import (
    resolve_embedding_input_modality,
)
from dlightrag.ai.fingerprints import ModelFingerprint
from dlightrag.ai.media import decode_image_base64
from dlightrag.ai.providers.embed_base import EmbedProvider
from dlightrag.ai.providers.embed_providers import (
    GeminiEmbedProvider,
    OpenAICompatibleEmbedProvider,
    VoyageEmbedProvider,
)
from dlightrag.ai.scheduler import ModelScheduler

_TEST_FINGERPRINT = ModelFingerprint(
    provider="test",
    model="test-model",
    endpoint_fingerprint=None,
)


def MultimodalEmbedder(**kwargs):
    scheduler = kwargs.pop("scheduler", ModelScheduler(max_concurrency=1))
    return _MultimodalEmbedder(
        fingerprint=_TEST_FINGERPRINT,
        scheduler=scheduler,
        **kwargs,
    )


async def test_embedding_fingerprint_is_canonical_text_for_storage_ports() -> None:
    embedder = MultimodalEmbedder(
        model="test-model",
        base_url="https://example.test/v1",
        api_key="key",
        dim=3,
        provider=OpenAICompatibleEmbedProvider(),
    )
    try:
        assert embedder.embedding_fingerprint == "test:test-model"
    finally:
        await embedder.aclose()


async def test_embedding_fingerprint_includes_endpoint_identity() -> None:
    embedder = _MultimodalEmbedder(
        model="test-model",
        base_url="https://example.test/v1",
        api_key="key",
        dim=3,
        provider=OpenAICompatibleEmbedProvider(),
        fingerprint=ModelFingerprint(
            provider="test",
            model="test-model",
            endpoint_fingerprint="endpoint-hash",
        ),
        scheduler=ModelScheduler(max_concurrency=1),
    )
    try:
        assert embedder.embedding_fingerprint == "test:test-model@endpoint-hash"
    finally:
        await embedder.aclose()


async def test_embedding_error_text_is_redacted_when_sensitive_capture_is_disabled() -> None:
    class Observation:
        updates: list[dict[str, Any]] = []

        def update(self, **kwargs: Any) -> None:
            self.updates.append(kwargs)

    class Telemetry:
        capture_sensitive_data = False
        observation = Observation()

        @asynccontextmanager
        async def observe(self, name: str, **_kwargs: Any):
            del name
            yield self.observation

    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=3,
        provider=VoyageEmbedProvider(),
        telemetry=Telemetry(),
    )
    embedder._client.post = AsyncMock(  # pyright: ignore[reportPrivateUsage]
        side_effect=RuntimeError("upstream echoed secret embedding input")
    )

    try:
        with pytest.raises(RuntimeError, match="secret embedding input"):
            await embedder.embed_texts(["secret"])
    finally:
        await embedder.aclose()

    assert Telemetry.observation.updates == [{"level": "ERROR", "status_message": "RuntimeError"}]


@pytest.mark.parametrize(
    ("provider", "configured", "resolved"),
    [
        (VoyageEmbedProvider(), "auto", "multimodal"),
        (VoyageEmbedProvider(), "text", "text"),
        (VoyageEmbedProvider(), "multimodal", "multimodal"),
        (OpenAICompatibleEmbedProvider(), "auto", "text"),
        (OpenAICompatibleEmbedProvider(), "text", "text"),
        (OpenAICompatibleEmbedProvider(), "multimodal", "multimodal"),
    ],
)
def test_resolve_embedding_input_modality(
    provider: EmbedProvider,
    configured: InputModality,
    resolved: ResolvedInputModality,
) -> None:
    assert resolve_embedding_input_modality(provider, configured) == resolved


async def test_native_multimodal_provider_can_be_forced_to_text() -> None:
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=1024,
        provider=VoyageEmbedProvider(),
        input_modality="text",
    )

    try:
        assert embedder.supports_images is False
        with pytest.raises(ValueError, match="does not support image"):
            await embedder.embed_index_fused([("x", Image.new("RGB", (1, 1), "white"))])
    finally:
        await embedder.aclose()


async def test_openai_compatible_auto_is_text_only() -> None:
    embedder = MultimodalEmbedder(
        model="qwen3-vl-embedding-2b",
        base_url="http://127.0.0.1:1234/v1",
        api_key="",
        dim=2048,
        provider=OpenAICompatibleEmbedProvider(),
        input_modality="auto",
    )

    try:
        assert embedder.supports_images is False
    finally:
        await embedder.aclose()


async def test_openai_compatible_explicit_multimodal_enables_images() -> None:
    embedder = MultimodalEmbedder(
        model="qwen3-vl-embedding-2b",
        base_url="http://127.0.0.1:1234/v1",
        api_key="",
        dim=2048,
        provider=OpenAICompatibleEmbedProvider(),
        input_modality="multimodal",
    )

    try:
        assert embedder.supports_images is True
    finally:
        await embedder.aclose()


def test_multimodal_embedder_builds_fused_voyage_payload() -> None:
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=1024,
        provider=VoyageEmbedProvider(),
    )

    assert embedder.supports_images is True
    payload = embedder._build_fused_payload(
        [("a bar chart", Image.new("RGB", (2, 2), "white"))], context="document"
    )
    content = payload["inputs"][0]["content"]
    assert content[0] == {"type": "text", "text": "a bar chart"}
    assert content[1]["type"] == "image_base64"
    assert content[1]["image_base64"].startswith("data:image/")


def test_fused_payload_degrades_to_image_only_when_description_blank() -> None:
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=1024,
        provider=VoyageEmbedProvider(),
    )

    payload = embedder._build_fused_payload(
        [("   ", Image.new("RGB", (2, 2), "white"))], context="document"
    )
    content = payload["inputs"][0]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "image_base64"


def test_image_capable_provider_builds_fused_payload() -> None:
    # Gemini Embedding 2 is a unified multimodal model: multiple parts in one
    # content aggregate into a single fused vector, so the fused index path works.
    embedder = MultimodalEmbedder(
        model="gemini-embedding-2",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key="key",
        dim=1536,
        provider=GeminiEmbedProvider(),
    )

    assert embedder.supports_images is True
    payload = embedder._build_fused_payload(
        [("x", Image.new("RGB", (2, 2), "white"))], context="document"
    )
    parts = payload["content"]["parts"]
    assert parts[0] == {"text": "x"}
    assert "inline_data" in parts[1]


def test_image_embedder_uses_asymmetric_by_default_for_capable_provider() -> None:
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=1024,
        provider=VoyageEmbedProvider(),
        asymmetric="auto",
    )

    image = Image.new("RGB", (1, 1), "white")
    index_payload = embedder._build_fused_payload([("chart", image)], context="document")
    query_payload = embedder._build_fused_payload([("chart", image)], context="query")

    assert embedder.supports_asymmetric is True
    assert index_payload["input_type"] == "document"
    assert query_payload["input_type"] == "query"


def test_image_embedder_bounds_oversized_images_before_send() -> None:
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=1024,
        provider=VoyageEmbedProvider(),
    )

    payload = embedder._build_fused_payload(
        [("", Image.new("RGB", (6000, 5000), "white"))], context="document"
    )

    data_uri = payload["inputs"][0]["content"][0]["image_base64"]
    raw, _ = decode_image_base64(data_uri)
    with Image.open(io.BytesIO(raw)) as decoded:
        assert max(decoded.size) <= 4096
        assert decoded.width * decoded.height <= 15_000_000


def test_image_embedder_can_disable_asymmetric_for_capable_provider() -> None:
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=1024,
        provider=VoyageEmbedProvider(),
        asymmetric="disable",
    )

    payload = embedder._build_fused_payload(
        [("chart", Image.new("RGB", (1, 1), "white"))], context="query"
    )

    assert embedder.supports_asymmetric is False
    assert "input_type" not in payload


def test_dimension_mismatch_raises() -> None:
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=1024,
        provider=VoyageEmbedProvider(),
    )

    with pytest.raises(ValueError, match="Expected embedding dim 1024"):
        embedder._validate_vectors([[0.1, 0.2, 0.3]])


async def test_embed_texts_posts_document_context() -> None:
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=3,
        provider=VoyageEmbedProvider(),
    )
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
    embedder._client.post = AsyncMock(return_value=response)

    result = await embedder.embed_texts(["hello"], context="document")

    assert result == [[0.1, 0.2, 0.3]]
    payload = embedder._client.post.call_args.kwargs["json"]
    assert payload["input_type"] == "document"


async def test_embedding_requests_share_scheduler_limit() -> None:
    scheduler = ModelScheduler(max_concurrency=1)
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    calls = 0
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=3,
        provider=VoyageEmbedProvider(),
        scheduler=scheduler,
    )

    async def post(payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        del payload
        calls += 1
        if calls == 1:
            first_started.set()
            await release_first.wait()
        return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

    embedder._post = post  # pyright: ignore[reportPrivateUsage]
    first = asyncio.create_task(embedder.embed_texts(["first"]))
    await first_started.wait()
    second = asyncio.create_task(embedder.embed_texts(["second"]))
    await asyncio.sleep(0)
    assert calls == 1

    release_first.set()
    assert await asyncio.gather(first, second) == [
        [[0.1, 0.2, 0.3]],
        [[0.1, 0.2, 0.3]],
    ]
    assert calls == 2


async def test_embed_query_images_batches_with_query_context() -> None:
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=3,
        provider=VoyageEmbedProvider(),
    )
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        "data": [{"embedding": [0.1, 0.2, 0.3]}, {"embedding": [0.4, 0.5, 0.6]}]
    }
    embedder._client.post = AsyncMock(return_value=response)

    result = await embedder.embed_query_images(
        [Image.new("RGB", (2, 2), "white"), Image.new("RGB", (2, 2), "black")]
    )

    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    payload = embedder._client.post.call_args.kwargs["json"]
    assert payload["input_type"] == "query"
    assert len(payload["inputs"]) == 2  # both images in ONE batched request


async def test_embed_query_images_builds_payload_off_event_loop(monkeypatch) -> None:
    from dlightrag.ai import embedding

    provider = VoyageEmbedProvider()
    loop_thread = threading.get_ident()
    preparation_threads: list[int] = []
    original_build_payload = provider.build_payload

    def encode_image(_image: Image.Image) -> str:
        preparation_threads.append(threading.get_ident())
        return "data:image/png;base64,AA=="

    def build_payload(*args, **kwargs):
        preparation_threads.append(threading.get_ident())
        return original_build_payload(*args, **kwargs)

    monkeypatch.setattr(embedding, "bounded_embedding_image_data_uri", encode_image)
    monkeypatch.setattr(provider, "build_payload", build_payload)
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=3,
        provider=provider,
    )
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
    embedder._client.post = AsyncMock(return_value=response)

    try:
        await embedder.embed_query_images([Image.new("RGB", (2, 2), "white")])
    finally:
        await embedder.aclose()

    assert preparation_threads
    assert all(thread_id != loop_thread for thread_id in preparation_threads)


async def test_embed_index_fused_builds_payload_off_event_loop(monkeypatch) -> None:
    from dlightrag.ai import embedding

    provider = VoyageEmbedProvider()
    loop_thread = threading.get_ident()
    preparation_threads: list[int] = []
    original_build_payload = provider.build_payload

    def encode_image(_image: Image.Image) -> str:
        preparation_threads.append(threading.get_ident())
        return "data:image/png;base64,AA=="

    def build_payload(*args, **kwargs):
        preparation_threads.append(threading.get_ident())
        return original_build_payload(*args, **kwargs)

    monkeypatch.setattr(embedding, "bounded_embedding_image_data_uri", encode_image)
    monkeypatch.setattr(provider, "build_payload", build_payload)
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=3,
        provider=provider,
    )
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
    embedder._client.post = AsyncMock(return_value=response)

    try:
        await embedder.embed_index_fused([("chart", Image.new("RGB", (2, 2), "white"))])
    finally:
        await embedder.aclose()

    assert preparation_threads
    assert all(thread_id != loop_thread for thread_id in preparation_threads)


async def test_probe_image_embedding_calls_query_embed() -> None:
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=3,
        provider=VoyageEmbedProvider(),
    )
    embedder.embed_query_images = AsyncMock(return_value=[[0.1, 0.2, 0.3]])  # type: ignore[method-assign]

    await embedder.probe_image_embedding()

    embedder.embed_query_images.assert_awaited_once()


async def test_image_probe_rejects_wrong_vector_count() -> None:
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=3,
        provider=VoyageEmbedProvider(),
    )
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        "data": [
            {"embedding": [0.1, 0.2, 0.3]},
            {"embedding": [0.4, 0.5, 0.6]},
        ]
    }
    embedder._client.post = AsyncMock(return_value=response)

    with pytest.raises(ValueError, match="Expected 1 embedding vector"):
        await embedder.probe_image_embedding()


async def test_embed_texts_rejects_wrong_vector_count() -> None:
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=3,
        provider=VoyageEmbedProvider(),
    )
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
    embedder._client.post = AsyncMock(return_value=response)

    with pytest.raises(ValueError, match="Expected 2 embedding vectors"):
        await embedder.embed_texts(["hello", "world"])


async def test_native_text_port_methods() -> None:
    """embed_documents/embed_query expose the raw-text port surface."""
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=3,
        provider=VoyageEmbedProvider(),
    )
    contexts: list[str] = []

    async def fake_request(payload: Any, *, expected_count: int, context: str, modality: str):
        del payload, modality
        contexts.append(context)
        return [[0.1] * 3 for _ in range(expected_count)]

    embedder._request_vectors = fake_request  # pyright: ignore[reportPrivateUsage]
    try:
        documents = await embedder.embed_documents(["alpha", "beta"])
        query = await embedder.embed_query("gamma")
    finally:
        await embedder.aclose()

    assert len(documents) == 2
    assert len(query) == 3
    assert contexts[0] == "document"
    assert contexts[-1] == "query"
