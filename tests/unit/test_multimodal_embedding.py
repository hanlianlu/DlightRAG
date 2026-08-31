# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for budgeted context-aware embedding execution."""

import asyncio
import io
import threading
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from lightrag.utils import TiktokenTokenizer
from PIL import Image

from dlightrag.engine.ai.contracts import InputModality, ResolvedInputModality
from dlightrag.engine.ai.embedding import MultimodalEmbedder as _MultimodalEmbedder
from dlightrag.engine.ai.embedding import resolve_embedding_input_modality
from dlightrag.engine.ai.fingerprints import ModelFingerprint
from dlightrag.engine.ai.media import decode_image_base64
from dlightrag.engine.ai.providers.embed_base import EmbedProvider, OutputDimensionPolicy
from dlightrag.engine.ai.providers.embed_providers import (
    GeminiEmbedProvider as _GeminiEmbedProvider,
)
from dlightrag.engine.ai.providers.embed_providers import (
    JinaEmbedProvider,
    OpenAICompatibleEmbedProvider,
)
from dlightrag.engine.ai.providers.embed_providers import (
    VoyageEmbedProvider as _VoyageEmbedProvider,
)
from dlightrag.engine.ai.scheduler import ModelScheduler

_TEST_FINGERPRINT = ModelFingerprint(
    provider="test",
    model="test-model",
    endpoint_fingerprint=None,
)


def _relaxed_test_dimensions(capabilities):
    return replace(
        capabilities,
        output_dimension=OutputDimensionPolicy(
            send_upstream=True,
            minimum=1,
            maximum=3072,
        ),
    )


class VoyageEmbedProvider(_VoyageEmbedProvider):
    """Keep transport tests compact while production tests cover real dimensions."""

    def capabilities(self, model: str):
        return _relaxed_test_dimensions(super().capabilities(model))


class GeminiEmbedProvider(_GeminiEmbedProvider):
    """Keep ordered request tests compact with two-dimensional fixtures."""

    def capabilities(self, model: str):
        return _relaxed_test_dimensions(super().capabilities(model))


class TwoInputVoyageProvider(VoyageEmbedProvider):
    """Expose a tiny provider input-count limit for transport splitting tests."""

    def capabilities(self, model: str):
        return replace(super().capabilities(model), max_inputs=2)


class TinyImageBudgetVoyageProvider(VoyageEmbedProvider):
    """Expose a tiny provider image-byte limit for transport splitting tests."""

    def capabilities(self, model: str):
        return replace(super().capabilities(model), max_image_bytes_per_request=100)


def MultimodalEmbedder(**kwargs):
    scheduler = kwargs.pop("scheduler", ModelScheduler(max_concurrency=4))
    return _MultimodalEmbedder(
        fingerprint=_TEST_FINGERPRINT,
        scheduler=scheduler,
        **kwargs,
    )


def _response(
    status: int,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    return httpx.Response(
        status,
        json=payload,
        headers=headers,
        request=httpx.Request("POST", "https://example.test/embeddings"),
    )


class RecordingTelemetry:
    capture_sensitive_data = False

    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)

    @asynccontextmanager
    async def observe(self, _name: str, **_kwargs: Any):
        yield self


def test_generic_compatible_protocol_requires_an_explicit_base_url() -> None:
    with pytest.raises(ValueError, match="requires an embedding base_url"):
        MultimodalEmbedder(
            model="private-model",
            base_url="",
            api_key="",
            dim=3,
            provider=OpenAICompatibleEmbedProvider(),
        )


async def test_embedding_fingerprint_is_canonical_text_for_storage_ports() -> None:
    embedder = MultimodalEmbedder(
        model="private-model",
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
        model="private-model",
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


@pytest.mark.parametrize(
    ("provider", "model", "configured", "resolved"),
    [
        (VoyageEmbedProvider(), "voyage-multimodal-3.5", "auto", "multimodal"),
        (VoyageEmbedProvider(), "voyage-multimodal-3.5", "text", "text"),
        (VoyageEmbedProvider(), "voyage-multimodal-3.5", "multimodal", "multimodal"),
        (JinaEmbedProvider(), "jina-embeddings-v4", "auto", "multimodal"),
        (JinaEmbedProvider(), "jina-embeddings-v5-omni-small", "auto", "text"),
        (OpenAICompatibleEmbedProvider(), "private-model", "auto", "text"),
    ],
)
def test_resolve_embedding_input_modality(
    provider: EmbedProvider,
    model: str,
    configured: InputModality,
    resolved: ResolvedInputModality,
) -> None:
    assert resolve_embedding_input_modality(provider, model, configured) == resolved


def test_required_multimodal_rejects_generic_and_non_fusing_models() -> None:
    with pytest.raises(ValueError, match=r"native single-vector text\+image fusion"):
        MultimodalEmbedder(
            model="private-model",
            base_url="https://example.test/v1",
            api_key="",
            dim=2048,
            provider=OpenAICompatibleEmbedProvider(),
            input_modality="multimodal",
        )
    with pytest.raises(ValueError, match=r"native single-vector text\+image fusion"):
        MultimodalEmbedder(
            model="jina-embeddings-v5-omni-small",
            base_url="https://api.jina.ai/v1",
            api_key="key",
            dim=1024,
            provider=JinaEmbedProvider(),
            input_modality="multimodal",
        )


async def test_native_multimodal_provider_can_be_forced_to_text() -> None:
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=1024,
        provider=VoyageEmbedProvider(),
        input_modality="text",
    )
    image = Image.new("RGB", (1, 1), "white")
    try:
        assert embedder.supports_images is False
        with pytest.raises(ValueError, match="native fused image"):
            await embedder.embed_index_fused([("x", image)])
    finally:
        image.close()
        await embedder.aclose()


def test_multimodal_embedder_builds_fused_voyage_payload() -> None:
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=1024,
        provider=VoyageEmbedProvider(),
    )
    image = Image.new("RGB", (2, 2), "white")
    try:
        payload = embedder._build_fused_payload(  # pyright: ignore[reportPrivateUsage]
            [("a bar chart", image)], context="document"
        )
    finally:
        image.close()
    content = payload["inputs"][0]["content"]
    assert content[0] == {"type": "text", "text": "a bar chart"}
    assert content[1]["type"] == "image_base64"
    assert content[1]["image_base64"].startswith("data:image/")
    assert payload["input_type"] == "document"


def test_fused_payload_degrades_to_image_only_when_description_blank() -> None:
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=1024,
        provider=VoyageEmbedProvider(),
    )
    image = Image.new("RGB", (2, 2), "white")
    try:
        payload = embedder._build_fused_payload(  # pyright: ignore[reportPrivateUsage]
            [("   ", image)], context="document"
        )
    finally:
        image.close()
    assert payload["inputs"][0]["content"][0]["type"] == "image_base64"


def test_image_embedder_bounds_oversized_images_before_send() -> None:
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=1024,
        provider=VoyageEmbedProvider(),
    )
    image = Image.new("RGB", (6000, 5000), "white")
    try:
        payload = embedder._build_fused_payload(  # pyright: ignore[reportPrivateUsage]
            [("", image)], context="document"
        )
    finally:
        image.close()
    data_uri = payload["inputs"][0]["content"][0]["image_base64"]
    raw, _ = decode_image_base64(data_uri)
    with Image.open(io.BytesIO(raw)) as decoded:
        assert max(decoded.size) <= 4096
        assert decoded.width * decoded.height <= 15_000_000


def test_known_dimension_policy_fails_before_any_request() -> None:
    with pytest.raises(ValueError, match="only supports dimensions"):
        MultimodalEmbedder(
            model="voyage-multimodal-3.5",
            base_url="https://api.voyageai.com/v1",
            api_key="key",
            dim=1536,
            provider=_VoyageEmbedProvider(),
        )


async def test_embed_texts_posts_document_context_and_uses_request_url() -> None:
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=3,
        provider=VoyageEmbedProvider(),
    )
    embedder._client.post = AsyncMock(  # pyright: ignore[reportPrivateUsage]
        return_value=_response(200, {"data": [{"embedding": [0.1, 0.2, 0.3]}]})
    )
    try:
        result = await embedder.embed_texts(["hello"], context="document")
    finally:
        await embedder.aclose()

    assert result == [[0.1, 0.2, 0.3]]
    call = embedder._client.post.call_args  # pyright: ignore[reportPrivateUsage]
    assert call.args[0] == "https://api.voyageai.com/v1/multimodalembeddings"
    assert call.kwargs["json"]["input_type"] == "document"


async def test_empty_batch_and_blank_text_are_rejected_without_network() -> None:
    embedder = MultimodalEmbedder(
        model="private-model",
        base_url="https://example.test/v1",
        api_key="",
        dim=3,
        provider=OpenAICompatibleEmbedProvider(),
    )
    embedder._client.post = AsyncMock()  # pyright: ignore[reportPrivateUsage]
    try:
        with pytest.raises(ValueError, match="at least one input"):
            await embedder.embed_texts([])
        with pytest.raises(ValueError, match="empty"):
            await embedder.embed_texts(["   "])
    finally:
        await embedder.aclose()
    embedder._client.post.assert_not_awaited()  # pyright: ignore[reportPrivateUsage]


async def test_lightrag_truncated_input_is_forwarded_without_local_token_guard() -> None:
    tokenizer = TiktokenTokenizer("gpt-4o-mini")
    original = "economic development and monetary policy " * 4_000
    span = tokenizer.truncate_by_token_limit(original, 8_192)
    text = original[span.start : span.end]
    provider = VoyageEmbedProvider()
    assert len(tokenizer.encode(text)) == 8_192

    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=3,
        provider=provider,
    )
    embedder._client.post = AsyncMock(  # pyright: ignore[reportPrivateUsage]
        return_value=_response(200, {"data": [{"embedding": [0.1, 0.2, 0.3]}]})
    )
    try:
        result = await embedder.embed_texts([text])
    finally:
        await embedder.aclose()

    assert result == [[0.1, 0.2, 0.3]]
    embedder._client.post.assert_awaited_once()  # pyright: ignore[reportPrivateUsage]


async def test_provider_input_count_auto_splits_and_preserves_order() -> None:
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=2,
        provider=TwoInputVoyageProvider(),
    )
    next_value = 1

    async def post(_url: str, *, json: dict[str, Any]) -> httpx.Response:
        nonlocal next_value
        vectors = []
        for _item in json["inputs"]:
            vectors.append({"embedding": [float(next_value), 1.0]})
            next_value += 1
        return _response(200, {"data": vectors})

    embedder._client.post = AsyncMock(side_effect=post)  # pyright: ignore[reportPrivateUsage]
    try:
        vectors = await embedder.embed_texts(["a", "b", "c", "d", "e"])
    finally:
        await embedder.aclose()

    assert vectors == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0], [5.0, 1.0]]
    assert embedder._client.post.await_count == 3  # pyright: ignore[reportPrivateUsage]


async def test_combined_image_byte_budget_auto_splits_requests() -> None:
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=2,
        provider=TinyImageBudgetVoyageProvider(),
    )

    async def post(_url: str, *, json: dict[str, Any]) -> httpx.Response:
        assert len(json["inputs"]) == 1
        return _response(200, {"data": [{"embedding": [1.0, 1.0]}]})

    embedder._client.post = AsyncMock(side_effect=post)  # pyright: ignore[reportPrivateUsage]
    images = [Image.new("RGB", (2, 2), "white"), Image.new("RGB", (2, 2), "black")]
    try:
        assert await embedder.embed_query_images(images) == [[1.0, 1.0], [1.0, 1.0]]
    finally:
        for image in images:
            image.close()
        await embedder.aclose()
    assert embedder._client.post.await_count == 2  # pyright: ignore[reportPrivateUsage]


async def test_gemini_multiple_inputs_become_ordered_single_input_requests() -> None:
    embedder = MultimodalEmbedder(
        model="gemini-embedding-2",
        base_url="",
        api_key="key",
        dim=2,
        provider=GeminiEmbedProvider(),
    )

    async def post(_url: str, *, json: dict[str, Any]) -> httpx.Response:
        text = json["content"]["parts"][0]["text"]
        value = 1.0 if text.endswith("one") else 2.0
        if value == 1.0:
            await asyncio.sleep(0.01)
        return _response(200, {"embedding": {"values": [value, 1.0]}})

    embedder._client.post = AsyncMock(side_effect=post)  # pyright: ignore[reportPrivateUsage]
    try:
        vectors = await embedder.embed_texts(["one", "two"], context="query")
    finally:
        await embedder.aclose()

    assert embedder.base_url == "https://generativelanguage.googleapis.com/v1beta"
    assert vectors == [[1.0, 1.0], [2.0, 1.0]]
    assert embedder._client.post.await_count == 2  # pyright: ignore[reportPrivateUsage]


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

    async def post(payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        nonlocal calls
        del payload
        calls += 1
        if calls == 1:
            first_started.set()
            await release_first.wait()
        return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}, 0

    embedder._post = post  # pyright: ignore[reportPrivateUsage]
    first = asyncio.create_task(embedder.embed_texts(["first"]))
    await first_started.wait()
    second = asyncio.create_task(embedder.embed_texts(["second"]))
    await asyncio.sleep(0)
    assert calls == 1

    release_first.set()
    try:
        assert await asyncio.gather(first, second) == [
            [[0.1, 0.2, 0.3]],
            [[0.1, 0.2, 0.3]],
        ]
    finally:
        await embedder.aclose()
    assert calls == 2


async def test_embed_query_images_batches_with_query_context() -> None:
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=3,
        provider=VoyageEmbedProvider(),
    )
    embedder._client.post = AsyncMock(  # pyright: ignore[reportPrivateUsage]
        return_value=_response(
            200,
            {
                "data": [
                    {"embedding": [0.1, 0.2, 0.3]},
                    {"embedding": [0.4, 0.5, 0.6]},
                ]
            },
        )
    )
    images = [Image.new("RGB", (2, 2), "white"), Image.new("RGB", (2, 2), "black")]
    try:
        result = await embedder.embed_query_images(images)
    finally:
        for image in images:
            image.close()
        await embedder.aclose()

    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    payload = embedder._client.post.call_args.kwargs["json"]  # pyright: ignore[reportPrivateUsage]
    assert payload["input_type"] == "query"
    assert len(payload["inputs"]) == 2


async def test_image_payload_preparation_runs_off_event_loop(monkeypatch) -> None:
    from dlightrag.engine.ai import embedding

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
    embedder._client.post = AsyncMock(  # pyright: ignore[reportPrivateUsage]
        return_value=_response(200, {"data": [{"embedding": [0.1, 0.2, 0.3]}]})
    )
    image = Image.new("RGB", (2, 2), "white")
    try:
        await embedder.embed_query_images([image])
    finally:
        image.close()
        await embedder.aclose()

    assert preparation_threads
    assert all(thread_id != loop_thread for thread_id in preparation_threads)


async def test_probe_checks_image_query_and_fused_document() -> None:
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=3,
        provider=VoyageEmbedProvider(),
    )
    embedder.embed_query_images = AsyncMock(return_value=[[0.1, 0.2, 0.3]])  # type: ignore[method-assign]
    embedder.embed_index_fused = AsyncMock(return_value=[[0.1, 0.2, 0.3]])  # type: ignore[method-assign]
    try:
        await embedder.probe_image_embedding()
    finally:
        await embedder.aclose()

    embedder.embed_query_images.assert_awaited_once()  # type: ignore[attr-defined]
    embedder.embed_index_fused.assert_awaited_once()  # type: ignore[attr-defined]
    assert embedder.embed_index_fused.await_args.args[0][0][0] == "DlightRAG fusion probe"  # type: ignore[attr-defined]


@pytest.mark.parametrize("failure", [httpx.ConnectError("down"), httpx.ReadError("reset")])
async def test_connection_failures_retry_at_most_twice(failure: httpx.TransportError) -> None:
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=3,
        provider=VoyageEmbedProvider(),
    )
    embedder._client.post = AsyncMock(  # pyright: ignore[reportPrivateUsage]
        side_effect=[failure, failure, _response(200, {"data": [{"embedding": [0.1, 0.2, 0.3]}]})]
    )
    with pytest.MonkeyPatch.context() as patch:
        sleep = AsyncMock()
        patch.setattr(asyncio, "sleep", sleep)
        try:
            assert await embedder.embed_texts(["hello"]) == [[0.1, 0.2, 0.3]]
        finally:
            await embedder.aclose()
    assert embedder._client.post.await_count == 3  # pyright: ignore[reportPrivateUsage]
    assert sleep.await_count == 2


@pytest.mark.parametrize("status", [408, 409, 429, 500, 503])
async def test_retryable_http_statuses_are_retried(status: int) -> None:
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=3,
        provider=VoyageEmbedProvider(),
    )
    embedder._client.post = AsyncMock(  # pyright: ignore[reportPrivateUsage]
        side_effect=[
            _response(status, {"error": "retry"}, headers={"Retry-After": "2"}),
            _response(200, {"data": [{"embedding": [0.1, 0.2, 0.3]}]}),
        ]
    )
    with pytest.MonkeyPatch.context() as patch:
        sleep = AsyncMock()
        patch.setattr(asyncio, "sleep", sleep)
        try:
            await embedder.embed_texts(["hello"])
        finally:
            await embedder.aclose()
    sleep.assert_awaited_once_with(2.0)


async def test_non_retryable_4xx_and_schema_errors_are_not_retried() -> None:
    for response in (
        _response(400, {"error": "bad input"}),
        _response(200, {"not_data": []}),
    ):
        embedder = MultimodalEmbedder(
            model="voyage-multimodal-3.5",
            base_url="https://api.voyageai.com/v1",
            api_key="key",
            dim=3,
            provider=VoyageEmbedProvider(),
        )
        embedder._client.post = AsyncMock(return_value=response)  # pyright: ignore[reportPrivateUsage]
        try:
            with pytest.raises((httpx.HTTPStatusError, ValueError)):
                await embedder.embed_texts(["hello"])
        finally:
            await embedder.aclose()
        assert embedder._client.post.await_count == 1  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    "vectors",
    [
        [[0.1, 0.2]],
        [[0.1, float("nan"), 0.3]],
        [[0.0, 0.0, 0.0]],
        [[True, 0.2, 0.3]],
    ],
)
def test_strict_vector_validation_rejects_bad_dimensions_values_and_norms(
    vectors: list[list[Any]],
) -> None:
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=3,
        provider=VoyageEmbedProvider(),
    )
    with pytest.raises(ValueError, match="dim|invalid"):
        embedder._validate_vectors(vectors)  # pyright: ignore[reportPrivateUsage]


async def test_usage_and_retry_counts_are_added_to_telemetry() -> None:
    telemetry = RecordingTelemetry()
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=3,
        provider=VoyageEmbedProvider(),
        telemetry=telemetry,
    )
    embedder._client.post = AsyncMock(  # pyright: ignore[reportPrivateUsage]
        return_value=_response(
            200,
            {
                "data": [{"embedding": [0.1, 0.2, 0.3]}],
                "usage": {"total_tokens": 7},
            },
        )
    )
    try:
        await embedder.embed_texts(["hello"])
    finally:
        await embedder.aclose()

    assert telemetry.updates[-1]["output"] == {
        "embedding_count": 1,
        "request_count": 1,
        "retry_count": 0,
        "usage": {"total_tokens": 7},
    }


async def test_embedding_error_text_is_redacted_when_sensitive_capture_is_disabled() -> None:
    telemetry = RecordingTelemetry()
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=3,
        provider=VoyageEmbedProvider(),
        telemetry=telemetry,
    )
    embedder._client.post = AsyncMock(  # pyright: ignore[reportPrivateUsage]
        side_effect=RuntimeError("upstream echoed secret embedding input")
    )
    try:
        with pytest.raises(RuntimeError, match="secret embedding input"):
            await embedder.embed_texts(["secret"])
    finally:
        await embedder.aclose()

    assert telemetry.updates == [{"level": "ERROR", "status_message": "RuntimeError"}]


async def test_native_text_port_methods() -> None:
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=3,
        provider=VoyageEmbedProvider(),
    )
    embedder._client.post = AsyncMock(  # pyright: ignore[reportPrivateUsage]
        side_effect=[
            _response(
                200,
                {
                    "data": [
                        {"embedding": [0.1, 0.2, 0.3]},
                        {"embedding": [0.4, 0.5, 0.6]},
                    ]
                },
            ),
            _response(200, {"data": [{"embedding": [0.7, 0.8, 0.9]}]}),
        ]
    )
    try:
        documents = await embedder.embed_documents(["alpha", "beta"])
        query = await embedder.embed_query("gamma")
    finally:
        await embedder.aclose()

    assert len(documents) == 2
    assert len(query) == 3
    assert embedder._client.post.await_args_list[0].kwargs["json"]["input_type"] == "document"  # pyright: ignore[reportPrivateUsage]
    assert embedder._client.post.await_args_list[1].kwargs["json"]["input_type"] == "query"  # pyright: ignore[reportPrivateUsage]
