# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unit-test fixtures; the root conftest isolates every suite from operator inputs."""

import functools
import json
import uuid
from collections.abc import Callable, Generator
from typing import Any

import langfuse as langfuse_sdk
import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from dlightrag.application.config import ObservabilitySettings
from dlightrag.engine.ai.capacity import ModelProfile
from dlightrag.engine.ai.media import MODEL_IMAGE_MAX_PIXELS
from dlightrag.engine.ai.structured_transport import JSON_SCHEMA_TRANSPORT_CACHE
from dlightrag.engine.answer.capabilities import AnswerCapabilities
from dlightrag.engine.answer.image_capability import AnswerImageCapability
from dlightrag.engine.answer.images import AnswerImagePolicy


def answer_image_policy(**overrides: int) -> AnswerImagePolicy:
    """Shipped answer transport policy for tests; images off unless opted in."""
    fields: dict[str, int] = {
        "max_images": 0,
        "max_total_bytes": 24_000_000,
        "max_bytes_per_image": 3_000_000,
        "max_pixels": MODEL_IMAGE_MAX_PIXELS,
        "max_px": 1536,
        "min_px": 1024,
        "quality": 89,
        "min_quality": 79,
    }
    return AnswerImagePolicy(**(fields | overrides))


def answer_model_profile(**overrides: int | bool | None) -> ModelProfile:
    """Resolved answer-model facts for tests that do not exercise the catalog."""
    fields: dict[str, int | bool | None] = {
        "context_window_tokens": 1_000_000,
        "max_input_tokens": None,
        "max_output_tokens": 128_000,
        "supports_images": True,
    }
    return ModelProfile(**(fields | overrides))  # type: ignore[arg-type]


def answer_capabilities(answer: AnswerImageCapability | None = None) -> AnswerCapabilities:
    """The capability snapshot a transport test's answers double reports."""
    return AnswerCapabilities(answer=answer, vlm_status="unknown")


@pytest.fixture(autouse=True)
def _fresh_json_schema_transport_cache():
    """Keep the process-wide json_schema transport cache out of test verdicts."""
    JSON_SCHEMA_TRANSPORT_CACHE.clear()
    yield
    JSON_SCHEMA_TRANSPORT_CACHE.clear()


class RecordingObservation:
    """One recorded Langfuse observation, including its ambient parent."""

    def __init__(self, client: RecordingLangfuse, kwargs: dict[str, Any]) -> None:
        self.client = client
        self.kwargs = kwargs
        self.parent = client.active[-1] if client.active else None
        self.updates: list[dict[str, Any]] = []

    def __enter__(self) -> RecordingObservation:
        self.client.active.append(self)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        popped = self.client.active.pop()
        assert popped is self

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)


class RecordingLangfuse:
    """A Langfuse client double that records observations and their nesting."""

    def __init__(self) -> None:
        self.observations: list[RecordingObservation] = []
        self.active: list[RecordingObservation] = []
        self.flushed = False
        self.shutdown_called = False

    def start_as_current_observation(self, **kwargs: Any) -> RecordingObservation:
        obs = RecordingObservation(self, kwargs)
        self.observations.append(obs)
        return obs

    def flush(self) -> None:
        self.flushed = True

    def shutdown(self) -> None:
        self.shutdown_called = True


@pytest.fixture
def reset_langfuse_client(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    """Isolate process tracing state, restoring it afterwards.

    The real ``langfuse.propagate_attributes`` reaches for a real client when a
    double is installed, so tests drive the same call shape through a neutral
    context manager unless they monkeypatch it themselves.
    """
    from contextlib import contextmanager

    from dlightrag.adapters.observability import langfuse as langfuse_state

    @contextmanager
    def _neutral_propagation(**_kwargs: Any) -> Generator[None]:
        yield

    monkeypatch.setattr("langfuse.propagate_attributes", _neutral_propagation)
    previous = langfuse_state.current_client()
    previous_sensitive = langfuse_state.trace_sensitive_enabled()
    langfuse_state.install_client(None, trace_sensitive=True)
    yield
    langfuse_state.install_client(previous, trace_sensitive=previous_sensitive)


class LangfuseExport:
    """What a real Langfuse client, started by ``init_tracing``, has exported."""

    def __init__(self, exporter: InMemorySpanExporter) -> None:
        self._exporter = exporter

    def spans(self) -> tuple[ReadableSpan, ...]:
        from dlightrag.adapters.observability import langfuse as langfuse_state

        client = langfuse_state.current_client()
        assert client is not None, "tracing is not started"
        client.flush()
        return self._exporter.get_finished_spans()

    @staticmethod
    def usage(span: ReadableSpan) -> dict[str, int]:
        """The usage keys the span carries, none when it carries none."""
        value = (span.attributes or {}).get(
            langfuse_sdk.LangfuseOtelSpanAttributes.OBSERVATION_USAGE_DETAILS
        )
        return {} if value is None else json.loads(str(value))


@pytest.fixture
def start_langfuse_export(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[Callable[..., LangfuseExport]]:
    """Start a real Langfuse client through ``init_tracing``, exporting into memory.

    Only the span exporter is the test's: the tracer provider, sampling, masking and
    attribute mapping are the product's. Langfuse keeps one resource manager per public
    key for the life of the process, so every client gets a key of its own.
    """
    from dlightrag.adapters.observability import langfuse as langfuse_state

    real_client = langfuse_sdk.Langfuse

    def start(**settings: Any) -> LangfuseExport:
        exporter = InMemorySpanExporter()
        monkeypatch.setattr(
            "langfuse.Langfuse", functools.partial(real_client, span_exporter=exporter)
        )
        langfuse_state.init_tracing(
            ObservabilitySettings(
                langfuse_public_key=f"pk-lf-{uuid.uuid4().hex}",
                langfuse_secret_key="sk-lf-test",
                langfuse_host="http://127.0.0.1:9",
                langfuse_environment="test",
                langfuse_release="9.9.9",
                **settings,
            )
        )
        return LangfuseExport(exporter)

    yield start
    langfuse_state.shutdown_tracing()
    langfuse_state.install_client(None, trace_sensitive=True)


@pytest.fixture
def langfuse_export(start_langfuse_export: Callable[..., LangfuseExport]) -> LangfuseExport:
    return start_langfuse_export()
