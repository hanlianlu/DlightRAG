# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unit-test fixtures; the root conftest isolates every suite from operator inputs."""

from collections.abc import Generator
from typing import Any

import pytest

from dlightrag.engine.ai.capacity import CONTEXT_POLICY_REVISION, ModelProfile
from dlightrag.engine.ai.catalog import current_model_catalog_revision
from dlightrag.engine.ai.fingerprints import ModelInvocationFingerprint
from dlightrag.engine.ai.media import MODEL_IMAGE_MAX_PIXELS
from dlightrag.engine.ai.structured_transport import JSON_SCHEMA_TRANSPORT_CACHE
from dlightrag.engine.answer.capabilities import AnswerCapabilities
from dlightrag.engine.answer.execution.input import (
    AnswerRunInput,
    AnswerRunRequest,
    PinnedModelProfile,
    new_resource_identity,
)
from dlightrag.engine.answer.image_capability import AnswerImageCapability
from dlightrag.engine.answer.images import AnswerImagePolicy
from dlightrag.engine.answer.resources.models import ResourceInput


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


async def prepare_test_answer_run_input(
    request: AnswerRunRequest,
    *,
    resources: list[ResourceInput] | None,  # noqa: ARG001
    idempotency_fingerprint: str,
) -> AnswerRunInput:
    """Pin one normalized request for tests that do not exercise model resolution."""
    return AnswerRunInput(
        query=request.query,
        workspaces=request.workspaces,
        history=request.history,
        retrieval=request.retrieval,
        filters=request.filters,
        semantic_highlights=request.semantic_highlights,
        links=request.links,
        attachments=request.attachments,
        history_attachments=request.history_attachments,
        pinned_models=(
            PinnedModelProfile(
                role="query",
                fingerprint=ModelInvocationFingerprint(
                    "openai", "test-model", None, "chat_completion"
                ),
                profile=answer_model_profile(),
            ),
        ),
        context_policy_revision=CONTEXT_POLICY_REVISION,
        model_catalog_revision=current_model_catalog_revision(),
        idempotency_fingerprint=idempotency_fingerprint,
        resource_identity=new_resource_identity(),
    )


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
