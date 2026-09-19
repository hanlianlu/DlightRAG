# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unit-test fixtures — isolate from the operator's .env and config.yaml.

Both are deployment inputs, not product contracts: a unit test that reads them
asserts whatever this checkout happens to be tuned to, so retuning config.yaml
breaks CI. Tests that mean to exercise a YAML config build their own file.
"""

import os
from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from dlightrag.application import config as config_module
from dlightrag.application.config import DlightragConfig
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
)
from dlightrag.engine.answer.image_capability import AnswerImageCapability
from dlightrag.engine.answer.images import AnswerImagePolicy
from dlightrag.engine.answer.resources.models import ResourceInput

_REPO_CONFIG_YAML = Path(__file__).resolve().parents[2] / "config.yaml"
# Bound before the fixture patches the name, otherwise the wrapper recurses.
_FIND_YAML_CONFIG = config_module._find_yaml_config


def _yaml_config_ignoring_repo_file() -> Path | None:
    """Resolve config.yaml as production does, minus this checkout's own file."""
    found = _FIND_YAML_CONFIG()
    if found is not None and found.resolve() == _REPO_CONFIG_YAML:
        return None
    return found


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


def answer_capability_view(
    answer: AnswerImageCapability | None = None,
) -> SimpleNamespace:
    """Read-only capability-view double for transport tests."""
    snapshot = AnswerCapabilities(answer=answer, vlm_status="unknown")
    return SimpleNamespace(read=AsyncMock(return_value=snapshot))


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
    )


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent .env and the repo's config.yaml from polluting unit tests."""
    monkeypatch.setitem(DlightragConfig.model_config, "env_file", None)
    monkeypatch.setattr(config_module, "_find_yaml_config", _yaml_config_ignoring_repo_file)
    for key in list(os.environ):
        if key.startswith("DLIGHTRAG_"):
            monkeypatch.delenv(key, raising=False)


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
