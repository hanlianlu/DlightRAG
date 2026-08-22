# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared test fixtures for dlightrag tests."""

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from dlightrag.adapters.postgres.answer_runs import (
    PGAnswerRunStore,
)
from dlightrag.ai.settings import (
    EmbeddingSettings,
    ModelCapacityOverrideSettings,
    ModelRoleSettings,
    ModelSettings,
)
from dlightrag.answer.routing import RoutingAcceptance
from dlightrag.config import DlightragConfig, reset_config, set_config
from dlightrag.runtime import (
    PendingArtifact,
    PendingArtifactReference,
    RunCreation,
    answer_run_request_fingerprint,
)


class FingerprintingAnswerRunStore(PGAnswerRunStore):
    """Test adapter for low-level suites whose raw request is the public input."""

    async def create_run(
        self,
        *,
        owner_id: str,
        request: Mapping[str, Any] | None = None,
        prepared_input: Mapping[str, Any] | None = None,
        idempotency_fingerprint: str | None = None,
        idempotency_key: str | None = None,
        artifacts: Sequence[PendingArtifact] = (),
        references: Sequence[PendingArtifactReference] = (),
    ) -> RunCreation:

        from dlightrag.agent.session.ids import SessionId

        if prepared_input is not None:
            return await super().create_run(
                owner_id=owner_id,
                prepared_input=dict(prepared_input),
                idempotency_fingerprint=(
                    idempotency_fingerprint or answer_run_request_fingerprint(prepared_input)
                ),
                idempotency_key=idempotency_key,
                artifacts=artifacts,
                references=references,
            )
        request = request or {}
        prepared: dict[str, Any] = {
            "session_id": SessionId.new().value,
            **dict(request),
        }
        return await super().create_run(
            owner_id=owner_id,
            prepared_input=prepared,
            idempotency_fingerprint=(
                idempotency_fingerprint or answer_run_request_fingerprint(request)
            ),
            idempotency_key=idempotency_key,
            artifacts=artifacts,
            references=references,
        )

    async def create_run_in(
        self,
        conn: Any,
        *,
        owner_id: str,
        request: Mapping[str, Any],
        idempotency_fingerprint: str | None = None,
        idempotency_key: str | None = None,
        artifacts: Sequence[PendingArtifact] = (),
        references: Sequence[PendingArtifactReference] = (),
        routing: RoutingAcceptance | None = None,
    ) -> RunCreation:

        from dlightrag.agent.session.ids import SessionId

        prepared: dict[str, Any] = {
            "session_id": SessionId.new().value,
            **dict(request),
        }
        return await super().create_run_in(
            conn,
            owner_id=owner_id,
            request=prepared,
            idempotency_fingerprint=(
                idempotency_fingerprint or answer_run_request_fingerprint(request)
            ),
            idempotency_key=idempotency_key,
            artifacts=artifacts,
            references=references,
            routing=routing,
        )


@pytest.fixture(autouse=True)
def _reset_config_singleton():
    """Reset the config singleton before each test."""
    reset_config()
    yield
    reset_config()


@pytest.fixture
def tmp_working_dir(tmp_path: Path) -> Path:
    """Create a temporary working directory structure."""
    working_dir = tmp_path / "dlightrag_storage"
    (working_dir / "artifacts" / "local").mkdir(parents=True)
    return working_dir


@pytest.fixture
def test_config(tmp_working_dir: Path) -> DlightragConfig:
    """Create a test config with temporary paths.

    Also sets the global singleton so that code calling get_config()
    directly (e.g. /health endpoint) gets the test config.
    """
    cfg = DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
        # type: ignore[call-arg]
        deployment={"working_dir": str(tmp_working_dir)},
        models={
            "chat": ModelRoleSettings(
                default=ModelSettings(
                    model="gpt-5.4-mini",
                    api_key=os.getenv("DLIGHTRAG_OPENAI_API_KEY", "test-key-for-unit-tests"),
                )
            ),
            "capacity_overrides": [
                ModelCapacityOverrideSettings(
                    provider="openai",
                    model="gpt-5.4-mini",
                    context_window_tokens=400_000,
                    max_output_tokens=128_000,
                    supports_images=True,
                    supports_tools=True,
                    supports_reasoning=True,
                )
            ],
            "embedding": EmbeddingSettings(
                provider="voyage",
                model="voyage-multimodal-3.5",
                api_key=os.getenv("DLIGHTRAG_OPENAI_API_KEY", "test-key-for-unit-tests"),
                startup_probe=False,
            ),
        },
    )
    set_config(cfg)
    return cfg
