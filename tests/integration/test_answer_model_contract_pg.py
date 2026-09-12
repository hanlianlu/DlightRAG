# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Obsolete Answer pins fail per Run, not at process startup."""

from dataclasses import replace
from unittest.mock import AsyncMock
from uuid import uuid7

import pytest

from dlightrag._compose import _compose
from dlightrag.engine.ai.capacity import CONTEXT_POLICY_REVISION, ModelProfile
from dlightrag.engine.ai.catalog import current_model_catalog_revision
from dlightrag.engine.ai.fingerprints import model_fingerprint
from dlightrag.engine.ai.settings import CHAT_MODEL_SELECTORS
from dlightrag.engine.answer.execution.input import (
    AnswerRunInput,
    PinnedModelProfile,
    model_reasoning_settings,
)
from dlightrag.engine.runtime.errors import RunSchemaError
from dlightrag.engine.runtime.records import Succeeded
from tests.integration import test_answer_run_coordinator_pg as coordinator_pg
from tests.integration.test_answer_run_coordinator_pg import _wait_for_status
from tests.unit.test_application import _CLOSE_ORDER, _Parts

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
store = coordinator_pg.store


@pytest.mark.parametrize(
    "incompatibility", ["four_pins", "reasoning", "endpoint", "policy", "model_catalog"]
)
async def test_incompatible_answer_does_not_block_startup_or_new_run(
    store, test_config, monkeypatch, incompatibility
):
    monkeypatch.setattr("dlightrag.adapters.postgres.runtime.PGRunStore", lambda **kw: store)
    components = _compose(test_config)
    executor = components.coordinator._executors["answer"]
    effects = AsyncMock(return_value=Succeeded(result={"answer": "current model result"}))
    executor._execute = effects
    pins = tuple(
        PinnedModelProfile(
            role=role,
            fingerprint=model_fingerprint(test_config.models.chat.resolve(role)),
            profile=ModelProfile(context_window_tokens=100_000),
            reasoning_settings=model_reasoning_settings(test_config.models.chat.resolve(role)),
        )
        for role in CHAT_MODEL_SELECTORS
    )
    current = AnswerRunInput(
        query="new",
        workspaces=("default",),
        pinned_models=pins,
        context_policy_revision=CONTEXT_POLICY_REVISION,
        model_catalog_revision=current_model_catalog_revision(),
        idempotency_fingerprint="new-request",
        agent_session_id=str(uuid7()),
        agent_lane_id="main",
    )
    old = replace(current, query="old", idempotency_fingerprint="old-request")
    match incompatibility:
        case "four_pins":
            old = replace(old, pinned_models=pins[:-1])
        case "reasoning":
            old = replace(
                old, pinned_models=(*pins[:-1], replace(pins[-1], reasoning_settings=None))
            )
        case "endpoint":
            old = replace(
                old,
                pinned_models=(
                    *pins[:-1],
                    replace(pins[-1], fingerprint=replace(pins[-1].fingerprint, model="obsolete")),
                ),
            )
        case "policy":
            old = replace(old, context_policy_revision="obsolete")
        case "model_catalog":
            old = replace(old, model_catalog_revision="obsolete")
    creation = await store.create_run(owner_id="owner-alpha", prepared_input=old.as_request())
    parts = _Parts()
    application = parts.application(test_config)
    application._components = replace(
        application._components,
        validate_active_runs=components.validate_active_runs,
        coordinator=components.coordinator,
    )
    try:
        await application.astart()
        assert application.health.is_ready is True
        failed = await _wait_for_status(
            store, owner_id="owner-alpha", run_id=creation.run.run_id, status="failed"
        )
        assert (
            failed.error_kind == "incompatible_answer_run"
            and "Start a new Run" in failed.error_message
        )
        effects.assert_not_awaited()
        accepted = await store.create_run(
            owner_id="owner-alpha", prepared_input=current.as_request()
        )
        components.coordinator.wake()
        await _wait_for_status(
            store, owner_id="owner-alpha", run_id=accepted.run.run_id, status="succeeded"
        )
        effects.assert_awaited_once()
    finally:
        await application.aclose()
        await components.models.aclose()


@pytest.mark.parametrize(
    "error", [RunSchemaError("run schema unavailable"), RuntimeError("storage read failed")]
)
async def test_active_run_storage_failure_still_aborts_startup(
    store, test_config, monkeypatch, error
):
    monkeypatch.setattr("dlightrag.adapters.postgres.runtime.PGRunStore", lambda **kw: store)
    components = _compose(test_config)
    original_requirements = store.iter_active_run_requirements

    async def broken_requirements():
        # Fail while reading storage, not in the operation-owned Answer validator.
        async for requirement in original_requirements():
            yield requirement
        raise error

    monkeypatch.setattr(store, "iter_active_run_requirements", broken_requirements)
    parts = _Parts()
    application = parts.application(test_config)
    application._components = replace(
        application._components, validate_active_runs=components.validate_active_runs
    )
    try:
        with pytest.raises(type(error), match=str(error)):
            await application.astart()
        assert parts.recorder.closed() == _CLOSE_ORDER
        assert application.health.is_closed is True
        assert application.health.is_ready is False
        assert "coordinator:start" not in parts.recorder.started()
    finally:
        await application.aclose()
        await components.models.aclose()
