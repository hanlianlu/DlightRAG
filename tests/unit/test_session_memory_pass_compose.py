# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Which model the composed Session Memory Pass asks, and how hard it is told to think.

The pass asks the `query` role's model at the reasoning level `memory_consolidation.reasoning`
sets (ADR 0045). `CompletionModel` reads only `reasoning`, and the `query` role ships with
`reasoning: off`, so the pass must set its own level or it runs without thinking: which the eval
(docs/memory-pass-eval.md) found recalls none of the owner's real preferences. The role's own
levels, ordinary or agentic, serve other work and do not decide it.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from dlightrag._compose import _session_memory_passes
from dlightrag.application.config.sections import MemoryConsolidationConfig
from dlightrag.engine.ai.reasoning import ReasoningLevel
from dlightrag.engine.ai.scheduler import ModelScheduler
from dlightrag.engine.ai.settings import ModelRoleOverrides, ModelRoleSettings, ModelSettings
from dlightrag.engine.ai.telemetry import NOOP_TELEMETRY


def _composed_pass_settings(
    config: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    query: ModelSettings,
    consolidation: MemoryConsolidationConfig,
) -> ModelSettings:
    """The settings `_session_memory_passes` builds its model from, for this role and setting."""
    chat = ModelRoleSettings(
        default=config.models.chat.default,
        roles=ModelRoleOverrides(
            # The role the pass used before: a distinct model proves which role it now asks.
            extract=ModelSettings(model="extract-model", api_key="k", reasoning="off"),
            query=query,
        ),
    )
    agent = config.answer.agent.model_copy(update={"memory_consolidation": consolidation})
    configured = config.model_copy(
        update={
            "models": config.models.model_copy(update={"chat": chat}),
            "answer": config.answer.model_copy(update={"agent": agent}),
        }
    )
    built: list[ModelSettings] = []

    class _Built:
        """Stands in for `CompletionModel`: remembers the settings the pass built it from."""

        def __init__(self, settings: ModelSettings, **_: Any) -> None:
            built.append(settings)

    monkeypatch.setattr("dlightrag.engine.ai.completion.CompletionModel", _Built)

    runner = _session_memory_passes(
        configured,
        run_store=cast(Any, object()),
        memory_store=cast(Any, object()),
        memory=cast(Any, object()),
        scheduler=ModelScheduler(max_concurrency=1),
        telemetry=NOOP_TELEMETRY,
    )

    assert runner is not None
    (settings,) = built
    return settings


@pytest.mark.parametrize(
    "role",
    [
        pytest.param({"reasoning": "off", "agentic_reasoning": "high"}, id="off-and-agentic-high"),
        pytest.param({"reasoning": "max"}, id="ordinary-max-inherited"),
        pytest.param({"reasoning": "off", "agentic_reasoning": None}, id="no-agentic-level"),
        pytest.param({}, id="no-reasoning-configured"),
    ],
)
@pytest.mark.parametrize(
    ("consolidation", "expected"),
    [
        pytest.param(MemoryConsolidationConfig(), "low", id="default"),
        pytest.param(MemoryConsolidationConfig(reasoning="medium"), "medium", id="explicit"),
        pytest.param(MemoryConsolidationConfig(reasoning="off"), "off", id="off"),
    ],
)
def test_the_pass_asks_the_query_role_at_the_level_memory_consolidation_sets(
    test_config: Any,
    monkeypatch: pytest.MonkeyPatch,
    role: dict[str, Any],
    consolidation: MemoryConsolidationConfig,
    expected: ReasoningLevel,
) -> None:
    query = ModelSettings.model_validate(
        {"model": "query-model", "api_key": "k", "temperature": 0.7, **role}
    )

    settings = _composed_pass_settings(
        test_config, monkeypatch, query=query, consolidation=consolidation
    )

    assert settings.model == "query-model"
    assert settings.temperature == 0.7
    assert settings.reasoning == expected
    # The role itself is not edited: the Agent's loop still reads its own levels.
    assert query.reasoning == role.get("reasoning")
    assert settings.agentic_reasoning == query.agentic_reasoning
