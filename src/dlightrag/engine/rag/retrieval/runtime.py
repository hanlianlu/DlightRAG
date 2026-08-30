# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Lifecycle for the AI-backed retrieval planner."""

import asyncio
from collections.abc import Callable
from functools import partial

from dlightrag.engine.ai.capacity import CONTEXT_POLICY, ModelProfile
from dlightrag.engine.ai.completion import CompletionModel
from dlightrag.engine.ai.scheduler import ModelScheduler
from dlightrag.engine.ai.settings import ModelSettings
from dlightrag.engine.ai.telemetry import Telemetry
from dlightrag.engine.rag.retrieval.planner import RetrievalPlanner
from dlightrag.engine.rag.workspace.lifecycle import await_shared_cleanup


class RetrievalPlannerRuntime:
    """Own the lazy planner model and profile-keyed planner cache."""

    def __init__(
        self,
        *,
        model_settings: ModelSettings,
        default_profile: Callable[[], ModelProfile],
        scheduler: ModelScheduler,
        telemetry: Telemetry,
    ) -> None:
        self._model_settings = model_settings
        self._default_profile = default_profile
        self._scheduler = scheduler
        self._telemetry = telemetry
        self._model: CompletionModel | None = None
        self._planners: dict[ModelProfile, RetrievalPlanner] = {}
        self._close_task: asyncio.Task[None] | None = None
        self._closed = False

    def planner_for(self, model_profile: ModelProfile | None = None) -> RetrievalPlanner:
        if self._closed:
            raise RuntimeError("Retrieval planner runtime is closed")
        profile = model_profile or self._default_profile()
        planner = self._planners.get(profile)
        if planner is not None:
            return planner
        if self._model is None:
            self._model = CompletionModel(
                self._model_settings,
                scheduler=self._scheduler,
                telemetry=self._telemetry,
            )
        planner = RetrievalPlanner(
            llm_func=partial(self._model, model_profile=profile),
            model_profile=profile,
            context_policy=CONTEXT_POLICY,
        )
        self._planners[profile] = planner
        return planner

    async def aclose(self) -> None:
        close_task = self._close_task
        if close_task is None:
            self._closed = True
            close_task = asyncio.create_task(self._close_resources())
            self._close_task = close_task
        await await_shared_cleanup(close_task)

    async def _close_resources(self) -> None:
        model, self._model = self._model, None
        self._planners.clear()
        if model is not None:
            await model.aclose()


__all__ = ["RetrievalPlannerRuntime"]
