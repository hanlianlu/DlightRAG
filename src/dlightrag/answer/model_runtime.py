# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Answer-owned lazy model and Web-search client lifecycle."""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from dlightrag.answer.images import AnswerImagePolicy
from dlightrag.answer.resources.images import QueryImageDescriber
from dlightrag.answer.synthesizer import AnswerSynthesizer
from dlightrag.answer.tools.web import ExaSearch
from dlightrag.engine.ai.capacity import CONTEXT_POLICY, ModelProfile
from dlightrag.engine.ai.completion import CompletionModel
from dlightrag.engine.ai.scheduler import ModelScheduler
from dlightrag.engine.ai.settings import ModelRole, ModelRoleSettings
from dlightrag.engine.ai.telemetry import Telemetry
from dlightrag.engine.ai.tool_model import ToolModel
from dlightrag.engine.rag.workspace.lifecycle import await_shared_cleanup

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AnswerModelRuntimeSettings:
    model_roles: ModelRoleSettings
    web_search_api_key: str | None
    query_image_limit: int


class AnswerModelRuntimeClosedError(RuntimeError):
    """Raised when Answer tries to create a model after shutdown."""


class AnswerModelRuntime:
    """Lazily construct and close every model used by Answer execution."""

    def __init__(
        self,
        *,
        settings: AnswerModelRuntimeSettings,
        scheduler: ModelScheduler,
        telemetry: Telemetry,
        answer_image_policy: Callable[[ModelProfile], AnswerImagePolicy],
        vlm_image_policy: Callable[[ModelProfile], AnswerImagePolicy],
        vlm_profile: Callable[[], ModelProfile],
    ) -> None:
        self._settings = settings
        self._scheduler = scheduler
        self._telemetry = telemetry
        self._answer_image_policy = answer_image_policy
        self._vlm_image_policy = vlm_image_policy
        self._vlm_profile = vlm_profile
        self._answer_synthesizers: dict[ModelProfile, AnswerSynthesizer] = {}
        self._answer_model: CompletionModel | None = None
        self._tool_models: dict[ModelRole, ToolModel] = {}
        self._vlm_model: CompletionModel | None = None
        self._web_search: ExaSearch | None = None
        self._closed = False
        self._close_task: asyncio.Task[asyncio.CancelledError | None] | None = None

    def answer_synthesizer(self, profile: ModelProfile) -> AnswerSynthesizer:
        self._ensure_open()
        if cached := self._answer_synthesizers.get(profile):
            return cached
        synthesizer = AnswerSynthesizer(
            model_func=None,
            image_policy=self._answer_image_policy(profile),
            model_profile=profile,
            context_policy=CONTEXT_POLICY,
        )
        if self._answer_model is None:
            self._answer_model = CompletionModel(
                self._settings.model_roles.resolve("query"),
                scheduler=self._scheduler,
                telemetry=self._telemetry,
            )
        synthesizer.model_func = self._answer_model
        self._answer_synthesizers[profile] = synthesizer
        return synthesizer

    def tool_model(self, role: ModelRole) -> ToolModel:
        """Return the configured tool wrapper for a selected child/model role."""
        self._ensure_open()
        if role not in self._tool_models:
            self._tool_models[role] = ToolModel(
                self._settings.model_roles.resolve(role),
                scheduler=self._scheduler,
                telemetry=self._telemetry,
            )
        return self._tool_models[role]

    def query_tool_model(self) -> ToolModel:
        return self.tool_model("query")

    def vlm_func(self) -> Callable[..., Any]:
        self._ensure_open()
        if self._vlm_model is None:
            self._vlm_model = CompletionModel(
                self._settings.model_roles.resolve("vlm"),
                scheduler=self._scheduler,
                telemetry=self._telemetry,
            )
        return self._vlm_model

    def query_image_describer(self) -> QueryImageDescriber:
        profile = self._vlm_profile()
        return QueryImageDescriber(
            vlm_func=self.vlm_func() if profile.supports_images else None,
            max_images=self._settings.query_image_limit if profile.supports_images else 0,
            image_policy=self._vlm_image_policy(profile),
        )

    def web_search(self) -> ExaSearch | None:
        self._ensure_open()
        key = self._settings.web_search_api_key
        if not key:
            return None
        if self._web_search is None:
            self._web_search = ExaSearch(key)
        return self._web_search

    def new_highlight_model(self) -> tuple[CompletionModel, Telemetry]:
        self._ensure_open()
        return (
            CompletionModel(
                self._settings.model_roles.resolve("keyword"),
                scheduler=self._scheduler,
                telemetry=self._telemetry,
            ),
            self._telemetry,
        )

    async def aclose(self) -> None:
        close_task = self._close_task
        if close_task is None:
            self._closed = True
            close_task = asyncio.create_task(
                self._close_components(),
                name="answer-model-runtime-close",
            )
            self._close_task = close_task
        resource_cancellation = await await_shared_cleanup(close_task)
        if resource_cancellation is not None:
            raise resource_cancellation

    async def _close_components(self) -> asyncio.CancelledError | None:
        components = (
            *self._tool_models.values(),
            self._answer_model,
            self._vlm_model,
            self._web_search,
        )
        self._tool_models.clear()
        self._answer_model = None
        self._vlm_model = None
        self._web_search = None
        self._answer_synthesizers.clear()
        cancellation: asyncio.CancelledError | None = None
        for component in components:
            if component is None:
                continue
            try:
                await component.aclose()
            except asyncio.CancelledError as exc:
                cancellation = cancellation or exc
            except Exception:
                logger.warning("Failed to close Answer model runtime component", exc_info=True)
        return cancellation

    def _ensure_open(self) -> None:
        if self._closed:
            raise AnswerModelRuntimeClosedError("Answer model runtime is closed")


__all__ = [
    "AnswerModelRuntime",
    "AnswerModelRuntimeClosedError",
    "AnswerModelRuntimeSettings",
]
