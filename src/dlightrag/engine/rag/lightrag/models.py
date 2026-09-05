# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""LightRAG adapters for AI-owned chat model lifecycles."""

import asyncio
import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from lightrag import RoleLLMConfig
from lightrag.utils import EmbeddingFunc, TruncatedResponse

from dlightrag.engine.ai.completion import CompletionModel
from dlightrag.engine.ai.scheduler import ModelScheduler
from dlightrag.engine.ai.settings import EmbeddingSettings, ModelRoleSettings
from dlightrag.engine.ai.telemetry import NOOP_TELEMETRY, Telemetry

logger = logging.getLogger(__name__)


def adapt_completion_for_lightrag(completion: Callable[..., Any]) -> Callable[..., Any]:
    """Bridge a messages-first AI model to LightRAG's prompt-first contract."""

    async def wrapper(
        prompt: str,
        *,
        system_prompt: str | None = None,
        hashing_kv: Any = None,  # noqa: ARG001
        history_messages: list[dict[str, Any]] | None = None,
        keyword_extraction: bool = False,  # noqa: ARG001
        enable_cot: bool = False,  # noqa: ARG001
        **kwargs: Any,
    ) -> Any:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        result = await completion(messages=messages, **kwargs)
        if getattr(result, "stop_reason", None) == "length":
            return TruncatedResponse(str(result))
        return result

    return wrapper


def build_lightrag_embedding(
    settings: EmbeddingSettings,
    embedder: Any,
) -> EmbeddingFunc:
    """Expose an AI embedding model through LightRAG's NumPy contract."""

    async def embed_func(texts: list[str], *, context: str = "document") -> Any:
        embed_context = "query" if context == "query" else "document"
        result = await embedder.embed_texts(texts, context=embed_context)
        return np.array(result)

    return EmbeddingFunc(
        embedding_dim=settings.dim,
        max_token_size=settings.max_token_size,
        func=embed_func,
        model_name=settings.model,
        supports_asymmetric=embedder.supports_asymmetric,
    )


class LightRagChatModels:
    """Own default and explicit role models exposed through LightRAG contracts."""

    def __init__(
        self,
        *,
        models: list[CompletionModel],
        default_func: Callable[..., Any],
        role_configs: dict[str, Any] | None,
    ) -> None:
        self._models = models
        self.default_func = default_func
        self.role_configs = role_configs

    @classmethod
    async def acreate(
        cls,
        settings: ModelRoleSettings,
        *,
        scheduler: ModelScheduler,
        telemetry: Telemetry = NOOP_TELEMETRY,
    ) -> LightRagChatModels:
        """Build all role models, closing earlier providers if construction fails."""
        models: list[CompletionModel] = []
        try:
            default_model = CompletionModel(
                settings.default,
                scheduler=scheduler,
                telemetry=telemetry,
            )
            models.append(default_model)
            default_func = adapt_completion_for_lightrag(default_model)

            role_configs: dict[str, Any] = {}
            for role, role_settings in settings.overrides.items():
                model = CompletionModel(
                    role_settings,
                    scheduler=scheduler,
                    telemetry=telemetry,
                )
                models.append(model)
                role_configs[role] = RoleLLMConfig(
                    func=adapt_completion_for_lightrag(model),
                    timeout=int(role_settings.timeout),
                    metadata={
                        "binding": role_settings.provider,
                        "model": role_settings.model,
                        "host": role_settings.base_url,
                    },
                )
            return cls(
                models=models,
                default_func=default_func,
                role_configs=role_configs or None,
            )
        except BaseException:
            try:
                await _close_chat_models(models)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Failed to close partially built chat models", exc_info=True)
            raise

    async def aclose(self) -> None:
        """Close every provider client while allowing sibling closes to finish."""
        await _close_chat_models(self._models)


async def _close_chat_models(models: list[CompletionModel]) -> None:
    results = await asyncio.gather(
        *(model.aclose() for model in models),
        return_exceptions=True,
    )
    failures: list[Exception] = []
    for result in results:
        if isinstance(result, asyncio.CancelledError):
            raise result
        if isinstance(result, Exception):
            failures.append(result)
        elif isinstance(result, BaseException):
            raise result
    if failures:
        raise ExceptionGroup("failed to close LightRAG chat models", failures)


__all__ = [
    "LightRagChatModels",
    "adapt_completion_for_lightrag",
    "build_lightrag_embedding",
]
