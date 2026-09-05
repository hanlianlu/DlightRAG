# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for LightRAG-aligned LLM role configuration."""

from types import MappingProxyType
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from dlightrag.application.config import DlightragConfig
from dlightrag.engine.ai.scheduler import ModelScheduler
from dlightrag.engine.ai.settings import (
    EmbeddingSettings,
    ModelRoleOverrides,
    ModelRoleSettings,
    ModelSettings,
    ModelsSettings,
)


def _cfg() -> DlightragConfig:
    return DlightragConfig(
        models=ModelsSettings(
            embedding=EmbeddingSettings(
                provider="voyage",
                model="voyage-multimodal-3.5",
                api_key="sk-test",
                dim=1024,
                startup_probe=False,
            ),
            chat=ModelRoleSettings(
                default=ModelSettings(
                    provider="openai", model="default-model", api_key="default-key"
                ),
                roles=ModelRoleOverrides(
                    keyword=ModelSettings(
                        provider="openai",
                        model="keyword-model",
                        api_key=None,
                    ),
                    query=ModelSettings(provider="openai", model="incomplete-query"),
                ),
            ),
        ),
    )


def test_root_exposes_only_complete_role_overrides() -> None:
    roles = _cfg().models.chat

    assert isinstance(roles.overrides, MappingProxyType)
    assert tuple(roles.overrides) == ("keyword",)
    assert roles.resolve("keyword").model == "keyword-model"
    assert roles.resolve("query").model == "default-model"


async def test_rag_chat_bundle_adapts_explicit_roles_and_closes_models(monkeypatch) -> None:
    from dlightrag.engine.ai.settings import ModelRoleOverrides, ModelRoleSettings, ModelSettings
    from dlightrag.engine.rag.lightrag import models as lightrag_models

    class FakeCompletionModel:
        instances: list[FakeCompletionModel] = []

        def __init__(self, settings: ModelSettings, **_kwargs: Any) -> None:
            self.settings = settings
            self.messages: list[dict[str, Any]] | None = None
            self.closed = False
            self.instances.append(self)

        async def __call__(self, messages: list[dict[str, Any]], **_kwargs: Any) -> str:
            self.messages = messages
            return self.settings.model

        async def aclose(self) -> None:
            self.closed = True

    monkeypatch.setattr(lightrag_models, "CompletionModel", FakeCompletionModel)
    scheduler = ModelScheduler(max_concurrency=2)
    bundle = await lightrag_models.LightRagChatModels.acreate(
        ModelRoleSettings(
            default=ModelSettings(provider="openai", model="default-model"),
            roles=ModelRoleOverrides(
                keyword=ModelSettings(
                    provider="openai",
                    model="keyword-model",
                    api_key=None,
                    base_url="https://models.example/v1",
                    timeout=30,
                )
            ),
        ),
        scheduler=scheduler,
    )

    result = await bundle.default_func(
        "question",
        system_prompt="system",
        history_messages=[{"role": "assistant", "content": "earlier"}],
    )

    assert result == "default-model"
    assert FakeCompletionModel.instances[0].messages == [
        {"role": "system", "content": "system"},
        {"role": "assistant", "content": "earlier"},
        {"role": "user", "content": "question"},
    ]
    assert bundle.role_configs is not None
    assert tuple(bundle.role_configs) == ("keyword",)
    keyword_config = bundle.role_configs["keyword"]
    assert keyword_config.timeout == 30
    assert keyword_config.metadata == {
        "binding": "openai",
        "model": "keyword-model",
        "host": "https://models.example/v1",
    }

    await bundle.aclose()

    assert [model.closed for model in FakeCompletionModel.instances] == [True, True]


async def test_rag_chat_bundle_closes_created_models_when_role_construction_fails(
    monkeypatch,
) -> None:
    from dlightrag.engine.ai.settings import ModelRoleOverrides, ModelRoleSettings, ModelSettings
    from dlightrag.engine.rag.lightrag import models as lightrag_models

    class FakeCompletionModel:
        instances: list[FakeCompletionModel] = []

        def __init__(self, settings: ModelSettings, **_kwargs: Any) -> None:
            if settings.model == "broken-role":
                raise RuntimeError("provider construction failed")
            self.closed = False
            self.instances.append(self)

        async def __call__(self, **_kwargs: Any) -> str:
            return "ok"

        async def aclose(self) -> None:
            self.closed = True

    monkeypatch.setattr(lightrag_models, "CompletionModel", FakeCompletionModel)

    with pytest.raises(RuntimeError, match="provider construction failed"):
        await lightrag_models.LightRagChatModels.acreate(
            ModelRoleSettings(
                default=ModelSettings(provider="openai", model="default-model"),
                roles=ModelRoleOverrides(
                    keyword=ModelSettings(provider="openai", model="broken-role", api_key=None)
                ),
            ),
            scheduler=ModelScheduler(max_concurrency=1),
        )

    assert len(FakeCompletionModel.instances) == 1
    assert FakeCompletionModel.instances[0].closed is True


async def test_rag_chat_bundle_preserves_construction_error_when_cleanup_fails(
    monkeypatch,
) -> None:
    from dlightrag.engine.ai.settings import ModelRoleOverrides, ModelRoleSettings, ModelSettings
    from dlightrag.engine.rag.lightrag import models as lightrag_models

    class FakeCompletionModel:
        def __init__(self, settings: ModelSettings, **_kwargs: Any) -> None:
            if settings.model == "broken-role":
                raise RuntimeError("construction failed")

        async def __call__(self, **_kwargs: Any) -> str:
            return "ok"

        async def aclose(self) -> None:
            raise RuntimeError("cleanup failed")

    monkeypatch.setattr(lightrag_models, "CompletionModel", FakeCompletionModel)

    with pytest.raises(RuntimeError, match="construction failed"):
        await lightrag_models.LightRagChatModels.acreate(
            ModelRoleSettings(
                default=ModelSettings(provider="openai", model="default-model"),
                roles=ModelRoleOverrides(
                    keyword=ModelSettings(provider="openai", model="broken-role", api_key=None)
                ),
            ),
            scheduler=ModelScheduler(max_concurrency=1),
        )


async def test_rag_embedding_adapter_injects_context_and_numpy_shape() -> None:
    from dlightrag.engine.ai.settings import EmbeddingSettings
    from dlightrag.engine.rag.lightrag.models import build_lightrag_embedding

    embedder = MagicMock()
    embedder.supports_asymmetric = True
    embedder.embed_texts = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    settings = EmbeddingSettings(
        provider="openai_compatible",
        model="embed-model",
        dim=3,
        max_token_size=2048,
    )

    embedding_func = build_lightrag_embedding(settings, embedder)
    result = await embedding_func.func(["hello"], context="query")

    assert result.tolist() == [[0.1, 0.2, 0.3]]
    assert embedding_func.embedding_dim == 3
    assert embedding_func.max_token_size == 2048
    assert embedding_func.supports_asymmetric is True
    embedder.embed_texts.assert_awaited_once_with(["hello"], context="query")
