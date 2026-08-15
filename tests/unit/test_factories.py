# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for root settings mapping and AI model factories."""

import operator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest
from dlightrag_ai.capacity import ModelProfile
from dlightrag_ai.completion import CompletionModel
from dlightrag_ai.providers.base import CompletionOutput
from dlightrag_ai.settings import ModelSettings
from dlightrag_ai.structured import StructuredOutput
from pydantic import BaseModel, ConfigDict

from dlightrag.config import (
    DlightragConfig,
    EmbeddingConfig,
    LLMConfig,
    LLMRolesConfig,
    ModelCapacityOverrideConfig,
    ModelConfig,
    RerankConfig,
)
from dlightrag.model_settings import (
    embedding_settings,
    model_profile_for_role,
    model_settings_for_role,
    rerank_scoring_model_settings,
    rerank_settings,
)


class DemoPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str


DEMO_STRUCTURED_OUTPUT = StructuredOutput(name="demo_plan", schema=DemoPlan)


class CapturingProvider:
    supports_native_json_schema: bool = False

    def __init__(self, seen: dict[str, Any], *, supports_native_json_schema: bool = False) -> None:
        self.seen = seen
        self.supports_native_json_schema = supports_native_json_schema

    async def complete(self, **kwargs: Any) -> str:
        self.seen.update(kwargs)
        return '{"answer": "ok"}'

    def stream(self, **kwargs: Any):  # pragma: no cover - not used
        raise AssertionError("stream should not be called")

    async def aclose(self) -> None:
        return None


def _capture_provider(
    monkeypatch: pytest.MonkeyPatch,
    *,
    supports_native_json_schema: bool = False,
) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        "dlightrag_ai.completion.get_provider",
        lambda *_args, **_kwargs: CapturingProvider(
            seen,
            supports_native_json_schema=supports_native_json_schema,
        ),
    )
    return seen


def _embedding_config() -> EmbeddingConfig:
    return EmbeddingConfig(
        provider="voyage",
        model="voyage-multimodal-3.5",
        api_key="sk-test",
        startup_probe=False,
    )


def test_root_maps_explicit_keyless_role_to_immutable_ai_settings() -> None:
    config = DlightragConfig(
        llm=LLMConfig(
            default=ModelConfig(model="default-model", api_key="default-key"),
            roles=LLMRolesConfig(
                query=ModelConfig(
                    model="local-query",
                    api_key=None,
                    base_url="http://host.docker.internal:8888/v1",
                    model_kwargs={"reasoning": {"enabled": False}},
                )
            ),
        ),
        embedding=_embedding_config(),
    )

    settings = model_settings_for_role(config, "query")

    assert settings.model == "local-query"
    assert settings.api_key is None
    assert settings.base_url == "http://host.docker.internal:8888/v1"
    with pytest.raises(TypeError):
        operator.setitem(settings.model_kwargs["reasoning"], "enabled", True)


def test_root_resolves_model_profiles_independently_per_role() -> None:
    config = DlightragConfig(
        llm=LLMConfig(
            default=ModelConfig(
                model="z-ai/glm-5.2",
                base_url="https://openrouter.ai/api/v1",
            ),
            roles=LLMRolesConfig(
                extract=ModelConfig(
                    model="deepseek-v4-flash",
                    base_url="https://api.deepseek.com",
                    api_key=None,
                ),
                query=ModelConfig(
                    model="private-query",
                    base_url="http://localhost:8888/v1",
                    api_key=None,
                ),
            ),
        ),
        model_capacity_overrides=[
            ModelCapacityOverrideConfig(
                provider="openai",
                model="private-query",
                base_url="http://localhost:8888/v1",
                context_window_tokens=262_144,
                max_input_tokens=200_000,
                max_output_tokens=32_768,
                supports_tools=True,
                supports_reasoning=True,
            )
        ],
        embedding=_embedding_config(),
    )

    assert model_profile_for_role(config, "keyword").max_output_tokens == 262_144
    assert model_profile_for_role(config, "extract").max_output_tokens == 393_216
    query = model_profile_for_role(config, "query")
    assert query.context_window_tokens == 262_144
    assert query.max_input_tokens == 200_000
    assert query.supports_tools is True


def test_root_consults_adapter_profile_before_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_profile = ModelProfile(context_window_tokens=123_456, supports_tools=True)
    metadata = MagicMock(return_value=adapter_profile)
    monkeypatch.setattr("dlightrag.model_settings.get_adapter_model_profile", metadata)
    config = DlightragConfig(
        llm=LLMConfig(
            default=ModelConfig(
                model="z-ai/glm-5.2",
                base_url="https://openrouter.ai/api/v1",
            )
        ),
        embedding=_embedding_config(),
    )

    profile = model_profile_for_role(config, "query")

    assert profile is adapter_profile
    metadata.assert_called_once_with(
        "openai",
        model="z-ai/glm-5.2",
        base_url="https://openrouter.ai/api/v1",
    )


def test_root_override_short_circuits_adapter_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = MagicMock(side_effect=AssertionError("override must win before adapter loading"))
    monkeypatch.setattr("dlightrag.model_settings.get_adapter_model_profile", metadata)
    config = DlightragConfig(
        llm=LLMConfig(default=ModelConfig(model="private-model")),
        model_capacity_overrides=[
            ModelCapacityOverrideConfig(
                provider="openai",
                model="private-model",
                context_window_tokens=200_000,
            )
        ],
        embedding=_embedding_config(),
    )

    profile = model_profile_for_role(config, "query")

    assert profile.context_window_tokens == 200_000
    metadata.assert_not_called()


def test_model_fingerprint_canonicalizes_endpoint_without_retaining_url() -> None:
    from dlightrag_ai.fingerprints import model_fingerprint

    first = model_fingerprint(
        ModelSettings(
            provider="openai",
            model="model-a",
            base_url="HTTPS://API.EXAMPLE.COM:443/v1/../v1/?token=secret",
        )
    )
    second = model_fingerprint(
        ModelSettings(
            provider="openai",
            model="model-a",
            base_url="https://api.example.com/v1",
        )
    )

    assert first == second
    assert first.provider == "openai"
    assert first.model == "model-a"
    assert first.endpoint_fingerprint is not None
    assert "example.com" not in first.endpoint_fingerprint


async def test_ai_completion_model_owns_provider_telemetry_and_lifecycle(monkeypatch) -> None:
    class Provider:
        closed = False

        async def complete(self, **kwargs: Any) -> CompletionOutput:
            assert kwargs["model"] == "model-a"
            return CompletionOutput("answer", usage_details={"input_tokens": 3})

        async def aclose(self) -> None:
            self.closed = True

    class Observation:
        updates: list[dict[str, Any]] = []

        def update(self, **kwargs: Any) -> None:
            self.updates.append(kwargs)

    class Telemetry:
        observation = Observation()
        capture_sensitive_data = False

        @asynccontextmanager
        async def observe(self, name: str, **kwargs: Any):
            assert name == "llm_model-a"
            assert kwargs["as_type"] == "generation"
            yield self.observation

    provider = Provider()
    monkeypatch.setattr(
        "dlightrag_ai.completion.get_provider",
        lambda *_args, **_kwargs: provider,
    )
    model = CompletionModel(
        ModelSettings(provider="openai", model="model-a", api_key="key"),
        telemetry=Telemetry(),
    )

    result = await model(messages=[{"role": "user", "content": "question"}])
    await model.aclose()

    assert result == "answer"
    assert provider.closed is True
    assert Telemetry.observation.updates == [
        {
            "output": {"text_length": 6},
            "usage_details": {"input_tokens": 3},
            "cost_details": None,
        }
    ]


def test_root_maps_embedding_settings_into_ai_factory(monkeypatch) -> None:
    from dlightrag_ai import embedding

    config = DlightragConfig(
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="embed-key",
            dim=768,
            input_modality="multimodal",
            asymmetric="require",
        ),
        embedding_request_timeout=45,
    )
    provider = MagicMock()
    provider.request_headers.return_value = {}
    monkeypatch.setattr(embedding, "get_embed_provider", lambda _name: provider)

    settings = embedding_settings(config)
    model = embedding.create_embedding_model(settings)

    assert settings.timeout == 45
    assert model.provider is provider
    assert model.dim == 768


def test_root_maps_rerank_settings_to_immutable_ai_value() -> None:
    config = DlightragConfig(
        rerank=RerankConfig(
            strategy="voyage_reranker",
            model="rerank-2.5",
            api_key="rerank-key",
            input_modality="multimodal",
            score_threshold=0.42,
            max_concurrency=3,
            batch_size=5,
            model_kwargs={"truncation": {"enabled": False}},
        ),
        embedding=_embedding_config(),
    )

    settings = rerank_settings(config)

    assert settings.strategy == "voyage_reranker"
    assert settings.score_threshold == 0.42
    assert settings.max_concurrency == 3
    assert settings.batch_size == 5
    assert config.rerank.model_kwargs == {"truncation": {"enabled": False}}
    assert "model_kwargs" not in type(settings).__slots__


def test_chat_rerank_scoring_settings_preserve_model_kwargs_and_temperature() -> None:
    config = DlightragConfig(
        rerank=RerankConfig(
            strategy="chat_llm_reranker",
            provider="openai",
            model="scoring-model",
            api_key="key",
            temperature=None,
            model_kwargs={"reasoning": {"enabled": False}},
        ),
        embedding=_embedding_config(),
    )

    scoring = rerank_scoring_model_settings(config)

    assert scoring.model_kwargs == {"reasoning": {"enabled": False}}
    assert scoring.temperature == 0.0


def test_rerank_scoring_settings_ignore_unrelated_role_overrides() -> None:
    config = DlightragConfig(
        llm=LLMConfig(
            default=ModelConfig(model="default-model", api_key="default-key"),
            roles=LLMRolesConfig(
                query=ModelConfig(model="query-model", api_key="query-key"),
                vlm=ModelConfig(model="vlm-model", api_key="vlm-key"),
            ),
        ),
        rerank=RerankConfig(strategy="chat_llm_reranker"),
        embedding=_embedding_config(),
    )

    assert rerank_scoring_model_settings(config).model == "default-model"


@pytest.mark.parametrize("api_key", ["", "   "])
def test_incomplete_independent_reranker_falls_back_to_default(api_key: str) -> None:
    config = DlightragConfig(
        llm=LLMConfig(default=ModelConfig(model="default-model", api_key="default-key")),
        rerank=RerankConfig(
            strategy="chat_llm_reranker",
            provider="openai",
            model="incomplete-reranker",
            api_key=api_key,
        ),
        embedding=_embedding_config(),
    )

    assert rerank_scoring_model_settings(config).model == "default-model"


async def test_structured_output_uses_openai_json_schema(monkeypatch) -> None:
    seen = _capture_provider(monkeypatch)
    model = CompletionModel(
        ModelSettings(provider="openai", model="gpt-5.4-mini", api_key="sk-test")
    )

    await model(
        messages=[{"role": "user", "content": "hi"}],
        structured_output=DEMO_STRUCTURED_OUTPUT,
    )

    response_format = seen["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "demo_plan"
    assert response_format["json_schema"]["strict"] is True


async def test_structured_output_auto_uses_json_object_for_compatible_endpoint(
    monkeypatch,
) -> None:
    seen = _capture_provider(monkeypatch)
    model = CompletionModel(
        ModelSettings(
            provider="openai",
            model="deepseek-v4-flash",
            api_key="sk-test",
            base_url="https://api.deepseek.com",
        )
    )

    await model(
        messages=[{"role": "user", "content": "hi"}],
        structured_output=DEMO_STRUCTURED_OUTPUT,
    )

    assert seen["response_format"] == {"type": "json_object"}


async def test_explicit_json_schema_overrides_custom_endpoint(monkeypatch) -> None:
    seen = _capture_provider(monkeypatch)
    model = CompletionModel(
        ModelSettings(
            provider="openai",
            model="schema-model",
            api_key="sk-test",
            base_url="https://llm.example.test/v1",
            structured_output="json_schema",
        )
    )

    await model(
        messages=[{"role": "user", "content": "hi"}],
        structured_output=DEMO_STRUCTURED_OUTPUT,
    )

    assert seen["response_format"]["type"] == "json_schema"
    assert seen["response_format"]["json_schema"]["name"] == "demo_plan"


@pytest.mark.parametrize(
    ("provider", "model_name"),
    [("anthropic", "claude-sonnet-4"), ("gemini", "gemini-2.5-flash")],
)
async def test_native_provider_auto_uses_json_schema(
    monkeypatch,
    provider,
    model_name,
) -> None:
    seen = _capture_provider(monkeypatch, supports_native_json_schema=True)
    model = CompletionModel(ModelSettings(provider=provider, model=model_name, api_key="sk-test"))

    await model(
        messages=[{"role": "user", "content": "hi"}],
        structured_output=DEMO_STRUCTURED_OUTPUT,
    )

    assert seen["response_format"]["type"] == "json_schema"
    assert seen["response_format"]["json_schema"]["name"] == "demo_plan"


async def test_openai_strict_schema_failure_retries_json_object(monkeypatch) -> None:
    seen: list[dict[str, Any]] = []

    class Provider:
        supports_native_json_schema = False

        async def complete(self, **kwargs: Any) -> str:
            seen.append(kwargs)
            if kwargs["response_format"]["type"] == "json_schema":
                raise RuntimeError("strict schemas unsupported")
            return '{"answer": "ok"}'

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(
        "dlightrag_ai.completion.get_provider",
        lambda *_args, **_kwargs: Provider(),
    )
    model = CompletionModel(
        ModelSettings(provider="openai", model="local-model", api_key="sk-test")
    )

    await model(
        messages=[{"role": "user", "content": "hi"}],
        structured_output=DEMO_STRUCTURED_OUTPUT,
    )

    assert [call["response_format"]["type"] for call in seen] == [
        "json_schema",
        "json_object",
    ]
