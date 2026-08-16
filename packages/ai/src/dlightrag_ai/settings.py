# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Immutable provider settings consumed by AI model factories."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from dlightrag_ai.contracts import AsymmetricMode, ChatProvider, InputModality

type ModelRole = Literal["extract", "keyword", "query", "vlm"]
MODEL_ROLE_NAMES: tuple[ModelRole, ...] = ("extract", "keyword", "query", "vlm")
type RerankStrategy = Literal[
    "chat_llm_reranker",
    "jina_reranker",
    "aliyun_reranker",
    "local_reranker",
    "voyage_reranker",
    "cohere_reranker",
    "azure_cohere",
]


def freeze_settings_value(value: Any) -> Any:
    """Recursively freeze collection-valued immutable settings."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): freeze_settings_value(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(freeze_settings_value(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(freeze_settings_value(item) for item in value)
    return value


def _frozen_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({key: freeze_settings_value(item) for key, item in value.items()})


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple | frozenset):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ModelSettings:
    """One fully resolved chat-model configuration."""

    provider: ChatProvider
    model: str
    api_key: str | None = None
    base_url: str | None = None
    structured_output: Literal["auto", "json_schema", "json_object"] = "auto"
    temperature: float | None = None
    timeout: float = 240.0
    max_retries: int = 3
    model_kwargs: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    agentic_model_kwargs: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_kwargs", _frozen_mapping(self.model_kwargs))
        object.__setattr__(
            self,
            "agentic_model_kwargs",
            _frozen_mapping(self.agentic_model_kwargs),
        )

    def model_kwargs_copy(self) -> dict[str, Any]:
        """Return a provider-safe mutable copy of ordinary model options."""
        return {key: _thaw(value) for key, value in self.model_kwargs.items()}

    def agentic_model_kwargs_copy(self) -> dict[str, Any]:
        """Return ordinary options overlaid by agent-specific options."""
        return {
            **self.model_kwargs_copy(),
            **{key: _thaw(value) for key, value in self.agentic_model_kwargs.items()},
        }


@dataclass(frozen=True, slots=True)
class ModelRoleSettings:
    """Default model plus the explicitly complete role overrides."""

    default: ModelSettings
    overrides: Mapping[ModelRole, ModelSettings] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "overrides", MappingProxyType(dict(self.overrides)))

    def resolve(self, role: ModelRole) -> ModelSettings:
        """Return a complete role override or the default model."""
        return self.overrides.get(role, self.default)


@dataclass(frozen=True, slots=True)
class EmbeddingSettings:
    """One fully resolved embedding model configuration."""

    provider: Literal["voyage", "gemini", "jina", "openai_compatible", "ollama"]
    model: str
    api_key: str | None = None
    base_url: str | None = None
    dim: int = 1024
    max_token_size: int = 8192
    input_modality: InputModality = "auto"
    asymmetric: AsymmetricMode = "auto"
    startup_probe: bool = True
    timeout: float = 120.0


@dataclass(frozen=True, slots=True)
class RerankSettings:
    """One fully resolved rerank orchestration and transport configuration."""

    enabled: bool = True
    strategy: RerankStrategy = "chat_llm_reranker"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    input_modality: InputModality = "auto"
    score_threshold: float | None = None
    max_concurrency: int = 8
    batch_size: int = 8


__all__ = [
    "EmbeddingSettings",
    "MODEL_ROLE_NAMES",
    "ModelRole",
    "ModelRoleSettings",
    "ModelSettings",
    "RerankSettings",
    "RerankStrategy",
    "freeze_settings_value",
]
