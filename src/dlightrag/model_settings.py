# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Map root Pydantic configuration into provider-neutral AI settings."""

from dlightrag_ai.settings import (
    MODEL_ROLE_NAMES,
    EmbeddingSettings,
    ModelRole,
    ModelRoleSettings,
    ModelSettings,
    RerankSettings,
)

from dlightrag.config import DlightragConfig, ModelConfig, RerankConfig


def _has_explicit_auth_setting(config: ModelConfig | RerankConfig) -> bool:
    if "api_key" not in config.model_fields_set:
        return False
    return config.api_key is None or bool(config.api_key.strip())


def model_settings_from_config(config: ModelConfig) -> ModelSettings:
    """Snapshot one root model block as immutable AI settings."""
    return ModelSettings(
        provider=config.provider,
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        structured_output=config.structured_output,
        temperature=config.temperature,
        timeout=config.timeout,
        max_retries=config.max_retries,
        model_kwargs=config.model_kwargs,
        agentic_model_kwargs=config.agentic_model_kwargs,
    )


def model_settings_for_role(config: DlightragConfig, role: ModelRole) -> ModelSettings:
    """Resolve one complete role override, otherwise snapshot the default model."""
    return model_role_settings(config).resolve(role)


def model_role_settings(config: DlightragConfig) -> ModelRoleSettings:
    """Snapshot the default model and every explicitly complete role override."""
    overrides: dict[ModelRole, ModelSettings] = {
        role: model_settings_from_config(role_config)
        for role in MODEL_ROLE_NAMES
        if (role_config := getattr(config.llm.roles, role)) is not None
        and _has_explicit_auth_setting(role_config)
    }
    return ModelRoleSettings(
        default=model_settings_from_config(config.llm.default),
        overrides=overrides,
    )


def rerank_scoring_model_settings(config: DlightragConfig) -> ModelSettings:
    """Resolve the independent chat reranker model or the default chat model."""
    rerank = config.rerank
    if rerank.provider and rerank.model and _has_explicit_auth_setting(rerank):
        return ModelSettings(
            provider=rerank.provider,
            model=rerank.model,
            api_key=rerank.api_key,
            base_url=rerank.base_url,
            temperature=rerank.temperature or 0.0,
            model_kwargs=rerank.model_kwargs,
        )
    return model_settings_from_config(config.llm.default)


def embedding_settings(config: DlightragConfig) -> EmbeddingSettings:
    """Snapshot root embedding configuration as immutable AI settings."""
    embedding = config.embedding
    return EmbeddingSettings(
        provider=embedding.provider,
        model=embedding.model,
        api_key=embedding.api_key,
        base_url=embedding.base_url,
        dim=embedding.dim,
        max_token_size=embedding.max_token_size,
        input_modality=embedding.input_modality,
        asymmetric=embedding.asymmetric,
        startup_probe=embedding.startup_probe,
        timeout=float(config.embedding_request_timeout),
    )


def rerank_settings(config: DlightragConfig) -> RerankSettings:
    """Snapshot root rerank configuration as immutable core settings."""
    rerank = config.rerank
    return RerankSettings(
        enabled=rerank.enabled,
        strategy=rerank.strategy,
        model=rerank.model,
        api_key=rerank.api_key,
        base_url=rerank.base_url,
        input_modality=rerank.input_modality,
        score_threshold=rerank.score_threshold,
        max_concurrency=rerank.max_concurrency,
        batch_size=rerank.batch_size,
    )


__all__ = [
    "embedding_settings",
    "model_role_settings",
    "model_settings_for_role",
    "model_settings_from_config",
    "rerank_settings",
    "rerank_scoring_model_settings",
]
