# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Map root Pydantic configuration into provider-neutral AI settings."""

from dlightrag_ai.capacity import ModelProfile
from dlightrag_ai.catalog import resolve_model_profile
from dlightrag_ai.fingerprints import (
    ModelFingerprint,
    model_fingerprint,
    normalized_endpoint_fingerprint,
)
from dlightrag_ai.providers import get_adapter_model_profile
from dlightrag_ai.settings import (
    MODEL_ROLE_NAMES,
    EmbeddingSettings,
    ModelRole,
    ModelRoleSettings,
    ModelSettings,
    RerankSettings,
)

from dlightrag.config import (
    DlightragConfig,
    ModelCapacityOverrideConfig,
    ModelConfig,
    RerankConfig,
)


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


def _profile_from_override(config: ModelCapacityOverrideConfig) -> ModelProfile:
    return ModelProfile(
        context_window_tokens=config.context_window_tokens,
        max_input_tokens=config.max_input_tokens,
        max_output_tokens=config.max_output_tokens,
        supports_images=config.supports_images,
        supports_tools=config.supports_tools,
        supports_reasoning=config.supports_reasoning,
    )


def _override_fingerprint(config: ModelCapacityOverrideConfig) -> ModelFingerprint:
    return ModelFingerprint(
        provider=config.provider,
        model=config.model,
        endpoint_fingerprint=normalized_endpoint_fingerprint(config.base_url),
    )


def model_profile_for_settings(
    config: DlightragConfig,
    settings: ModelSettings,
    *,
    adapter_profile: ModelProfile | None = None,
) -> ModelProfile:
    """Resolve immutable capacity facts for one fully resolved model endpoint."""
    fingerprint = model_fingerprint(settings)
    override = next(
        (
            _profile_from_override(candidate)
            for candidate in config.model_capacity_overrides
            if _override_fingerprint(candidate) == fingerprint
        ),
        None,
    )
    if override is not None:
        return resolve_model_profile(fingerprint, override=override)
    if adapter_profile is None:
        adapter_profile = get_adapter_model_profile(
            settings.provider,
            model=settings.model,
            base_url=settings.base_url,
        )
    return resolve_model_profile(
        fingerprint,
        adapter_profile=adapter_profile,
    )


def model_profile_for_role(config: DlightragConfig, role: ModelRole) -> ModelProfile:
    """Resolve one role's model settings and independent capacity profile."""
    return model_profile_for_settings(config, model_settings_for_role(config, role))


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
    "model_profile_for_role",
    "model_profile_for_settings",
    "model_role_settings",
    "model_settings_for_role",
    "model_settings_from_config",
    "rerank_settings",
    "rerank_scoring_model_settings",
]
