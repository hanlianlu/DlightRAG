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
from dlightrag_rag.settings import RagSettings
from dlightrag_rag.workspaces import normalize_workspace

from dlightrag.access import (
    AccessRule,
    AccessSettings,
    AuthenticationSettings,
)
from dlightrag.answer.capabilities import (
    AnswerCapabilitySettings,
    AnswerImagePolicySettings,
)
from dlightrag.answer.executor import (
    AnswerExecutorSettings,
    AnswerResourceSettings,
)
from dlightrag.answer.highlights import SemanticHighlightSettings
from dlightrag.answer.model_runtime import AnswerModelRuntimeSettings
from dlightrag.answer.resources.images import MAX_QUERY_IMAGES
from dlightrag.config import (
    DlightragConfig,
    ModelCapacityOverrideConfig,
    ModelConfig,
    RerankConfig,
)
from dlightrag.services.retrieval import RetrievalSettings


def access_settings(config: DlightragConfig) -> AccessSettings:
    """Snapshot root authorization configuration into immutable Access settings."""
    return AccessSettings(
        mode=config.access_control.mode,
        rules=tuple(
            AccessRule(
                claim=rule.claim,
                value=rule.value,
                workspaces=tuple(
                    workspace if workspace == "*" else normalize_workspace(workspace)
                    for workspace in rule.workspaces
                ),
                actions=tuple(rule.actions),
            )
            for rule in config.access_control.rules
        ),
    )


def authentication_settings(
    config: DlightragConfig,
    *,
    audience: str | None = None,
) -> AuthenticationSettings:
    """Snapshot root bearer configuration into immutable Access settings."""
    configured_audience = audience if audience is not None else config.jwt_audience
    if isinstance(configured_audience, list):
        resolved_audience: str | tuple[str, ...] | None = tuple(configured_audience)
    else:
        resolved_audience = configured_audience
    return AuthenticationSettings(
        mode=config.auth_mode,
        api_token=config.api_auth_token,
        jwt_verification_key=config.jwt_verification_key,
        jwt_jwks_url=config.jwt_jwks_url,
        jwt_issuer=config.jwt_issuer,
        jwt_audience=resolved_audience,
        jwt_algorithm=config.jwt_algorithm,
    )


def answer_capability_settings(config: DlightragConfig) -> AnswerCapabilitySettings:
    """Snapshot root Answer capability and image policy configuration."""
    answer = config.answer
    return AnswerCapabilitySettings(
        images=AnswerImagePolicySettings(
            max_images=int(answer.max_images),
            max_total_bytes=answer.image_max_total_bytes,
            max_bytes_per_image=answer.image_max_bytes,
            max_pixels=answer.image_max_pixels,
            max_px=answer.image_max_px,
            min_px=answer.image_min_px,
            quality=answer.image_quality,
            min_quality=answer.image_min_quality,
        ),
        web_search_enabled=bool(config.web_search.api_key),
        rerank_enabled=config.rerank.enabled,
        rerank_strategy=config.rerank.strategy,
    )


def semantic_highlight_settings(config: DlightragConfig) -> SemanticHighlightSettings:
    """Snapshot root semantic highlight configuration into Answer settings."""
    highlights = config.citations.highlights
    return SemanticHighlightSettings(
        enabled=highlights.enabled,
        timeout=highlights.timeout,
        max_concurrency=highlights.max_concurrency,
        batch_size=highlights.batch_size,
        max_input_chars=highlights.max_input_chars,
        cache_size=highlights.cache_size,
    )


def answer_model_runtime_settings(config: DlightragConfig) -> AnswerModelRuntimeSettings:
    """Snapshot Answer model roles and Web-search configuration."""
    return AnswerModelRuntimeSettings(
        model_roles=model_role_settings(config),
        web_search_api_key=config.web_search.api_key,
        query_image_limit=MAX_QUERY_IMAGES,
    )


def answer_resource_settings(config: DlightragConfig) -> AnswerResourceSettings:
    """Snapshot Answer attachment and current-image resource limits."""
    answer = config.answer
    return AnswerResourceSettings(
        max_attachments=answer.max_attachments,
        max_attachment_bytes=answer.max_attachment_bytes,
        max_total_attachment_bytes=answer.max_total_attachment_bytes,
        image_max_bytes=answer.image_max_bytes,
        image_max_pixels=answer.image_max_pixels,
    )


def answer_executor_settings(config: DlightragConfig) -> AnswerExecutorSettings:
    """Snapshot durable Answer execution policy."""
    return AnswerExecutorSettings(
        default_top_k=config.top_k,
        default_chunk_top_k=config.chunk_top_k,
        max_agent_turns=config.max_agent_turns,
        semantic_highlights=semantic_highlight_settings(config),
    )


def retrieval_settings(config: DlightragConfig) -> RetrievalSettings:
    """Snapshot caller-awaited Retrieval policy."""
    return RetrievalSettings(
        default_top_k=config.top_k,
        default_chunk_top_k=config.chunk_top_k,
        timeout_seconds=config.retrieval_timeout,
        query_image_limit=MAX_QUERY_IMAGES,
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


def rag_settings(config: DlightragConfig) -> RagSettings:
    """Snapshot root product configuration into immutable RAG settings."""
    docling = config.parser_sidecars.docling
    return RagSettings(
        model_roles=model_role_settings(config),
        embedding=embedding_settings(config),
        rerank=rerank_settings(config),
        rerank_scoring_model=rerank_scoring_model_settings(config),
        read_only=config.is_reader,
        input_root=config.input_dir_path,
        parser_rules=config.parser_rules,
        docling_active=docling is not None,
        docling_code_formula_preset=docling.code_formula_preset if docling else None,
        parser_min_image_pixel=config.parser_sidecars.vlm.min_image_pixel,
        chunk_options=config.parser.chunk_options,
        extraction_language=config.extraction.language,
        entity_type_prompt_file=config.extraction.entity_type_prompt_file,
        entity_extraction_use_json=config.extraction.use_json,
        chunk_p_token_size=config.chunk_p_token_size,
        kg_entity_types=tuple(config.kg_entity_types),
        kg_chunk_pick_method=config.kg_chunk_pick_method,
        max_entity_tokens=config.max_entity_tokens,
        max_relation_tokens=config.max_relation_tokens,
        max_total_tokens=config.max_total_tokens,
        direct_visual_top_k=config.direct_visual_top_k,
        rrf_k=config.rrf_k,
        thumb_cache_size=config.visual_assets.thumb_cache_size,
        thumb_max_px=config.visual_assets.thumb_max_px,
        ingestion_replace_default=config.ingestion_replace_default,
        retain_remote_source_files=config.retain_remote_source_files,
        url_ingest_max_bytes=config.url_ingest_max_bytes,
        url_ingest_private_host_allowlist=tuple(config.url_ingest_private_host_allowlist),
        blob_connection_string=config.blob_connection_string,
        azure_sas_expiry=config.azure_sas_expiry,
        s3_presign_expiry=config.s3_presign_expiry,
        s3_region=config.s3_region,
        rag_pipeline_max_async=config.rag_pipeline_max_async,
        embedding_func_max_async=config.embedding_func_max_async,
        embedding_batch_num=config.embedding_batch_num,
        max_parallel_insert=config.max_parallel_insert,
        max_parallel_parse_native=config.max_parallel_parse_native,
        max_parallel_parse_mineru=config.max_parallel_parse_mineru,
        max_parallel_parse_docling=config.max_parallel_parse_docling,
        max_parallel_analyze=config.max_parallel_analyze,
        queue_size_parse=config.queue_size_parse,
        queue_size_analyze=config.queue_size_analyze,
        queue_size_insert=config.queue_size_insert,
    )


__all__ = [
    "embedding_settings",
    "model_profile_for_role",
    "model_profile_for_settings",
    "model_role_settings",
    "model_settings_for_role",
    "model_settings_from_config",
    "rag_settings",
    "rerank_settings",
    "rerank_scoring_model_settings",
]
