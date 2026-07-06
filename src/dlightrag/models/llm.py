# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Model factory functions.

Uses the provider registry (openai, anthropic, gemini) to build callables.
Provides _adapt_for_lightrag() to bridge to LightRAG's (prompt, system_prompt) signature.
"""

import logging
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np

from dlightrag.config import DlightragConfig, ModelConfig
from dlightrag.models.llm_roles import LIGHTRAG_ROLE_NAMES, model_for_role
from dlightrag.models.providers import get_provider
from dlightrag.models.structured import StructuredOutput

logger = logging.getLogger(__name__)


def _adapt_for_lightrag(completion_func: Callable) -> Callable:
    """Wrap messages-first callable for LightRAG's (prompt, system_prompt, ...) signature."""

    async def wrapper(
        prompt: str,
        *,
        system_prompt: str | None = None,
        hashing_kv: Any = None,  # noqa: ARG001
        history_messages: list[dict[str, Any]] | None = None,
        keyword_extraction: bool = False,  # noqa: ARG001
        enable_cot: bool = False,  # noqa: ARG001
        token_tracker: Any = None,  # noqa: ARG001
        use_azure: bool = False,  # noqa: ARG001
        azure_deployment: str | None = None,  # noqa: ARG001
        api_version: str | None = None,  # noqa: ARG001
        **kwargs: Any,
    ) -> Any:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        return await completion_func(messages=messages, **kwargs)

    return wrapper


def _make_completion_func(cfg: ModelConfig, default_api_key: str | None = None) -> partial:
    """Build a messages-first completion callable from config."""
    api_key = cfg.api_key or default_api_key
    provider = get_provider(
        cfg.provider,
        api_key=api_key,
        base_url=cfg.base_url,
        timeout=cfg.timeout,
        max_retries=cfg.max_retries,
    )

    async def completion_wrapper(messages: list[dict[str, Any]], **kw: Any) -> Any:
        stream = kw.pop("stream", False)
        response_format = kw.pop("response_format", None)
        max_tokens = kw.pop("max_tokens", None)
        structured_output = kw.pop("structured_output", None)
        if structured_output is not None:
            if not isinstance(structured_output, StructuredOutput):
                raise TypeError("structured_output must be a StructuredOutput")
            response_format = response_format or structured_output.response_format_for_provider(
                cfg.provider
            )
        model_kwargs = {**cfg.model_kwargs, **kw}
        if stream:
            return provider.stream(
                messages=messages,
                model=cfg.model,
                temperature=cfg.temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                model_kwargs=model_kwargs,
            )
        try:
            return await provider.complete(
                messages=messages,
                model=cfg.model,
                temperature=cfg.temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                model_kwargs=model_kwargs,
            )
        except Exception:
            if (
                structured_output is not None
                and cfg.provider == "openai"
                and isinstance(response_format, dict)
                and response_format.get("type") == "json_schema"
            ):
                logger.warning(
                    "Strict structured output failed for %s; retrying with json_object",
                    cfg.model,
                    exc_info=True,
                )
                return await provider.complete(
                    messages=messages,
                    model=cfg.model,
                    temperature=cfg.temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    model_kwargs=model_kwargs,
                )
            raise

    from dlightrag.observability import wrap_chat_func

    traced_func = wrap_chat_func(
        completion_wrapper,
        name=f"llm_{cfg.model}",
        model=cfg.model,
        model_parameters={"temperature": cfg.temperature} if cfg.temperature is not None else None,
    )

    return partial(traced_func)


def _queue_managed_completion_func(
    completion_func: Callable,
    *,
    max_async: int,
    timeout: float,
    queue_name: str,
) -> Callable:
    """Apply LightRAG's priority queue wrapper to a messages-first callable."""
    from lightrag.utils import priority_limit_async_func_call

    return priority_limit_async_func_call(
        max_async,
        llm_timeout=timeout,
        queue_name=queue_name,
    )(completion_func)


def get_default_model_func(config: DlightragConfig) -> Callable:
    """Messages-first callable for the configured default LLM."""
    return _make_completion_func(config.llm.default)


def get_extract_model_func(config: DlightragConfig) -> Callable:
    """Messages-first extract callable using role config or default LLM."""
    return _make_completion_func(
        model_for_role(config, "extract"),
        default_api_key=config.llm.default.api_key,
    )


def get_keyword_model_func(config: DlightragConfig) -> Callable:
    """Messages-first keyword callable using role config or default LLM."""
    return _make_completion_func(
        model_for_role(config, "keyword"),
        default_api_key=config.llm.default.api_key,
    )


def get_planner_model_func(config: DlightragConfig, *, bounded: bool = True) -> Callable:
    """Messages-first planner callable using keyword role config or default LLM."""
    cfg = model_for_role(config, "keyword")
    func = _make_completion_func(cfg, default_api_key=config.llm.default.api_key)
    if not bounded:
        return func
    return _queue_managed_completion_func(
        func,
        max_async=config.max_async,
        timeout=cfg.timeout,
        queue_name="DlightRAG planner LLM func",
    )


def get_query_model_func(config: DlightragConfig, *, bounded: bool = True) -> Callable:
    """Messages-first query callable for AnswerEngine.

    Uses ``config.llm.roles.query`` if set, otherwise ``config.llm.default``.
    """
    cfg = model_for_role(config, "query")
    func = _make_completion_func(cfg, default_api_key=config.llm.default.api_key)
    if not bounded:
        return func
    return _queue_managed_completion_func(
        func,
        max_async=config.max_async,
        timeout=cfg.timeout,
        queue_name="DlightRAG query LLM func",
    )


def get_vlm_model_func(config: DlightragConfig, *, bounded: bool = True) -> Callable:
    """Messages-first VLM callable for vlm_parser, multimodal_query, and unified extractor.

    Uses ``config.llm.roles.vlm`` if set, otherwise ``config.llm.default``.
    """
    cfg = model_for_role(config, "vlm")
    func = _make_completion_func(cfg, default_api_key=config.llm.default.api_key)
    if not bounded:
        return func
    return _queue_managed_completion_func(
        func,
        max_async=config.max_async,
        timeout=cfg.timeout,
        queue_name="DlightRAG VLM LLM func",
    )


def _lightrag_adapted(
    factory: Callable[[DlightragConfig], Callable],
) -> Callable[[DlightragConfig], Callable]:
    """Wrap a messages-first model factory to produce a LightRAG-compatible callable."""

    def wrapper(config: DlightragConfig) -> Callable:
        return _adapt_for_lightrag(factory(config))

    wrapper.__doc__ = f"LightRAG-adapted version of {factory.__name__}."
    wrapper.__name__ = f"{factory.__name__}_for_lightrag"
    return wrapper


get_default_model_func_for_lightrag = _lightrag_adapted(get_default_model_func)


def build_role_llm_configs(config: DlightragConfig) -> dict[str, Any] | None:
    """Build LightRAG ``role_llm_configs`` from LightRAG-aligned role names."""
    from lightrag import RoleLLMConfig

    overrides: dict[str, Any] = {}
    for role in LIGHTRAG_ROLE_NAMES:
        role_cfg: ModelConfig | None = getattr(config.llm.roles, role)
        if role_cfg is None:
            continue
        completion = _make_completion_func(role_cfg, default_api_key=config.llm.default.api_key)
        overrides[role] = RoleLLMConfig(
            func=_adapt_for_lightrag(completion),
            timeout=int(role_cfg.timeout),
        )

    return overrides or None


def get_embedding_func(config: DlightragConfig, *, embedder: Any | None = None) -> Any:
    """Build LightRAG EmbeddingFunc from config.

    Uses the provider-aware multimodal embedder so text and image vectors
    share one model space and the LightRAG wrapper can inject query/document
    context for asymmetric providers.
    """
    from lightrag.utils import EmbeddingFunc

    active_embedder: Any = embedder if embedder is not None else get_multimodal_embedder(config)

    cfg = config.embedding

    async def embed_func(texts: list[str], *, context: str = "document") -> np.ndarray:
        embed_context = "query" if context == "query" else "document"
        result = await active_embedder.embed_texts(texts, context=embed_context)
        # LightRAG's EmbeddingFunc validates via result.size — requires numpy
        return np.array(result)

    from dlightrag.observability import wrap_embedding_func

    traced_embed_func = wrap_embedding_func(embed_func, name=f"embed_{cfg.model}")

    return EmbeddingFunc(
        embedding_dim=cfg.dim,
        max_token_size=cfg.max_token_size,
        func=traced_embed_func,
        model_name=cfg.model,
        supports_asymmetric=active_embedder.supports_asymmetric,
    )


def get_multimodal_embedder(config: DlightragConfig) -> Any:
    """Build a provider-aware multimodal embedder for text and images."""
    from dlightrag.models.multimodal_embedding import MultimodalEmbedder
    from dlightrag.models.providers.embed_providers import detect_embed_provider

    cfg = config.embedding
    return MultimodalEmbedder(
        model=cfg.model,
        api_key=cfg.api_key or "",
        base_url=cfg.base_url or "",
        dim=cfg.dim,
        provider=detect_embed_provider(cfg.model, provider=cfg.provider, base_url=cfg.base_url),
        asymmetric=cfg.asymmetric,
        batch_size=config.embedding_func_max_async,
        timeout=float(config.embedding_request_timeout),
    )


def get_chat_rerank_scoring_config(config: DlightragConfig) -> ModelConfig:
    """Return the messages-first model config used by chat_llm_reranker."""
    rc = config.rerank
    if rc.provider and rc.model:
        return ModelConfig(
            provider=rc.provider,
            model=rc.model,
            api_key=rc.api_key,
            base_url=rc.base_url,
            temperature=rc.temperature or 0.0,
            model_kwargs=rc.model_kwargs,
        )
    return config.llm.roles.vlm or config.llm.roles.query or config.llm.default


def get_rerank_func(
    config: DlightragConfig,
    *,
    supports_vision: bool | None = None,
) -> Callable | None:
    """Build multimodal rerank callable from config.

    For chat_llm_reranker: uses an independent rerank model if provider/model
    are set in rerank config. Otherwise it reuses the most capable configured
    chat role in order: vlm, query, then default. When *supports_vision* is
    provided, it is the startup probe result for that selected scoring model.
    """
    from dlightrag.models.rerank import build_rerank_func

    rc = config.rerank
    if not rc.enabled:
        return None

    # For chat_llm_reranker: build scoring callable (messages-first, NOT LightRAG-adapted)
    if rc.strategy == "chat_llm_reranker":
        scoring_cfg = get_chat_rerank_scoring_config(config)
        if supports_vision is not None:
            if rc.input_modality == "auto":
                rc = rc.model_copy(
                    update={"input_modality": "multimodal" if supports_vision else "text"}
                )
            elif rc.input_modality == "multimodal" and not supports_vision:
                raise ValueError(
                    "chat_llm_reranker input_modality=multimodal but the selected scoring "
                    "model does not support image input"
                )

        scoring_func = _make_completion_func(
            scoring_cfg,
            default_api_key=config.llm.default.api_key,
        )
        return build_rerank_func(rc, ingest_func=scoring_func)

    return build_rerank_func(rc)


__all__ = [
    "build_role_llm_configs",
    "get_chat_rerank_scoring_config",
    "get_default_model_func",
    "get_default_model_func_for_lightrag",
    "get_embedding_func",
    "get_extract_model_func",
    "get_keyword_model_func",
    "get_multimodal_embedder",
    "get_planner_model_func",
    "get_query_model_func",
    "get_rerank_func",
    "get_vlm_model_func",
]
