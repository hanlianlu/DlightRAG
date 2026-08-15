# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Provider registry with lazy SDK loading."""

import importlib
from typing import Any, cast

from dlightrag_ai.capacity import ModelProfile
from dlightrag_ai.providers.base import CompletionProvider

_PROVIDER_CLASSES: dict[str, str] = {
    "openai": "dlightrag_ai.providers.openai_compatible.OpenAICompatibleProvider",
    "anthropic": "dlightrag_ai.providers.anthropic_native.AnthropicProvider",
    "gemini": "dlightrag_ai.providers.gemini_native.GeminiProvider",
}


def _provider_class(provider: str) -> type[CompletionProvider]:
    qualified = _PROVIDER_CLASSES.get(provider)
    if qualified is None:
        available = ", ".join(sorted(_PROVIDER_CLASSES))
        raise ValueError(f"Unknown provider {provider!r}. Available: {available}")
    module_path, class_name = qualified.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return cast(type[CompletionProvider], getattr(module, class_name))


def get_adapter_model_profile(
    provider: str,
    *,
    model: str,
    base_url: str | None,
) -> ModelProfile | None:
    """Read optional static model facts from the selected provider adapter."""
    provider_class = _provider_class(provider)
    return provider_class.declared_model_profile(model=model, base_url=base_url)


def get_provider(
    provider: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 120.0,
    max_retries: int = 3,
) -> CompletionProvider:
    """Lazy-load and instantiate a provider by string name."""
    cls = cast(Any, _provider_class(provider))
    return cls(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)


__all__ = ["get_adapter_model_profile", "get_provider"]
