# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Provider registry with lazy SDK loading."""

from __future__ import annotations

import importlib

from dlightrag.models.providers.base import CompletionProvider

_PROVIDER_CLASSES: dict[str, str] = {
    "openai": "dlightrag.models.providers.openai_compat.OpenAICompatProvider",
    "anthropic": "dlightrag.models.providers.anthropic_native.AnthropicProvider",
    "gemini": "dlightrag.models.providers.gemini_native.GeminiProvider",
}


def get_provider(
    provider: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 120.0,
    max_retries: int = 3,
) -> CompletionProvider:
    """Lazy-load and instantiate a provider by string name."""
    qualified = _PROVIDER_CLASSES.get(provider)
    if qualified is None:
        available = ", ".join(sorted(_PROVIDER_CLASSES))
        raise ValueError(f"Unknown provider {provider!r}. Available: {available}")
    module_path, class_name = qualified.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)


__all__ = ["get_provider"]
