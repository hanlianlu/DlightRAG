# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Run-scoped answer Resource Registry and bounded reads."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import (
        EXTRACTION_TEXT,
        ResourceAdmissionError,
        ResourceCursorError,
        ResourceDecodeError,
        ResourceInput,
        ResourceManifestEntry,
        ResourceNotFoundError,
        ResourceReadResult,
        ResourceRegistryError,
        TextWindowLocator,
        VisualHandle,
    )
    from .registry import ResourceRegistry, UrlTextFallback

__all__ = [
    "EXTRACTION_TEXT",
    "ResourceAdmissionError",
    "ResourceCursorError",
    "ResourceDecodeError",
    "ResourceInput",
    "ResourceManifestEntry",
    "ResourceNotFoundError",
    "ResourceReadResult",
    "ResourceRegistry",
    "ResourceRegistryError",
    "TextWindowLocator",
    "UrlTextFallback",
    "VisualHandle",
]


def __getattr__(name: str) -> Any:
    # Resource data contracts do not require the registry's conversion backends.
    if name in {"ResourceRegistry", "UrlTextFallback"}:
        from . import registry

        return getattr(registry, name)
    if name in {
        "EXTRACTION_TEXT",
        "ResourceAdmissionError",
        "ResourceCursorError",
        "ResourceDecodeError",
        "ResourceInput",
        "ResourceManifestEntry",
        "ResourceNotFoundError",
        "ResourceReadResult",
        "ResourceRegistryError",
        "TextWindowLocator",
        "VisualHandle",
    }:
        from . import models

        return getattr(models, name)
    raise AttributeError(name)
