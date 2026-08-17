# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Request-local answer resource registry and bounded reads."""

from dlightrag.answer.resources.models import (
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
from dlightrag.answer.resources.registry import ResourceRegistry, UrlTextFallback

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
