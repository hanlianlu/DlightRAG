# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Request-local resource inputs, manifest entries, locators, and results.

Full resource bytes never enter model context; only bounded read/inspection
results derived from these types are exposed to the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

EXTRACTION_TEXT = "text"


class ResourceRegistryError(Exception):
    """Base error for request-local resource registry failures."""


class ResourceAdmissionError(ResourceRegistryError):
    """Raised when a resource violates count or byte admission limits."""


class ResourceNotFoundError(ResourceRegistryError):
    """Raised when an unknown resource id is read or materialized."""


class ResourceCursorError(ResourceRegistryError):
    """Raised when a continuation cursor is unknown or bound to another read."""


class ResourceDecodeError(ResourceRegistryError):
    """Raised when resource bytes are not decodable, mismatched text."""


@dataclass(frozen=True)
class ResourceInput:
    """Immutable answer resource: either inline bytes or an inert HTTPS link.

    Exactly one of ``content`` or ``url`` is supplied by the caller. Links stay
    inert until an explicit read materializes them under full SSRF revalidation.
    """

    filename: str | None = None
    content: bytes | None = None
    url: str | None = None
    declared_mime: str | None = None


@dataclass(frozen=True)
class ResourceManifestEntry:
    """Compact, model-safe description of a registered resource."""

    resource_id: str
    filename: str | None
    declared_mime: str | None
    source: Literal["bytes", "link"]
    byte_size: int | None


@dataclass(frozen=True)
class TextWindowLocator:
    """Structural, human-readable locator for a returned text window."""

    unit: Literal["line"]
    start: int
    end: int


@dataclass(frozen=True)
class VisualHandle:
    """Opaque, request-local reference to an inspectable visual region."""

    handle_id: str
    label: str | None = None


@dataclass(frozen=True)
class ResourceReadResult:
    """Bounded evidence returned for one resource read."""

    resource_id: str
    locator: TextWindowLocator | None
    content: str
    extraction_status: str
    has_more: bool
    next_cursor: str | None
    visual_handles: tuple[VisualHandle, ...] = field(default_factory=tuple)


__all__ = [
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
]
