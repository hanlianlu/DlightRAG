# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Request-local resource inputs, manifest entries, locators, and results.

Full resource bytes never enter model context; only bounded read/inspection
results derived from these types are exposed to the model.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
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
    """Immutable answer resource: inline bytes, an inert HTTPS link, or a loader.

    Exactly one of ``content``, ``url``, or ``loader`` is supplied by the caller.
    Links stay inert until an explicit read materializes them under full SSRF
    revalidation. ``loader`` is an authorized, request-local async callable used
    for durable server-owned bytes (e.g. prior Web attachments) that must stay
    lazy: the registry invokes it only when the model reads or inspects the
    resource, so no path or provider locator is ever exposed.
    """

    filename: str | None = None
    content: bytes | None = None
    url: str | None = None
    declared_mime: str | None = None
    loader: Callable[[], Awaitable[bytes]] | None = None


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
    """Structural, human-readable locator for a returned text window.

    ``start``/``end`` are 1-based line numbers and always describe the physical
    lines the window covers. When a single line is larger than one observation
    budget it is split into character sub-windows on that one line; ``char_start``
    and ``char_end`` then carry the 1-based inclusive character span within the
    line. They stay ``None`` for whole-line windows.
    """

    unit: Literal["line"]
    start: int
    end: int
    char_start: int | None = None
    char_end: int | None = None


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
