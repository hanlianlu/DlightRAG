# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Run-scoped Resource inputs, manifest entries, locators, and results.

Full resource bytes never enter model context; only bounded read/view
results derived from these types are exposed to the model.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal

EXTRACTION_TEXT = "text"


class ResourceRegistryError(Exception):
    """Base error for run-scoped Resource Registry failures."""


class ResourceAdmissionError(ResourceRegistryError):
    """Raised when a resource violates count or byte admission limits."""


class ResourceNotFoundError(ResourceRegistryError):
    """Raised when an unknown resource id is read or materialized."""


class ResourceCursorError(ResourceRegistryError):
    """Raised when a continuation cursor is unknown or bound to another read."""


class RenderedReadTargetError(ResourceRegistryError):
    """Raised when ``rendered=true`` names a target that is not a Web Resource."""

    def __init__(self, resource_id: str) -> None:
        super().__init__(
            f"rendered=true reads a URL or a Web Resource; {resource_id} is not a Web Resource"
        )


class NoAdmittedBytesError(ResourceRegistryError):
    """Raised when a Web Resource holds no bytes of its own to copy.

    Nothing may fetch or render them for the copy, so the message names the call that makes them.
    """

    def __init__(self, resource_id: str, *, rendered: bool) -> None:
        if rendered:
            message = (
                f"{resource_id} holds only the Agent Browser's rendering, which is not bytes the "
                'Resource admitted; open the page with browser(action="navigate", url=...), '
                'capture it with browser(action="capture"), and materialize the capture\'s '
                "resource_id"
            )
        else:
            message = (
                f"{resource_id} holds no admitted bytes yet; read(resource_id={resource_id!r}) "
                "acquires them, then materialize copies them"
            )
        super().__init__(message)


class ResourceDecodeError(ResourceRegistryError):
    """Raised when resource bytes are not decodable, mismatched text."""


class ResourceNotConvertedError(ResourceRegistryError):
    """Raised when text needs a conversion view this Run may not build.

    An adopted Resource reads text only through the view stored with it:
    converting it here would record a view the Run that registered it never had.
    """

    def __init__(self, filename: str, media_type: str | None = None) -> None:
        super().__init__(f"{filename} has no stored conversion view")
        self.filename = filename
        self.media_type = media_type


@dataclass(frozen=True)
class ResourceInput:
    """Immutable answer resource: inline bytes, an inert HTTPS link, or a loader.

    Exactly one of ``content``, ``url``, or ``loader`` is supplied by the caller.
    Links stay inert until an explicit read materializes them under full SSRF
    revalidation. ``loader`` is an authorized, execution-local async callable used
    for durable server-owned bytes (e.g. prior Web attachments) that must stay
    lazy: the registry invokes it only when the model reads or views the
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
    """Opaque, run-scoped reference to an viewable visual region."""

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
    evidence_available: bool = True
    note: str | None = None
    rendered: bool = False
    """Whether the text is the Web Resource's rendered representation, not its snapshot."""


#: The handle families this system mints. A durable handle is whatever its minter
#: declares it to be: the Resource registry mints ``res-…`` for prepared, fetched,
#: evidence-backed, and spilled bytes, and publication mints ``artifact-…`` for a
#: Published Artifact. Declaring them here is what lets the alias binder, the registry's
#: mint, and publication's mint agree without any of them matching on a literal, and it
#: is why a handle a Tool taught the model stays readable after adoption whatever family
#: it belongs to.
PREPARED_RESOURCE_HANDLE_PREFIX = "res-"
PUBLISHED_ARTIFACT_HANDLE_PREFIX = "artifact-"
RESOURCE_HANDLE_PREFIXES: tuple[str, ...] = (
    PREPARED_RESOURCE_HANDLE_PREFIX,
    PUBLISHED_ARTIFACT_HANDLE_PREFIX,
)


def is_resource_handle(value: str) -> bool:
    """Return whether one string is a durable Resource handle this system mints."""
    return value.startswith(RESOURCE_HANDLE_PREFIXES)


__all__ = [
    "EXTRACTION_TEXT",
    "PREPARED_RESOURCE_HANDLE_PREFIX",
    "PUBLISHED_ARTIFACT_HANDLE_PREFIX",
    "RESOURCE_HANDLE_PREFIXES",
    "is_resource_handle",
    "NoAdmittedBytesError",
    "ResourceAdmissionError",
    "ResourceCursorError",
    "ResourceDecodeError",
    "ResourceInput",
    "ResourceManifestEntry",
    "ResourceNotConvertedError",
    "ResourceNotFoundError",
    "RenderedReadTargetError",
    "ResourceReadResult",
    "ResourceRegistryError",
    "TextWindowLocator",
    "VisualHandle",
]
