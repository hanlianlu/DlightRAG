# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The Agent Browser port: what a Run asks of its leased browser, and what it answers.

The Agent Browser is a deployment capability reached through a port (ADR 0032). The
engine states what it needs here and an adapter provides it, so neither Resource
reading nor Run policy imports a browser driver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True, slots=True)
class BrowserHolder:
    """The Run claim a browser lease is fenced by: the fields of its ``RunSession``."""

    owner_id: str
    run_id: str
    worker_id: str
    fencing_epoch: int


@dataclass(frozen=True, slots=True)
class RenderedPage:
    """One page as the browser serialized it after its scripts ran."""

    requested_url: str
    final_url: str
    html: bytes
    """UTF-8 of the serialized DOM, whatever the page's own ``<meta charset>`` says."""
    status: int | None
    """The last main-frame navigation response, when the browser saw one."""


type AgentBrowserFailure = Literal[
    "not_configured",
    "busy",
    "unreachable",
    "disconnected",
    "timeout",
    "navigation_failed",
    "http_status",
    "download",
    "too_large",
    "no_text",
    "final_url_refused",
]

#: What the model reads when a render fails, one stable sentence per reason. Page URLs
#: and driver error text never enter it: a message names the reason and, where the
#: reason has one, its number or its network error token.
_PUBLIC_MESSAGES: dict[AgentBrowserFailure, str] = {
    "not_configured": "The Agent Browser is not available in this Run.",
    "busy": (
        "Every Agent Browser is in use by other Runs, so this page was not rendered. "
        "Try again later or work from the direct read."
    ),
    "unreachable": "The Agent Browser is unreachable, so this page was not rendered.",
    "disconnected": (
        "The Agent Browser disconnected while rendering; the next rendered read starts a "
        "fresh browser."
    ),
    "timeout": "The page did not finish loading within {seconds:g} seconds in the Agent Browser.",
    "navigation_failed": "The Agent Browser could not load the page{detail}.",
    "http_status": "The page answered HTTP {status} to the Agent Browser.",
    "download": "The URL starts a download, not a page; a rendered read cannot read it.",
    "too_large": "The rendered page exceeds {limit} bytes.",
    "no_text": "The rendered page produced no text.",
    "final_url_refused": (
        "The rendered page ended at a URL this deployment does not admit; nothing was admitted."
    ),
}


class AgentBrowserError(Exception):
    """A render the Agent Browser could not give, with the sentence the model reads."""

    def __init__(self, reason: AgentBrowserFailure, public_message: str) -> None:
        super().__init__(public_message)
        self.reason: AgentBrowserFailure = reason
        self.public_message = public_message


def browser_failure(reason: AgentBrowserFailure, **fields: object) -> AgentBrowserError:
    """The error for ``reason``, with the fields its sentence names (``seconds``, ``status``,
    ``limit``, or ``detail`` for a network error token)."""
    if reason == "navigation_failed":
        detail = fields.get("detail")
        fields = {"detail": f" ({detail})" if detail else ""}
    return AgentBrowserError(reason, _PUBLIC_MESSAGES[reason].format(**fields))


type BrowserSandbox = Literal["chromium", "unavailable"]
"""Whether Chromium ran inside its own sandbox for one lease."""


class LeasedBrowser(Protocol):
    """One browser a Run holds until it is closed."""

    @property
    def sandbox(self) -> BrowserSandbox: ...

    async def render(
        self, url: str, *, navigation_timeout: float, settle_timeout: float, max_bytes: int
    ) -> RenderedPage: ...

    async def aclose(self) -> None:
        """Disconnect the browser, then release its lease."""
        ...


class BrowserProvider(Protocol):
    """The port that leases a Run's browser endpoint."""

    async def lease(self, holder: BrowserHolder, *, wait_seconds: float) -> LeasedBrowser: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True, slots=True)
class AgentBrowserSettings:
    """How a Run uses its browser; the pool itself is the adapter's configuration."""

    lease_wait_seconds: float
    navigation_timeout_seconds: float
    settle_timeout_seconds: float
    idle_release_seconds: float
    max_page_bytes: int


__all__ = [
    "AgentBrowserError",
    "AgentBrowserFailure",
    "AgentBrowserSettings",
    "BrowserHolder",
    "BrowserProvider",
    "BrowserSandbox",
    "LeasedBrowser",
    "RenderedPage",
    "browser_failure",
]
