# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The Agent Browser port: what a Run asks of its leased browser, and what it answers.

The Agent Browser is a deployment capability reached through a port (ADR 0032). The
engine states what it needs here and an adapter provides it, so neither Resource
reading nor Run policy imports a browser driver.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import SecretStr

if TYPE_CHECKING:
    from dlightrag.engine.answer.agent_browser.accounts import AgentAccountsBinding


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


#: The most files one call delivers from its pages; a page that downloads more has the rest refused.
MAX_DOWNLOADS_PER_CALL = 4


@dataclass(frozen=True, slots=True)
class PageLimits:
    """The bounds an Agent Page works within, from the settings of its Run."""

    navigation_timeout: float
    """Seconds a navigation, a wait for text, a screenshot, or one download's save may take."""
    action_timeout: float
    """Seconds one element action, one snapshot, or one find may take."""
    settle_timeout: float
    """Seconds, in all, an acting call waits for the page to load and go quiet."""
    snapshot_depth: int
    max_download_bytes: int
    """The most one download may hold; the transfer stops once it is over."""


@dataclass(frozen=True, slots=True)
class PageState:
    """Where an Agent Session's active page is."""

    url: str
    title: str
    """One line of at most 200 characters."""


@dataclass(frozen=True, slots=True)
class DownloadedFile:
    """One file a page downloaded, as the browser delivered it."""

    url: str
    """As the browser reports it: an HTTP(S), ``blob:``, or ``data:`` URL."""
    suggested_filename: str
    content: bytes


@dataclass(frozen=True, slots=True)
class DownloadRefusal:
    """One download the Agent Page did not deliver, and why."""

    suggested_filename: str
    reason: Literal["too_large", "timeout", "failed", "limit"]


@dataclass(frozen=True, slots=True)
class PageDialog:
    """One JavaScript dialog a page showed, and whether it was told yes."""

    kind: str
    """alert, confirm, beforeunload, or prompt."""
    message: str
    """One line of at most 200 characters."""
    accepted: bool


@dataclass(frozen=True, slots=True)
class PageEvents:
    """What happened in the Agent Page around one call, besides the call itself."""

    downloads: tuple[DownloadedFile, ...] = ()
    refused_downloads: tuple[DownloadRefusal, ...] = ()
    dialogs: tuple[PageDialog, ...] = ()
    """The dialogs the page showed, at most five."""
    new_page: bool = False
    """A popup or a new tab became the active page."""
    returned: bool = False
    """The active page closed, and an earlier page is active again."""
    closed: bool = False
    """The last page closed: nothing is open until the next navigate."""
    http_status: int | None = None
    """The active page's last main-frame navigation response during the call."""


@dataclass(frozen=True, slots=True)
class PageObservation:
    """What an acting call left: the page, what happened around it, and its snapshot."""

    page: PageState | None
    """None when the call closed the last page."""
    events: PageEvents
    snapshot: str | None


@dataclass(frozen=True, slots=True)
class FoundElements:
    """The elements of the active page that match a query."""

    page: PageState
    events: PageEvents
    lines: tuple[str, ...]
    """The snapshot lines that match, at most ``limit``, each cut to 300 characters."""
    total: int
    """How many lines matched, shown or not."""


@dataclass(frozen=True, slots=True)
class PageScreenshot:
    """The active page's pixels."""

    page: PageState
    events: PageEvents
    png: bytes


@dataclass(frozen=True, slots=True)
class PageCapture:
    """The active page as the browser serializes it now."""

    page: PageState
    events: PageEvents
    html: bytes
    """UTF-8 of the serialized DOM, as the page stands."""


@dataclass(frozen=True, slots=True)
class UploadFile:
    """One workspace file offered to a page's file input."""

    name: str
    mime_type: str
    content: bytes


#: What stands in for a filled password in every text a page or a mail yields.
PASSWORD_MASK = "********"  # noqa: S105 - what replaces a password, not a password


class FilledPasswords:
    """The passwords DlightRAG filled into one Agent Session's pages in this Run (ADR 0034).

    Every text a page or a mail yields passes through it before the tool sees it. A generated
    password holds only characters that HTML, JSON and URL encoding leave unchanged, so the one
    spelling it has is all there is to find.
    """

    def __init__(self) -> None:
        self._values: set[str] = set()

    def add(self, password: SecretStr) -> None:
        # The empty value a cleanup fills back in is nothing to hide, and replacing it would put
        # the mask between every character of every text.
        if value := password.get_secret_value():
            self._values.add(value)

    def __bool__(self) -> bool:
        return bool(self._values)

    def found_in(self, text: str) -> bool:
        return any(value in text for value in self._values)

    def redact(self, text: str) -> str:
        """``text`` with every filled password replaced by the mask."""
        for value in self._values:
            text = text.replace(value, PASSWORD_MASK)
        return text

    def redact_bytes(self, data: bytes) -> bytes:
        """The same over UTF-8, for a page's serialized HTML."""
        for value in self._values:
            data = data.replace(value.encode(), PASSWORD_MASK.encode())
        return data

    def __repr__(self) -> str:
        return f"FilledPasswords({len(self._values)} filled)"


@dataclass(frozen=True, slots=True)
class CredentialForm:
    """What a sign-up form's fields say before DlightRAG fills them."""

    password_limit: int | None
    """The smallest positive ``maxlength`` of the password fields, or None when none has one."""
    email: str
    username: str
    """The fields' values now, empty for a field the call named no ref for."""


@dataclass(frozen=True, slots=True)
class CredentialFill:
    """One value DlightRAG fills into one field of the active page."""

    ref: str
    kind: Literal["email", "username", "password"]
    value: SecretStr


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
    "no_page",
    "page_closed",
    "page_lost",
    "stale_ref",
    "not_actionable",
    "invalid_key",
    "no_history",
    "not_file_input",
    "wait_timeout",
    "action_failed",
    "wrong_site",
    "not_password_field",
    "not_text_field",
    "field_rejected",
    "password_shown",
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

#: What the model reads when a ``browser`` call fails. A reason missing here keeps its
#: sentence above, which a call and a render say in the same words. Driver error text
#: enters only through ``action_failed``, and only its first line.
_PAGE_MESSAGES: dict[AgentBrowserFailure, str] = {
    "no_page": (
        'No page is open in this Agent Session. Start with browser(action="navigate", '
        "url=...); a Run that resumed after an interruption starts with no open page."
    ),
    "page_closed": 'The page closed. Start again with browser(action="navigate", url=...).',
    "page_lost": (
        "This Agent Session's page was lost when the Agent Browser disconnected. Navigate "
        "again; earlier refs no longer apply."
    ),
    "disconnected": (
        "The Agent Browser disconnected, and every open page of this Run was lost. Navigate "
        "again to start a fresh page."
    ),
    "stale_ref": (
        "No element on the current page has ref {ref}. Refs come from the latest snapshot or "
        "find of this page; call snapshot or find again."
    ),
    "not_actionable": (
        "The element {ref} could not be {verb} within {seconds:g} seconds; it may be hidden, "
        "disabled, or covered. Call snapshot to see the page again."
    ),
    "invalid_key": (
        "{key} is not a key the browser knows. Use names such as Enter, Tab, Escape, "
        "ArrowDown, or Control+A."
    ),
    "no_history": "There is no earlier page in this page's history.",
    "not_file_input": "Element {ref} is not a file input and did not open a file chooser.",
    "wait_timeout": '"{text}" did not {change} within {seconds:g} seconds.',
    "action_failed": "The browser could not {action} {target}: {detail}.",
    "busy": "Every Agent Browser is in use by other Runs, so no page was opened. Try again later.",
    "unreachable": "The Agent Browser is unreachable, so no page was opened.",
    "wrong_site": (
        "Field {ref} is not on an https page of {site}, so nothing was filled: register and "
        "login fill only the fields of the page's own site."
    ),
    "not_password_field": "Element {ref} is not a password field, so nothing was filled.",
    "not_text_field": "Element {ref} is not a text or email field, so nothing was filled.",
    "field_rejected": (
        "The page changed the value filled into {ref}, so nothing was recorded and the fields "
        "were cleared."
    ),
    "password_shown": "The page shows a filled password as text, so no screenshot was taken.",
}


class AgentBrowserError(Exception):
    """A render or a ``browser`` call the Agent Browser could not give, with the sentence the model reads."""

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


def page_failure(reason: AgentBrowserFailure, **fields: object) -> AgentBrowserError:
    """The error for ``reason`` as a ``browser`` call reports it.

    ``fields`` are the ones its sentence names: ``ref``, ``verb``, ``key``, ``text`` and
    ``change``, ``action``, ``target`` and ``detail``, ``seconds``, ``site``. A reason a
    render reports in the same words is the render's error.
    """
    template = _PAGE_MESSAGES.get(reason)
    if template is None:
        return browser_failure(reason, **fields)
    return AgentBrowserError(reason, template.format(**fields))


class AgentPage(Protocol):
    """An Agent Session's anonymous context in the Run's browser, driven as one active page.

    A call acts on the active page and answers with what it left. A popup or a new tab
    becomes the active page after the call that opened it, or at the start of the next call
    that names no ref when it opened later; a call that names a ref acts on the page the ref
    came from.
    """

    def current_url(self) -> str | None: ...

    async def navigate(self, url: str) -> PageObservation: ...

    async def back(self) -> PageObservation: ...

    async def snapshot(self) -> PageObservation: ...

    async def find(self, query: str, *, limit: int) -> FoundElements: ...

    async def wait(
        self, *, text: str | None, text_gone: str | None, seconds: float | None
    ) -> PageObservation: ...

    async def click(self, ref: str) -> PageObservation: ...

    async def type_text(self, ref: str, text: str, *, submit: bool) -> PageObservation: ...

    async def select(self, ref: str, values: tuple[str, ...]) -> PageObservation: ...

    async def press(self, key: str, *, ref: str | None) -> PageObservation: ...

    async def scroll(
        self, *, direction: Literal["up", "down"], ref: str | None
    ) -> PageObservation: ...

    async def upload(self, ref: str, files: tuple[UploadFile, ...]) -> PageObservation: ...

    async def screenshot(self, *, full_page: bool) -> PageScreenshot: ...

    async def capture(self) -> PageCapture: ...

    async def credential_form(
        self,
        *,
        site: str,
        password_refs: tuple[str, ...],
        email_ref: str | None,
        username_ref: str | None,
    ) -> CredentialForm:
        """Check that every ref is a field of the right kind on a page of ``site``, and read
        what the form says: the password limit, and what the email and username fields hold."""
        ...

    async def fill_credentials(
        self, fills: tuple[CredentialFill, ...], *, site: str
    ) -> PageObservation:
        """Fill each value into its field, once every ref has passed the checks of
        ``credential_form``; a failure clears what was filled and fills nothing more."""
        ...

    async def aclose(self) -> None:
        """Close the context. Failing, it logs; it never raises."""
        ...


class LeasedBrowser(Protocol):
    """One browser a Run holds until it is closed."""

    async def render(
        self, url: str, *, navigation_timeout: float, settle_timeout: float
    ) -> RenderedPage: ...

    async def open_page(self, limits: PageLimits, passwords: FilledPasswords) -> AgentPage:
        """Open an Agent Page that adds the passwords it fills to ``passwords`` and redacts
        them from every text it returns."""
        ...

    async def aclose(self) -> None:
        """Disconnect the browser, then release its lease."""
        ...


class BrowserProvider(Protocol):
    """The port that leases a Run's browser endpoint."""

    async def lease(self, holder: BrowserHolder, *, wait_seconds: float) -> LeasedBrowser: ...

    async def aclose(self) -> None: ...


class BrowserLeases(Protocol):
    """The shared record of which Run holds each endpoint, which a provider claims through."""

    async def register_endpoints(self, endpoints: Sequence[str]) -> None: ...

    async def claim(
        self, holder: BrowserHolder, endpoints: Sequence[str], exclude: Sequence[str] = ()
    ) -> str | None: ...

    async def release(self, holder: BrowserHolder, endpoint: str) -> None: ...


@dataclass(frozen=True, slots=True)
class AgentBrowserSettings:
    """The Agent Browser a deployment configures.

    The pool's endpoints, the egress proxy, whether Chromium is launched inside its own
    sandbox, and the connect timeout are what its provider is built from. The other values
    are the waits and bounds a Run keeps: for a free browser, for a page to load, settle
    and answer an action, for a download, and before it gives an idle browser back.
    """

    endpoints: tuple[str, ...]
    egress_proxy: str
    chromium_sandbox: bool
    connect_timeout_seconds: float
    lease_wait_seconds: float
    navigation_timeout_seconds: float
    settle_timeout_seconds: float
    action_timeout_seconds: float
    snapshot_depth: int
    max_download_bytes: int
    idle_release_seconds: float


@dataclass(frozen=True, slots=True)
class AgentBrowserBinding:
    """The Agent Browser a deployment composed: its provider, the settings it runs under, and
    the Agent Accounts its Runs may register and log in with (ADR 0034), when it allows them."""

    provider: BrowserProvider
    settings: AgentBrowserSettings
    accounts: AgentAccountsBinding | None = None


__all__ = [
    "MAX_DOWNLOADS_PER_CALL",
    "PASSWORD_MASK",
    "AgentBrowserBinding",
    "AgentBrowserError",
    "AgentBrowserFailure",
    "AgentBrowserSettings",
    "BrowserHolder",
    "BrowserLeases",
    "BrowserProvider",
    "AgentPage",
    "CredentialFill",
    "CredentialForm",
    "DownloadRefusal",
    "DownloadedFile",
    "FilledPasswords",
    "FoundElements",
    "PageLimits",
    "LeasedBrowser",
    "PageCapture",
    "PageDialog",
    "PageEvents",
    "PageObservation",
    "PageScreenshot",
    "PageState",
    "RenderedPage",
    "UploadFile",
    "browser_failure",
    "page_failure",
]
