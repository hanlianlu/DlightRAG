# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The Agent Browser port: what a Run asks of its leased browser, and what it answers.

The Agent Browser is a deployment capability reached through a port (ADR 0032). The
engine states what it needs here and an adapter provides it, so neither Resource
reading nor Run policy imports a browser driver.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import SecretStr

from dlightrag.engine.answer.agent_browser.accounts import AgentAccountsBinding
from dlightrag.engine.answer.agent_browser.block import Block
from dlightrag.engine.answer.agent_browser.live import LiveBrowserPage, LiveControlState
from dlightrag.engine.answer.agent_browser.passwords import FilledPasswords


@dataclass(frozen=True, slots=True)
class BrowserHolder:
    """The Run claim a browser lease is fenced by: the fields of its ``RunSession``."""

    owner_id: str
    run_id: str
    worker_id: str
    fencing_epoch: int


@dataclass(frozen=True, slots=True)
class StorageState:
    """One origin's web storage, as a move to another browser carries it."""

    local: Mapping[str, str] = field(default_factory=dict)
    """The origin's ``localStorage``; its keys and values, which are always strings."""
    session: Mapping[str, str] = field(default_factory=dict)
    """The origin's ``sessionStorage``, which the context would otherwise lose with its page."""


@dataclass(frozen=True, slots=True)
class SessionState:
    """What a move to another browser carries from the one being left (ADR 0032).

    The session of the page that was being driven: the context's cookies, each origin's web
    storage, and where the page was. What a session is *not* is the page's DOM: a form that
    was half filled is gone with the context it was typed into, and the model reads that and
    fills it again. Contexts are anonymous and this state is never written anywhere but the
    new context; it dies with the Run as the session it came from does.
    """

    url: str
    """Where the page was, so the browser that follows can revisit it; empty when none was open."""
    cookies: tuple[Mapping[str, Any], ...] = ()
    """The context's cookies, in Playwright's own ``add_cookies`` shape."""
    origins: Mapping[str, StorageState] = field(default_factory=dict)
    """Each origin's web storage, by origin, as the pages that were open held it."""


@dataclass(frozen=True, slots=True)
class RenderedPage:
    """One page as the browser serialized it after its scripts ran."""

    requested_url: str
    final_url: str
    html: bytes
    """UTF-8 of the serialized DOM, whatever the page's own ``<meta charset>`` says."""
    status: int | None
    """The last main-frame navigation response, when the browser saw one."""
    headless: bool = True
    """Whether the browser that rendered this page runs without a display."""
    settled: bool = True
    """Whether the page went quiet within the settle wait before it was serialized."""


#: The most files one call delivers from its pages; a page that downloads more has the rest refused.
MAX_DOWNLOADS_PER_CALL = 4
#: The most dialogs one call reports; a page that raises more has the rest counted, not hidden.
MAX_DIALOGS_PER_CALL = 20
#: The most download refusals one call reports; a page that raises more has the rest counted.
MAX_REFUSALS_PER_CALL = 20
#: The grammar of a ref: the name one snapshot line prints for an element — ``eN``, or ``fN``
#: before it for an element in a frame the page holds — which the next call acts on. The
#: adapter that prints a ref and the tool that accepts one read it here, so the two cannot
#: drift apart.
REF_PATTERN = r"(?:f[0-9]+)?e[0-9]+"


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
    """The page's title on one line, cut to 200 characters with the cut marked and counted."""


@dataclass(frozen=True, slots=True)
class DownloadedFile:
    """One file a page downloaded, as the browser delivered it."""

    url: str
    """As the browser reports it: an HTTP(S), ``blob:``, or ``data:`` URL."""
    suggested_filename: str
    content: bytes
    """The file's bytes, with every filled password masked as in a capture's HTML."""


@dataclass(frozen=True, slots=True)
class DownloadRefusal:
    """One download the Agent Page did not deliver, and why."""

    suggested_filename: str
    reason: Literal["too_large", "timeout", "failed", "limit"]


#: What a call may tell a dialog it meets: accept it, or dismiss it.
DialogVerdict = Literal["accept", "dismiss"]


@dataclass(frozen=True, slots=True)
class PageDialog:
    """One JavaScript dialog a page showed, and what became of it."""

    kind: str
    """alert, confirm, beforeunload, or prompt."""
    message: str
    """One line, cut to 200 characters with the cut marked."""
    answer: Literal["accepted", "dismissed", "waiting"]
    """What the dialog was told, or that nobody has answered it and the page is stopped."""
    default: str = ""
    """A prompt's default text, which nothing has typed; empty for every other dialog."""


@dataclass(frozen=True, slots=True)
class PageEvents:
    """What happened in the Agent Page around one call, besides the call itself."""

    downloads: tuple[DownloadedFile, ...] = ()
    refused_downloads: tuple[DownloadRefusal, ...] = ()
    """The downloads the pages started and did not deliver, each with the reason it was refused.

    At most ``MAX_REFUSALS_PER_CALL`` of them, with ``omitted_refusals`` counting the rest: a
    page can start downloads without end, so what the model reads is bounded and says what it
    left out rather than a list that reads as the whole of them.
    """
    omitted_refusals: int = 0
    """How many download refusals ``refused_downloads`` does not hold, past the bound."""
    dialogs: tuple[PageDialog, ...] = ()
    """The dialogs the pages showed since the last call, in the order they showed them.

    At most ``MAX_DIALOGS_PER_CALL`` of them, with ``omitted_dialogs`` counting the rest: a
    page can raise dialogs without end, so what the model reads is bounded and says what it
    left out. The dialog each page is stopped on follows them, and the bound never drops it.
    """
    omitted_dialogs: int = 0
    """How many dialogs the pages showed that ``dialogs`` does not hold, past the bound."""
    new_page: bool = False
    """A popup or a new tab became the active page."""
    returned: bool = False
    """The active page closed, and an earlier page is active again."""
    closed: bool = False
    """The last page closed: nothing is open until the next navigate."""
    http_status: int | None = None
    """The active page's last main-frame navigation response during the call."""
    settled: bool | None = None
    """Whether the page went quiet within the settle wait before this call read it.

    True when it did, False when the wait ran out first — what the call read may be a page
    that was still loading — and None when no wait was made, which claims neither."""


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
    """The snapshot lines that match, at most ``limit``, each cut to 300 characters with the
    cut marked and counted."""
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
    "egress_refused",
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
    "blocked",
    "verification_required",
    "wrong_site",
    "not_password_field",
    "not_text_field",
    "field_rejected",
    "password_held",
    "password_too_long",
    "password_limit_too_small",
    "taken_over",
    "expired",
    "dialog_waiting",
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
    "egress_refused": (
        "This deployment's egress proxy refused the connection, so the site was never "
        "reached. The Agent Browser leaves through that proxy, and it admits public "
        "destinations on ports 80 and 443 only; work from the direct read or another source."
    ),
    "download": "The URL starts a download, not a page; a rendered read cannot read it.",
    "too_large": "The rendered page exceeds {limit} bytes.",
    "no_text": "The rendered page produced no text.",
    "expired": (
        "This live browser session reached the hard cap one session may last, so there is "
        "nothing live to watch. Start the preview again for a fresh session."
    ),
    "final_url_refused": (
        "The rendered page ended at a URL this deployment does not admit; nothing was admitted."
    ),
    "blocked": (
        "The page was refused for being automated: a bot filter answered the Agent Browser "
        "instead of the page, so nothing was read.{detail} Do not try to defeat the check. "
        "A Run whose deployment leases headed browsers can ask for one with headed=true on "
        "the next call, and keeps it for the rest of the Run."
    ),
    "verification_required": (
        "The page shows a human-verification check, so nothing was read.{detail} That path "
        "stops here: never solve, bypass, or outsource the check."
    ),
    "password_too_long": (
        "This site's password field takes at most {limit} characters and {asked} were asked "
        "for; ask for {limit} or fewer."
    ),
    #: The field takes fewer characters than the composition leaves room for, so no ask this
    #: mechanism can honour fits it: there is no shorter password to ask for, and the sentence
    #: says so rather than advising an ask that cannot be made.
    "password_limit_too_small": (
        "This site's password field takes at most {limit} characters, and the shortest password "
        "this mechanism makes is {minimum}: no registration can succeed on this site."
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
        "Field {ref} is not on an https page of {site}, so nothing was filled: an Agent "
        "Account is filled only into the fields of the page's own site."
    ),
    "not_password_field": "Element {ref} is not a password field, so nothing was filled.",
    "not_text_field": "Element {ref} is not a text or email field, so nothing was filled.",
    "field_rejected": (
        "The page changed the value filled into {ref}, so nothing was recorded and the fields "
        "were cleared."
    ),
    "password_held": (
        "The page holds a filled password, so no screenshot was taken. Take it after the form "
        "is submitted."
    ),
    "taken_over": (
        "The person watching this Run has taken control of its browser, so this page action was "
        "refused. It is theirs until they hand it back; the Run goes on and other work is "
        "unaffected."
    ),
    "dialog_waiting": (
        'The page is waiting on a {kind} dialog nobody has answered: "{message}".{detail} '
        'Answer it with dialog="accept" or dialog="dismiss" on the call that reads the page '
        "again; until then the page stays as it is."
    ),
}


class AgentBrowserError(Exception):
    """A render or a ``browser`` call the Agent Browser could not give, with the sentence the model reads."""

    def __init__(self, reason: AgentBrowserFailure, public_message: str) -> None:
        super().__init__(public_message)
        self.reason: AgentBrowserFailure = reason
        self.public_message = public_message


def browser_failure(reason: AgentBrowserFailure, **fields: object) -> AgentBrowserError:
    """The error for ``reason``, with the fields its sentence names (``seconds``, ``limit`` with
    ``asked`` or ``minimum``, ``marker`` for a refusal's evidence, or ``detail`` for a network
    error token)."""
    if reason in ("blocked", "verification_required"):
        marker = fields.get("marker")
        fields = {"detail": f' The page carries "{marker}".' if marker else ""}
    elif reason == "navigation_failed":
        detail = fields.get("detail")
        fields = {"detail": f" ({detail})" if detail else ""}
    return AgentBrowserError(reason, _PUBLIC_MESSAGES[reason].format(**fields))


def refusal_failure(block: Block) -> AgentBrowserError:
    """The error for a page that was refused for being automated, with the evidence it used."""
    return browser_failure(block.reason, marker=block.marker)


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

    async def answer_dialogs(self, verdict: DialogVerdict | None) -> None:
        """Say what the pages' dialogs are answered with for the call that follows.

        ``verdict`` answers both the dialog each page is stopped on now — the one an earlier
        call met and nobody answered — and every dialog the call meets, and each page's own
        waiting dialog is kept until then. ``None`` is the default and answers nothing: a
        dialog the agent has not answered leaves its page waiting, and the call that met it
        reports the dialog rather than deciding for the agent. A call that finds the page it
        acts on waiting and carries no verdict does nothing else.
        """
        ...

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

    async def clear_fields(self, refs: tuple[str, ...], *, site: str) -> None:
        """Empty the fields ``refs`` name, once each is known to be a field of a page of ``site``,
        so that a form DlightRAG filled is not left holding what nothing recorded."""
        ...

    async def aclose(self) -> None:
        """Close the context. Failing, it logs; it never raises."""
        ...

    async def capture_session(self) -> SessionState:
        """What this page's session holds now: its URL, its origins' web storage.

        The per-page half of what a move carries; the context's cookies are the
        lease's (``LeasedBrowser.capture_session``). Read before anything closes: a closed
        context holds nothing.
        """
        ...

    async def restore_session(self, session: SessionState) -> None:
        """Give this page the storage state of a page that was carried over.

        Called once, before the navigation that revisits the URL, so the state is in place
        for the page's own scripts.
        """
        ...


class LeasedBrowser(Protocol):
    """One browser a Run holds until it is closed."""

    @property
    def headless(self) -> bool:
        """Whether this browser runs without a display; a headed lease never goes back."""
        ...

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

    async def capture_session(self) -> SessionState:
        """The session of the page this lease is driving: its cookies and their storage.

        A move reads this from the browser it is leaving and restores it into the
        browser it goes to; a lease driving no page still answers its context's cookies.
        """
        ...

    async def restore_session(self, session: SessionState) -> None:
        """Put a carried-over session into the page this lease is about to drive."""
        ...


class BrowserProvider(Protocol):
    """The port that leases a Run's browser endpoint."""

    async def lease(
        self, holder: BrowserHolder, *, wait_seconds: float, headed: bool = False
    ) -> LeasedBrowser:
        """A browser for ``holder``: headless, or a headed one that runs with a display.

        A deployment that leases no headed browser refuses a headed lease as not configured.
        The lease is the Run's for as long as it holds it, so a Run that has one headed never
        asks again: there is no counter here to spend.
        """
        ...

    def page_for(self, holder: BrowserHolder) -> LiveBrowserPage | None:
        """The live surface of the browser ``holder`` leases here, or None when this process
        leases it none. It is read-only: the lease's claim, exclusion and expiry are not its,
        and it answers for its own driving seat."""
        ...

    async def aclose(self) -> None: ...


class BrowserLeases(Protocol):
    """The shared record of which Run holds each endpoint, which a provider claims through."""

    async def register_endpoints(self, endpoints: Sequence[str]) -> None: ...

    async def claim(
        self, holder: BrowserHolder, endpoints: Sequence[str], exclude: Sequence[str] = ()
    ) -> str | None: ...

    async def release(self, holder: BrowserHolder, endpoint: str) -> None: ...

    async def holder_for(self, *, owner_id: str, run_id: str) -> BrowserHolder | None:
        """The holder that Run leases an endpoint through, or None while it holds none.

        A read-only question: the answer is the fenced holder of a running Run whose lease
        has not expired, which is what a live session is opened against.
        """
        ...

    async def set_control(self, holder: BrowserHolder, state: LiveControlState) -> None:
        """Record who drives the holder's browser, for operators.

        Called on a control handover only, never per input event.
        """
        ...


@dataclass(frozen=True, slots=True)
class AgentBrowserSettings:
    """The Agent Browser a deployment configures.

    The pool's endpoints, the egress proxy, whether Chromium is launched inside its own
    sandbox, and the connect timeout are what its provider is built from. The other values
    are the waits and bounds a Run keeps: for a free browser, for a page to load, settle
    and answer an action, for a download, and before it gives an idle browser back.
    """

    endpoints: tuple[str, ...]
    headed_endpoints: tuple[str, ...]
    """The subset of ``endpoints`` whose members run a headed browser with a display server.

    Empty in a deployment that leases headless browsers alone, which is then the only thing
    a Run may ask for (ADR 0032)."""
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

    @property
    def headed(self) -> bool:
        """Whether this deployment leases a headed browser at all."""
        return bool(self.headed_endpoints)


@dataclass(frozen=True, slots=True)
class AgentBrowserBinding:
    """The Agent Browser a deployment composed: its provider, the settings it runs under, and
    the Agent Accounts its Runs log in with and may register (ADR 0034)."""

    provider: BrowserProvider
    settings: AgentBrowserSettings
    accounts: AgentAccountsBinding


__all__ = [
    "MAX_DIALOGS_PER_CALL",
    "MAX_DOWNLOADS_PER_CALL",
    "MAX_REFUSALS_PER_CALL",
    "REF_PATTERN",
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
    "DialogVerdict",
    "DownloadRefusal",
    "DownloadedFile",
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
    "SessionState",
    "StorageState",
    "UploadFile",
    "browser_failure",
    "page_failure",
    "refusal_failure",
]
