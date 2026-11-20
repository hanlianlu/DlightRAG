# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The ``browser`` tool: an Agent Session drives a page of the Run's Agent Browser (ADR 0032).

It is one tool with an ``action`` because its actions share one piece of state, the active
page and the refs of its latest snapshot. Actions that change the page answer with a bounded
snapshot; ``capture`` and a file a page downloads become Resources, and everything else a
page yields is context, never evidence. The tool is not read-only and never replays: each
call runs alone, and a call pending at a crash settles its outcome as unknown.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import mimetypes
import unicodedata
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import timedelta
from hashlib import sha256
from typing import Annotated, Any, Literal, Self, cast, get_args
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    StringConstraints,
    create_model,
    model_validator,
)

from dlightrag.engine.agent.environment.access import (
    AccessScheduler,
    PathAccess,
)
from dlightrag.engine.agent.environment.errors import (
    TOOL_RESULT_MAX_BYTES,
    TOOL_RESULT_MAX_LINES,
    PathRejected,
)
from dlightrag.engine.agent.environment.execution import ExecutionEnvironment
from dlightrag.engine.agent.tool_content import ToolResourceAttachmentPart, ToolTextPart
from dlightrag.engine.agent.tools.contracts import (
    AgentTool,
    CommittedOutput,
    ResourceAttachmentBytes,
    ToolDeclaration,
    ToolEffects,
    ToolResult,
    ToolRuntime,
)
from dlightrag.engine.agent.tools.files import (
    ImagePreparer,
    ResourceReader,
    ResourceReadRequest,
    SpillWriter,
    agentspace_integrity_refusal,
    head_excerpt,
    spill_continuation,
    within_result_bounds,
)
from dlightrag.engine.agent.tools.listing import escape_path
from dlightrag.engine.answer.agent_browser.accounts import (
    PASSWORD_LENGTH,
    PASSWORD_MIN_LENGTH,
    AgentAccount,
    RunAgentAccounts,
    SessionAccounts,
    account_site,
    generate_password,
    has_mailbox,
)
from dlightrag.engine.answer.agent_browser.contracts import (
    MAX_DOWNLOADS_PER_CALL,
    REF_PATTERN,
    AgentBrowserError,
    AgentPage,
    CredentialFill,
    DialogVerdict,
    PageEvents,
    PageObservation,
    PageState,
    UploadFile,
    browser_failure,
    page_failure,
)
from dlightrag.engine.answer.agent_browser.elision import (
    elide,
)
from dlightrag.engine.answer.agent_browser.mailbox import (
    INBOX_MESSAGE_BYTES,
    INBOX_MESSAGES,
    AgentMailboxError,
    MailListing,
    inbox_text,
)
from dlightrag.engine.answer.agent_browser.run import (
    RunAgentBrowser,
)
from dlightrag.engine.answer.resources.models import ResourceRegistryError
from dlightrag.engine.answer.resources.registry import (
    BROWSER_CAPTURE,
    BROWSER_DOWNLOAD,
    BrowserResourceInput,
    ResourceEffectOwner,
    ResourceRegistry,
    browser_capture_filename,
    browser_download_filename,
    browser_download_media_type,
)
from dlightrag.engine.credential_cipher import UnreadableEnvelope
from dlightrag.engine.public_http import (
    PublicHttpPolicyError,
    avalidate_public_http_url,
    validate_public_http_url,
)

logger = logging.getLogger(__name__)

BrowserAction = Literal[
    "navigate",
    "snapshot",
    "find",
    "back",
    "wait",
    "click",
    "type",
    "select",
    "press",
    "scroll",
    "upload",
    "screenshot",
    "capture",
    "register",
    "login",
    "inbox",
]
BROWSER_ACTIONS: tuple[BrowserAction, ...] = get_args(BrowserAction)

#: The most characters one register makes, and this mechanism's own bound rather than a rule about
#: the password an agent should choose. Every character of one is drawn, held in this process's
#: memory, filled into the page, and sealed into the envelope the account stores: at this bound the
#: envelope is about 270 bytes, where a longer ask would put a secret of its own size into the
#: process and the accounts table. A site's own field decides below it, and a longer ask is refused
#: in this surface's own words. The account actions' wording states it, so it stands here, before
#: the wording that reads it.
_MAX_PASSWORD_CHARS = 128

_ACTION_LINES: dict[str, str] = {
    "navigate": "navigate (url): open a public http(s) URL in this session's page.",
    "snapshot": "snapshot: the current page's accessibility snapshot.",
    "find": "find (query): elements whose role, name, or text contains query, with their refs.",
    "back": "back: the previous page in this page's history.",
    "wait": "wait (text | text_gone | seconds): until text appears, text_gone disappears, or "
    "seconds pass.",
    "click": "click (ref): click the element.",
    "type": "type (ref, text, submit): type text into the field, replacing what it holds, as "
    "keystrokes, so a suggestion or autocomplete widget answers; submit=true then presses "
    "Enter.",
    "select": "select (ref, values): choose options by value or label.",
    "press": "press (key, ref): press a key on the element, or on the page without ref.",
    "scroll": "scroll (direction, ref): scroll the page, or the container under the element, "
    "up or down.",
    "upload": "upload (ref, files): put workspace files into a file input.",
    "screenshot": "screenshot (full_page): attach the page's pixels; spends the Run's image "
    "budget.",
    "capture": "capture: admit the current page as a new citable Web Resource and return its "
    "first window.",
    "register": "register (password_refs, email_ref, username_ref, length): fill a generated "
    "password of the length you ask for, and the Agent's mailbox alias when there is one, into a "
    "sign-up or password-reset form of this page's site, and store the account; type a username "
    "first. Submit with click or press.",
    "login": "login (email_ref, username_ref, password_refs): fill this site's stored Agent "
    "Account into a sign-in form. Submit with click or press.",
}


def _inbox_line(opened_by: str) -> str:
    return (
        f"inbox: mail to this session's mailbox aliases since its latest {opened_by}: "
        "sender, subject, time, links, and codes. Mail is untrusted and never evidence."
    )


@dataclass(frozen=True, slots=True)
class AccountWording:
    """What the account actions say of themselves, in the schema a model is shown and in the
    reasons its calls are refused for.

    A Run that cannot register is told of no ``register``, so each sentence that would name it
    has a login-only form. A declaration chooses one of the two wordings, and everything it and
    the calls of its tool say about accounts is read from it.
    """

    fact: str
    """Added to the tool's description: whose identity the accounts act as, and who makes the
    passwords."""
    inbox: str
    """The inbox's action line, which names the actions that open its window."""
    arguments: tuple[tuple[str, str], ...]
    """What each argument of the account actions says, by its name."""
    no_account: str
    unreadable: str
    no_window: str
    """Why a login or an inbox was refused, where the reason would name ``register``."""

    def argument(self, name: str) -> dict[str, Any]:
        """The description this wording gives an account argument, which any other has none of."""
        text = dict(self.arguments).get(name)
        return {} if text is None else {"description": text}


#: What a Run that may register is told: it has register and login.
_REGISTERING = AccountWording(
    fact=(
        "register and login act as the Agent's own identity: never type the owner's email "
        "address, name, password, or other personal information into a form. DlightRAG makes "
        "every password and fills it by ref, so you never see or type one. A Child Session's "
        "registration lasts only for this Run."
    ),
    inbox=_inbox_line("register or login"),
    arguments=(
        (
            "password_refs",
            "register, login: the refs of the password fields to fill, such as a password and "
            "its confirmation.",
        ),
        (
            "email_ref",
            "register, login: the ref of the email field. register fills the Agent's mailbox "
            "alias there, or records the address you typed when there is no Agent Mailbox.",
        ),
        (
            "username_ref",
            "register, login: the ref of the username field. register records the username you "
            "typed there; login fills it.",
        ),
        (
            "length",
            f"register: how many characters the generated password has, {PASSWORD_MIN_LENGTH} to "
            f"{_MAX_PASSWORD_CHARS}; {PASSWORD_LENGTH} when omitted. {_MAX_PASSWORD_CHARS} is the "
            "most this mechanism will fill and seal, whatever the site takes. Ask for the longest "
            "the site's own password field allows, and never a shorter password than the task "
            "needs: read the maxlength, minlength, pattern, and required format off the page, "
            "which a snapshot prints beside the field's ref. Every password DlightRAG makes "
            "holds a lowercase letter, an uppercase letter, a digit, and one of -._, at any "
            "length, so a site's character-set rules are met. A field that takes fewer characters "
            "than you ask for is refused, and the refusal says how many it takes.",
        ),
    ),
    no_account=(
        'No Agent Account exists for {site}. Register one with browser(action="register", ...).'
    ),
    unreadable=(
        "The stored password for {site} can no longer be opened. Recover the account with the "
        'site\'s password reset: on its reset request form, call browser(action="login", '
        "email_ref=...) without password_refs to fill the account's address, submit it, read the "
        "reset mail with inbox, open its link with navigate, and call register on the reset form."
    ),
    no_window=(
        "inbox shows mail only after register or login in this Agent Session, and it has done "
        "neither in this Run."
    ),
)
#: What a Run that cannot register is told: it has login alone, no password to make and no
#: registration to name.
_LOGIN_ONLY = AccountWording(
    fact=(
        "login acts as the Agent's own identity: never type the owner's email address, name, "
        "password, or other personal information into a form. DlightRAG fills every stored "
        "password by ref, so you never see or type one. This Run does not open new accounts, so "
        "never create an account on any site by filling a sign-up form yourself."
    ),
    inbox=_inbox_line("login"),
    arguments=(
        ("password_refs", "login: the refs of the password fields to fill."),
        ("email_ref", "login: the ref of the email field."),
        ("username_ref", "login: the ref of the username field."),
    ),
    no_account="No Agent Account exists for {site}.",
    unreadable="The stored password for {site} can no longer be opened.",
    no_window=(
        "inbox shows mail only after login in this Agent Session, and it has not logged in "
        "during this Run."
    ),
)


def _wording(may_register: bool) -> AccountWording:
    return _REGISTERING if may_register else _LOGIN_ONLY


def _action_line(action: str, wording: AccountWording) -> str:
    """One action's line in the schema, where only the inbox's depends on the Run's accounts."""
    return wording.inbox if action == "inbox" else _ACTION_LINES[action]


_REF = Annotated[str, StringConstraints(pattern=rf"^{REF_PATTERN}$", max_length=32)]
_KEY = Annotated[str, StringConstraints(pattern=r"^[\x21-\x7e]{1,64}$")]
_URL = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8192)]
_QUERY = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
_GONE = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]
#: Typed text keeps its spaces, so nothing strips it.
_TEXT = Annotated[str, StringConstraints(max_length=10_000)]
_VALUE = Annotated[str, StringConstraints(max_length=1000)]
_PATH = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4096)]
_SECONDS = Annotated[float, Field(gt=0, allow_inf_nan=False)]

#: Each argument: its annotation, its field arguments, and the actions that read it. The
#: order is the schema's. An action reads only what it names, so every argument is optional.
_FIELDS: dict[str, tuple[Any, dict[str, Any], frozenset[str]]] = {
    "url": (
        _URL,
        {"description": "navigate: the public http(s) URL to open."},
        frozenset({"navigate"}),
    ),
    "query": (
        _QUERY,
        {"description": "find: text to look for in element roles, names, and text."},
        frozenset({"find"}),
    ),
    #: The most matching lines one find answers with, which the call may ask for.
    "limit": (
        Annotated[int, Field(ge=1)],
        {
            "description": (
                "find: how many matching lines to return, 1 to 100; 30 when omitted. The "
                "result says how many matched in all."
            )
        },
        frozenset({"find"}),
    ),
    "ref": (
        _REF,
        {
            "description": "The element's ref from the latest snapshot or find of this page, such as e12."
        },
        frozenset({"click", "type", "select", "press", "scroll", "upload"}),
    ),
    "text": (
        _TEXT,
        {
            "description": (
                "type: the text typed into the field, replacing what it holds. wait: text to "
                "wait for."
            )
        },
        frozenset({"type", "wait"}),
    ),
    "submit": (
        bool,
        {"description": "type: press Enter after the text."},
        frozenset({"type"}),
    ),
    "values": (
        list[_VALUE],
        {
            "min_length": 1,
            "max_length": 20,
            "description": "select: option values or labels to choose.",
        },
        frozenset({"select"}),
    ),
    "key": (
        _KEY,
        {"description": "press: a key such as Enter, Tab, Escape, ArrowDown, or Control+A."},
        frozenset({"press"}),
    ),
    "direction": (
        Literal["up", "down"],
        {"description": "scroll: up or down; down when omitted."},
        frozenset({"scroll"}),
    ),
    "text_gone": (
        _GONE,
        {"description": "wait: text to wait to disappear."},
        frozenset({"wait"}),
    ),
    "seconds": (
        _SECONDS,
        {
            "description": (
                "wait: seconds to wait, up to 30 in one call; call wait again to wait longer."
            )
        },
        frozenset({"wait"}),
    ),
    "full_page": (
        bool,
        {"description": "screenshot: the whole page instead of its visible part."},
        frozenset({"screenshot"}),
    ),
    "headed": (
        bool,
        {
            "description": (
                "navigate: make this Run's browser one with a display, for a page that "
                "refuses an automated browser; the session it carries moves with it, and the "
                "Run keeps that browser for the rest of the Run. Needs a deployment that "
                "leases headed browsers."
            )
        },
        frozenset({"navigate"}),
    ),
    "dialog": (
        Literal["accept", "dismiss"],
        {
            "description": (
                "Every action that reads or drives a page: what to do with a page dialog this "
                "call meets — accept it or dismiss it. Omitted, a dialog is left waiting and "
                "reported, and the page stays as it is until a later call answers it."
            )
        },
        frozenset(BROWSER_ACTIONS) - {"inbox"},
    ),
    "files": (
        list[_PATH],
        {
            "min_length": 1,
            "max_length": 10,
            "description": "upload: workspace paths of the files to put into the file input.",
        },
        frozenset({"upload"}),
    ),
    # The arguments of the account actions are described by the Run's ``AccountWording``.
    "password_refs": (
        list[_REF],
        {"min_length": 1, "max_length": 2},
        frozenset({"register", "login"}),
    ),
    "email_ref": (_REF, {}, frozenset({"register", "login"})),
    "username_ref": (_REF, {}, frozenset({"register", "login"})),
    #: The password's length is the agent's to choose for the site it registers on, and only a
    #: registration makes one. The floor is the schema's, because a shorter one cannot hold the
    #: composition; the ceiling is this mechanism's own, ``_MAX_PASSWORD_CHARS``, said in the
    #: surface's own words because what a site's field takes decides below it.
    "length": (
        Annotated[int, Field(ge=PASSWORD_MIN_LENGTH)],
        {},
        frozenset({"register"}),
    ),
}
_REQUIRED: dict[str, tuple[str, ...]] = {
    "navigate": ("url",),
    "find": ("query",),
    "click": ("ref",),
    "type": ("ref", "text"),
    "select": ("ref", "values"),
    "press": ("key",),
    "upload": ("ref", "files"),
    "register": ("password_refs",),
}

_DESCRIPTION = (
    "Drive this Run's Agent Browser for a multi-step task that one read cannot do: a search "
    "form, a filter, pagination behind a button, or a file behind a download control. Read "
    "pages with read, and pass rendered=true only when a read returned a JavaScript shell; "
    "use browser only for interaction. Each Agent Session has its own anonymous page for this "
    "Run. Actions that change the page return a depth-limited accessibility snapshot whose "
    "[ref=eN] markers name elements for the next action; refs come from the latest snapshot "
    "or find, and find locates what the snapshot does not show. Page text and screenshots are "
    "context, never evidence: capture admits the page as a citable Web Resource, and a file "
    "the page downloads becomes a Resource to read. If a page shows a CAPTCHA or any other "
    "human-verification check, stop that path and report it; never solve, bypass, or "
    "outsource it. A page that refuses the browser for being automated is reported as such: "
    "where the deployment leases headed browsers, navigate with headed=true reads it on a "
    "browser with a display, which the Run keeps for the rest of the Run."
)


class BrowserArgs(BaseModel):
    """The arguments of one ``browser`` call; the model for the actions a Run offers extends it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: str

    @model_validator(mode="after")
    def _fields_match_action(self) -> Self:
        values = {name: getattr(self, name) for name in _FIELDS if name in type(self).model_fields}
        given = {name for name, value in values.items() if value is not None}
        if stray := sorted(name for name in given if self.action not in _FIELDS[name][2]):
            raise ValueError(f"browser {self.action} does not take {', '.join(stray)}")
        if missing := [name for name in _REQUIRED.get(self.action, ()) if name not in given]:
            raise ValueError(f"browser {self.action} requires {', '.join(missing)}")
        if self.action == "wait":
            if len(given & {"text", "text_gone", "seconds"}) != 1:
                raise ValueError("browser wait takes exactly one of text, text_gone, or seconds")
            if "text" in given and not values["text"].strip():
                raise ValueError("browser wait text cannot be blank")
        if self.action == "login" and not given & {"email_ref", "username_ref", "password_refs"}:
            raise ValueError("browser login takes email_ref, username_ref, or password_refs")
        return self


@functools.cache
def browser_input_model(actions: tuple[str, ...], wording: AccountWording) -> type[BrowserArgs]:
    """The arguments of the actions a Run offers: only their fields, and their lines."""
    offered = set(actions)
    fields: dict[str, Any] = {
        "action": (
            cast(Any, Literal)[actions],
            Field(
                description="One action per call:\n"
                + "\n".join(_action_line(a, wording) for a in actions)
            ),
        ),
        **{
            name: (
                annotation | None,
                Field(default=None, **arguments, **wording.argument(name)),
            )
            for name, (annotation, arguments, readers) in _FIELDS.items()
            if readers & offered
        },
    }
    return create_model("BrowserArgs", __base__=BrowserArgs, **fields)


def browser_declaration(*, upload: bool, may_register: bool, mailbox: bool) -> ToolDeclaration:
    """The tool a Run offers. ``upload`` needs an Agent Workspace, so only ``trust`` has it,
    ``register`` needs a Run that may register, which is the deployment's allowance and the
    owner's switch together, and ``inbox`` needs an Agent Mailbox. ``login`` is every Run's, since
    a Run with a browser has Agent Accounts.

    No configured value appears in the description or the schema, so changing a timeout or
    the depth never changes the plan a Run is pinned to.
    """
    wording = _wording(may_register)
    offered = {"upload": upload, "register": may_register, "inbox": mailbox}
    actions = tuple(action for action in BROWSER_ACTIONS if offered.get(action, True))
    return ToolDeclaration(
        name="browser",
        description=f"{_DESCRIPTION} {wording.fact}",
        input_model=browser_input_model(actions, wording),
        replay_policy="never",
        read_only=False,
        contract_version=1,
    )


@dataclass(frozen=True, slots=True)
class BrowserRequest:
    """One validated call, with the defaults its actions assume."""

    action: BrowserAction
    url: str | None = None
    query: str | None = None
    ref: str | None = None
    text: str | None = None
    submit: bool = False
    limit: int | None = None
    values: tuple[str, ...] = ()
    key: str | None = None
    direction: Literal["up", "down"] = "down"
    text_gone: str | None = None
    seconds: float | None = None
    full_page: bool = False
    headed: bool = False
    dialog: DialogVerdict | None = None
    files: tuple[str, ...] = ()
    password_refs: tuple[str, ...] = ()
    email_ref: str | None = None
    username_ref: str | None = None
    length: int | None = None
    """The generated password's length, or None for the generator's own default."""

    @classmethod
    def from_args(cls, args: BrowserArgs) -> BrowserRequest:
        # A model that does not offer an action has none of its fields.
        given = {name: getattr(args, name, None) for name in _FIELDS}
        return cls(
            action=cast(BrowserAction, args.action),
            url=given["url"],
            query=given["query"],
            ref=given["ref"],
            text=given["text"],
            submit=bool(given["submit"]),
            limit=given["limit"],
            values=tuple(given["values"] or ()),
            key=given["key"],
            direction=given["direction"] or "down",
            text_gone=given["text_gone"],
            seconds=given["seconds"],
            full_page=bool(given["full_page"]),
            headed=bool(given["headed"]),
            dialog=given["dialog"],
            files=tuple(given["files"] or ()),
            password_refs=tuple(given["password_refs"] or ()),
            email_ref=given["email_ref"],
            username_ref=given["username_ref"],
            length=given["length"],
        )


@dataclass(frozen=True, slots=True)
class BrowserToolHost:
    """What the Run's ``browser`` tool reaches, shared by its Parent and every Child."""

    browser: RunAgentBrowser
    registry: ResourceRegistry
    read_resource: ResourceReader
    """The Run's own reader, which a capture is read through as ``read`` would read it."""
    accounts: RunAgentAccounts
    """The Run's Agent Accounts, which login and, where the Run may register, register act on."""


NEW_PAGE = "A new tab opened and is now the active page."
RETURNED = "The active page closed; the previous page is active again."
CLOSED = "The page closed itself; navigate to open a new one."
NAVIGATE_REFUSED = (
    "browser navigate refused this URL: {exc}. It opens public http(s) URLs without embedded "
    "credentials."
)
FOUND = '{total} element(s) match "{query}"'
FOUND_CUT = " (first {shown} shown)"
NOTHING_FOUND = 'No element matches "{query}".'
SPILLED = "browser snapshot exceeded {max_bytes} UTF-8 bytes or {max_lines} lines ({size} bytes); its head follows."
UNAVAILABLE = (
    "The action completed, but its full snapshot is unavailable: it exceeds {max_bytes} UTF-8 "
    "bytes or {max_lines} lines and this Run cannot keep it. Do not repeat the action; use find "
    "to locate an element. Its head follows."
)
UPLOAD_TOO_BIG = "upload sends at most {limit} MiB in one call."
UPLOAD_NOT_FILE = "upload needs regular workspace files: {path} is not one."
DOWNLOADED = (
    "Downloaded {filename} ({mime}, {size:,} bytes) as {resource_id} (browser_download); "
    "read(resource_id='{resource_id}') reads it."
)
DIALOG = 'The page showed a {kind} dialog: "{message}" ({answer}).'
DIALOG_DEFAULT = 'Its default text is "{default}"; accepting takes that as the answer.'
DIALOG_WAITING = (
    'The page showed a {kind} dialog: "{message}" and it is waiting for an answer: the page '
    "is stopped until a call answers it. Every call of this Session reports it until then. "
    'Call browser again with dialog="accept" or dialog="dismiss" to answer it.'
)
NOT_SETTLED = (
    "The page had not gone quiet within {seconds:g} seconds of this call: what follows was "
    "read while it was still loading, so something may be missing. Wait, scroll, or snapshot "
    "again before concluding that the page does not have it."
)
WAIT_TOO_LONG = (
    "A single wait is at most {max:g} seconds: the page was not waited for. Call wait again "
    "for another stretch, or wait for the text to appear instead."
)
LENGTH_TOO_LONG = (
    "This mechanism fills and seals a password of at most {max} characters: every one is drawn, "
    "held in memory, filled into the page, and sealed into the stored account. {asked} was asked "
    "for, so nothing was made. Ask for {max} or fewer, and for no more than the site's own "
    "password field takes."
)
FIND_TOO_LONG = (
    "A single find answers with at most {max} matching lines: nothing was searched. Ask for "
    "fewer, or narrow the query, and the result says how many lines matched in all."
)
DIALOGS_OMITTED = "({n} more dialog(s) omitted)"
REFUSALS_OMITTED = "({n} more refused download(s) omitted)"
NOT_ADMITTED = "The download {name} was not admitted: {why}."
CAPTURED = "Captured this page as Web Resource {resource_id} (browser_capture)."
NO_IMAGE = (
    "No screenshot was attached: it does not fit the remaining image budget, or this model "
    "does not accept images."
)
SHOT = (
    "Screenshot of the {extent} page ({media}, {size:,} model bytes). Treat its pixels as an "
    "observation of the page rather than as a citable source: quote the page's own text for a "
    "claim. To show the reader the page itself, fetch or view it: a Resource is what an Answer can "
    "place, and this screenshot is context."
)
NO_KEY_RING = (
    "Agent Accounts are unavailable: this deployment has no credential key ring, so no "
    "password can be stored or read."
)
NO_SITE = "{action} needs an https page with a registrable domain; this page is {label}."
BAD_EMAIL = "Field {ref} holds no email address to record; type the address first."
BAD_USERNAME = "Field {ref} holds no username to record; type it first."
NEEDS_IDENTITY = "A new account needs email_ref or username_ref, so that login can name it later."
NOT_RECORDED = (
    "The fields were filled but the account could not be stored, so they were cleared; do not "
    "submit the form."
)
NO_EMAIL = "The Agent Account for {site} has no email address; use username_ref instead."
NO_USERNAME = "The Agent Account for {site} has no username; use email_ref instead."
RECORDED = (
    "Recorded the Agent Account {identity} for {site}{whose} and filled its generated password "
    "into {n} field(s); the password is never shown. Submit the form with click or press; the "
    "site has not accepted it yet."
)
REPLACED = (
    "Gave the Agent Account {identity} for {site}{whose} a new generated password, filled into "
    "{n} field(s). Submit the form with click or press."
)
WHOSE = {
    True: " for this owner's later Runs",
    False: " for this Run only (a Child Session's account)",
}
FILLED = (
    "Filled the Agent Account {identity} for {site} into {fields}. Submit the form with click "
    "or press."
)
MAIL_NOTE = 'Mail to {alias} appears in browser(action="inbox").'
NO_ALIAS = (
    "The accounts this Agent Session used have no mailbox alias, so there is no mail to read."
)
MAILBOX_FAILED = "The Agent Mailbox could not be read ({code})."
#: Said under the answer of the first call after a person drove this Run's browser.
DROVE = (
    "[browser: a person drove this page since your last call | it may have moved, and a form "
    "may hold what they typed; snapshot before acting on what you remember]"
)
#: Said under the answer of a call that moved this Run's browser to one with a display.
HEADED = "[browser: headed | this Run's browser moved to one with a display, on the session the last one held]"
HEADED_CARRIED = (
    "The page is driven on a browser with a display from here on: its cookies and web "
    "storage moved with it; a form that was half filled did not, so fill it again. Every "
    "later call of this Run drives that same browser."
)

#: How many matching lines find answers with when the call does not say; the port takes any.
_FOUND_SHOWN = 30
#: How many matching lines one find answers with; a longer list is a narrower query's to make.
_MAX_FOUND = 100
#: How long one wait may last; a longer stretch is the agent's to ask for again.
_MAX_WAIT_SECONDS = 30.0
_FRAME_URL_CHARS = 500
#: The longest email address, and the longest username, an account records.
_MAX_EMAIL_CHARS = 254
_MAX_USERNAME_CHARS = 128
#: The stored time is the bucket's, so an inbox's window opens a little before the Session
#: signed in, for the clocks' skew.
_CLOCK_SKEW = timedelta(seconds=120)
#: Playwright's own limit on the files one call hands a page.
_MAX_UPLOAD_MIB = 50


def browser_tool(
    host: BrowserToolHost,
    *,
    environment: ExecutionEnvironment | None,
    scheduler: AccessScheduler,
    spill: SpillWriter | None,
    image_preparer: ImagePreparer | None,
    child: bool,
) -> AgentTool:
    """Bind the ``browser`` declaration to the Run's browser, Resources, and workspace.

    Without an ``image_preparer`` the model takes no images, so a screenshot is refused. A
    ``child`` Session's registrations last only for the Run (ADR 0034).
    """

    wording = _wording(host.accounts.may_register)

    async def execute(raw: BaseModel, runtime: ToolRuntime) -> ToolResult:
        call = _Call(
            host,
            BrowserRequest.from_args(cast(BrowserArgs, raw)),
            runtime,
            wording=wording,
            environment=environment,
            scheduler=scheduler,
            spill=spill,
            image_preparer=image_preparer,
            child=child,
        )
        return await call.run()

    return browser_declaration(
        upload=environment is not None,
        may_register=host.accounts.may_register,
        mailbox=has_mailbox(host.accounts),
    ).bind(execute)


class _Call:
    """One ``browser`` call: the action asked for, and the pieces of the Run it uses."""

    def __init__(
        self,
        host: BrowserToolHost,
        request: BrowserRequest,
        runtime: ToolRuntime,
        *,
        wording: AccountWording,
        environment: ExecutionEnvironment | None,
        scheduler: AccessScheduler,
        spill: SpillWriter | None,
        image_preparer: ImagePreparer | None,
        child: bool,
    ) -> None:
        self._host = host
        self._request = request
        self._runtime = runtime
        self._wording = wording
        self._environment = environment
        self._scheduler = scheduler
        self._spill = spill
        self._image_preparer = image_preparer
        self._child = child
        self._scope = runtime.execution_scope
        self._owner = ResourceEffectOwner(self._scope, runtime.intent_id)

    async def run(self) -> ToolResult:
        if self._host.browser.taken_over():
            # A person is driving this Run's browser. The call is refused and the Run goes on:
            # what they are doing on the page is not the agent's to interrupt (ADR 0032).
            return ToolResult.text(page_failure("taken_over").public_message, is_error=True)
        if subject := self._subject():
            await self._runtime.emit_update(ToolResult.text("", subject=subject))
        before = self._host.browser.headed
        try:
            result = await self._act()
        except AgentBrowserError as exc:
            return ToolResult.text(exc.public_message, is_error=True)
        except ResourceRegistryError as exc:
            return ToolResult.text(str(exc), is_error=True)
        if self._request.headed and before is False and self._host.browser.headed is True:
            # The call moved the Run to a browser with a display, and its answer says so.
            result = self._noted(result, f"{HEADED}\n{HEADED_CARRIED}")
        if not result.is_error and self._host.browser.take_drive_notice():
            # A person drove this page and handed the seat back: the page is not the one the
            # agent last saw, and the first call that learns it says so. A call that failed
            # spent nothing, so the next one that reaches the page still carries the notice.
            result = self._noted(result, DROVE)
        return result

    def _noted(self, result: ToolResult, note: str) -> ToolResult:
        """The answer of a call with one more fact about the browser it was made on."""
        first = result.parts[0] if result.parts else None
        text = "" if first is None else getattr(first, "text", "")
        head = ToolTextPart(f"{text}\n{note}" if text else note)
        return replace(result, parts=(head, *result.parts[1:]))

    def _subject(self) -> str:
        """What the call acts on, for the live row: never typed text."""
        request = self._request
        if request.action == "navigate":
            return _label(request.url)
        if request.action == "find":
            return request.query or ""
        if request.action == "inbox":
            window = self._session_accounts().inbox_window()
            return "" if window is None else ", ".join(window.aliases)
        current = self._host.browser.current_url(self._scope)
        if current is None:
            return ""
        if request.ref is not None:
            return f"{request.ref} · {_label(current)}"
        return _label(current)

    async def _act(self) -> ToolResult:
        request = self._request
        match request.action:
            case "navigate":
                url = cast(str, request.url)
                try:
                    # Only the first URL is checked here; the deployment's network confines
                    # what the page loads after it.
                    validate_public_http_url(url)
                    await avalidate_public_http_url(url)
                except PublicHttpPolicyError as exc:
                    return ToolResult.text(NAVIGATE_REFUSED.format(exc=exc), is_error=True)
                return await self._acting(
                    lambda p: p.navigate(url), open_page=True, headed=request.headed
                )
            case "back":
                return await self._acting(lambda p: p.back())
            case "snapshot":
                return await self._acting(lambda p: p.snapshot())
            case "wait":
                if request.seconds is not None and request.seconds > _MAX_WAIT_SECONDS:
                    return ToolResult.text(
                        WAIT_TOO_LONG.format(max=_MAX_WAIT_SECONDS), is_error=True
                    )
                return await self._acting(
                    lambda p: p.wait(
                        text=request.text, text_gone=request.text_gone, seconds=request.seconds
                    )
                )
            case "click":
                return await self._acting(lambda p: p.click(cast(str, request.ref)))
            case "type":
                return await self._acting(
                    lambda p: p.type_text(
                        cast(str, request.ref), cast(str, request.text), submit=request.submit
                    )
                )
            case "select":
                return await self._acting(
                    lambda p: p.select(cast(str, request.ref), request.values)
                )
            case "press":
                return await self._acting(
                    lambda p: p.press(cast(str, request.key), ref=request.ref)
                )
            case "scroll":
                return await self._acting(
                    lambda p: p.scroll(direction=request.direction, ref=request.ref)
                )
            case "upload":
                return await self._upload()
            case "find":
                return await self._find()
            case "screenshot":
                return await self._screenshot()
            case "capture":
                return await self._capture()
            case "register":
                return await self._with_accounts(self._register)
            case "login":
                return await self._with_accounts(self._login)
            case "inbox":
                return await self._inbox()

    async def _on_page[T](
        self,
        call: Callable[[AgentPage], Awaitable[T]],
        *,
        open_page: bool = False,
        headed: bool = False,
    ) -> T:
        """Run ``call`` on this Session's page, with this call's verdict for its dialogs."""

        async def with_verdict(page: AgentPage) -> T:
            await page.answer_dialogs(self._request.dialog)
            return await call(page)

        return await self._host.browser.with_page(
            self._scope, with_verdict, open_page=open_page, headed=headed
        )

    async def _acting(
        self,
        call: Callable[[AgentPage], Awaitable[PageObservation]],
        *,
        open_page: bool = False,
        headed: bool = False,
    ) -> ToolResult:
        """A call that answers with the page and its snapshot."""
        return await self._reported(await self._on_page(call, open_page=open_page, headed=headed))

    async def _reported(self, observation: PageObservation, *lines: str) -> ToolResult:
        """The frame and notes of what a call left, then ``lines``, then its bounded snapshot."""
        report = "\n".join((await self._report(observation.page, observation.events), *lines))
        if observation.snapshot is None:
            return ToolResult.text(report)
        return await self._bounded(report, observation.snapshot)

    async def _find(self) -> ToolResult:
        query = cast(str, self._request.query)
        limit = self._request.limit or _FOUND_SHOWN
        if limit > _MAX_FOUND:
            # A sentence the model can act on, as a long wait's is, and refused before the
            # page is touched: this surface never answers with a validation dump.
            return ToolResult.text(FIND_TOO_LONG.format(max=_MAX_FOUND), is_error=True)
        found = await self._on_page(lambda p: p.find(query, limit=limit))
        report = await self._report(found.page, found.events)
        if not found.lines:
            return ToolResult.text("\n".join((report, NOTHING_FOUND.format(query=query))))
        count = FOUND.format(total=found.total, query=query)
        if found.total > len(found.lines):
            count += FOUND_CUT.format(shown=len(found.lines))
        return ToolResult.text("\n".join((report, f"{count}:", *found.lines)))

    async def _screenshot(self) -> ToolResult:
        request = self._request
        shot = await self._on_page(lambda p: p.screenshot(full_page=request.full_page))
        report = await self._report(shot.page, shot.events)
        label = _label(shot.page.url)
        prepared = (
            None
            if self._image_preparer is None
            else await asyncio.to_thread(self._image_preparer, shot.png, f"screenshot of {label}")
        )
        if prepared is None:
            return ToolResult.text("\n".join((report, NO_IMAGE)), is_error=True)
        digest = sha256(prepared.data).hexdigest()
        resource_id = (
            "res-shot-"
            + sha256(
                f"{self._scope}\0{self._runtime.intent_id.value}\0{digest}".encode()
            ).hexdigest()[:32]
        )
        text = SHOT.format(
            extent="whole" if request.full_page else "visible",
            media=prepared.media_type,
            size=len(prepared.data),
        )
        return ToolResult(
            parts=(
                ToolTextPart("\n".join((report, text))),
                ToolResourceAttachmentPart(
                    resource_id,
                    "screenshot",
                    prepared.media_type,
                    digest,
                    len(prepared.data),
                    data=prepared.data,
                    source=None,
                ),
            ),
            effects=ToolEffects(
                attached_resources=(
                    ResourceAttachmentBytes(
                        resource_id,
                        "screenshot",
                        prepared.media_type,
                        source_locator=label or resource_id,
                        content=prepared.data,
                    ),
                )
            ),
        )

    async def _capture(self) -> ToolResult:
        host = self._host
        cap = await self._on_page(lambda p: p.capture())
        report = await self._report(cap.page, cap.events)
        try:
            resource_id = await host.registry.admit_browser_resource(
                BrowserResourceInput(
                    BROWSER_CAPTURE,
                    cap.html,
                    browser_capture_filename(cap.page.url),
                    "text/html; charset=utf-8",
                    cap.page.url,
                ),
                effect_owner=self._owner,
                index=len(cap.events.downloads),
            )
        except ResourceRegistryError as exc:
            return ToolResult.text("\n".join((report, str(exc))), is_error=True)
        read = await host.read_resource(
            ResourceReadRequest(resource_id=resource_id, url=None, focus=None, cursor=None),
            self._runtime,
        )
        # What a read of the capture settles and cites is kept; only its words are framed.
        text = "\n".join((report, CAPTURED.format(resource_id=resource_id), read.text_content))
        return replace(read, parts=(ToolTextPart(text),))

    def _session_accounts(self) -> SessionAccounts:
        """The Agent Accounts of the Session this call runs in."""
        return self._host.accounts.session(self._scope, child=self._child)

    async def _with_accounts(
        self, act: Callable[[AgentPage, SessionAccounts, str], Awaitable[ToolResult]]
    ) -> ToolResult:
        """Run register or login on the page the Session's earlier calls opened."""
        if not self._host.accounts.available():
            return ToolResult.text(NO_KEY_RING, is_error=True)
        accounts = self._session_accounts()

        async def on_page(page: AgentPage) -> ToolResult:
            current = page.current_url() or ""
            site = account_site(current)
            if site is None:
                return ToolResult.text(
                    NO_SITE.format(action=self._request.action, label=_label(current)),
                    is_error=True,
                )
            return await act(page, accounts, site)

        return await self._on_page(on_page)

    async def _register(self, page: AgentPage, accounts: SessionAccounts, site: str) -> ToolResult:
        """Fill a new generated password of the length the call asked for, and the account's
        email, into a sign-up form.

        The account is recorded as soon as its fields are filled, before the site accepts the
        form; a password reset is a registration on a site whose account exists. A length past
        the most one call makes is refused in this surface's own words, and a site whose own
        field takes fewer characters than the call asked for is refused before anything is
        filled or stored, naming both numbers: a password is never shortened to fit a field,
        since the length is the agent's to choose for the site it meets. A field that takes
        fewer than the shortest password this mechanism can make leaves no ask that fits it, so
        that refusal names the mechanism's own minimum and the site as one no registration can
        succeed on, rather than advising an ask that cannot be made.
        """
        request = self._request
        length = request.length or PASSWORD_LENGTH
        if length > _MAX_PASSWORD_CHARS:
            # A sentence the model can act on, as a wait's or a find's is, and refused before
            # the form is read: this surface never answers with a validation dump.
            return ToolResult.text(
                LENGTH_TOO_LONG.format(max=_MAX_PASSWORD_CHARS, asked=length), is_error=True
            )
        existing = await accounts.registration_target(site)
        form = await page.credential_form(
            site=site,
            password_refs=request.password_refs,
            email_ref=request.email_ref,
            username_ref=request.username_ref,
        )
        if (limit := form.password_limit) is not None and limit < length:
            if limit < PASSWORD_MIN_LENGTH:
                # No ask this mechanism can honour is short enough for the field, so the agent is
                # told the site cannot be registered on instead of being sent to ask again.
                raise browser_failure(
                    "password_limit_too_small", limit=limit, minimum=PASSWORD_MIN_LENGTH
                )
            raise browser_failure("password_too_long", limit=limit, asked=length)
        # What an account already holds is kept unless the form names the field again.
        email = existing.email if existing else None
        username = existing.username if existing else None
        email_fills: list[CredentialFill] = []
        if request.email_ref is not None:
            # An address the account has, or one the mailbox mints, is filled by DlightRAG.
            email = email or accounts.new_alias(site)
            if email is not None:
                email_fills.append(CredentialFill(request.email_ref, "email", SecretStr(email)))
            else:
                # With no Agent Mailbox the Agent typed an address of its own, which stays.
                email = _typed_email(form.email)
                if email is None:
                    return ToolResult.text(BAD_EMAIL.format(ref=request.email_ref), is_error=True)
        if request.username_ref is not None:
            username = _typed_username(form.username)
            if username is None:
                return ToolResult.text(BAD_USERNAME.format(ref=request.username_ref), is_error=True)
        identity = email or username
        if identity is None:
            return ToolResult.text(NEEDS_IDENTITY, is_error=True)

        password = generate_password(length)
        fills = (
            *email_fills,
            *(CredentialFill(ref, "password", password) for ref in request.password_refs),
        )
        observation = await page.fill_credentials(fills, site=site)
        recorded: AgentAccount | None = None
        try:
            recorded = await accounts.record(
                site, existing=existing, email=email, username=username, password=password
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("An Agent Account could not be recorded (%s)", type(exc).__name__)
        if recorded is None:
            # The form must not be left holding a password nothing stored.
            with suppress(AgentBrowserError):
                await page.clear_fields(tuple(fill.ref for fill in fills), site=site)
            return ToolResult.text(NOT_RECORDED, is_error=True)
        accounts.signed_in(recorded)
        notes = [
            (REPLACED if existing is not None else RECORDED).format(
                identity=identity,
                site=site,
                whose=WHOSE[recorded.persistent],
                n=len(request.password_refs),
            )
        ]
        if (alias := accounts.alias(recorded)) is not None:
            notes.append(MAIL_NOTE.format(alias=alias))
        return await self._reported(observation, *notes)

    async def _login(self, page: AgentPage, accounts: SessionAccounts, site: str) -> ToolResult:
        """Fill the account this site has for the Agent into a sign-in form."""
        request = self._request
        account = await accounts.login_target(site)
        if account is None:
            return ToolResult.text(self._wording.no_account.format(site=site), is_error=True)
        fills: list[CredentialFill] = []
        filled: list[str] = []
        if request.email_ref is not None:
            if account.email is None:
                return ToolResult.text(NO_EMAIL.format(site=site), is_error=True)
            fills.append(CredentialFill(request.email_ref, "email", SecretStr(account.email)))
            filled.append("email")
        if request.username_ref is not None:
            if account.username is None:
                return ToolResult.text(NO_USERNAME.format(site=site), is_error=True)
            fills.append(
                CredentialFill(request.username_ref, "username", SecretStr(account.username))
            )
            filled.append("username")
        if request.password_refs:
            try:
                password = accounts.password(account)
            except UnreadableEnvelope:
                return ToolResult.text(self._wording.unreadable.format(site=site), is_error=True)
            fills.extend(CredentialFill(ref, "password", password) for ref in request.password_refs)
            filled.append(f"{len(request.password_refs)} password field(s)")
        observation = await page.fill_credentials(tuple(fills), site=site)
        accounts.signed_in(account)
        if request.password_refs:
            # A sign-in is a login that filled the password. The reset request, which fills the
            # address alone, is how an account whose password cannot be opened recovers.
            try:
                await accounts.mark_used(account)
            except Exception as exc:
                # The form is filled, and the day Settings shows is not worth failing it for.
                logger.warning(
                    "An Agent Account's last use could not be recorded (%s)", type(exc).__name__
                )
        note = FILLED.format(
            identity=account.email or account.username, site=site, fields=", ".join(filled)
        )
        return await self._reported(observation, note)

    async def _inbox(self) -> ToolResult:
        """The mail this Agent Session's mailbox aliases received since it last registered or
        logged in.

        It needs no page and leases nothing. The text is mail, which anyone who learns an alias
        can write: untrusted context, said so, and never Evidence.
        """
        window = self._session_accounts().inbox_window()
        if window is None:
            return ToolResult.text(self._wording.no_window, is_error=True)
        if not window.aliases:
            return ToolResult.text(NO_ALIAS, is_error=True)
        listings: list[tuple[str, MailListing]] = []
        try:
            for alias in window.aliases:
                listing = await window.mailbox.messages(
                    alias,
                    since=window.since - _CLOCK_SKEW,
                    limit=INBOX_MESSAGES,
                    max_bytes=INBOX_MESSAGE_BYTES,
                )
                listings.append((alias, listing))
        except AgentMailboxError as exc:
            logger.warning("The Agent Mailbox could not be read (%s)", exc.code)
            return ToolResult.text(MAILBOX_FAILED.format(code=exc.code), is_error=True)
        passwords = self._host.browser.filled_passwords(self._scope)
        return ToolResult.text(inbox_text(window, listings, passwords))

    async def _upload(self) -> ToolResult:
        environment = cast(ExecutionEnvironment, self._environment)
        files = await self._agentspace_files(environment)
        if isinstance(files, ToolResult):
            return files
        return await self._acting(lambda p: p.upload(cast(str, self._request.ref), files))

    async def _agentspace_files(
        self, environment: ExecutionEnvironment
    ) -> tuple[UploadFile, ...] | ToolResult:
        """The workspace files an upload sends, or the result that says why it cannot."""
        if blocked := agentspace_integrity_refusal(environment):
            return blocked
        files: list[UploadFile] = []
        total = 0
        for name in self._request.files:
            try:
                path = environment.resolve(name)
            except PathRejected as exc:
                return ToolResult.text(str(exc), is_error=True)
            async with self._scheduler.hold(PathAccess(path=str(path), kind="read")):
                # A command that held the workspace while this waited may have latched it.
                if blocked := agentspace_integrity_refusal(environment):
                    return blocked
                if environment.stat_kind(path) != "file":
                    return ToolResult.text(
                        UPLOAD_NOT_FILE.format(path=escape_path(name)), is_error=True
                    )
                # A file is measured before it is read, so one that is too big is never loaded.
                total += path.stat().st_size
                if total > _MAX_UPLOAD_MIB * 1024 * 1024:
                    return ToolResult.text(
                        UPLOAD_TOO_BIG.format(limit=_MAX_UPLOAD_MIB), is_error=True
                    )
                content = environment.read_bytes(path)
            media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            files.append(UploadFile(path.name, media_type, content))
        return tuple(files)

    async def _report(self, page: PageState | None, events: PageEvents) -> str:
        """The frame that names the page, then a note for everything the pages did.

        The files they downloaded are admitted as Resources of this call as they are noted.
        """
        action = self._request.action
        if page is None:
            frame = f"[browser: {action} | the page closed]"
        else:
            url = elide(page.url, _FRAME_URL_CHARS)
            status = ""
            if events.http_status is not None and events.http_status >= 400:
                status = f" | HTTP {events.http_status}"
            frame = f"[browser: {action} | page: {url} | title: {page.title}{status}]"
        notes = [
            note
            for flag, note in (
                (events.new_page, NEW_PAGE),
                (events.returned, RETURNED),
                (events.closed, CLOSED),
            )
            if flag
        ]
        if events.settled is False:
            notes.append(NOT_SETTLED.format(seconds=self._host.browser.limits.settle_timeout))
        notes.extend(self._dialogs(events))
        notes.extend(await self._admit_downloads(events))
        notes.extend(self._refusals(events))
        return "\n".join((frame, *notes))

    def _dialogs(self, events: PageEvents) -> list[str]:
        """One note for every dialog the call met, waiting or answered, and one for the rest.

        The dialogs past the bound are counted rather than listed: a page can raise them
        without end, and a list that looked complete would be read as the whole of them.
        """
        notes: list[str] = []
        for dialog in events.dialogs:
            if dialog.answer == "waiting":
                notes.append(DIALOG_WAITING.format(kind=dialog.kind, message=dialog.message))
            else:
                notes.append(
                    DIALOG.format(kind=dialog.kind, message=dialog.message, answer=dialog.answer)
                )
            if dialog.default:
                notes.append(DIALOG_DEFAULT.format(default=dialog.default))
        if events.omitted_dialogs:
            notes.append(DIALOGS_OMITTED.format(n=events.omitted_dialogs))
        return notes

    async def _admit_downloads(self, events: PageEvents) -> list[str]:
        """Admit what the pages downloaded as Resources of this call, one note for each."""
        notes = []
        for index, file in enumerate(events.downloads):
            filename = browser_download_filename(file.suggested_filename)
            media_type = browser_download_media_type(filename, file.content)
            try:
                resource_id = await self._host.registry.admit_browser_resource(
                    BrowserResourceInput(
                        BROWSER_DOWNLOAD, file.content, filename, media_type, file.url
                    ),
                    effect_owner=self._owner,
                    index=index,
                )
            except ResourceRegistryError as exc:
                notes.append(NOT_ADMITTED.format(name=filename, why=exc))
                continue
            notes.append(
                DOWNLOADED.format(
                    filename=filename,
                    mime=media_type,
                    size=len(file.content),
                    resource_id=resource_id,
                )
            )
        return notes

    def _refusals(self, events: PageEvents) -> list[str]:
        """One note for every download the call refused, and one for the rest the bound left.

        The refusals past the bound are counted rather than listed: a page can start downloads
        without end, and a list that looked complete would be read as the whole of them.
        """
        limits = self._host.browser.limits
        reasons = {
            "too_large": f"it exceeds {limits.max_download_bytes} bytes",
            "timeout": f"it did not finish within {limits.navigation_timeout:g} seconds",
            "failed": "it could not be downloaded",
            "limit": f"one call admits at most {MAX_DOWNLOADS_PER_CALL} downloads",
        }
        notes = [
            NOT_ADMITTED.format(
                name=browser_download_filename(refusal.suggested_filename),
                why=reasons[refusal.reason],
            )
            for refusal in events.refused_downloads
        ]
        if events.omitted_refusals:
            notes.append(REFUSALS_OMITTED.format(n=events.omitted_refusals))
        return notes

    async def _bounded(self, report: str, snapshot: str) -> ToolResult:
        """The report and the snapshot, with the snapshot kept in full when it is too big.

        This is not ``preview_or_spill``: the frame and the notes of the call always reach
        the model whole, only the snapshot is kept and cut to its head, and a Run that cannot
        keep it still reports the action as completed rather than as a failure.
        """
        whole = "\n".join((report, snapshot))
        if within_result_bounds(whole):
            return ToolResult.text(whole)
        bounds = {"max_bytes": TOOL_RESULT_MAX_BYTES, "max_lines": TOOL_RESULT_MAX_LINES}
        excerpt = head_excerpt(snapshot).rstrip()
        if (receipt := await self._spilled(snapshot)) is None:
            # The action completed, so this is not a failure: reporting one would invite a
            # repeat, a second submit among them, and the head carries refs to act on.
            return ToolResult.text("\n".join((report, UNAVAILABLE.format(**bounds), excerpt)))
        protected = spill_continuation(receipt.resource_id)
        notice = SPILLED.format(size=len(snapshot.encode("utf-8")), **bounds)
        return ToolResult.text(
            "\n".join((report, notice, excerpt, protected)),
            protected_text=protected,
            effects=ToolEffects(committed_outputs=(receipt,)),
        )

    async def _spilled(self, snapshot: str) -> CommittedOutput | None:
        """The snapshot kept in full in the workspace, or None where nothing can keep it."""
        if self._spill is None:
            return None
        try:
            return await self._spill(snapshot)
        except OSError:
            return None


def _typed_email(text: str) -> str | None:
    """The address a field holds, when it can be one: a local part and a domain around one
    ``@``, no whitespace or control character, a sane length."""
    address = text.strip()
    local, _, domain = address.partition("@")
    if (
        address.count("@") == 1
        and local
        and domain
        and not any(c.isspace() or unicodedata.category(c) == "Cc" for c in address)
        and len(address) <= _MAX_EMAIL_CHARS
    ):
        return address
    return None


def _typed_username(text: str) -> str | None:
    """The username a field holds, when it can be one: 1 to 128 characters, none a control."""
    username = text.strip()
    if 1 <= len(username) <= _MAX_USERNAME_CHARS and not any(
        unicodedata.category(c) == "Cc" for c in username
    ):
        return username
    return None


def _label(url: str | None) -> str:
    """A page by where it is, never by its query: the host and path, or the scheme of a non-page."""
    if not url:
        return ""
    try:
        parts = urlsplit(url)
    except ValueError:
        return ""
    if parts.scheme not in {"http", "https"}:
        return parts.scheme
    return f"{parts.hostname or ''}{parts.path}"


__all__ = ["BrowserToolHost", "browser_declaration", "browser_tool"]
