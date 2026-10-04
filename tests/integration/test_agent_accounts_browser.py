# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Agent Accounts driven through the ``browser`` tool in a real Chromium (ADR 0034).

What stands in for the public Web is a loopback proxy that terminates TLS, so a test drives
``https`` pages of ``example.com`` and ``example.org`` and observes the requests the browser
sends, the rows the tool stores, and what the tool answers. No database is needed: accounts go to
a store held in memory, and the Resources the tool admits go to a recording sink.

A generated password is compared in code and never put into an assertion, a message or a log: a
test counts where it appears and asserts the count, so a failure reports a number.
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import pytest
from pydantic import SecretStr

from dlightrag.adapters.agent_mailbox import S3AgentMailbox
from dlightrag.engine.agent.environment import AccessScheduler
from dlightrag.engine.agent.tool_content import tool_content_attachments
from dlightrag.engine.agent.tools import AgentTool, ToolResult
from dlightrag.engine.answer.agent_browser import (
    ACCOUNT_LABEL,
    PASSWORD_MASK,
    AgentAccountsBinding,
    AgentAccountStore,
    AgentMailbox,
    BrowserHolder,
    RunAgentAccounts,
    RunAgentBrowser,
    StoredAgentAccount,
    generate_password,
    owner_alias,
)
from dlightrag.engine.answer.resources.registry import (
    FetchedResourceBytes,
    ResourceEffectOwner,
    ResourceRegistry,
)
from dlightrag.engine.answer.tools import browser as browser_module
from dlightrag.engine.answer.tools.browser import BrowserToolHost, browser_tool
from dlightrag.engine.answer.tools.resources import make_resource_reader
from dlightrag.engine.credential_cipher import CredentialCipher
from tests.support.agent_browser import (
    MemoryAccountStore,
    Served,
    StubMailbox,
    WebProxy,
    browser_settings,
    launched_chromium,
    web_proxy,
)
from tests.support.dns import public_dns
from tests.support.loopback import bypass_proxies, loopback_certificate
from tests.support.resources import preparer
from tests.support.s3 import S3Stub, StoredObject, s3_stub
from tests.tool_helpers import recording_tool_runtime

pytestmark = pytest.mark.asyncio

OWNER = "owner"
HOLDER = BrowserHolder(OWNER, "11111111-1111-1111-1111-111111111111", "worker", 1)
SHOP = "https://example.com"
SITE = "example.com"
KEYRING = '{"active": "test", "keys": {"test": "YWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWE="}}'
OTHER_KEYRING = (
    '{"active": "next", "keys": {"next": "YmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmI="}}'
)
EMAIL, HANDLE = "shopper@example.com", "shopper-77"
DOMAIN = "orliantra.cc"


def page(title: str, body: str) -> Served:
    return Served(f"<html><head><title>{title}</title></head><body>{body}</body></html>")


def signup(*, password: str = 'type="password"', confirm: bool = True) -> Served:
    """A sign-up form posting to /join. ``password`` is the attributes of its password field."""
    return page(
        "Join",
        f"""<form action="/join" method="post">
<input aria-label="Handle" name="handle">
<input aria-label="Email" name="email" type="email">
<input aria-label="Password" name="password" {password}>
{'<input aria-label="Confirmation" name="confirm" type="password">' if confirm else ""}
<button type="submit">Join</button></form>""",
    )


SIGNIN = page(
    "Sign in",
    """<form action="/session" method="post">
<input aria-label="Email" name="email" type="email">
<input aria-label="Password" name="password" type="password">
<button type="submit">Sign in</button></form>""",
)
# The form that asks a site to mail a password-reset link.
FORGOT = page(
    "Forgot",
    """<form action="/sent" method="post">
<input aria-label="Email" name="email" type="email">
<button type="submit">Send reset mail</button></form>""",
)
# A page that mirrors what is typed into its password fields into its own attributes, title,
# links and dialogs, and submits one of them with GET: every way a driver or a page prints a value.
MIRROR = page(
    "Mirror",
    """<input aria-label="Handle" name="handle">
<input aria-label="Secret" type="password" id="pw" oninput="mirror(this)">
<form action="/search" method="get">
<input aria-label="Again" type="password" name="password" oninput="mirror(this)">
<button type="submit">Go</button></form>
<button onclick="alert('echo ' + document.getElementById('pw').value)">Alert</button>
<a id="dl" download href="#">Save</a>
<script>function mirror(input) {
  input.setAttribute('value', input.value);
  if (input.id === 'pw') {
    document.title = 'Mirror ' + input.value;
    document.getElementById('dl').href = '/files/' + encodeURIComponent(input.value) + '.csv';
  } }</script>""",
)
EMBED = page(
    "Embed",
    """<input aria-label="Handle" name="handle">
<input aria-label="Email" type="email">
<input aria-label="Password" type="password">
<iframe src="https://pay.example.org/card"></iframe>
<iframe src="https://accounts.example.com/card"></iframe>""",
)
PAGES = {
    f"{SHOP}/signup": signup(),
    f"{SHOP}/signin": SIGNIN,
    f"{SHOP}/forgot": FORGOT,
    f"{SHOP}/sent": page("Sent", "<p>Check your mail</p>"),
    f"{SHOP}/reset?token=abc123": signup(),
    f"{SHOP}/mirror": MIRROR,
    # A password field restyled to show its characters, which Chromium ignores: the guard reads
    # what a field holds and never how it is drawn.
    f"{SHOP}/revealed": signup(
        password='type="password" style="-webkit-text-security: none"', confirm=False
    ),
    f"{SHOP}/embed": EMBED,
    f"{SHOP}/short": signup(password='type="password" maxlength="14"', confirm=False),
    f"{SHOP}/tiny": signup(password='type="password" maxlength="10"', confirm=False),
    f"{SHOP}/locked": signup(password='type="password" disabled', confirm=False),
    # Its handler drops every capital letter of a value it is given, so what it holds is not what
    # was filled.
    f"{SHOP}/stripping": signup(
        password="""type="password" oninput="this.value = this.value.replace(/[A-Z]/g, '')" """,
        confirm=False,
    ),
    f"{SHOP}/join": page("Welcome", "<p>Thanks for joining</p>"),
    f"{SHOP}/verify?token=abc123": page("Verified", "<p>Your account is confirmed</p>"),
    f"{SHOP}/session": page("Dashboard", "<p>Your dashboard</p>"),
    "https://pay.example.org/card": page(
        "Card", '<input aria-label="Other secret" type="password">'
    ),
    "https://accounts.example.com/card": page(
        "Account", '<input aria-label="Same-site secret" type="password">'
    ),
    "http://example.com/signup": signup(),
}


@pytest.fixture(autouse=True)
def _hosts_resolve_public(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)


@dataclass
class Browsing:
    """A Run's ``browser`` tool over a real Chromium, and everything it leaves behind."""

    tools: dict[bool, AgentTool]
    run: RunAgentBrowser
    proxy: WebProxy
    store: AgentAccountStore
    cipher: CredentialCipher
    admitted: list[tuple[FetchedResourceBytes, ResourceEffectOwner | None]]
    #: The text of every result and live update a call produced: where a password could show.
    transcript: list[str] = field(default_factory=list)
    #: The passwords the test knows, which no call may show: each call is checked as it returns,
    #: before the test reads its text, so a leak fails with a count and never prints the text.
    watching: list[SecretStr] = field(default_factory=list)

    async def call(
        self, scope: str = "parent", *, child: bool = False, **arguments: Any
    ) -> ToolResult:
        tool, updates = self.tools[child], []
        runtime = recording_tool_runtime(updates, tool_name="browser", execution_scope=scope)
        result = await tool.execute(tool.input_model.model_validate(arguments), runtime)
        texts = [repr(result), result.text_content, *(repr(u) for u in updates)]
        self.transcript += texts
        for password in self.watching:
            assert_hidden(password, *texts)
        return result

    def watch(self, password: SecretStr) -> SecretStr:
        """Keep ``password`` out of every text from now on, and check the ones so far."""
        self.watching.append(password)
        assert_hidden(password, *self.transcript)
        return password

    async def form(self, url: str, scope: str = "parent", *, child: bool = False) -> dict[str, str]:
        """Open ``url`` and name the refs of the fields and buttons its snapshot shows."""
        opened = await self.call(scope, child=child, action="navigate", url=url)
        text = opened.text_content
        found = (
            ("handle", 'textbox "Handle"'),
            ("email", 'textbox "Email"'),
            ("password", 'textbox "Password"'),
            ("confirm", 'textbox "Confirmation"'),
            ("button", "button "),
        )
        return {name: ref_of(text, label) for name, label in found if f"{label}" in text}

    @property
    def rows(self) -> list[StoredAgentAccount]:
        """What the in-memory store holds."""
        assert isinstance(self.store, MemoryAccountStore)
        return list(self.store.rows.values())

    def stored_password(self, site: str = SITE) -> SecretStr:
        """The password of the owner's account on ``site``, opened from its stored envelope and
        watched from then on."""
        assert isinstance(self.store, MemoryAccountStore)
        row = self.store.rows[(OWNER, site)]
        opened = self.cipher.open(
            row.envelope, label=ACCOUNT_LABEL, binding=(OWNER, site, row.account_id)
        )
        return self.watch(opened)

    async def register(
        self, form: dict[str, str], *, email: str | None = EMAIL, handle: str = HANDLE, **more: Any
    ) -> ToolResult:
        """Type a handle, and an address unless the mailbox supplies one, then register."""
        await self.call(action="type", ref=form["handle"], text=handle)
        if email is not None:
            await self.call(action="type", ref=form["email"], text=email)
        refs = [form["password"], *([form["confirm"]] if "confirm" in form else [])]
        return await self.call(
            action="register",
            password_refs=refs,
            email_ref=form["email"],
            username_ref=form["handle"],
            **more,
        )


def ref_of(snapshot: str, text: str) -> str:
    """The ref on the first line of ``snapshot`` that holds ``text``."""
    for line in snapshot.splitlines():
        if text in line and (marker := re.search(r"\[ref=([^\]]+)\]", line)):
            return marker.group(1)
    raise AssertionError(f"no ref for {text!r} in:\n{snapshot}")


def shows_mask(snapshot: str, label: str, ref: str) -> bool:
    """Whether ``snapshot`` shows the textbox ``label`` holding the mask. The driver quotes a value
    YAML would read as something else, and the mask, which starts with an asterisk, is one."""
    pattern = rf'textbox "{label}"( \[active\])? \[ref={ref}\]: "?{re.escape(PASSWORD_MASK)}"?$'
    return re.search(pattern, snapshot, re.MULTILINE) is not None


def leaks(password: SecretStr, *texts: str | bytes) -> int:
    """How many times ``password`` appears in ``texts``: the only thing a test says about it."""
    needle = password.get_secret_value()
    return sum(
        text.count(needle.encode()) if isinstance(text, bytes) else text.count(needle)
        for text in texts
    )


def posts(proxy: WebProxy, path: str) -> list[dict[str, list[str]]]:
    """The form fields of every POST the browser sent to ``path``, as ``parse_qs`` reads them."""
    return [
        parse_qs(request.body.decode())
        for request in proxy.requests
        if request.method == "POST" and request.target == f"{SHOP}{path}"
    ]


def posted(proxy: WebProxy, path: str) -> dict[str, list[str]]:
    """The form fields of the one POST the browser sent to ``path``."""
    (fields,) = posts(proxy, path)
    return fields


def assert_hidden(password: SecretStr, *texts: str | bytes) -> None:
    """Fail, saying how often and never what, if ``password`` is in any of ``texts``."""
    count = leaks(password, *texts)
    assert count == 0


def size(password: SecretStr) -> int:
    return len(password.get_secret_value())


def assert_sent(fields: dict[str, list[str]], password: SecretStr, *names: str) -> None:
    """Fail, saying which fields and never what, unless each named field of a POST carries
    exactly ``password``."""
    wrong = [name for name in names if fields[name] != [password.get_secret_value()]]
    assert wrong == []


@asynccontextmanager
async def browsing(
    directory: Path,
    *,
    store: AgentAccountStore | None = None,
    keyring: str | None = KEYRING,
    mailbox: AgentMailbox | None = None,
    pages: dict[str, Served] | None = None,
    **bounds: Any,
) -> AsyncIterator[Browsing]:
    """A Run's browser tool over a Chromium that reaches ``pages`` through a TLS-terminating proxy."""
    admitted: list[tuple[FetchedResourceBytes, ResourceEffectOwner | None]] = []

    async def sink(fetched: FetchedResourceBytes, owner: ResourceEffectOwner | None) -> None:
        admitted.append((fetched, owner))

    cipher = CredentialCipher(None if keyring is None else SecretStr(keyring))
    accounts_store = MemoryAccountStore() if store is None else store
    async with AsyncExitStack() as stack:
        proxy = await stack.enter_async_context(
            web_proxy(PAGES if pages is None else pages, tls=loopback_certificate(directory))
        )
        provider = await stack.enter_async_context(launched_chromium(proxy))
        settings = browser_settings(**{"navigation": 5, "settle": 0.3, "action": 3, **bounds})
        run = RunAgentBrowser(provider, HOLDER, settings)
        stack.push_async_callback(run.aclose)
        registry = await stack.enter_async_context(ResourceRegistry(fetched_bytes_sink=sink))
        accounts = RunAgentAccounts(
            owner_id=OWNER, binding=AgentAccountsBinding(accounts_store, cipher, mailbox)
        )
        host = BrowserToolHost(run, registry, make_resource_reader(registry, 4000), accounts)
        tools = {
            child: browser_tool(
                host,
                environment=None,
                scheduler=AccessScheduler(),
                spill=None,
                image_preparer=preparer(3),
                child=child,
            )
            for child in (False, True)
        }
        yield Browsing(tools, run, proxy, accounts_store, cipher, admitted)


# -- register ---------------------------------------------------------------------------------


@pytest.mark.parametrize("mailbox", [False, True], ids=["typed address", "mailbox alias"])
async def test_register_fills_a_generated_password_and_never_shows_it(
    tmp_path: Path, mailbox: bool
) -> None:
    async with browsing(tmp_path, mailbox=StubMailbox(DOMAIN) if mailbox else None) as web:
        form = await web.form(f"{SHOP}/signup")
        # With an Agent Mailbox the address is the owner's alias for the site, which DlightRAG
        # fills, and without one it is the address the Agent typed.
        email = owner_alias(OWNER, SITE, DOMAIN) if mailbox else EMAIL

        registered = await web.register(form, email=None if mailbox else EMAIL)

        assert not registered.is_error, registered.text_content
        password = web.stored_password()
        text = registered.text_content
        assert text.startswith(f"[browser: register | page: {SHOP}/signup | title: Join]\n")
        assert (
            f"Recorded the Agent Account {email} for {SITE} for this owner's later Runs and "
            "filled its generated password into 2 field(s); the password is never shown."
        ) in text
        assert (f'Mail to {email} appears in browser(action="inbox").' in text) is mailbox
        # The snapshot it returns shows the fields filled and the password as the mask.
        assert shows_mask(text, "Password", form["password"])
        assert shows_mask(text, "Confirmation", form["confirm"])
        (row,) = web.rows
        assert (row.owner_id, row.site, row.email, row.username) == (OWNER, SITE, email, HANDLE)
        assert size(password) == 20
        # A later snapshot shows the mask too.
        shown = await web.call(action="snapshot")
        assert shown.text_content.count(PASSWORD_MASK) == 2

        await web.call(action="click", ref=form["button"])

        # The site received the password in both fields, and the model never did.
        fields = posted(web.proxy, "/join")
        assert_sent(fields, password, "password", "confirm")
        assert (fields["email"], fields["handle"]) == ([email], [HANDLE])


async def test_registering_again_fills_a_new_password_and_the_address_the_account_has(
    tmp_path: Path,
) -> None:
    async with browsing(tmp_path) as web:
        await web.register(await web.form(f"{SHOP}/signup"))
        before = web.stored_password()

        # A password-reset form is a sign-up form of a site whose account exists.
        form = await web.form(f"{SHOP}/signup")
        await web.call(action="type", ref=form["handle"], text=HANDLE)
        reset = await web.call(
            action="register",
            password_refs=[form["password"], form["confirm"]],
            email_ref=form["email"],
            username_ref=form["handle"],
        )

        renewed = web.stored_password()
        assert not reset.is_error, reset.text_content
        assert (
            f"Gave the Agent Account {EMAIL} for {SITE} for this owner's later Runs a new "
            "generated password, filled into 2 field(s). Submit the form with click or press."
        ) in reset.text_content
        assert renewed != before
        await web.call(action="click", ref=form["button"])
        # The address was filled by DlightRAG, since the account already had one.
        fields = posted(web.proxy, "/join")
        assert_sent(fields, renewed, "password", "confirm")
        assert fields["email"] == [EMAIL]


async def test_every_page_text_is_redacted(tmp_path: Path) -> None:
    async with browsing(tmp_path) as web:
        opened = await web.call(action="navigate", url=f"{SHOP}/mirror")
        handle, secret, again = (
            ref_of(opened.text_content, f'textbox "{label}"')
            for label in ("Handle", "Secret", "Again")
        )
        await web.call(action="type", ref=handle, text=HANDLE)

        registered = await web.call(
            action="register", password_refs=[secret, again], username_ref=handle
        )

        assert not registered.is_error, registered.text_content
        password = web.stored_password()
        value = password.get_secret_value()
        # A file the page links by a percent-encoded address, which a site names after the
        # password in its file name and echoes in its rows.
        web.proxy.add(
            f"{SHOP}/files/{value}.csv",
            Served(
                f"id,password\n1,{value}\n".encode(),
                headers={
                    "content-type": "text/csv",
                    "content-disposition": f'attachment; filename="{value}.csv"',
                },
            ),
        )
        # The page mirrored the value into its title and its attributes as it was filled.
        assert f"title: Mirror {PASSWORD_MASK}]" in registered.text_content
        # A query that starts as the password does matches nothing, and the lines that do match
        # show the mask.
        found = await web.call(action="find", query="Secret")
        assert shows_mask(found.text_content, "Secret", secret)
        probe = await web.call(action="find", query=value[:6])
        # The answer echoes the query, so the test compares in code and never prints it.
        matched_nothing = probe.text_content.splitlines()[-1].startswith("No element matches")
        assert matched_nothing
        # The serialized page and its address, a dialog it opens and a file it links to.
        await web.call(action="capture")
        (capture, _) = web.admitted[-1]
        assert_hidden(password, capture.content, capture.url)
        assert f'value="{PASSWORD_MASK}"'.encode() in capture.content
        alerted = await web.call(action="click", ref=await ref_in(web, 'button "Alert"'))
        assert f'a alert dialog: "echo {PASSWORD_MASK}" (accepted).' in alerted.text_content
        await web.call(action="click", ref=await ref_in(web, 'link "Save"'))
        (download, _) = web.admitted[-1]
        assert_hidden(password, download.url, download.filename, download.content)
        assert download.url == f"{SHOP}/files/{PASSWORD_MASK}.csv"
        # The name was masked before it was made safe, and the rows as the file was read.
        assert download.filename == "________.csv"
        echoed_masked = download.content == f"id,password\n1,{PASSWORD_MASK}\n".encode()
        assert echoed_masked
        # A form that submits with GET puts the password into the page's own address.
        went = await web.call(action="click", ref=await ref_in(web, 'button "Go"'))
        assert f"page: {SHOP}/search?password={PASSWORD_MASK} |" in went.text_content
        # The browser did send the password to the site in that address, so the mask is what hid it.
        asked = [request.target for request in web.proxy.requests]
        assert leaks(password, *asked) == 2


async def ref_in(web: Browsing, label: str) -> str:
    """The ref of the element of the current page whose snapshot line holds ``label``."""
    found = await web.call(action="find", query=label.split('"')[1])
    return ref_of(found.text_content, label)


async def test_a_password_fills_only_password_fields_of_the_pages_site(tmp_path: Path) -> None:
    async with browsing(tmp_path) as web:
        opened = await web.call(action="navigate", url=f"{SHOP}/embed")
        text = opened.text_content
        handle, email, password = (
            ref_of(text, f'textbox "{label}"') for label in ("Handle", "Email", "Password")
        )
        other, same = (
            ref_of(text, 'textbox "Other secret"'),
            ref_of(text, 'textbox "Same-site secret"'),
        )

        elsewhere = await web.call(action="register", password_refs=[other], username_ref=handle)
        not_a_password = await web.call(
            action="register", password_refs=[email], username_ref=handle
        )
        not_text = await web.call(action="register", password_refs=[password], email_ref=password)

        assert elsewhere.is_error and elsewhere.text_content == (
            f"Field {other} is not on an https page of {SITE}, so nothing was filled: register "
            "and login fill only the fields of the page's own site."
        )
        assert not_a_password.text_content == (
            f"Element {email} is not a password field, so nothing was filled."
        )
        assert not_text.text_content == (
            f"Element {password} is not a text or email field, so nothing was filled."
        )
        # Nothing was made, filled or stored.
        assert (web.rows, bool(web.run.filled_passwords("parent"))) == ([], False)

        await web.call(action="type", ref=handle, text=HANDLE)
        registered = await web.call(
            action="register", password_refs=[password], username_ref=handle
        )
        assert not registered.is_error, registered.text_content
        # The stored password goes into a frame of the same site, and into no other site's.
        stolen = await web.call(action="login", password_refs=[other])
        accepted = await web.call(action="login", password_refs=[same])
        assert stolen.is_error and "is not on an https page of example.com" in stolen.text_content
        assert not accepted.is_error, accepted.text_content
        lines = {
            label: ln
            for ln in accepted.text_content.splitlines()
            for label in ("Same-site secret", "Other secret")
            if label in ln
        }
        assert shows_mask(lines["Same-site secret"], "Same-site secret", same)
        assert lines["Other secret"].endswith(f"[ref={other}]")

        # A page that is not https has no site to hold an account.
        form = await web.form("http://example.com/signup")
        plain = await web.register(form)
        assert plain.is_error and plain.text_content == (
            "register and login need an https page with a registrable domain; this page is "
            "example.com/signup."
        )
        assert len(web.rows) == 1


@pytest.mark.parametrize(
    ("page_path", "length", "stored"), [("short", 14, True), ("tiny", 10, False)]
)
async def test_maxlength_shortens_the_password_and_below_12_refuses(
    tmp_path: Path, page_path: str, length: int, stored: bool
) -> None:
    async with browsing(tmp_path) as web:
        form = await web.form(f"{SHOP}/{page_path}")

        registered = await web.register(form)

        assert registered.is_error is not stored
        if stored:
            await web.call(action="click", ref=form["button"])
            sent = [len(value) for value in posted(web.proxy, "/join")["password"]]
            assert (sent, size(web.stored_password())) == ([length], length)
        else:
            assert registered.text_content == (
                f"This site's password field takes at most {length} characters, fewer than the "
                "12 an Agent Account needs, so nothing was filled."
            )
            assert (web.rows, bool(web.run.filled_passwords("parent"))) == ([], False)


# -- a registration that does not complete ------------------------------------------------------


@pytest.fixture
def generated(monkeypatch: pytest.MonkeyPatch) -> list[SecretStr]:
    """Every password the tool generates, which a flow that stores nothing gives no other way to learn."""
    made: list[SecretStr] = []
    real = browser_module.generate_password

    def recording(length: int = 20) -> SecretStr:
        made.append(password := real(length))
        return password

    monkeypatch.setattr(browser_module, "generate_password", recording)
    return made


async def is_empty(web: Browsing, label: str) -> bool:
    """Whether the page's snapshot shows its textbox ``label`` without a value. What a field
    holds is never read out, since a page can leave a variant of the password in it."""
    shown = await web.call(action="snapshot")
    (line,) = [ln for ln in shown.text_content.splitlines() if f'textbox "{label}"' in ln]
    return "]: " not in line


@pytest.mark.parametrize(
    ("page_path", "sentence"),
    [
        (
            "locked",
            "The element {ref} could not be filled within 1 seconds; it may be hidden, disabled, "
            "or covered. Call snapshot to see the page again.",
        ),
        (
            "stripping",
            "The page changed the value filled into {ref}, so nothing was recorded and the "
            "fields were cleared.",
        ),
    ],
)
async def test_a_failed_fill_leaks_nothing_and_records_nothing(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    generated: list[SecretStr],
    page_path: str,
    sentence: str,
) -> None:
    caplog.set_level(logging.DEBUG)
    async with browsing(tmp_path, action=1.0) as web:
        form = await web.form(f"{SHOP}/{page_path}")

        failed = await web.register(form)

        # The driver's own error for a failed fill quotes the value in its call log; none of that
        # reaches the result or any log, and the password was never stored.
        (password,) = generated
        web.watch(password)
        assert_hidden(password, caplog.text)
        assert failed.is_error and failed.text_content == sentence.format(ref=form["password"])
        assert web.rows == []
        # No field is left holding what was filled, nor what the page made of it.
        assert await is_empty(web, "Password")


async def test_a_password_that_cannot_be_stored_is_cleared_from_the_form(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, generated: list[SecretStr]
) -> None:
    class Down(MemoryAccountStore):
        async def save(self, account: StoredAgentAccount) -> None:
            raise ConnectionError("the database is down")

    caplog.set_level(logging.DEBUG)
    async with browsing(tmp_path, store=Down()) as web:
        form = await web.form(f"{SHOP}/signup")

        failed = await web.register(form)

        (password,) = generated
        web.watch(password)
        assert_hidden(password, caplog.text)
        assert failed.is_error and failed.text_content == (
            "The fields were filled but the account could not be stored, so they were cleared; "
            "do not submit the form."
        )
        assert "An Agent Account could not be recorded (ConnectionError)" in caplog.text
        assert web.rows == []
        assert [await is_empty(web, label) for label in ("Password", "Confirmation")] == [
            True,
            True,
        ]
        assert_hidden(password, caplog.text)


@pytest.mark.parametrize(
    ("typed", "refs", "sentence"),
    [
        (
            {"handle": HANDLE},
            ("email",),
            "Field {email} holds no email address to record; type the address first.",
        ),
        (
            {"handle": HANDLE, "email": "not an address"},
            ("email",),
            "Field {email} holds no email address to record; type the address first.",
        ),
        (
            {"handle": HANDLE, "email": "shopper\x07@example.com"},
            ("email",),
            "Field {email} holds no email address to record; type the address first.",
        ),
        (
            {"handle": HANDLE, "email": "@ab"},
            ("email",),
            "Field {email} holds no email address to record; type the address first.",
        ),
        (
            {"handle": HANDLE, "email": "ab@"},
            ("email",),
            "Field {email} holds no email address to record; type the address first.",
        ),
        (
            {"email": EMAIL},
            ("email", "handle"),
            "Field {handle} holds no username to record; type it first.",
        ),
        (
            {"handle": HANDLE, "email": EMAIL},
            (),
            "A new account needs email_ref or username_ref, so that login can name it later.",
        ),
    ],
)
async def test_register_refuses_what_it_cannot_record_and_fills_nothing(
    tmp_path: Path, typed: dict[str, str], refs: tuple[str, ...], sentence: str
) -> None:
    async with browsing(tmp_path) as web:
        form = await web.form(f"{SHOP}/signup")
        for name, value in typed.items():
            await web.call(action="type", ref=form[name], text=value)
        named: dict[str, Any] = {
            "email_ref" if name == "email" else "username_ref": form[name] for name in refs
        }

        refused = await web.call(
            action="register", password_refs=[form["password"], form["confirm"]], **named
        )

        assert refused.is_error and refused.text_content == sentence.format(**form)
        assert (web.rows, bool(web.run.filled_passwords("parent"))) == ([], False)


# -- login ------------------------------------------------------------------------------------


async def test_login_fills_the_stored_account_in_a_fresh_session(tmp_path: Path) -> None:
    async with browsing(tmp_path) as first:
        await first.register(await first.form(f"{SHOP}/signup"))
        store, password = first.store, first.stored_password()

    # Another Run of the same owner: a new browser with nothing in it, over the same store.
    async with browsing(tmp_path, store=store) as later:
        later.watch(password)
        form = await later.form(f"{SHOP}/signin")
        assert [row.last_used_at for row in await store.summaries(owner_id=OWNER)] == [None]

        filled = await later.call(
            action="login", email_ref=form["email"], password_refs=[form["password"]]
        )

        assert not filled.is_error, filled.text_content
        # The owner's Settings show the day this login filled the account.
        assert [row.last_used_at is not None for row in await store.summaries(owner_id=OWNER)] == [
            True
        ]
        text = filled.text_content
        assert (
            f"Filled the Agent Account {EMAIL} for {SITE} into email, 1 password field(s). "
            "Submit the form with click or press."
        ) in text
        assert shows_mask(text, "Password", form["password"])
        await later.call(action="click", ref=form["button"])
        fields = posted(later.proxy, "/session")
        assert fields["email"] == [EMAIL]
        assert_sent(fields, password, "password")


async def test_a_login_whose_last_use_cannot_be_recorded_still_fills_the_form(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    class Down(MemoryAccountStore):
        async def mark_used(self, account: StoredAgentAccount) -> None:
            raise ConnectionError("the database is down")

    caplog.set_level(logging.WARNING)
    async with browsing(tmp_path, store=Down()) as web:
        await web.register(await web.form(f"{SHOP}/signup"))
        password = web.stored_password()
        form = await web.form(f"{SHOP}/signin")

        filled = await web.call(
            action="login", email_ref=form["email"], password_refs=[form["password"]]
        )

        assert not filled.is_error, filled.text_content
        assert shows_mask(filled.text_content, "Password", form["password"])
        assert "An Agent Account's last use could not be recorded (ConnectionError)" in caplog.text
        assert_hidden(password, caplog.text)


async def test_login_says_what_the_account_lacks_and_that_a_deployment_without_a_ring_has_none(
    tmp_path: Path,
) -> None:
    async with browsing(tmp_path) as web:
        signin = await web.form(f"{SHOP}/signin")
        nothing = await web.call(action="login", email_ref=signin["email"])
        assert nothing.is_error and nothing.text_content == (
            'No Agent Account exists for example.com. Register one with browser(action="register", ...).'
        )

        # An account of a username alone has no address to fill.
        form = await web.form(f"{SHOP}/signup")
        await web.call(action="type", ref=form["handle"], text=HANDLE)
        await web.call(
            action="register",
            password_refs=[form["password"], form["confirm"]],
            username_ref=form["handle"],
        )
        signin = await web.form(f"{SHOP}/signin")
        no_email = await web.call(action="login", email_ref=signin["email"])
        assert no_email.text_content == (
            "The Agent Account for example.com has no email address; use username_ref instead."
        )
        store = web.store

    async with browsing(tmp_path, store=store, keyring=None) as keyless:
        signin = await keyless.form(f"{SHOP}/signin")
        form = await keyless.form(f"{SHOP}/signup")
        refused = [
            await keyless.call(action="login", email_ref=signin["email"]),
            await keyless.register(form),
        ]
        assert [r.text_content for r in refused] == [
            "Agent Accounts are unavailable: this deployment has no credential key ring, so "
            "no password can be stored or read."
        ] * 2
        assert all(r.is_error for r in refused)


async def test_a_childs_registration_is_run_scoped(tmp_path: Path) -> None:
    async with browsing(tmp_path, mailbox=StubMailbox(DOMAIN)) as web:
        await web.register(await web.form(f"{SHOP}/signup"), email=None)
        (owners_row,) = web.rows
        owners, owners_alias = web.stored_password(), owner_alias(OWNER, SITE, DOMAIN)

        # Child A registers an account of its own for this Run, under an address of its own that
        # is not the owner's, and the store never sees it.
        form = await web.form(f"{SHOP}/signup", "child-a", child=True)
        await web.call("child-a", child=True, action="type", ref=form["handle"], text="child-a")
        registered = await web.call(
            "child-a",
            child=True,
            action="register",
            password_refs=[form["password"], form["confirm"]],
            email_ref=form["email"],
            username_ref=form["handle"],
        )
        recorded = re.search(
            rf"Recorded the Agent Account ([a-z2-7]{{16}}@{re.escape(DOMAIN)}) for {SITE} for "
            r"this Run only \(a Child Session's account\)\.? ?",
            registered.text_content,
        )
        assert recorded is not None and recorded.group(1) != owners_alias
        alias = recorded.group(1)
        assert web.rows == [owners_row]
        await web.call("child-a", child=True, action="click", ref=form["button"])
        # The site's copy of the Child's password is the only one a test can learn, and from here
        # on no call may show it either.
        (join,) = posts(web.proxy, "/join")
        web.watch(SecretStr(join["password"][0]))

        # Its login fills its own account, and another Child's falls back to the owner's.
        last_used = []
        for scope in ("child-a", "child-b"):
            signin = await web.form(f"{SHOP}/signin", scope, child=True)
            await web.call(
                scope,
                child=True,
                action="login",
                email_ref=signin["email"],
                password_refs=[signin["password"]],
            )
            await web.call(scope, child=True, action="click", ref=signin["button"])
            last_used.append(
                [row.last_used_at is not None for row in await web.store.summaries(owner_id=OWNER)]
            )
        # A login with a Child's own account writes nothing, and one with the owner's account
        # records its use, which adds the Child no authority.
        assert last_used == [[False], [True]]

        own, fallback = posts(web.proxy, "/session")
        assert own["email"] == [alias]
        # Its own password is the one it registered with, and not the owner's.
        reused = (
            own["password"] == join["password"],
            own["password"] == [owners.get_secret_value()],
        )
        assert reused == (True, False)
        assert fallback["email"] == [owners_alias]
        assert_sent(fallback, owners, "password")
        assert web.rows == [owners_row]


# -- screenshots ------------------------------------------------------------------------------


@pytest.mark.parametrize("page_path", ["signup", "revealed"])
async def test_a_screenshot_waits_until_the_form_holding_a_password_is_gone(
    tmp_path: Path, page_path: str
) -> None:
    async with browsing(tmp_path) as web:
        form = await web.form(f"{SHOP}/{page_path}")
        await web.register(form)
        web.stored_password()

        # A field drawn as dots holds the password as much as one drawn as text, so neither is
        # photographed.
        held = await web.call(action="screenshot")
        assert held.is_error and held.text_content == (
            "The page holds a filled password, so no screenshot was taken. Take it after the "
            "form is submitted."
        )
        assert tool_content_attachments(held.parts) == () and held.effects.attached_resources == ()

        await web.call(action="click", ref=form["button"])
        taken = await web.call(action="screenshot")

        assert not taken.is_error and len(tool_content_attachments(taken.parts)) == 1


# -- the inbox --------------------------------------------------------------------------------


def mail(body: str, *, subject: str = "Confirm your account") -> bytes:
    headers = f"From: Shop <noreply@example.com>\r\nSubject: {subject}\r\n"
    return f"{headers}Content-Type: text/plain; charset=utf-8\r\n\r\n{body}".encode()


def delivered(minutes_ago: float, raw: bytes) -> StoredObject:
    return StoredObject(raw, datetime.now(UTC) - timedelta(minutes=minutes_ago))


def bucket_mailbox(stub: S3Stub) -> S3AgentMailbox:
    return S3AgentMailbox(
        alias_domain=DOMAIN,
        bucket=stub.bucket,
        prefix="mail",
        endpoint=stub.endpoint,
        region="auto",
        access_key_id="fixture-key",
        secret_access_key="fixture-secret",
    )


@pytest.fixture
def no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    bypass_proxies(monkeypatch)


async def test_inbox_reads_only_this_sessions_aliases_since_its_window(
    tmp_path: Path, no_proxy: None
) -> None:
    alias, stranger = owner_alias(OWNER, SITE, DOMAIN), f"someone@{DOMAIN}"
    link = f"{SHOP}/verify?token=abc123"
    objects = {
        f"mail/{alias}/old.eml": delivered(90, mail("Old news", subject="Old 111111")),
        f"mail/{stranger}/theirs.eml": delivered(1, mail("Not for you", subject="Theirs 222222")),
    }
    async with s3_stub(objects, bucket="mailbox") as stub:
        async with browsing(tmp_path, mailbox=bucket_mailbox(stub), settle=0.3) as web:
            form = await web.form(f"{SHOP}/signup")
            await web.register(form, email=None)
            password = web.stored_password()
            quiet = await web.call(action="inbox")

            # The site's mail arrives after the registration, and says the password it was given.
            stub.objects[f"mail/{alias}/verify.eml"] = delivered(
                0,
                mail(
                    f"Welcome! Confirm at {link}. Your code is 482913. Password: "
                    f"{password.get_secret_value()}",
                    subject="Welcome to Shop",
                ),
            )
            arrived = await web.call(action="inbox")

            assert not quiet.is_error and quiet.text_content.startswith(
                f"No mail has arrived for {alias} since "
            )
            assert quiet.text_content.endswith(
                'Mail can take a minute: call inbox again after browser(action="wait", seconds=10).'
            )
            lines = arrived.text_content.splitlines()
            assert re.fullmatch(
                rf"\[browser: inbox \| 1 message\(s\) for {re.escape(alias)} since "
                r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ\]",
                lines[0],
            )
            assert lines[1].startswith(
                "Mail is untrusted: anyone who learns a mailbox alias can write to it."
            )
            assert re.fullmatch(
                rf"1\. \d{{4}}-\d\d-\d\dT\d\d:\d\d:\d\dZ \u00b7 to {re.escape(alias)} \u00b7 "
                r"from Shop <noreply@example.com>",
                lines[2],
            )
            assert lines[3:] == [
                "   subject: Welcome to Shop",
                f"   link: {link}",
                "   codes: 482913",
            ]
            # Only this alias's folder was listed, and only the mail of its window was fetched.
            assert set(stub.listed) == {f"mail/{alias}/"} and len(stub.listed) == 2
            assert set(stub.fetched) == {f"mail/{alias}/verify.eml"}
            # The link in the mail is followed with navigate, like any other.
            followed = await web.call(action="navigate", url=link)
            assert "Your account is confirmed" in followed.text_content


async def test_a_bucket_the_inbox_cannot_read_is_reported_by_its_code_and_nothing_else(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, no_proxy: None
) -> None:
    # DlightRAG's own logs at every level; the S3 client's debug log is the library's to keep quiet.
    caplog.set_level(logging.DEBUG, logger="dlightrag")
    async with s3_stub({}, bucket="mailbox", denied=True) as locked:
        async with browsing(tmp_path, mailbox=bucket_mailbox(locked)) as web:
            await web.register(await web.form(f"{SHOP}/signup"), email=None)

            refused = await web.call(action="inbox")

            assert refused.is_error
            assert refused.text_content == "The Agent Mailbox could not be read (AccessDenied)."
            assert "The Agent Mailbox could not be read (AccessDenied)" in caplog.text
            # Neither the endpoint nor a key is anywhere a log or the result could carry it.
            seen = f"{caplog.text} {refused.text_content} {web.transcript}"
            assert not any(
                secret in seen for secret in (locked.endpoint, "fixture-key", "fixture-secret")
            )


async def test_an_account_whose_key_is_lost_recovers_through_its_reset_mail(
    tmp_path: Path, no_proxy: None
) -> None:
    alias, link = owner_alias(OWNER, SITE, DOMAIN), f"{SHOP}/reset?token=abc123"
    # An earlier Run registered with the owner's alias under a ring that this deployment no
    # longer has.
    key_id, envelope = CredentialCipher(SecretStr(KEYRING)).seal(
        generate_password(), label=ACCOUNT_LABEL, binding=(OWNER, SITE, "account")
    )
    store = MemoryAccountStore()
    store.rows[(OWNER, SITE)] = StoredAgentAccount(
        OWNER, SITE, "account", alias, None, key_id, envelope
    )
    async with s3_stub({}, bucket="mailbox") as bucket:
        mailbox = bucket_mailbox(bucket)
        async with browsing(tmp_path, store=store, keyring=OTHER_KEYRING, mailbox=mailbox) as web:
            signin = await web.form(f"{SHOP}/signin")
            unreadable = await web.call(
                action="login", email_ref=signin["email"], password_refs=[signin["password"]]
            )
            assert unreadable.is_error and unreadable.text_content == (
                "The stored password for example.com can no longer be opened. Recover the account "
                "with the site's password reset: on its reset request form, call "
                'browser(action="login", email_ref=...) without password_refs to fill the '
                "account's address, submit it, read the reset mail with inbox, open its link "
                "with navigate, and call register on the reset form."
            )
            assert not web.run.filled_passwords("parent")

            # The path it names: login fills the address alone and opens the inbox window.
            forgot = await web.form(f"{SHOP}/forgot")
            filled = await web.call(action="login", email_ref=forgot["email"])
            assert not filled.is_error, filled.text_content
            await web.call(action="click", ref=forgot["button"])
            assert posted(web.proxy, "/sent")["email"] == [alias]
            bucket.objects[f"mail/{alias}/reset.eml"] = delivered(
                0, mail(f"Reset your password at {link}", subject="Password reset")
            )
            arrived = await web.call(action="inbox")
            assert f"   link: {link}" in arrived.text_content

            # Its link opens the reset form, where register gives the same account a password the
            # deployment's ring seals.
            reset = await web.form(link)
            replaced = await web.call(
                action="register", password_refs=[reset["password"], reset["confirm"]]
            )
            assert not replaced.is_error, replaced.text_content
            (row,) = web.rows
            assert (row.account_id, row.key_id) == ("account", "next")
            web.stored_password()
