# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Agent Accounts driven through the ``browser`` tool in a real Chromium (ADR 0034).

What stands in for the public Web is a loopback proxy that terminates TLS, so a test drives
``https://*.example`` pages and observes the requests the browser sends, the rows the tool stores,
and what the tool answers. No database is needed: accounts go to a store held in memory, and the
Resources the tool admits go to a recording sink.

A generated password is compared in code and never put into an assertion, a message or a log: a
test counts where it appears and asserts the count, so a failure reports a number.
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import pytest
from pydantic import SecretStr

from dlightrag.engine.agent.environment import AccessScheduler
from dlightrag.engine.agent.tool_content import tool_content_attachments
from dlightrag.engine.agent.tools import AgentTool, ToolResult
from dlightrag.engine.answer.agent_browser import (
    ACCOUNT_LABEL,
    PASSWORD_MASK,
    AgentAccountsBinding,
    AgentAccountStore,
    BrowserHolder,
    RunAgentAccounts,
    RunAgentBrowser,
    StoredAgentAccount,
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
    WebProxy,
    browser_settings,
    launched_chromium,
    web_proxy,
)
from tests.support.dns import public_dns
from tests.support.loopback import loopback_certificate
from tests.support.resources import preparer
from tests.tool_helpers import recording_tool_runtime

pytestmark = pytest.mark.asyncio

OWNER = "owner"
HOLDER = BrowserHolder(OWNER, "11111111-1111-1111-1111-111111111111", "worker", 1)
SHOP = "https://shop.example"
SITE = "shop.example"
KEYRING = '{"active": "test", "keys": {"test": "YWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWE="}}'
OTHER_KEYRING = (
    '{"active": "next", "keys": {"next": "YmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmI="}}'
)
EMAIL, HANDLE = "shopper@example.com", "shopper-77"


def page(title: str, body: str) -> Served:
    return Served(f"<html><head><title>{title}</title></head><body>{body}</body></html>")


def signup(*, password: str = 'type="password"', confirm: bool = True, extra: str = "") -> Served:
    """A sign-up form posting to /join. ``password`` is the attributes of its password field."""
    return page(
        "Join",
        f"""<form action="/join" method="post">
<input aria-label="Handle" name="handle">
<input aria-label="Email" name="email" type="email">
<input aria-label="Password" name="password" {password}>
{'<input aria-label="Confirmation" name="confirm" type="password">' if confirm else ""}
<button type="submit">Join</button></form>{extra}""",
    )


SIGNIN = page(
    "Sign in",
    """<form action="/session" method="post">
<input aria-label="Email" name="email" type="email">
<input aria-label="Password" name="password" type="password">
<button type="submit">Sign in</button></form>""",
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
<a id="dl" download="x.csv" href="#">Save</a>
<script>function mirror(input) {
  input.setAttribute('value', input.value);
  if (input.id === 'pw') {
    document.title = 'Mirror ' + input.value;
    document.getElementById('dl').href = '/files/' + input.value + '.csv';
  } }</script>""",
)
# A password field a page can turn into a text field, as a "show password" button does.
TOGGLE = page(
    "Toggle",
    """<input aria-label="Handle" name="handle">
<input aria-label="Password" id="pw" type="password">
<button onclick="document.getElementById('pw').type = 'text'">Show</button>""",
)
EMBED = page(
    "Embed",
    """<input aria-label="Handle" name="handle">
<input aria-label="Email" type="email">
<input aria-label="Password" type="password">
<iframe src="https://pay.other.example/card"></iframe>
<iframe src="https://accounts.shop.example/card"></iframe>""",
)
PAGES = {
    f"{SHOP}/signup": signup(),
    f"{SHOP}/signin": SIGNIN,
    f"{SHOP}/mirror": MIRROR,
    f"{SHOP}/toggle": TOGGLE,
    f"{SHOP}/embed": EMBED,
    f"{SHOP}/short": signup(password='type="password" maxlength="14"', confirm=False),
    f"{SHOP}/tiny": signup(password='type="password" maxlength="10"', confirm=False),
    f"{SHOP}/locked": signup(password='type="password" disabled', confirm=False),
    # Its handler drops the asterisk of every value it is given, so what it holds is not what was filled.
    f"{SHOP}/stripping": signup(
        password="""type="password" oninput="this.value = this.value.replace(/[*]/g, '')" """,
        confirm=False,
    ),
    f"{SHOP}/join": page("Welcome", "<p>Thanks for joining</p>"),
    f"{SHOP}/session": page("Dashboard", "<p>Your dashboard</p>"),
    "https://pay.other.example/card": page(
        "Card", '<input aria-label="Other secret" type="password">'
    ),
    "https://accounts.shop.example/card": page(
        "Account", '<input aria-label="Same-site secret" type="password">'
    ),
    "http://shop.example/signup": signup(),
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
        self, form: dict[str, str], *, email: str = EMAIL, handle: str = HANDLE, **more: Any
    ) -> ToolResult:
        """Type an address and a handle into the sign-up form, then register."""
        await self.call(action="type", ref=form["handle"], text=handle)
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
    YAML would read as something else, a password that starts with an asterisk among them, and
    the mask keeps the quotes."""
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
            owner_id=OWNER, binding=AgentAccountsBinding(accounts_store, cipher)
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


async def test_register_fills_a_generated_password_and_never_shows_it(tmp_path: Path) -> None:
    async with browsing(tmp_path) as web:
        form = await web.form(f"{SHOP}/signup")

        registered = await web.register(form)

        assert not registered.is_error, registered.text_content
        password = web.stored_password()
        text = registered.text_content
        assert text.startswith(f"[browser: register | page: {SHOP}/signup | title: Join]\n")
        assert (
            f"Recorded the Agent Account {EMAIL} for {SITE} for this owner's later Runs and "
            "filled its generated password into 2 field(s); the password is never shown."
        ) in text
        # The snapshot it returns shows the fields filled and the password as the mask.
        assert shows_mask(text, "Password", form["password"])
        assert shows_mask(text, "Confirmation", form["confirm"])
        (row,) = web.rows
        assert (row.owner_id, row.site, row.email, row.username) == (OWNER, SITE, EMAIL, HANDLE)
        assert size(password) == 20
        # A later snapshot shows the mask too.
        shown = await web.call(action="snapshot")
        assert shown.text_content.count(PASSWORD_MASK) == 2

        await web.call(action="click", ref=form["button"])

        # The site received the password in both fields, and the model never did.
        fields = posted(web.proxy, "/join")
        assert_sent(fields, password, "password", "confirm")
        assert (fields["email"], fields["handle"]) == ([EMAIL], [HANDLE])


async def test_registering_again_resets_the_password_and_keeps_the_account(tmp_path: Path) -> None:
    async with browsing(tmp_path) as web:
        await web.register(await web.form(f"{SHOP}/signup"))
        (first,) = web.rows
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
        (second,) = web.rows
        assert (second.account_id, second.email) == (first.account_id, first.email)
        assert second.envelope != first.envelope
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
        web.proxy.add(
            f"{SHOP}/files/{value}.csv", Served(b"a,b\n1,2\n", headers={"content-type": "text/csv"})
        )
        # The page mirrored the value into its title and its attributes as it was filled.
        assert f"title: Mirror {PASSWORD_MASK}]" in registered.text_content
        # A query that starts as the password does matches nothing, and the lines that do match
        # show the mask.
        found = await web.call(action="find", query="Secret")
        assert shows_mask(found.text_content, "Secret", secret)
        probe = await web.call(action="find", query=value[:6])
        assert probe.text_content.splitlines()[-1] == f'No element matches "{value[:6]}".'
        # The serialized page and its address, a dialog it opens and a file it links to.
        await web.call(action="capture")
        (capture, _) = web.admitted[-1]
        assert f'value="{PASSWORD_MASK}"'.encode() in capture.content
        alerted = await web.call(action="click", ref=await ref_in(web, 'button "Alert"'))
        assert f'a alert dialog: "echo {PASSWORD_MASK}" (accepted).' in alerted.text_content
        await web.call(action="click", ref=await ref_in(web, 'link "Save"'))
        (download, _) = web.admitted[-1]
        assert download.url == f"{SHOP}/files/{PASSWORD_MASK}.csv"
        # A form that submits with GET puts the password into the page's own address.
        went = await web.call(action="click", ref=await ref_in(web, 'button "Go"'))
        assert f"page: {SHOP}/search?password={PASSWORD_MASK} |" in went.text_content
        # The browser did send the password to the site in that address, so the mask is what hid it.
        asked = [request.target for request in web.proxy.requests]
        assert leaks(password, *asked) == 2
        assert_hidden(password, capture.content, capture.url, download.url)


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
        assert stolen.is_error and "is not on an https page of shop.example" in stolen.text_content
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
        form = await web.form("http://shop.example/signup")
        plain = await web.register(form)
        assert plain.is_error and plain.text_content == (
            "register and login need an https page with a registrable domain; this page is "
            "shop.example/signup."
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


async def value_of(web: Browsing, label: str) -> str | None:
    """What the page's snapshot shows in its textbox ``label``: the value, or None when it is empty."""
    shown = await web.call(action="snapshot")
    (line,) = [ln for ln in shown.text_content.splitlines() if f'textbox "{label}"' in ln]
    _, separator, value = line.partition("]: ")
    return value if separator else None


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
        assert await value_of(web, "Password") is None


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
        assert [await value_of(web, label) for label in ("Password", "Confirmation")] == [
            None,
            None,
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

        filled = await later.call(
            action="login", email_ref=form["email"], password_refs=[form["password"]]
        )

        assert not filled.is_error, filled.text_content
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


async def test_login_says_what_the_account_lacks_and_what_the_deployment_cannot_open(
    tmp_path: Path,
) -> None:
    async with browsing(tmp_path) as web:
        signin = await web.form(f"{SHOP}/signin")
        nothing = await web.call(action="login", email_ref=signin["email"])
        assert nothing.is_error and nothing.text_content == (
            'No Agent Account exists for shop.example. Register one with browser(action="register", ...).'
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
            "The Agent Account for shop.example has no email address; use username_ref instead."
        )
        store = web.store

    async with browsing(tmp_path, store=store, keyring=OTHER_KEYRING) as lost:
        signin = await lost.form(f"{SHOP}/signin")
        unreadable = await lost.call(action="login", password_refs=[signin["password"]])
        assert unreadable.is_error and unreadable.text_content == (
            "The stored password for shop.example can no longer be opened. Recover the account "
            f"with the site's password reset: request the reset mail for {HANDLE}, open its link "
            "with navigate, and call register on the reset form."
        )
        assert not lost.run.filled_passwords("parent")

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
    async with browsing(tmp_path) as web:
        await web.register(await web.form(f"{SHOP}/signup"))
        (owners_row,) = web.rows
        owners = web.stored_password()

        # Child A registers an account of its own for this Run, which the store never sees.
        form = await web.form(f"{SHOP}/signup", "child-a", child=True)
        for name, value in (("handle", "child-a-handle"), ("email", "child-a@example.com")):
            await web.call("child-a", child=True, action="type", ref=form[name], text=value)
        registered = await web.call(
            "child-a",
            child=True,
            action="register",
            password_refs=[form["password"], form["confirm"]],
            email_ref=form["email"],
            username_ref=form["handle"],
        )
        assert (
            "Recorded the Agent Account child-a@example.com for shop.example for this Run only (a Child Session's account)"
            in registered.text_content
        )
        assert web.rows == [owners_row]
        await web.call("child-a", child=True, action="click", ref=form["button"])

        # Its login fills its own account, and another Child's falls back to the owner's.
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

        (join,) = posts(web.proxy, "/join")
        own, fallback = posts(web.proxy, "/session")
        assert own["email"] == ["child-a@example.com"]
        assert own["password"] == join["password"] and own["password"] != [
            owners.get_secret_value()
        ]
        assert fallback["email"] == [EMAIL]
        assert_sent(fallback, owners, "password")
        assert web.rows == [owners_row]


# -- screenshots ------------------------------------------------------------------------------


async def test_screenshots_stop_when_a_password_shows(tmp_path: Path) -> None:
    async with browsing(tmp_path) as web:
        opened = await web.call(action="navigate", url=f"{SHOP}/toggle")
        handle, password, show = (
            ref_of(opened.text_content, label)
            for label in ('textbox "Handle"', 'textbox "Password"', 'button "Show"')
        )
        await web.call(action="type", ref=handle, text=HANDLE)
        await web.call(action="register", password_refs=[password], username_ref=handle)
        web.stored_password()

        # The browser draws a password field's value as dots, so there is nothing to refuse yet.
        before = await web.call(action="screenshot")
        assert not before.is_error and len(tool_content_attachments(before.parts)) == 1

        await web.call(action="click", ref=show)
        after = await web.call(action="screenshot")

        assert after.is_error and after.text_content == (
            "The page shows a filled password as text, so no screenshot was taken."
        )
        assert (
            tool_content_attachments(after.parts) == () and after.effects.attached_resources == ()
        )
