# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The length of a generated Agent Account password, which the agent chooses for the site.

``register`` makes exactly as many characters as the call asks for, and the generator's own
twenty when it asks for none; the platform keeps them random and never shows them (ADR 0034).
A site whose own password field takes fewer characters than the call asked for is refused in
the browser failure vocabulary — naming the field's limit and the length asked for — with
nothing filled and nothing stored, because a password is never shortened to fit. The agent
asks for the length the site allows instead, which the snapshot prints beside the field's ref.
A field that takes fewer than the shortest password the composition leaves room for leaves no
ask that fits it, and that refusal names the mechanism's own minimum and the site as one no
registration can succeed on.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import SecretStr

from dlightrag.application.config.sections import DlightragConfig
from dlightrag.application.settings.projections import agent_browser_settings
from dlightrag.engine.answer.agent_browser.accounts import (
    ACCOUNT_LABEL,
    PASSWORD_LENGTH,
    PASSWORD_MIN_LENGTH,
    AgentAccountsBinding,
    RunAgentAccounts,
)
from dlightrag.engine.answer.agent_browser.contracts import (
    BrowserHolder,
    CredentialFill,
    CredentialForm,
    PageObservation,
    PageState,
    SessionState,
)
from dlightrag.engine.answer.agent_browser.run import (
    RunAgentBrowser,
)
from dlightrag.engine.answer.resources.registry import ResourceRegistry
from dlightrag.engine.answer.tools.browser import (
    BrowserToolHost,
    browser_tool,
)
from dlightrag.engine.credential_cipher import CredentialCipher
from tests.support.agent_browser import (
    FakeLease,
    FakePage,
    FakeProvider,
    MemoryAccountStore,
    browser_settings,
)
from tests.support.dns import public_dns
from tests.support.resources import call

HOLDER = BrowserHolder("owner", "11111111-1111-1111-1111-111111111111", "worker-1", 1)
URL = "https://shop.example.com/signup"
SITE = "example.com"
KEYRING = json.dumps(
    {"active": "test", "keys": {"test": "YWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWE="}}
)


class _SignUpPage(FakePage):
    """A sign-up form that states the smallest password ``maxlength`` it was given, and remembers
    how long each password filled into it was. Never the password itself: what a test says about
    one is how many characters it has."""

    def __init__(self, limit: int | None) -> None:
        super().__init__(session=SessionState(url=URL), snapshot='- textbox "Password" [ref=e1]')
        self.limit = limit
        self.forms = 0
        self.filled: list[str] = []
        self.lengths: list[int] = []

    async def credential_form(
        self,
        *,
        site: str,
        password_refs: tuple[str, ...],
        email_ref: str | None,
        username_ref: str | None,
    ) -> CredentialForm:
        self.forms += 1
        return CredentialForm(self.limit, "", "handle")

    async def fill_credentials(
        self, fills: tuple[CredentialFill, ...], *, site: str
    ) -> PageObservation:
        for fill in fills:
            self.filled.append(fill.kind)
            if fill.kind == "password":
                self.lengths.append(len(fill.value.get_secret_value()))
        return PageObservation(PageState(URL, "Sign up"), self.events, self.snapshot)


async def _reads_nothing(request: Any, runtime: Any) -> Any:
    raise AssertionError("the browser tool reads no Resource here")


def _tool(page: _SignUpPage, store: MemoryAccountStore, *, settings: Any = None) -> Any:
    """The ``browser`` tool of a Run whose browser may register an Agent Account.

    ``settings`` is what the deployment composed; the tests that drive the configuration surface
    pass what it projects, and the rest take the double's own.
    """
    accounts = RunAgentAccounts(
        owner_id="owner",
        binding=AgentAccountsBinding(
            store, CredentialCipher(SecretStr(KEYRING)), registration_allowed=True
        ),
        may_register=True,
    )
    host = BrowserToolHost(
        RunAgentBrowser(
            FakeProvider(FakeLease(page=page)),
            HOLDER,
            browser_settings() if settings is None else settings,
        ),
        ResourceRegistry(),
        _reads_nothing,
        accounts,
    )
    return browser_tool(
        host,
        environment=None,
        scheduler=None,  # pyright: ignore[reportArgumentType]
        spill=None,
        image_preparer=None,
        child=False,
    )


def _stored_length(store: MemoryAccountStore) -> int:
    """How many characters the one stored account's password has, opened from its envelope."""
    (row,) = store.rows.values()
    opened = CredentialCipher(SecretStr(KEYRING)).open(
        row.envelope, label=ACCOUNT_LABEL, binding=(row.owner_id, row.site, row.account_id)
    )
    return len(opened.get_secret_value())


@pytest.fixture(autouse=True)
def _public_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """``shop.example.com`` is a public host, as the URL check the tool makes requires."""
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)


async def test_the_length_the_call_asks_for_is_the_one_generated_and_stored() -> None:
    page, store = _SignUpPage(limit=None), MemoryAccountStore()
    tool = _tool(page, store)
    await call(tool, action="navigate", url=URL)

    registered = await call(
        tool, action="register", password_refs=["e1"], username_ref="e2", length=14
    )

    assert registered.is_error is False
    # The one password was made at exactly the length asked for, and that is what was stored.
    assert (page.filled, page.lengths) == (["password"], [14])
    assert _stored_length(store) == 14


async def test_a_call_that_asks_for_no_length_gets_the_generators_own_default() -> None:
    page, store = _SignUpPage(limit=None), MemoryAccountStore()
    tool = _tool(page, store)
    await call(tool, action="navigate", url=URL)

    registered = await call(tool, action="register", password_refs=["e1"], username_ref="e2")

    assert registered.is_error is False
    assert page.lengths == [PASSWORD_LENGTH]
    assert _stored_length(store) == PASSWORD_LENGTH


@pytest.mark.parametrize(("asked", "length"), [(None, 20), (18, 18)], ids=["omitted", "asked-for"])
async def test_a_site_that_takes_fewer_characters_than_asked_for_refuses_and_names_both(
    asked: int | None, length: int
) -> None:
    page, store = _SignUpPage(limit=10), MemoryAccountStore()
    tool = _tool(page, store)
    await call(tool, action="navigate", url=URL)

    refused = await call(
        tool,
        action="register",
        password_refs=["e1"],
        username_ref="e2",
        **({} if asked is None else {"length": asked}),
    )

    # One stable sentence, naming the field's own limit and the length that was asked for.
    assert refused.is_error is True
    assert refused.text_content == (
        f"This site's password field takes at most 10 characters and {length} were asked for; "
        "ask for 10 or fewer."
    )
    # Nothing was filled and nothing was stored: no password is shortened to fit a field.
    assert (page.filled, page.lengths, store.rows) == ([], [], {})


@pytest.mark.parametrize("asked", [None, PASSWORD_LENGTH + 1], ids=["omitted", "asked-for"])
async def test_a_site_capped_below_the_mechanisms_own_minimum_names_the_minimum(
    asked: int | None,
) -> None:
    """A field that takes fewer characters than the composition leaves room for leaves no ask
    that fits it. The refusal says the two numbers that are true — the field's limit and the
    shortest password this mechanism makes — and that the site cannot be registered on, rather
    than advising an ask the schema itself would refuse and the agent would retry forever."""
    page, store = _SignUpPage(limit=4), MemoryAccountStore()
    tool = _tool(page, store)
    await call(tool, action="navigate", url=URL)

    refused = await call(
        tool,
        action="register",
        password_refs=["e1"],
        username_ref="e2",
        **({} if asked is None else {"length": asked}),
    )

    assert refused.is_error is True
    assert refused.text_content == (
        "This site's password field takes at most 4 characters, and the shortest password this "
        "mechanism makes is 6: no registration can succeed on this site."
    )
    assert (page.forms, page.filled, page.lengths, store.rows) == (1, [], [], {})


async def test_asking_for_the_length_the_site_allows_registers() -> None:
    page, store = _SignUpPage(limit=10), MemoryAccountStore()
    tool = _tool(page, store)
    await call(tool, action="navigate", url=URL)

    registered = await call(
        tool, action="register", password_refs=["e1"], username_ref="e2", length=10
    )

    assert registered.is_error is False
    assert f"Recorded the Agent Account handle for {SITE}" in registered.text_content
    assert page.lengths == [10]
    assert _stored_length(store) == 10


async def test_a_length_past_the_most_one_call_makes_is_refused_in_the_surface_s_words() -> None:
    page, store = _SignUpPage(limit=None), MemoryAccountStore()
    tool = _tool(page, store)
    await call(tool, action="navigate", url=URL)

    refused = await call(
        tool, action="register", password_refs=["e1"], username_ref="e2", length=200
    )

    # A sentence the model can act on rather than a validation dump, and the form is never read:
    # the ceiling is the mechanism's own bound on what it will fill and seal, and it says so.
    assert refused.is_error is True
    assert refused.text_content == (
        "This mechanism fills and seals a password of at most 128 characters: every one is "
        "drawn, held in memory, filled into the page, and sealed into the stored account. 200 "
        "was asked for, so nothing was made. Ask for 128 or fewer, and for no more than the "
        "site's own password field takes."
    )
    assert (page.forms, page.filled, page.lengths, store.rows) == (0, [], [], {})


async def test_a_site_that_takes_the_shortest_password_this_mechanism_makes_registers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A site whose field takes exactly the shortest password the composition leaves room for
    registers, on the configuration surface a default deployment builds.

    ``test_the_password_floor_is_gone_and_no_deployment_can_state_one`` reads *names*: it can only
    catch a floor that comes back under one of the names it lists, so a differently-named setting,
    or a floor held inside the register path itself, would pass it unnoticed. This drives the path
    with a length that any floor above the composition's own would refuse or round up, and reads
    the length where it lands — the field that was filled and the envelope that was stored — so a
    floor is caught however it is called and wherever it is configured. The settings are what
    ``agent_browser_settings`` projects from a default ``DlightragConfig``, so the deployment's
    whole configuration surface stands between a legal ask and its password.
    """
    monkeypatch.setenv(
        "DLIGHTRAG_ANSWER__AGENT__BROWSER__ENDPOINTS", '["ws://agent-browser-1:3000/"]'
    )
    monkeypatch.setenv(
        "DLIGHTRAG_ANSWER__AGENT__BROWSER__EGRESS_PROXY", "http://agent-browser-egress:3128"
    )
    settings = agent_browser_settings(DlightragConfig())  # pyright: ignore[reportCallIssue]
    assert settings is not None
    page, store = _SignUpPage(limit=PASSWORD_MIN_LENGTH), MemoryAccountStore()
    tool = _tool(page, store, settings=settings)
    await call(tool, action="navigate", url=URL)

    registered = await call(
        tool,
        action="register",
        password_refs=["e1"],
        username_ref="e2",
        length=PASSWORD_MIN_LENGTH,
    )

    assert registered.is_error is False, registered.text_content
    assert page.lengths == [PASSWORD_MIN_LENGTH]
    assert _stored_length(store) == PASSWORD_MIN_LENGTH
