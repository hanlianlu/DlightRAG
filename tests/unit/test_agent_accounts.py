# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Agent Accounts: the key of a site, the generated password, and one Run's view of its accounts.

A generated password is never put into an assertion: a test compares in code and asserts a
count or a flag, so a failure reports a number and never the value.
"""

from __future__ import annotations

import html
import json
import re
import string
import subprocess
import sys
import sysconfig
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import quote, quote_plus

import pytest
from pydantic import SecretStr

from dlightrag.engine.answer.agent_browser import (
    ACCOUNT_LABEL,
    AgentAccount,
    AgentAccountsBinding,
    RunAgentAccounts,
    account_site,
    generate_password,
    owner_alias,
    run_alias,
)
from dlightrag.engine.credential_cipher import CredentialCipher, UnreadableEnvelope
from tests.support.agent_browser import MemoryAccountStore, StubMailbox

KEYRING = json.dumps(
    {"active": "test", "keys": {"test": "YWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWE="}}
)
OTHER_KEYRING = json.dumps(
    {"active": "next", "keys": {"next": "YmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmI="}}
)
OWNER = "owner"
SITE = "example.com"
_ROOT = Path(__file__).resolve().parents[2]
_IMPORT_PATHS = [
    str(_ROOT / "src"),
    str(_ROOT / "packages/memory/src"),
    sysconfig.get_path("purelib"),
]


@pytest.mark.parametrize(
    ("url", "site"),
    [
        ("https://www.example.co.uk/signup", "example.co.uk"),
        ("https://accounts.example.com/login?next=/", "example.com"),
        ("https://www.example.com.:8443/", "example.com"),
        ("https://www.xn--bcher-kva.com/", "xn--bcher-kva.com"),
        # The private section of the list: a host of the platform is its own site.
        ("https://alice.github.io/", "alice.github.io"),
        ("https://bob.github.io/", "bob.github.io"),
        ("https://github.io/", None),
        ("https://co.uk/", None),
        # A name under no public suffix has no registrable domain.
        ("https://shop.example/", None),
        ("https://localhost/", None),
        ("http://example.com/", None),
        ("https://127.0.0.1/", None),
        ("https://[::1]/", None),
        ("about:blank", None),
        ("", None),
    ],
)
def test_an_account_is_keyed_by_the_registrable_domain_of_an_https_page(
    url: str, site: str | None
) -> None:
    assert account_site(url) == site


def test_a_site_is_read_from_the_bundled_list_with_no_socket_and_no_cache_file(
    tmp_path: Path,
) -> None:
    # A fresh interpreter, since the list is read at the first lookup and kept after it. The
    # environment names a list to fetch and a directory to cache it in, and the lookup uses neither.
    cache = tmp_path / "cache"
    script = f"""
import json, sys
sys.path[:0] = {_IMPORT_PATHS!r}
from dlightrag.engine.answer.agent_browser import account_site

sockets = []
def refuse(event, args):
    if event.startswith("socket."):
        sockets.append(event)
        raise AssertionError(event)
sys.addaudithook(refuse)
sites = [account_site(url) for url in ("https://alice.github.io/", "https://www.example.co.uk/")]
print(json.dumps([sites, sockets]))
"""

    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", script],
        cwd=tmp_path,
        env={
            "HOME": str(tmp_path),
            "PYTHON_DOTENV_DISABLED": "1",
            "TLDEXTRACT_CACHE": str(cache),
            "TLDEXTRACT_PUBLIC_SUFFIX_LIST_URLS": "https://psl.invalid/public_suffix_list.dat",
        },
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [["alice.github.io", "example.co.uk"], []]
    assert not cache.exists()


def well_formed(password: str, length: int) -> bool:
    """Whether a password has the length, only letters, digits and ``-._``, a letter or a digit
    at each end, and every class a site's composition rule asks for."""
    allowed = set(string.ascii_letters + string.digits + "-._")
    return (
        len(password) == length
        and set(password) <= allowed
        and password[0].isalnum()
        and password[-1].isalnum()
        and any(c.islower() for c in password)
        and any(c.isupper() for c in password)
        and any(c.isdigit() for c in password)
        and any(c in "-._" for c in password)
    )


def test_a_generated_password_has_the_length_asked_and_every_class_and_never_repeats() -> None:
    passwords = [generate_password().get_secret_value() for _ in range(200)]
    shorter = generate_password(14).get_secret_value()

    formed = sum(well_formed(password, 20) for password in passwords)
    distinct = len(set(passwords))
    assert (formed, distinct) == (200, 200)
    assert well_formed(shorter, 14) is True


def test_no_encoding_a_page_or_a_driver_uses_changes_a_generated_password() -> None:
    unchanged = 0
    for _ in range(200):
        value = generate_password().get_secret_value()
        unchanged += (
            quote(value, safe="") == value
            and quote_plus(value) == value
            and html.escape(value) == value
            and json.dumps(value) == f'"{value}"'
        )

    assert unchanged == 200


def accounts(
    store: MemoryAccountStore, *, keyring: str | None = KEYRING, mailbox: StubMailbox | None = None
) -> RunAgentAccounts:
    cipher = CredentialCipher(None if keyring is None else SecretStr(keyring))
    return RunAgentAccounts(owner_id=OWNER, binding=AgentAccountsBinding(store, cipher, mailbox))


async def register(
    run: RunAgentAccounts,
    scope: str,
    *,
    site: str = SITE,
    child: bool = False,
    email: str | None = "a@x.example",
    existing: AgentAccount | None = None,
) -> tuple[AgentAccount, SecretStr]:
    password = generate_password()
    account = await run.session(scope, child=child).record(
        site,
        existing=existing,
        email=email,
        username=None if email else "handle",
        password=password,
    )
    return account, password


async def test_a_parents_account_is_sealed_for_its_owner_site_and_account() -> None:
    store, cipher = MemoryAccountStore(), CredentialCipher(SecretStr(KEYRING))
    run = accounts(store)

    account, password = await register(run, "parent")

    (row,) = store.rows.values()
    assert (row.owner_id, row.site, row.email, row.username) == (OWNER, SITE, "a@x.example", None)
    assert account.persistent
    opened = cipher.open(row.envelope, label=ACCOUNT_LABEL, binding=(OWNER, SITE, row.account_id))
    assert opened == password and run.session("parent", child=False).password(account) == password
    for other in ((OWNER, "other.example", row.account_id), ("someone", SITE, row.account_id)):
        with pytest.raises(UnreadableEnvelope):
            cipher.open(row.envelope, label=ACCOUNT_LABEL, binding=other)


async def test_a_reset_replaces_the_password_and_keeps_the_account_id_and_the_email() -> None:
    store = MemoryAccountStore()
    run = accounts(store)
    await register(run, "parent")
    (before,) = store.rows.values()

    existing = await run.session("parent", child=False).registration_target(SITE)
    assert existing is not None
    renewed, password = await register(run, "parent", email=existing.email, existing=existing)

    (after,) = store.rows.values()
    assert (after.account_id, after.email) == (before.account_id, before.email)
    assert after.envelope != before.envelope
    assert run.session("parent", child=False).password(renewed) == password


async def test_a_childs_registration_stays_in_the_run_and_serves_only_its_own_session() -> None:
    store = MemoryAccountStore()
    run = accounts(store)

    _, own = await register(run, "child-a", child=True, email="alias-a@x.example")
    _, siblings = await register(run, "child-b", child=True, email="alias-b@x.example")

    assert store.rows == {}
    first_session, second_session = (
        run.session("child-a", child=True),
        run.session("child-b", child=True),
    )
    first = await first_session.registration_target(SITE)
    second = await second_session.registration_target(SITE)
    assert first is not None and second is not None
    assert (first.email, second.email) == ("alias-a@x.example", "alias-b@x.example")
    assert first_session.password(first) == own and second_session.password(second) == siblings
    assert not first.persistent
    # What a Child registered is not the owner's account, so a parent finds nothing.
    assert await run.session("parent", child=False).registration_target(SITE) is None


async def test_a_childs_login_prefers_its_own_account_then_the_owners() -> None:
    run = accounts(MemoryAccountStore())
    _, owners = await register(run, "parent", email="owner@x.example")
    await register(run, "child-a", child=True, email="alias@x.example")

    own = await run.session("child-a", child=True).login_target(SITE)
    child_b = run.session("child-b", child=True)
    fallback = await child_b.login_target(SITE)
    parent = await run.session("parent", child=False).login_target(SITE)

    assert own is not None and fallback is not None and parent is not None
    assert (own.email, fallback.email, parent.email) == (
        "alias@x.example",
        "owner@x.example",
        "owner@x.example",
    )
    assert fallback.persistent and child_b.password(fallback) == owners
    assert await run.session("parent", child=False).login_target("other.example") is None


async def test_an_account_whose_key_the_ring_lost_cannot_be_opened() -> None:
    store = MemoryAccountStore()
    await register(accounts(store), "parent")
    session = accounts(store, keyring=OTHER_KEYRING).session("parent", child=False)

    account = await session.login_target(SITE)

    assert account is not None
    with pytest.raises(UnreadableEnvelope):
        session.password(account)


def test_without_a_ring_accounts_are_unavailable() -> None:
    assert accounts(MemoryAccountStore()).available()
    assert not accounts(MemoryAccountStore(), keyring=None).available()


def test_an_owners_alias_is_stable_for_a_site_and_differs_for_every_other_owner_and_site() -> None:
    alias = owner_alias("owner", "shop.example", "orliantra.cc")

    assert re.fullmatch(r"[a-z2-7]{16}@orliantra\.cc", alias)
    assert owner_alias("owner", "shop.example", "orliantra.cc") == alias
    distinct = {
        alias,
        owner_alias("another", "shop.example", "orliantra.cc"),
        owner_alias("owner", "other.example", "orliantra.cc"),
    }
    assert len(distinct) == 3
    # It names nothing of the owner or the site.
    assert "owner" not in alias and "shop" not in alias


def test_a_childs_alias_is_random_on_the_same_domain() -> None:
    aliases = {run_alias("orliantra.cc") for _ in range(50)}

    assert len(aliases) == 50
    assert all(re.fullmatch(r"[a-z2-7]{16}@orliantra\.cc", alias) for alias in aliases)
    assert owner_alias("owner", SITE, "orliantra.cc") not in aliases


async def test_the_inbox_window_opens_at_each_sign_in_and_names_only_mailbox_aliases() -> None:
    run = accounts(MemoryAccountStore(), mailbox=StubMailbox("orliantra.cc"))
    session = run.session("parent", child=False)
    alias, other = session.new_alias(SITE), session.new_alias("other.example")
    assert session.inbox_window() is None

    before = datetime.now(UTC)
    first, _ = await register(run, "parent", email=alias)
    session.signed_in(first)
    window = session.inbox_window()

    assert window is not None and window.aliases == (alias,)
    assert timedelta(0) <= window.since - before < timedelta(seconds=5)
    # A later sign-in moves the window on and keeps the aliases of the Run. An address the Agent
    # typed is no alias, though it ends in the mailbox's domain. Another Session has no window.
    typed, _ = await register(run, "parent", site="typed.example", email="info@orliantra.cc")
    second, _ = await register(run, "parent", site="other.example", email=other)
    session.signed_in(typed)
    session.signed_in(second)
    session.signed_in(first)
    moved = session.inbox_window()
    assert moved is not None and moved.aliases == (alias, other) and moved.since >= window.since
    assert run.session("child", child=True).inbox_window() is None


async def test_a_deployment_with_no_mailbox_keeps_no_inbox_window() -> None:
    run = accounts(MemoryAccountStore())
    session = run.session("parent", child=False)
    account, _ = await register(run, "parent", email="a@orliantra.cc")

    session.signed_in(account)

    assert session.inbox_window() is None and session.alias(account) is None
    assert session.new_alias(SITE) is None
