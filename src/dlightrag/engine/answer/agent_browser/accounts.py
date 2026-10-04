# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Agent Accounts: the identities an Agent registers on third-party sites (ADR 0034).

DlightRAG makes each account's password, seals it under the deployment key ring, and fills it
into a page by reference, so a password is never in model context. A parent Agent Session's
accounts persist for its owner; a Child's last for its Run and live only in this process.
"""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import secrets
import string
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import ClassVar, Protocol
from urllib.parse import urlsplit

import tldextract
from pydantic import SecretStr

from dlightrag.engine.answer.agent_browser.mailbox import AgentMailbox, InboxWindow
from dlightrag.engine.credential_cipher import CredentialCipher, UnreadableEnvelope

#: An account's envelope is sealed under this label, so it never opens as a Connection Grant.
ACCOUNT_LABEL = "dlightrag-agent-account-v1"
PASSWORD_LENGTH = 20
#: A site that caps a password's length gets a shorter one, never shorter than this.
MIN_PASSWORD_LENGTH = 12
#: The punctuation of a password: with the letters and digits, the characters no encoding
#: changes (HTML, JSON and form escaping, and RFC 3986's percent-encoding, leave them as they
#: are), so a password has one spelling in every text a page and a driver print.
_SYMBOLS = "-._"
_ALNUM = string.ascii_letters + string.digits
_ALPHABET = _ALNUM + _SYMBOLS


#: The Public Suffix List snapshot the package bundles, private section included. It is read
#: from the package and never fetched or cached, so the list a deployment runs is the one its
#: pin ships, whatever the environment says.
_PUBLIC_SUFFIXES = tldextract.TLDExtract(
    suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True
)


def account_site(url: str) -> str | None:
    """The key of an Agent Account: the registrable domain of an https URL.

    That is the Public Suffix List's eTLD+1, private section included, so ``alice.github.io``
    and ``bob.github.io`` are different sites. A URL of another scheme, of an IP address, or of
    a host with no registrable part, such as a bare public suffix, has none.
    """
    parts = urlsplit(url)
    host = parts.hostname
    if parts.scheme != "https" or not host:
        return None
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return _PUBLIC_SUFFIXES.extract_str(host).top_domain_under_public_suffix or None
    return None


def generate_password(length: int = PASSWORD_LENGTH) -> SecretStr:
    """``length`` characters drawn with ``secrets`` from ``[A-Za-z0-9._-]``, holding at least one
    lowercase letter, one uppercase letter, one digit and one of ``-._``, so the usual composition
    rules pass.

    It begins and ends with a letter or a digit: Chromium trims a dot from either end of a
    downloaded file's name, and a trimmed password is a spelling nothing redacts.
    """
    inner = [
        secrets.choice(string.ascii_lowercase),
        secrets.choice(string.ascii_uppercase),
        secrets.choice(string.digits),
        secrets.choice(_SYMBOLS),
        *(secrets.choice(_ALPHABET) for _ in range(length - 6)),
    ]
    secrets.SystemRandom().shuffle(inner)
    return SecretStr(f"{secrets.choice(_ALNUM)}{''.join(inner)}{secrets.choice(_ALNUM)}")


def owner_alias(owner_id: str, site: str, domain: str) -> str:
    """The address an owner registers with on a site, on the Agent Mailbox's domain.

    It is the same every time for one owner and site, so mail for the account keeps reaching it
    in later Runs, and it names nothing of the owner: 16 characters of ``[a-z2-7]`` from a hash.
    The hash is unkeyed, so a mailbox alias survives the rotation of the key ring.
    """
    digest = hashlib.sha256(f"dlightrag-agent-alias-v1\0{owner_id}\0{site}".encode()).digest()
    return f"{base64.b32encode(digest[:10]).decode().lower()}@{domain}"


def run_alias(domain: str) -> str:
    """A random address on the Agent Mailbox's domain, which no other registration shares."""
    return f"{base64.b32encode(secrets.token_bytes(10)).decode().lower()}@{domain}"


@dataclass(frozen=True, slots=True)
class StoredAgentAccount:
    """One owner's account on one site, as the store keeps it: its password stays sealed."""

    persistent: ClassVar[bool] = True

    owner_id: str
    site: str
    account_id: str
    """Minted at the first registration and kept by every reset after it."""
    email: str | None
    username: str | None
    key_id: str
    envelope: str = field(repr=False)

    @property
    def binding(self) -> tuple[str, str, str]:
        """What its envelope is sealed to: its owner, its site, and this account."""
        return (self.owner_id, self.site, self.account_id)


@dataclass(frozen=True, slots=True)
class ChildAccount:
    """A Child's account on one site, which lives in this process until its Run settles."""

    persistent: ClassVar[bool] = False

    site: str
    email: str | None
    username: str | None
    password: SecretStr = field(repr=False)


#: An account as ``register`` and ``login`` see it: the owner's, or a Child's own.
type AgentAccount = StoredAgentAccount | ChildAccount


class AgentAccountStore(Protocol):
    """The owner-scoped durable record of Agent Accounts."""

    async def account(self, *, owner_id: str, site: str) -> StoredAgentAccount | None: ...

    async def save(self, account: StoredAgentAccount) -> None:
        """Insert the account, or replace the owner's account on that site."""
        ...

    async def sealed_under(
        self, *, key_ids: Sequence[str], after: tuple[str, str] = ("", ""), limit: int
    ) -> tuple[StoredAgentAccount, ...]:
        """At most ``limit`` accounts sealed under ``key_ids``, in the order of their owner and
        site, that come after the ``(owner_id, site)`` ``after``; the default starts at the
        first."""
        ...

    async def reseal(self, account: StoredAgentAccount, *, key_id: str, envelope: str) -> bool:
        """Replace the envelope ``account`` holds, unless it changed meanwhile."""
        ...


@dataclass(frozen=True, slots=True)
class AgentAccountsBinding:
    """What a deployment composed for Agent Accounts: where they are kept, the key ring, and the
    Agent Mailbox that delivers their mail, when it has one."""

    store: AgentAccountStore
    cipher: CredentialCipher
    mailbox: AgentMailbox | None = None


class _Mailboxed(Protocol):
    @property
    def mailbox(self) -> AgentMailbox | None: ...


def has_mailbox(accounts: _Mailboxed | None) -> bool:
    """Whether a deployment or a Run composed Agent Accounts with an Agent Mailbox that delivers
    their mail. A mailbox belongs to the accounts: there is none without them."""
    return accounts is not None and accounts.mailbox is not None


class SessionAccounts:
    """The Agent Accounts one parent Agent Session acts on: the owner's, sealed under the key ring.

    It also keeps the Session's inbox window, which only a registration or a login opens. A
    Child's rules differ in what it registers and which account its login prefers, and are in
    ``ChildSessionAccounts``.
    """

    def __init__(self, owner_id: str, binding: AgentAccountsBinding) -> None:
        self._owner_id = owner_id
        self._store = binding.store
        self._cipher = binding.cipher
        self._mailbox = binding.mailbox
        self._window: InboxWindow | None = None

    async def registration_target(self, site: str) -> AgentAccount | None:
        """The account a registration on ``site`` replaces the password of, if there is one."""
        return await self._owned(site)

    def new_alias(self, site: str) -> str | None:
        """The mailbox alias a registration on ``site`` fills in, the owner's for the site, or
        None where the deployment has no Agent Mailbox."""
        if self._mailbox is None:
            return None
        return owner_alias(self._owner_id, site, self._mailbox.alias_domain)

    async def login_target(self, site: str) -> AgentAccount | None:
        """The account a login on ``site`` fills."""
        return await self._owned(site)

    async def record(
        self,
        site: str,
        *,
        existing: AgentAccount | None,
        email: str | None,
        username: str | None,
        password: SecretStr,
    ) -> AgentAccount:
        """Keep a new account, or the new password of ``existing``, sealed and stored for the
        owner under the account id the first registration minted."""
        account_id = (
            existing.account_id if isinstance(existing, StoredAgentAccount) else uuid.uuid4().hex
        )
        key_id, envelope = self._cipher.seal(
            password, label=ACCOUNT_LABEL, binding=(self._owner_id, site, account_id)
        )
        stored = StoredAgentAccount(
            self._owner_id, site, account_id, email, username, key_id, envelope
        )
        await self._store.save(stored)
        return stored

    def password(self, account: AgentAccount) -> SecretStr:
        """The account's password. An envelope no key opens raises ``UnreadableEnvelope``."""
        if isinstance(account, ChildAccount):
            return account.password
        return self._cipher.open(account.envelope, label=ACCOUNT_LABEL, binding=account.binding)

    def alias(self, account: AgentAccount) -> str | None:
        """The account's address when DlightRAG minted it on the Agent Mailbox's domain.

        The domain does not say so: an Agent with no mailbox typed an address of its own, which
        may end in the same domain, and reading its folder would read mail it is not owed. The
        owner's account has a mailbox alias exactly when its address is the one worked out for
        its owner and site. A Child's takes only the address its session minted, so with a
        mailbox its address is one.
        """
        if self._mailbox is None or account.email is None:
            return None
        if isinstance(account, ChildAccount):
            return account.email
        minted = owner_alias(account.owner_id, account.site, self._mailbox.alias_domain)
        return account.email if account.email == minted else None

    def signed_in(self, account: AgentAccount) -> None:
        """The Session registered or logged in with ``account`` now: its inbox window opens here,
        and keeps the mailbox aliases of the accounts it used earlier in this Run."""
        if self._mailbox is None:
            return
        aliases = self._window.aliases if self._window is not None else ()
        if (alias := self.alias(account)) is not None and alias not in aliases:
            aliases = (*aliases, alias)
        self._window = InboxWindow(self._mailbox, datetime.now(UTC), aliases)

    def inbox_window(self) -> InboxWindow | None:
        """The window the Session reads mail through, or None before it has signed in."""
        return self._window

    async def _owned(self, site: str) -> StoredAgentAccount | None:
        return await self._store.account(owner_id=self._owner_id, site=site)


class ChildSessionAccounts(SessionAccounts):
    """The Agent Accounts one Child Session acts on: its own, kept in this process until the Run
    settles, and the owner's, which it may sign in with, as capability and not authority."""

    def __init__(self, owner_id: str, binding: AgentAccountsBinding) -> None:
        super().__init__(owner_id, binding)
        self._own: dict[str, ChildAccount] = {}

    async def registration_target(self, site: str) -> AgentAccount | None:
        """Only its own account, never the owner's."""
        return self._own.get(site)

    def new_alias(self, site: str) -> str | None:
        """A random mailbox alias, so an account that goes with the Run never takes the owner's."""
        return None if self._mailbox is None else run_alias(self._mailbox.alias_domain)

    async def login_target(self, site: str) -> AgentAccount | None:
        """Its own account first, else the owner's."""
        return self._own.get(site) or await super().login_target(site)

    async def record(
        self,
        site: str,
        *,
        existing: AgentAccount | None,
        email: str | None,
        username: str | None,
        password: SecretStr,
    ) -> AgentAccount:
        """Keep a new account, or the new password of ``existing``, in this process alone."""
        account = self._own[site] = ChildAccount(site, email, username, password)
        return account


class RunAgentAccounts:
    """One Research Run's Agent Accounts, as each of its Agent Sessions sees them."""

    def __init__(self, *, owner_id: str, binding: AgentAccountsBinding) -> None:
        self._owner_id = owner_id
        self._binding = binding
        self.mailbox = binding.mailbox
        self._sessions: dict[str, SessionAccounts] = {}

    def available(self) -> bool:
        """Whether the key ring can seal a password. Without one, register and login fail closed."""
        return self._binding.cipher.active_key_id is not None

    def session(self, scope: str, *, child: bool) -> SessionAccounts:
        """The accounts of the Agent Session whose tool calls run in ``scope``: the same object
        each time, so what a registration or a login opens stays open for its next call."""
        if scope not in self._sessions:
            kind = ChildSessionAccounts if child else SessionAccounts
            self._sessions[scope] = kind(self._owner_id, self._binding)
        return self._sessions[scope]


async def reseal_agent_accounts(
    store: AgentAccountStore, cipher: CredentialCipher, *, limit: int = 100
) -> int:
    """One pass over the accounts sealed under a retired key, ``limit`` of them at a time:
    re-seal each password under the active key.

    An envelope no key opens is left alone, and its account recovers through the site's reset
    mail. A page is read after the last account of the one before, so the ones left alone
    never keep a later account from being read. Returns how many accounts it re-sealed.
    """
    retired = cipher.retired_key_ids
    if not retired:
        return 0
    resealed = 0
    after = ("", "")
    while True:
        page = await store.sealed_under(key_ids=retired, after=after, limit=limit)
        for account in page:
            try:
                password = cipher.open(
                    account.envelope, label=ACCOUNT_LABEL, binding=account.binding
                )
            except UnreadableEnvelope:
                continue
            key_id, envelope = cipher.seal(password, label=ACCOUNT_LABEL, binding=account.binding)
            if await store.reseal(account, key_id=key_id, envelope=envelope):
                resealed += 1
        if len(page) < limit:
            return resealed
        after = (page[-1].owner_id, page[-1].site)


__all__ = [
    "ACCOUNT_LABEL",
    "MIN_PASSWORD_LENGTH",
    "PASSWORD_LENGTH",
    "AgentAccount",
    "AgentAccountStore",
    "AgentAccountsBinding",
    "ChildAccount",
    "RunAgentAccounts",
    "SessionAccounts",
    "StoredAgentAccount",
    "account_site",
    "generate_password",
    "has_mailbox",
    "owner_alias",
    "reseal_agent_accounts",
    "run_alias",
]
