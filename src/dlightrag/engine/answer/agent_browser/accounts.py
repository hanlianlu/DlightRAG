# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Agent Accounts: the identities an Agent registers on third-party sites (ADR 0034).

DlightRAG makes each account's password, seals it under the deployment key ring, and fills it
into a page by reference, so a password is never in model context. A parent Agent Session's
accounts persist for its owner; a Child's last for its Run and live only in this process.
"""

from __future__ import annotations

import functools
import ipaddress
import secrets
import string
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, cast
from urllib.parse import urlsplit

from publicsuffixlist import PublicSuffixList
from pydantic import SecretStr

from dlightrag.engine.credential_cipher import CredentialCipher, UnreadableEnvelope

#: An account's envelope is sealed under this label, so it never opens as a Connection Grant.
ACCOUNT_LABEL = "dlightrag-agent-account-v1"
PASSWORD_LENGTH = 20
#: A site that caps a password's length gets a shorter one, never shorter than this.
MIN_PASSWORD_LENGTH = 12
#: The punctuation HTML, JSON and URL encoding leave unchanged, so a password has one spelling.
_SYMBOLS = "*-._"
_ALPHABET = string.ascii_letters + string.digits + _SYMBOLS


@functools.cache
def _public_suffixes() -> PublicSuffixList:
    """The Public Suffix List pinned in the wheel, which nothing here ever fetches."""
    return PublicSuffixList()


def account_site(url: str) -> str | None:
    """The key of an Agent Account: the registrable domain of an https URL.

    That is the Public Suffix List's eTLD+1, private section included, so ``alice.github.io``
    and ``bob.github.io`` are different sites. A URL of another scheme, of an IP address, or of
    a host with no registrable part has none.
    """
    parts = urlsplit(url)
    host = parts.hostname
    if parts.scheme != "https" or not host:
        return None
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return _public_suffixes().privatesuffix(host)
    return None


def generate_password(length: int = PASSWORD_LENGTH) -> SecretStr:
    """``length`` characters drawn with ``secrets``, holding at least one lowercase letter,
    one uppercase letter, one digit and one ``*``, so the usual composition rules pass."""
    characters = [
        secrets.choice(string.ascii_lowercase),
        secrets.choice(string.ascii_uppercase),
        secrets.choice(string.digits),
        "*",
        *(secrets.choice(_ALPHABET) for _ in range(length - 4)),
    ]
    secrets.SystemRandom().shuffle(characters)
    return SecretStr("".join(characters))


@dataclass(frozen=True, slots=True)
class StoredAgentAccount:
    """One owner's account on one site, as the store keeps it."""

    owner_id: str
    site: str
    account_id: str
    """Minted at the first registration and kept by every reset after it."""
    email: str | None
    username: str | None
    key_id: str
    envelope: str = field(repr=False)


class AgentAccountStore(Protocol):
    """The owner-scoped durable record of Agent Accounts."""

    async def account(self, *, owner_id: str, site: str) -> StoredAgentAccount | None: ...

    async def save(self, account: StoredAgentAccount) -> None:
        """Insert the account, or replace the owner's account on that site."""
        ...

    async def sealed_under(
        self, *, key_ids: Sequence[str], limit: int
    ) -> tuple[StoredAgentAccount, ...]: ...

    async def reseal(self, account: StoredAgentAccount, *, key_id: str, envelope: str) -> bool:
        """Replace the envelope ``account`` holds, unless it changed meanwhile."""
        ...


@dataclass(frozen=True, slots=True)
class AgentAccount:
    """One account as ``register`` and ``login`` see it; its password stays sealed or in memory."""

    site: str
    email: str | None
    username: str | None
    stored: StoredAgentAccount | None = field(default=None, repr=False)
    """The owner's account, which later Runs find. None for a Child's."""
    run_password: SecretStr | None = field(default=None, repr=False)
    """A Child's password, which lives only as long as its Run."""

    @property
    def persistent(self) -> bool:
        return self.stored is not None

    @property
    def identity(self) -> str:
        """What names the account to a person: its email address, else its username."""
        return cast(str, self.email or self.username)


@dataclass(frozen=True, slots=True)
class AgentAccountsBinding:
    """What a deployment composed for Agent Accounts: where they are kept, and the key ring."""

    store: AgentAccountStore
    cipher: CredentialCipher


class RunAgentAccounts:
    """One Research Run's Agent Accounts: the owner's persistent ones, sealed under the key
    ring, and its Children's Run-scoped ones, held in this process until the Run settles."""

    def __init__(self, *, owner_id: str, binding: AgentAccountsBinding) -> None:
        self._owner_id = owner_id
        self._store = binding.store
        self._cipher = binding.cipher
        #: A Child's registrations by the Agent Session that made them and the site.
        self._children: dict[tuple[str, str], AgentAccount] = {}

    def available(self) -> bool:
        """Whether the key ring can seal a password. Without one, register and login fail closed."""
        return self._cipher.active_key_id is not None

    async def registration_target(
        self, scope: str, site: str, *, child: bool
    ) -> AgentAccount | None:
        """The account a registration on ``site`` replaces the password of, if there is one.

        A parent resets the owner's account; a Child only ever replaces its own, never the owner's.
        """
        if child:
            return self._children.get((scope, site))
        return await self._owned(site)

    async def login_target(self, scope: str, site: str, *, child: bool) -> AgentAccount | None:
        """The account a login on ``site`` fills: a Child's own first, else the owner's."""
        if child and (own := self._children.get((scope, site))) is not None:
            return own
        return await self._owned(site)

    async def record(
        self,
        scope: str,
        site: str,
        *,
        child: bool,
        existing: AgentAccount | None,
        email: str | None,
        username: str | None,
        password: SecretStr,
    ) -> AgentAccount:
        """Keep a new account, or the new password of ``existing``.

        A parent's account is sealed and stored for its owner under the account id the first
        registration minted. A Child's goes no further than this process.
        """
        if child:
            account = AgentAccount(site, email, username, run_password=password)
            self._children[(scope, site)] = account
            return account
        account_id = (
            existing.stored.account_id if existing and existing.stored else uuid.uuid4().hex
        )
        key_id, envelope = self._cipher.seal(
            password, label=ACCOUNT_LABEL, binding=(self._owner_id, site, account_id)
        )
        stored = StoredAgentAccount(
            self._owner_id, site, account_id, email, username, key_id, envelope
        )
        await self._store.save(stored)
        return AgentAccount(site, email, username, stored=stored)

    def password(self, account: AgentAccount) -> SecretStr:
        """The account's password. An envelope no key opens raises ``UnreadableEnvelope``."""
        if account.run_password is not None:
            return account.run_password
        stored = cast(StoredAgentAccount, account.stored)
        return self._cipher.open(
            stored.envelope,
            label=ACCOUNT_LABEL,
            binding=(stored.owner_id, stored.site, stored.account_id),
        )

    async def _owned(self, site: str) -> AgentAccount | None:
        stored = await self._store.account(owner_id=self._owner_id, site=site)
        if stored is None:
            return None
        return AgentAccount(site, stored.email, stored.username, stored=stored)


async def reseal_agent_accounts(
    store: AgentAccountStore, cipher: CredentialCipher, *, limit: int = 100
) -> int:
    """One bounded pass: re-seal the passwords under a retired key with the active key.

    An envelope no key opens is left alone, and its account recovers through the site's reset
    mail. Returns how many accounts it re-sealed.
    """
    retired = cipher.retired_key_ids
    if not retired:
        return 0
    resealed = 0
    for account in await store.sealed_under(key_ids=retired, limit=limit):
        binding = (account.owner_id, account.site, account.account_id)
        try:
            password = cipher.open(account.envelope, label=ACCOUNT_LABEL, binding=binding)
        except UnreadableEnvelope:
            continue
        key_id, envelope = cipher.seal(password, label=ACCOUNT_LABEL, binding=binding)
        if await store.reseal(account, key_id=key_id, envelope=envelope):
            resealed += 1
    return resealed


__all__ = [
    "ACCOUNT_LABEL",
    "MIN_PASSWORD_LENGTH",
    "PASSWORD_LENGTH",
    "AgentAccount",
    "AgentAccountStore",
    "AgentAccountsBinding",
    "RunAgentAccounts",
    "StoredAgentAccount",
    "account_site",
    "generate_password",
    "reseal_agent_accounts",
]
