# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Agent Accounts above the stores: what an owner manages of them in Settings, and the writer's
upkeep of their envelopes.

An owner sees the sites the Agent registered on, the identity it registered under, and two
dates, switches the Agent's new sign-ups on or off, and removes an account (ADR 0034). Nothing
here reads a password: DlightRAG makes and seals each one, and no view carries it.

A deployment rotates its key ring by adding a key and making it active; every account envelope
still sealed under an older key is opened with that key and sealed again under the new one
(ADR 0034), so the old key can go. Connections do the same for their Grants in a loop of their
own, because that loop also refreshes OAuth grants and the two share nothing but the ring.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import re
from dataclasses import dataclass
from typing import Protocol

from dlightrag.application.errors import ApplicationInputError, ApplicationNotFoundError
from dlightrag.engine.answer.agent_browser import (
    AgentAccountStore,
    AgentAccountSummary,
    reseal_agent_accounts,
)
from dlightrag.engine.credential_cipher import CredentialCipher

logger = logging.getLogger(__name__)

#: How often a writer looks for envelopes under a retired key, as often as Connections do.
_MAINTENANCE_SECONDS = 60.0

#: What names an account: the registrable domain of its site, in lowercase.
_SITE = re.compile(r"[a-z0-9_-]{1,63}(\.[a-z0-9_-]{1,63})*")
_SITE_MAX_CHARS = 253


class AgentAccountSettingsStore(Protocol):
    """Whether each owner lets the Agent register new accounts; an owner who never chose has them
    on."""

    async def registration_enabled(self, *, owner_id: str) -> bool: ...

    async def set_registration_enabled(self, *, owner_id: str, enabled: bool) -> bool: ...


@dataclass(frozen=True, slots=True)
class AgentRegistrationView:
    """Whether new sign-ups are possible: the deployment allows them, and the owner's switch."""

    allowed: bool
    enabled: bool


@dataclass(frozen=True, slots=True)
class AgentAccountView:
    """One account as its owner sees it, with its times in UTC."""

    site: str
    email: str | None
    username: str | None
    created_at: str
    last_used_at: str | None


@dataclass(frozen=True, slots=True)
class AgentAccountsView:
    """An owner's Agent Accounts, the switch for new sign-ups, and whether the deployment has
    Agent Accounts at all, which it has where an Agent Browser is configured."""

    available: bool
    registration: AgentRegistrationView
    accounts: tuple[AgentAccountView, ...]


def _utc(moment: datetime.datetime) -> str:
    return moment.astimezone(datetime.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _view_of(summary: AgentAccountSummary) -> AgentAccountView:
    return AgentAccountView(
        site=summary.site,
        email=summary.email,
        username=summary.username,
        created_at=_utc(summary.created_at),
        last_used_at=None if summary.last_used_at is None else _utc(summary.last_used_at),
    )


class AgentAccounts:
    """An owner's Agent Accounts in Settings: the list, the switch, and removal.

    An account is removable whether or not the deployment still has Agent Accounts, and the
    switch is stored whether or not the deployment allows sign-ups: what the Agent may do is the
    deployment's allowance and the owner's switch together.
    """

    def __init__(
        self,
        *,
        store: AgentAccountStore,
        settings_store: AgentAccountSettingsStore,
        available: bool,
        registration_allowed: bool,
    ) -> None:
        self._store = store
        self._settings = settings_store
        self._available = available
        self._registration_allowed = registration_allowed

    async def view(self, *, owner_id: str) -> AgentAccountsView:
        return AgentAccountsView(
            available=self._available,
            registration=AgentRegistrationView(
                allowed=self._registration_allowed,
                enabled=await self._settings.registration_enabled(owner_id=owner_id),
            ),
            accounts=tuple(
                _view_of(summary) for summary in await self._store.summaries(owner_id=owner_id)
            ),
        )

    async def registration(self, *, owner_id: str) -> bool:
        """Whether a Run accepted for the owner now may register: the deployment allows it and
        the owner has not turned it off. The Run keeps this answer, whatever is switched next."""
        return self._registration_allowed and await self._settings.registration_enabled(
            owner_id=owner_id
        )

    async def set_registration(self, *, owner_id: str, enabled: bool) -> AgentAccountsView:
        await self._settings.set_registration_enabled(owner_id=owner_id, enabled=enabled)
        return await self.view(owner_id=owner_id)

    async def remove(self, *, owner_id: str, site: str) -> AgentAccountsView:
        """Remove the owner's account on ``site``. Another owner's account there is not theirs
        to remove, so it is as unknown as one nobody has."""
        if len(site) > _SITE_MAX_CHARS or _SITE.fullmatch(site) is None:
            raise ApplicationInputError("site must be a lowercase hostname")
        if not await self._store.delete(owner_id=owner_id, site=site):
            raise ApplicationNotFoundError("This owner has no Agent Account for that site")
        return await self.view(owner_id=owner_id)


class AgentAccountMaintenance:
    """Re-seals Agent Account envelopes under the active key, now and then once a minute."""

    def __init__(self, *, store: AgentAccountStore, cipher: CredentialCipher) -> None:
        self._store = store
        self._cipher = cipher
        self._task: asyncio.Task[None] | None = None

    async def maintain(self) -> int:
        """One bounded pass; how many accounts it re-sealed."""
        return await reseal_agent_accounts(self._store, self._cipher)

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(
                self._maintain_forever(), name="agent-account-maintenance"
            )

    async def aclose(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _maintain_forever(self) -> None:
        while True:
            try:
                await self.maintain()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # A store that is down says only what kind of failure it was, never its text.
                logger.warning("Agent Account maintenance failed (%s)", type(exc).__name__)
            await asyncio.sleep(_MAINTENANCE_SECONDS)


__all__ = [
    "AgentAccountMaintenance",
    "AgentAccountSettingsStore",
    "AgentAccountView",
    "AgentAccounts",
    "AgentAccountsView",
    "AgentRegistrationView",
]
