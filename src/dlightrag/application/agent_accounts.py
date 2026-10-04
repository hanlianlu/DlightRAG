# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Writer maintenance for Agent Accounts: re-seal their passwords after a key ring rotation.

A deployment rotates its key ring by adding a key and making it active; every account envelope
still sealed under an older key is opened with that key and sealed again under the new one
(ADR 0034), so the old key can go. Connections do the same for their Grants in a loop of their
own, because that loop also refreshes OAuth grants and the two share nothing but the ring.
"""

from __future__ import annotations

import asyncio
import logging

from dlightrag.engine.answer.agent_browser import AgentAccountStore, reseal_agent_accounts
from dlightrag.engine.credential_cipher import CredentialCipher

logger = logging.getLogger(__name__)

#: How often a writer looks for envelopes under a retired key, as often as Connections do.
_MAINTENANCE_SECONDS = 60.0


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


__all__ = ["AgentAccountMaintenance"]
