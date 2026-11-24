# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Rotation over interchangeable credentials for one provider and operation.

A provider that accepts several API keys can multiply its quota and spread a per-key rate
limit. This module owns only that availability policy: which credential to ask next, which
ones said "not now", and when one is worth probing again. The caller runs the operation and
reports the outcome, so the policy is reusable by any provider whose credentials are
interchangeable and whose failures split the same two ways — a failure about this call,
which the next call may not repeat, and a failure about the credential, which it will.

State is per (provider, operation) pool, because one leg's credit says nothing about the
other's: a provider out of search credit may still serve an extraction.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field

#: How long a credential that said it is out of reach is left alone. Long enough that a
#: burst does not pay a dead credential's round trip, short enough that a top-up or a lifted
#: rate limit is picked up within a minute of it happening.
DEFAULT_COOLDOWN_SECONDS = 60.0


@dataclass(frozen=True, slots=True)
class Credential:
    """One interchangeable credential; ``index`` is its 1-based place, never its secret."""

    index: int
    secret: str = field(repr=False)


class CredentialPool:
    """Rotate over one provider's interchangeable credentials for one operation.

    A call walks :meth:`attempts` and reports each outcome with :meth:`release`.
    ``cooling=True`` means a credential-level failure (refused, out of credit, over a rate
    limit) and parks that credential for ``cooldown_seconds``; ``cooling=False`` means the
    credential answered or failed about this call, and clears any cooldown it carried.

    Rotation asks the credential leased longest ago first, so load spreads across the pool
    instead of draining the first key — which is also what keeps a per-key rate limit from
    greeting every call. When every credential is cooling, :meth:`attempts` yields exactly
    one probe: the credential whose cooldown ends first.
    """

    def __init__(
        self,
        secrets: Sequence[str],
        *,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._credentials = _unique_credentials(secrets)
        self._cooldown_seconds = cooldown_seconds
        self._clock = clock or time.monotonic
        #: The rotation order, least recently leased first.
        self._order = list(self._credentials)
        self._cooling_until: dict[int, float] = {}

    @property
    def credentials(self) -> tuple[Credential, ...]:
        """Every credential in configured order; a caller builds one client per item."""
        return self._credentials

    def __len__(self) -> int:
        return len(self._credentials)

    def attempts(self) -> Iterator[Credential]:
        """Yield the credentials this call should try, in order, and nothing when empty.

        The candidates are settled when iteration starts: a credential another concurrent
        call cools mid-walk is still tried here, because it was eligible when this call
        began. Each is yielded once, so a failed credential is never retried in the walk.
        """
        now = self._clock()
        ready = [
            credential
            for credential in tuple(self._order)
            if self._cooling_until.get(credential.index, 0.0) <= now
        ]
        if ready:
            for credential in ready:
                self._lease(credential)
                yield credential
            return
        if self._credentials:
            probe = min(self._order, key=lambda item: self._cooling_until[item.index])
            self._lease(probe)
            yield probe

    def release(self, credential: Credential, *, cooling: bool) -> None:
        """Report one attempt's outcome.

        A probe that fails again stays out for a fresh window, measured from the answer that
        refused it. An answer, or a failure about this call, clears the cooldown.
        """
        if cooling:
            self._cooling_until[credential.index] = self._clock() + self._cooldown_seconds
        else:
            self._cooling_until.pop(credential.index, None)

    def _lease(self, credential: Credential) -> None:
        self._order.remove(credential)
        self._order.append(credential)


def _unique_credentials(secrets: Sequence[str]) -> tuple[Credential, ...]:
    """The configured secrets, deduplicated and numbered for a log line and never rendered.

    A blank secret is a configuration error rather than a credential that can never answer:
    a pool that silently carried one would spend a call on it first.
    """
    unique: list[str] = []
    for secret in secrets:
        text = secret.strip()
        if not text:
            raise ValueError("Credential rotation refuses a blank secret")
        if text not in unique:
            unique.append(text)
    return tuple(Credential(index, secret) for index, secret in enumerate(unique, start=1))


__all__ = [
    "DEFAULT_COOLDOWN_SECONDS",
    "Credential",
    "CredentialPool",
]
