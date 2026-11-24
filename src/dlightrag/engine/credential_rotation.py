# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Rotation over interchangeable credentials for one provider and operation.

A provider that accepts several API keys can multiply its quota and spread a per-key
rate limit across them. This module owns only the availability policy: which credential
to ask next, which ones said "not now", and when one is worth probing again. The caller
runs its own operation and reports the outcome, so the policy is reusable by any provider
whose credentials are interchangeable and whose failures split the same two ways — a
failure about this call, which the next call may not repeat, and a failure about the
credential, which it will.

State is per (provider, operation) pool rather than global, because one leg's credit
says nothing about the other's: a provider out of search credit may still serve an
extraction, exactly as the ordered provider chain treats the two operations.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field

#: How long a credential that said it is out of reach is left alone. Long enough that a
#: burst does not pay a dead credential's round trip on every call, short enough that a
#: top-up or a lifted rate limit is picked up within a minute of it happening.
DEFAULT_COOLDOWN_SECONDS = 60.0


@dataclass(frozen=True, slots=True)
class Credential:
    """One interchangeable credential.

    ``index`` is its 1-based position in the pool: the only part of it a log line or a
    diagnostic may carry. ``secret`` never renders, and never reaches a log record.
    """

    index: int
    secret: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class CredentialLease:
    """One credential a caller should try, and whether trying it tests a recovery.

    A probe is the single attempt that goes out when every credential is cooling: it is
    what notices a top-up or a lifted rate limit that happened a second after a burst,
    rather than waiting for a timer nobody reads. A caller stops after a failed probe.
    """

    credential: Credential
    probe: bool = False


class CredentialPool:
    """Rotate over one provider's interchangeable credentials for one operation.

    A call walks :meth:`attempts` and reports each outcome with :meth:`release`, which
    is what keeps the pool's memory of a credential current: ``cooling=True`` means a
    credential-level failure (refused, out of credit, over a rate limit) and parks that
    credential for ``cooldown_seconds``; ``cooling=False`` means the credential answered
    or failed about this call, and clears any cooldown it carried.

    Rotation asks the credential leased longest ago first, so load spreads across the
    pool instead of draining the first key before touching the second — which is also
    what keeps a per-key rate limit from greeting every call. When every credential is
    cooling, :meth:`attempts` yields exactly one probe, the credential whose cooldown
    ends first; the chain's own order settles a tie.
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
        #: ``index -> the monotonic time the credential may be asked again``.
        self._cooling_until: dict[int, float] = {}

    @property
    def credentials(self) -> tuple[Credential, ...]:
        """Every credential in configured order; a caller builds one client per item."""
        return self._credentials

    def __len__(self) -> int:
        return len(self._credentials)

    def attempts(self) -> Iterator[CredentialLease]:
        """Yield the credentials this call should try, in order, and nothing when empty.

        The candidates are settled when iteration starts: a credential that another
        concurrent call cools mid-walk is still tried here rather than skipped, because
        it was eligible when this call began and its answer is the only evidence that
        would clear it. A credential that failed a credential-level check during this
        same walk is not retried, because ``attempts`` yields each one once.
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
                yield CredentialLease(credential)
            return
        if not self._credentials:
            return
        probe = min(
            self._order,
            key=lambda credential: self._cooling_until[credential.index],
        )
        self._lease(probe)
        yield CredentialLease(probe, probe=True)

    def release(self, credential: Credential, *, cooling: bool) -> None:
        """Report one attempt's outcome.

        A probe that fails again stays out for a fresh window, measured from the answer
        that refused it rather than from the burst that first parked it. An answer, or a
        failure about this call, clears the cooldown: the credential is eligible again,
        and the rotation order already put it behind the credentials this call skipped.
        """
        if credential not in self._credentials:
            raise ValueError("Credential does not belong to this pool")
        if cooling:
            self._cooling_until[credential.index] = self._clock() + self._cooldown_seconds
        else:
            self._cooling_until.pop(credential.index, None)

    def _lease(self, credential: Credential) -> None:
        self._order.remove(credential)
        self._order.append(credential)


def _unique_credentials(secrets: Sequence[str]) -> tuple[Credential, ...]:
    """The configured secrets, deduplicated, numbered for a log line and never rendered.

    A blank secret is a configuration error rather than a credential that can never
    answer: a pool that silently carried one would spend a call on it first.
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
    "CredentialLease",
    "CredentialPool",
]
