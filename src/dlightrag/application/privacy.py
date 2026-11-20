# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Conversation Contribution: whether an owner lets their conversations be used to evaluate and
train models (ADR 0041).

The permission is off until the owner turns it on, and it is one fact: the moment it began.
Turning it off clears the moment and ends the permission at once, and turning it on again begins
a new one that covers only the conversations started after. An owner with nothing stored has never
been asked, which reads as off here and everywhere else. Only the operator's trace export
(scripts/export_traces.py) reads the setting, straight from the table, and it takes a Run only if
its owner contributes now and the Run's Agent Session started at or after that moment: a Session
starts at the earlier of its own creation and the acceptance of its first remaining Run, and the
Run's Session is the one its routing row names, never a Child's. The owner's word for a Session is
"conversation": a conversation and everything forked or followed up from it share one, and a Run
submitted without a parent starts a new one.

It is the owner's own state, like their Memory and the judgements they give an answer, so no
access action guards it. It is not a monitoring switch: monitoring has no off state and its traces
keep bounded conversation text, which this setting does not change (ADR 0039).
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class PrivacySettings:
    """One owner's Conversation Contribution as it stands.

    ``since`` is when the permission began, and it is None exactly when ``contribute`` is false:
    the permission is the moment, and the flag only spells it. An owner who never chose
    contributes nothing.
    """

    contribute: bool
    since: datetime.datetime | None


class PrivacySettingsStore(Protocol):
    """Where each owner's setting is kept: one row for an owner who has chosen, none for one who
    has not."""

    async def read_privacy_settings(self, *, owner_id: str) -> PrivacySettings:
        """The owner's setting. An owner with no row is not contributing."""
        ...

    async def write_privacy_settings(self, *, owner_id: str, contribute: bool) -> PrivacySettings:
        """Keep the owner's choice and answer what is kept.

        One atomic write decides the moment from what is stored when it runs: the database's
        clock when the permission turns on after being off or absent, unchanged when it stays on,
        none when it turns off. A write is made once and never replayed, since a replay arriving
        after a later turn-off would give back a permission the owner had withdrawn.
        """
        ...


class Privacy:
    """An owner's Conversation Contribution in Settings: read it, and turn it on or off."""

    def __init__(self, *, store: PrivacySettingsStore) -> None:
        self._store = store

    async def settings(self, *, owner_id: str) -> PrivacySettings:
        """The owner's setting, read afresh: a request never meets an earlier answer."""
        return await self._store.read_privacy_settings(owner_id=owner_id)

    async def set_contribute(self, *, owner_id: str, contribute: bool) -> PrivacySettings:
        """Turn the permission on or off and answer what is kept.

        The value is the state the owner wants, not a toggle, so a page that is out of date
        cannot flip it by accident. Turning it on while it is on keeps its moment, turning it off
        clears it, and turning it on again begins a new permission at that time.
        """
        return await self._store.write_privacy_settings(owner_id=owner_id, contribute=contribute)


__all__ = ["Privacy", "PrivacySettings", "PrivacySettingsStore"]
