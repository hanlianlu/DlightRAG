# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The agent session store contract and its closed commit outcomes.

Every mutating operation returns an explicit commit outcome value; expected
conflicts are values, never database exceptions (M3-D3). Each successful
transaction increments ``session_version`` exactly once and allocates one
contiguous entry-sequence range for every entry it appends (M3-D16).
``append`` can never settle an effect.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from dlightrag.agent.session.effects import EffectSettlement, HostUpdateT
from dlightrag.agent.session.entries import SessionEntry
from dlightrag.agent.session.ids import IntentId, SessionId
from dlightrag.agent.session.projection import ContextProjection

type SessionProgressClass = Literal["live", "prelude"]


@dataclass(frozen=True, slots=True)
class SessionCommit:
    """One committed append: new version and the contiguous sequences written."""

    version: int
    appended_sequences: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EffectCommit:
    """One committed settlement: new version, sequences, and the settled intent."""

    version: int
    appended_sequences: tuple[int, ...]
    intent_id: IntentId
    outcome: str


@dataclass(frozen=True, slots=True)
class VersionConflict:
    """The expected session version no longer matches the stored version."""

    expected_version: int
    current_version: int


@dataclass(frozen=True, slots=True)
class LeaseLost:
    """The caller's lease no longer owns this session."""


@dataclass(frozen=True, slots=True)
class EffectMissing:
    """No unsettled intent with this id exists in the session."""

    intent_id: IntentId


@dataclass(frozen=True, slots=True)
class EffectAlreadySettled:
    """The intent was already settled; load and fold the committed settlement."""

    intent_id: IntentId


@dataclass(frozen=True, slots=True)
class EffectContractChanged:
    """The stored intent no longer matches the settlement's tool contract."""

    intent_id: IntentId


@dataclass(frozen=True, slots=True)
class EvidenceConflict:
    """A host update collided with existing evidence or resource identity."""


type AppendCommit = SessionCommit | VersionConflict | LeaseLost
type SettleCommit = (
    EffectCommit
    | VersionConflict
    | LeaseLost
    | EffectMissing
    | EffectAlreadySettled
    | EffectContractChanged
    | EvidenceConflict
)


@dataclass(frozen=True, slots=True)
class AgentSessionSnapshot:
    """One session's folded facts: entries, version, and active projection."""

    session_id: SessionId
    version: int
    entries: tuple[SessionEntry, ...]
    active_projection: ContextProjection | None


class AgentSessionStore[HostUpdateT](Protocol):
    """Durable journal storage for one agent session.

    The PostgreSQL adapter commits host updates, ordered result entries,
    projection, settlement, session version, and — for live settlements —
    durable run progress in one transaction. Recovery prelude settlements use
    ``progress="prelude"`` and must not advance durable progress.
    """

    async def load(self, session_id: SessionId) -> AgentSessionSnapshot:
        """Return the current snapshot for one session."""
        ...

    async def append(
        self,
        *,
        session_id: SessionId,
        expected_version: int,
        entries: Sequence[SessionEntry],
        projection: ContextProjection | None = None,
    ) -> AppendCommit:
        """Append ordered entries atomically; never settles an effect."""
        ...

    async def settle_effect(
        self,
        *,
        session_id: SessionId,
        expected_version: int,
        intent_id: IntentId,
        settlement: EffectSettlement[HostUpdateT],
        entries: Sequence[SessionEntry],
        projection: ContextProjection | None = None,
        progress: SessionProgressClass = "live",
    ) -> SettleCommit:
        """Settle one existing unsettled intent atomically with its results."""
        ...


__all__ = [
    "AgentSessionSnapshot",
    "AgentSessionStore",
    "AppendCommit",
    "EffectAlreadySettled",
    "EffectCommit",
    "EffectContractChanged",
    "EffectMissing",
    "EvidenceConflict",
    "HostUpdateT",
    "LeaseLost",
    "SessionCommit",
    "SessionProgressClass",
    "SettleCommit",
    "VersionConflict",
]
