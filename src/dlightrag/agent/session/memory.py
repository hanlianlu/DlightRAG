# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""An in-memory agent session store with the durable store's exact semantics.

The adapter implements the same version, effect, and projection rules as the
PostgreSQL adapter so orchestrator behavior is testable without a database.
Host updates are :class:`NoHostUpdate`: nothing outside the session changes.
"""

from collections.abc import Sequence
from dataclasses import replace

from dlightrag.agent.session.effects import EffectSettlement
from dlightrag.agent.session.entries import (
    EffectIntentEntry,
    EffectResultEntry,
    SessionEntry,
)
from dlightrag.agent.session.ids import IntentId, SessionId
from dlightrag.agent.session.projection import ContextProjection
from dlightrag.agent.session.store import (
    AgentSessionSnapshot,
    AppendCommit,
    EffectAlreadySettled,
    EffectCommit,
    EffectMissing,
    NoHostUpdate,
    SessionCommit,
    SessionProgressClass,
    SettleCommit,
    VersionConflict,
)


class InMemoryAgentSessionStore:
    """One process-local journal with one version per committed transaction."""

    def __init__(self, *, initial_version: int = 0) -> None:
        if initial_version < 0:
            raise ValueError("initial_version cannot be negative")
        self._sessions: dict[SessionId, _Session] = {}
        self._initial_version = initial_version

    async def load(self, session_id: SessionId) -> AgentSessionSnapshot:
        session = self._sessions.get(session_id)
        if session is None:
            return AgentSessionSnapshot(
                session_id=session_id,
                version=self._initial_version,
                entries=(),
                active_projection=None,
            )
        return AgentSessionSnapshot(
            session_id=session_id,
            version=session.version,
            entries=tuple(session.entries),
            active_projection=session.active_projection,
        )

    async def append(
        self,
        *,
        session_id: SessionId,
        expected_version: int,
        entries: Sequence[SessionEntry],
        projection: ContextProjection | None = None,
    ) -> AppendCommit:
        session = self._sessions.setdefault(session_id, _Session())
        if session.version != expected_version:
            return VersionConflict(
                expected_version=expected_version, current_version=session.version
            )
        appended = self._write(session, entries, projection=projection)
        return SessionCommit(version=session.version, appended_sequences=appended)

    async def settle_effect(
        self,
        *,
        session_id: SessionId,
        expected_version: int,
        intent_id: IntentId,
        settlement: EffectSettlement[NoHostUpdate],
        entries: Sequence[SessionEntry],
        projection: ContextProjection | None = None,
        progress: SessionProgressClass = "live",
    ) -> SettleCommit:
        del progress
        session = self._sessions.get(session_id)
        if session is None:
            return EffectMissing(intent_id=intent_id)
        if session.version != expected_version:
            return VersionConflict(
                expected_version=expected_version, current_version=session.version
            )
        intent_entry = self._unsettled_intent(session, intent_id)
        if intent_entry is None:
            settled = session.settled_intents.get(intent_id)
            if settled is not None:
                return EffectAlreadySettled(intent_id=intent_id)
            return EffectMissing(intent_id=intent_id)
        if any(
            isinstance(entry, EffectResultEntry) and entry.intent_id != intent_id
            for entry in entries
        ):
            raise ValueError("settlement entries must belong to the settled intent")
        appended = self._write(session, entries, projection=projection)
        session.settled_intents[intent_id] = intent_entry
        return EffectCommit(
            version=session.version,
            appended_sequences=appended,
            intent_id=intent_id,
            outcome=settlement.outcome,
        )

    def _unsettled_intent(self, session: _Session, intent_id: IntentId) -> EffectIntentEntry | None:
        for entry in session.entries:
            if isinstance(entry, EffectIntentEntry) and entry.intent_id == intent_id:
                if intent_id not in session.settled_intents:
                    return entry
                return None
        return None

    def _write(
        self,
        session: _Session,
        entries: Sequence[SessionEntry],
        *,
        projection: ContextProjection | None,
    ) -> tuple[int, ...]:
        if not entries:
            raise ValueError("a session transaction requires at least one entry")
        sequences = tuple(
            range(session.last_sequence + 1, session.last_sequence + len(entries) + 1)
        )
        stamped = [
            replace(entry, sequence=sequence)
            for entry, sequence in zip(entries, sequences, strict=True)
        ]
        session.entries.extend(stamped)
        session.last_sequence = sequences[-1]
        session.version += 1
        if projection is not None:
            session.active_projection = projection
        return sequences


class _Session:
    def __init__(self) -> None:
        self.entries: list[SessionEntry] = []
        self.settled_intents: dict[IntentId, EffectIntentEntry] = {}
        self.last_sequence = 0
        self.version = 0
        self.active_projection: ContextProjection | None = None


__all__ = ["InMemoryAgentSessionStore"]
