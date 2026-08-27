# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable Web Conversation records and persistence port."""

import datetime
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from dlightrag.application.answer_runs.routing import RoutingAcceptance
from dlightrag.runtime import AnswerRunRecord, PendingArtifact, PendingArtifactReference


@dataclass(frozen=True, slots=True)
class LinkedTurn:
    """One conversation entry and the authoritative run state behind it."""

    turn_id: str
    turn_number: int
    submission_id: str
    created_at: datetime.datetime
    run: AnswerRunRecord
    conversation_id: str = ""

    @property
    def answer_run_id(self) -> str:
        return self.run.run_id


@dataclass(frozen=True, slots=True)
class ConversationSnapshot:
    principal_id: str
    conversation_id: str
    content_revision: int
    title: str | None
    created_at: datetime.datetime
    updated_at: datetime.datetime
    agent_session_id: str
    agent_lane_id: str
    turns: tuple[LinkedTurn, ...]


@dataclass(frozen=True, slots=True)
class ConversationSummary:
    """Application projection of one durable conversation row."""

    conversation_id: str
    title: str | None
    created_at: datetime.datetime
    updated_at: datetime.datetime
    forked_from_conversation_id: str | None = None
    forked_from_title: str | None = None


@dataclass(frozen=True, slots=True)
class AnswerTurnCreation:
    """The run and conversation entry one submission durably created."""

    turn: LinkedTurn
    summary: dict[str, Any]
    replayed: bool


class ConversationSubmissionConflict(RuntimeError):
    """One principal reused a submission id for different accepted input."""


class WebConversationUnavailableError(RuntimeError):
    """Durable Web Conversation storage cannot currently be reached."""

    detail = "Web conversation storage is unavailable"


class WebConversationSchemaError(RuntimeError):
    """The durable Web Conversation schema is incompatible with this revision."""


class WebConversationStore(Protocol):
    """Persistence operations owned by Web Conversations."""

    async def prune_empty_conversations(self, *, batch_size: int = 500) -> int: ...
    async def create_conversation(self, principal_id: str) -> dict[str, Any]: ...
    async def list_conversations(self, principal_id: str) -> list[dict[str, Any]]: ...
    async def rename_conversation(
        self, principal_id: str, conversation_id: str, *, title: str
    ) -> dict[str, Any] | None: ...
    async def delete_conversation(self, principal_id: str, conversation_id: str) -> bool: ...
    async def delete_all_conversations(self, principal_id: str) -> int: ...
    async def snapshot(
        self, principal_id: str, conversation_id: str, *, window_turns: int = 100
    ) -> ConversationSnapshot | None: ...
    async def find_turn_by_run(self, principal_id: str, run_id: str) -> LinkedTurn | None: ...
    async def replay_answer_turn(
        self,
        *,
        principal_id: str,
        conversation_id: str,
        submission_id: str,
        idempotency_fingerprint: str,
    ) -> AnswerTurnCreation | None: ...
    async def create_answer_turn(
        self,
        *,
        principal_id: str,
        conversation_id: str,
        submission_id: str,
        request: Mapping[str, Any],
        idempotency_fingerprint: str,
        artifacts: Sequence[PendingArtifact],
        references: Sequence[PendingArtifactReference],
        title_hint: str | None,
        routing: RoutingAcceptance | None = None,
        create_conversation: bool = False,
        forked_from_conversation_id: str | None = None,
    ) -> AnswerTurnCreation | None: ...


__all__ = [
    "AnswerTurnCreation",
    "ConversationSnapshot",
    "ConversationSummary",
    "ConversationSubmissionConflict",
    "LinkedTurn",
    "WebConversationSchemaError",
    "WebConversationStore",
    "WebConversationUnavailableError",
]
