# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Browser-safe contracts for durable Web conversations."""

import datetime
from dataclasses import dataclass
from typing import Any

from pydantic import Field, field_validator

from dlightrag.answer.client_contracts import ClientContractModel
from dlightrag.runtime import AnswerRunRecord, AnswerRunStatus
from dlightrag.web.presentation import AnswerPresentation


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
class AnswerTurnCreation:
    """The run and the conversation entry one submission durably created."""

    turn: LinkedTurn
    summary: dict[str, Any]
    replayed: bool


class ConversationSubmissionConflict(RuntimeError):
    """One principal reused a submission id with a different conversation or input."""


class WebConversationUnavailableError(RuntimeError):
    """Durable Web conversation storage cannot currently be reached."""

    detail = "Web conversation storage is unavailable"


class WebConversationSchemaError(RuntimeError):
    """The durable Web conversation schema is incompatible with this revision."""


class ConversationSummary(ClientContractModel):
    conversation_id: str
    title: str | None = None
    created_at: datetime.datetime
    updated_at: datetime.datetime
    forked_from_conversation_id: str | None = None
    forked_from_title: str | None = None


class ConversationAttachmentReference(ClientContractModel):
    attachment_id: str
    ordinal: int
    kind: str
    filename: str
    mime_type: str
    byte_size: int
    url: str
    thumbnail_url: str | None = None
    label: str


class ConversationTurn(ClientContractModel):
    turn_id: str
    turn_number: int
    answer_run_id: str
    submission_id: str
    status: AnswerRunStatus
    cancel_requested: bool = False
    user_text: str
    assistant_text: str
    user_attachments: list[ConversationAttachmentReference] = Field(default_factory=list)
    presentation: AnswerPresentation | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)
    error_kind: str | None = None
    error_message: str | None = None
    created_at: datetime.datetime


class AnswerRunDescriptor(ClientContractModel):
    """What the browser needs to follow one accepted submission."""

    run_id: str
    status: AnswerRunStatus
    cancel_requested: bool = False
    turn_id: str
    turn_number: int
    submission_id: str
    events_url: str
    status_url: str
    cancel_url: str
    conversation: ConversationSummary
    parent_run_id: str | None = None
    continuation_kind: str | None = None


class ConversationHistory(ClientContractModel):
    conversation: ConversationSummary
    turns: list[ConversationTurn]


class RenameConversationRequest(ClientContractModel):
    title: str = Field(min_length=1, max_length=120)

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("title must not be blank")
        return normalized


__all__ = [
    "AnswerRunDescriptor",
    "AnswerTurnCreation",
    "ConversationAttachmentReference",
    "ConversationHistory",
    "ConversationSnapshot",
    "ConversationSubmissionConflict",
    "ConversationSummary",
    "ConversationTurn",
    "LinkedTurn",
    "RenameConversationRequest",
    "WebConversationUnavailableError",
    "WebConversationSchemaError",
]
