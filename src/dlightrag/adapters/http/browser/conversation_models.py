# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Browser Pydantic presentation models for Web Conversations."""

import datetime
from typing import Any

from pydantic import Field, field_validator

from dlightrag.adapters.http.browser.presentation import AnswerPresentation
from dlightrag.application.answer_runs import AnswerRunStatus
from dlightrag.application.answer_runs.client_contracts import ClientContractModel


class ConversationSummary(ClientContractModel):
    conversation_id: str
    title: str | None = None
    created_at: datetime.datetime
    updated_at: datetime.datetime
    forked_from_conversation_id: str | None = None
    forked_from_title: str | None = None


class ConversationPage(ClientContractModel):
    items: list[ConversationSummary]
    next_cursor: str | None = None


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
    "ConversationAttachmentReference",
    "ConversationHistory",
    "ConversationPage",
    "ConversationSummary",
    "ConversationTurn",
    "RenameConversationRequest",
]
