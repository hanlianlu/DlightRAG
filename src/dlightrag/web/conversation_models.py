# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Browser-safe contracts for durable Web conversations."""

import datetime

from pydantic import Field, field_validator

from dlightrag.core.client_contracts import ClientContractModel


class ConversationSummary(ClientContractModel):
    conversation_id: str
    title: str | None = None
    created_at: datetime.datetime
    updated_at: datetime.datetime


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
    user_text: str
    assistant_text: str
    user_attachments: list[ConversationAttachmentReference] = Field(default_factory=list)
    answer_html: str
    created_at: datetime.datetime


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
    "ConversationAttachmentReference",
    "ConversationHistory",
    "ConversationSummary",
    "ConversationTurn",
    "RenameConversationRequest",
]
