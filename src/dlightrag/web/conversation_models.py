# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Browser-safe contracts for durable Web conversations."""

import datetime
from typing import Any

from pydantic import Field, field_validator

from dlightrag.core.client_contracts import ClientContractModel


class ConversationSummary(ClientContractModel):
    conversation_id: str
    title: str | None = None
    created_at: datetime.datetime
    updated_at: datetime.datetime


class ConversationImageReference(ClientContractModel):
    image_id: str
    ordinal: int
    mime_type: str
    url: str
    thumbnail_url: str
    label: str


class ConversationDocumentReference(ClientContractModel):
    attachment_id: str
    ordinal: int
    filename: str
    mime_type: str
    byte_size: int
    url: str
    label: str
    parse_summary: str | None = None


class ConversationTurn(ClientContractModel):
    turn_id: str
    turn_number: int
    user_text: str
    assistant_text: str
    user_images: list[ConversationImageReference] = Field(default_factory=list)
    user_documents: list[ConversationDocumentReference] = Field(default_factory=list)
    answer_sources: dict[str, Any] = Field(default_factory=dict)
    answer_html: str
    queried_workspaces: list[str] = Field(default_factory=list)
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
    "ConversationDocumentReference",
    "ConversationHistory",
    "ConversationImageReference",
    "ConversationSummary",
    "ConversationTurn",
    "RenameConversationRequest",
]
