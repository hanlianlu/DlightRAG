# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Typed browser-facing SSE event payloads."""

from typing import Any, Literal

from pydantic import Field

from dlightrag.core.client_contracts import ClientContractModel
from dlightrag.web.conversation_models import ConversationSummary


class AnswerMetaEvent(ClientContractModel):
    history_kept: int


class AnswerProgressEvent(ClientContractModel):
    phase: Literal["planning", "searching", "generating", "saving"]


class AnswerDoneEvent(ClientContractModel):
    html: str
    answer: str
    current_image_ids: list[str] = Field(default_factory=list)
    current_attachment_ids: list[str] = Field(default_factory=list)
    image_descriptions: dict[str, str] = Field(default_factory=dict)
    answer_images: list[dict[str, Any]] = Field(default_factory=list)
    answer_blocks: list[dict[str, Any]] = Field(default_factory=list)
    conversation_saved: bool
    conversation_save_reason: str | None = None
    conversation: ConversationSummary | None = None


class AnswerTraceEvent(ClientContractModel):
    trace: dict[str, Any]


class AnswerErrorEvent(ClientContractModel):
    message: str
    error_kind: str


__all__ = [
    "AnswerDoneEvent",
    "AnswerErrorEvent",
    "AnswerMetaEvent",
    "AnswerProgressEvent",
    "AnswerTraceEvent",
]
