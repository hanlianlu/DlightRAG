# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable Web Conversation records, persistence port, and lifecycle."""

from .models import (
    AnswerTurnCreation,
    ConversationSnapshot,
    ConversationSubmissionConflict,
    ConversationSummary,
    LinkedTurn,
    WebConversationSchemaError,
    WebConversationStore,
    WebConversationUnavailableError,
)
from .service import WebAnswerSubmission, WebAttachment, WebConversationService

__all__ = [
    "AnswerTurnCreation",
    "ConversationSnapshot",
    "ConversationSubmissionConflict",
    "ConversationSummary",
    "LinkedTurn",
    "WebAnswerSubmission",
    "WebAttachment",
    "WebConversationSchemaError",
    "WebConversationService",
    "WebConversationStore",
    "WebConversationUnavailableError",
]
