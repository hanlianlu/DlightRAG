# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable Web Conversation records, persistence port, and lifecycle."""

from .models import (
    CONVERSATION_PAGE_DEFAULT_LIMIT,
    CONVERSATION_PAGE_MAX_LIMIT,
    AnswerTurnCreation,
    ConversationCursor,
    ConversationCursorCodec,
    ConversationCursorError,
    ConversationPage,
    ConversationPageRequest,
    ConversationRowPage,
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
    "CONVERSATION_PAGE_DEFAULT_LIMIT",
    "CONVERSATION_PAGE_MAX_LIMIT",
    "ConversationCursor",
    "ConversationCursorCodec",
    "ConversationCursorError",
    "ConversationPage",
    "ConversationPageRequest",
    "ConversationRowPage",
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
