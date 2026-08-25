# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Public HTTP client surface for DlightRAG."""

from dlightrag.runtime.errors import AnswerRunCancelledError, AnswerRunFailedError
from dlightrag.sdk.attachments import AnswerAttachmentUpload
from dlightrag.sdk.client import (
    EVENT_READ_IDLE_SECONDS,
    MAX_RECONNECT_ATTEMPTS,
    STATUS_POLL_SECONDS,
    AnswerArtifact,
    AnswerArtifactIssue,
    AnswerPart,
    AnswerResult,
    AnswerRunClient,
    AnswerRunDescriptor,
    AnswerStreamEvent,
    ArtifactOutcome,
    EvidenceImage,
    ProfileMemoryReceipt,
    ProfileMemorySettings,
    SyncAnswerRunClient,
    parse_sse_frames,
)
from dlightrag.sdk.http import (
    CLIENT_TIMEOUT_ENV,
    DEFAULT_API_URL,
    DEFAULT_CLIENT_TIMEOUT,
    api_url,
    auth_headers,
    auth_token,
    client_timeout,
    json_headers,
)
from dlightrag.sdk.requests import query_image_blocks_from_urls

__all__ = [
    "CLIENT_TIMEOUT_ENV",
    "DEFAULT_API_URL",
    "DEFAULT_CLIENT_TIMEOUT",
    "EVENT_READ_IDLE_SECONDS",
    "MAX_RECONNECT_ATTEMPTS",
    "STATUS_POLL_SECONDS",
    "AnswerArtifact",
    "AnswerArtifactIssue",
    "AnswerAttachmentUpload",
    "AnswerPart",
    "AnswerResult",
    "AnswerRunCancelledError",
    "AnswerRunClient",
    "AnswerRunDescriptor",
    "AnswerStreamEvent",
    "AnswerRunFailedError",
    "ArtifactOutcome",
    "EvidenceImage",
    "ProfileMemoryReceipt",
    "ProfileMemorySettings",
    "SyncAnswerRunClient",
    "api_url",
    "auth_headers",
    "auth_token",
    "client_timeout",
    "json_headers",
    "parse_sse_frames",
    "query_image_blocks_from_urls",
]
