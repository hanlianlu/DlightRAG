# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Canonical transport-neutral client payload contracts.

These contracts are the one client-facing request vocabulary shared by the
REST, Web, and MCP transports. They stay transport-neutral: no HTTP, browser,
or MCP types may leak into these models, so every adapter projects them into
its own surface instead of diverging.

Independent requests re-send any caller-managed conversation history they need.
An accepted Answer run pins its bounded history for recovery and server-owned
continuation. The message ceiling bounds request size (~50 prior turns); the
planner independently truncates to the configured token budget before prompting
the model.
"""

from collections.abc import Sequence
from typing import Any, Literal, cast
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dlightrag.engine.ai.capacity import ModelProfile
from dlightrag.engine.answer.mode import AnswerMode
from dlightrag.engine.answer.resources.images import MAX_QUERY_IMAGES

#: The agent efforts a caller may choose, ordered from least to most.
AnswerEffort = Literal["low", "high", "max"]
ANSWER_EFFORT_LEVELS: tuple[AnswerEffort, ...] = ("low", "high", "max")


def offered_answer_efforts(profile: ModelProfile) -> tuple[AnswerEffort, ...]:
    """Return the named agent efforts one answering profile can express.

    An uncatalogued endpoint resolves to a best-effort reasoning profile that maps
    every level, so it offers all three and the provider stays the judge. A
    catalogued profile offers only the levels whose provider value it names: a model
    whose ladder stops below a level never advertises it, so the control cannot
    promise something reasoning resolution would have to clamp.

    An empty tuple means the model states no effort at all, and no caller may offer
    a picker or honor a choice.
    """
    levels = None if profile.reasoning is None else profile.reasoning.levels
    if levels is None:
        return ()
    return tuple(level for level in ANSWER_EFFORT_LEVELS if levels.value(level) is not None)


def normalize_answer_effort(value: Any) -> AnswerEffort | None:
    """Return one accepted agent effort, or raise for a value outside the three.

    One entry point owns the three levels so the browser control, every public
    transport, and the engine's pinned input can never drift apart.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("effort is not a supported level")
    text = value.strip().lower()
    if not text:
        return None
    if text not in ANSWER_EFFORT_LEVELS:
        raise ValueError("effort is not a supported level")
    return cast(AnswerEffort, text)


MAX_HISTORY_MESSAGES = 100
MAX_HISTORY_CONTENT_CHARS = 16000
MAX_BM25_QUERY_CHARS = 1024


class ClientContractModel(BaseModel):
    """Base model for public client contracts."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


class ConversationMessage(ClientContractModel):
    """One prior, caller-supplied conversation message (stateless).

    Callers own conversation persistence, so prior turns are re-sent on each
    answer request and never stored. Historical files are not accepted here;
    re-send them as current ``attachments`` when a follow-up depends on them.
    """

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_HISTORY_CONTENT_CHARS)


def conversation_history_as_dicts(
    history: Sequence[ConversationMessage] | None,
) -> list[dict[str, Any]] | None:
    """Project caller history messages to the engine's message-dict shape.

    Returns ``None`` for empty history so callers pass the stateless-default
    straight through to the planner and answer engine.
    """
    if not history:
        return None
    return [{"role": message.role, "content": message.content} for message in history]


class ImageURL(ClientContractModel):
    url: str
    detail: Literal["auto", "low", "high"] | None = None


class QueryImage(ClientContractModel):
    type: Literal["image_url"]
    image_url: ImageURL


class AnswerAttachmentLink(ClientContractModel):
    """Public HTTP(S) reference to an answer attachment resolved on explicit read.

    Discovered links are inert handles; full scheme/credential/host validation is
    repeated when the resource is actually read. Only ``http`` and ``https`` are
    admitted and embedded credentials are rejected. Identity is not rewritten.
    """

    url: str
    filename: str | None = None

    @field_validator("url")
    @classmethod
    def _validate_http_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("attachment url must use HTTP or HTTPS")
        if not parsed.hostname:
            raise ValueError("attachment url must include a host")
        if parsed.username or parsed.password:
            raise ValueError("attachment url must not include credentials")
        return value


class QueryRequestContract(ClientContractModel):
    """Shared transport-neutral fields for client query requests."""

    query: str
    top_k: int | None = Field(default=None, ge=1)
    chunk_top_k: int | None = Field(default=None, ge=1)
    federated_rerank: bool = False


class RetrieveRequestContract(QueryRequestContract):
    """Shared transport-neutral contract for retrieve requests."""

    bm25_query: str | None = Field(default=None, max_length=MAX_BM25_QUERY_CHARS)
    query_images: list[QueryImage] | None = Field(default=None, max_length=MAX_QUERY_IMAGES)


class AnswerRequestContract(QueryRequestContract):
    """Shared transport-neutral contract for answer requests.

    Answer inputs never accept ``query_images``; user files and HTTPS references
    arrive as ``attachments`` and become request-local resources read on demand.
    """

    attachments: list[AnswerAttachmentLink] | None = None
    semantic_highlights: bool = False
    history: list[ConversationMessage] | None = Field(default=None, max_length=MAX_HISTORY_MESSAGES)
    mode: AnswerMode | None = None
    effort: AnswerEffort | None = None


def model_dump_json_safe(value: Any) -> Any:
    """Return plain JSON-ready data from Pydantic models and containers."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, list):
        return [model_dump_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [model_dump_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): model_dump_json_safe(item) for key, item in value.items()}
    return value


__all__ = [
    "ANSWER_EFFORT_LEVELS",
    "offered_answer_efforts",
    "AnswerEffort",
    "normalize_answer_effort",
    "ClientContractModel",
    "ConversationMessage",
    "AnswerAttachmentLink",
    "AnswerRequestContract",
    "ImageURL",
    "MAX_BM25_QUERY_CHARS",
    "MAX_QUERY_IMAGES",
    "MAX_HISTORY_CONTENT_CHARS",
    "MAX_HISTORY_MESSAGES",
    "QueryImage",
    "QueryRequestContract",
    "RetrieveRequestContract",
    "conversation_history_as_dicts",
    "model_dump_json_safe",
]
