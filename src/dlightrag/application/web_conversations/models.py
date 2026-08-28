# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable Web Conversation records and persistence port."""

import base64
import binascii
import datetime
import hashlib
import hmac
import json
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID

from dlightrag.application.answer_runs.routing import RoutingAcceptance
from dlightrag.engine.runtime import AnswerRunRecord, PendingArtifact, PendingArtifactReference


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


CONVERSATION_PAGE_DEFAULT_LIMIT = 50
CONVERSATION_PAGE_MAX_LIMIT = 100
_CURSOR_MAC_BYTES = 16
_BASE64URL_CHARACTERS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
)


class ConversationCursorError(ValueError):
    """An opaque conversation page cursor is malformed or fails integrity checking."""


@dataclass(frozen=True, slots=True)
class ConversationCursor:
    """The complete ordering key for the next newest-first page."""

    updated_at: datetime.datetime
    conversation_id: UUID

    def __post_init__(self) -> None:
        if not isinstance(self.updated_at, datetime.datetime):
            raise ValueError("conversation cursor timestamp must be a datetime")
        if self.updated_at.tzinfo is None or self.updated_at.utcoffset() is None:
            raise ValueError("conversation cursor timestamp must include a timezone")
        if not isinstance(self.conversation_id, UUID):
            raise ValueError("conversation cursor id must be a UUID")


@dataclass(frozen=True, slots=True)
class ConversationPageRequest:
    """One hard-bounded application request for conversation summaries."""

    limit: int = CONVERSATION_PAGE_DEFAULT_LIMIT
    cursor: ConversationCursor | None = None

    def __post_init__(self) -> None:
        if isinstance(self.limit, bool) or not isinstance(self.limit, int):
            raise ValueError("conversation page limit must be an integer")
        if not 1 <= self.limit <= CONVERSATION_PAGE_MAX_LIMIT:
            raise ValueError(
                f"conversation page limit must be between 1 and {CONVERSATION_PAGE_MAX_LIMIT}"
            )
        if self.cursor is not None and not isinstance(self.cursor, ConversationCursor):
            raise ValueError("conversation cursor must contain paired ordering fields")


@dataclass(frozen=True, slots=True)
class ConversationRowPage:
    """Bounded persistence result, including the measured physical fetch size."""

    items: tuple[Mapping[str, Any], ...]
    has_more: bool
    fetched_rows: int


@dataclass(frozen=True, slots=True)
class ConversationPage:
    """Newest-first application page with a paired continuation key."""

    items: tuple[ConversationSummary, ...]
    next_cursor: ConversationCursor | None
    fetched_rows: int


class ConversationCursorCodec:
    """Encode paired ordering facts as an opaque, integrity-checked token.

    The cursor deliberately carries no principal. Ownership remains a mandatory
    query predicate derived from authentication on every page.
    """

    def __init__(self, secret: bytes | None = None) -> None:
        self._secret = secret or secrets.token_bytes(32)

    def encode(self, cursor: ConversationCursor) -> str:
        payload = json.dumps(
            {
                "conversation_id": str(cursor.conversation_id),
                "updated_at": _canonical_cursor_timestamp(cursor.updated_at),
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        encoded = _base64url_encode(payload)
        mac = hmac.new(self._secret, payload, hashlib.sha256).digest()[:_CURSOR_MAC_BYTES]
        return f"{encoded}.{_base64url_encode(mac)}"

    def decode(self, token: str) -> ConversationCursor:
        try:
            encoded, encoded_mac = token.split(".")
            if not encoded or not encoded_mac:
                raise ValueError
            payload = _base64url_decode(encoded)
            supplied_mac = _base64url_decode(encoded_mac)
            expected_mac = hmac.new(self._secret, payload, hashlib.sha256).digest()[
                :_CURSOR_MAC_BYTES
            ]
            if len(supplied_mac) != _CURSOR_MAC_BYTES or not hmac.compare_digest(
                supplied_mac, expected_mac
            ):
                raise ValueError
            decoded = json.loads(payload)
            if not isinstance(decoded, dict) or set(decoded) != {
                "conversation_id",
                "updated_at",
            }:
                raise ValueError
            conversation_id_text = decoded["conversation_id"]
            timestamp_text = decoded["updated_at"]
            if not isinstance(conversation_id_text, str) or not isinstance(timestamp_text, str):
                raise ValueError
            conversation_id = UUID(conversation_id_text)
            if str(conversation_id) != conversation_id_text:
                raise ValueError
            updated_at = datetime.datetime.fromisoformat(timestamp_text.replace("Z", "+00:00"))
            if _canonical_cursor_timestamp(updated_at) != timestamp_text:
                raise ValueError
            return ConversationCursor(
                updated_at=updated_at,
                conversation_id=conversation_id,
            )
        except (binascii.Error, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            raise ConversationCursorError("invalid conversation page cursor") from exc


def _canonical_cursor_timestamp(value: datetime.datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("conversation cursor timestamp must include a timezone")
    return value.astimezone(datetime.UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _base64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _base64url_decode(value: str) -> bytes:
    if not value or any(character not in _BASE64URL_CHARACTERS for character in value):
        raise ValueError("invalid base64url")
    decoded = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    if _base64url_encode(decoded) != value:
        raise ValueError("non-canonical base64url")
    return decoded


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
    async def list_conversations(
        self,
        principal_id: str,
        *,
        page: ConversationPageRequest,
    ) -> ConversationRowPage: ...
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
    "CONVERSATION_PAGE_DEFAULT_LIMIT",
    "CONVERSATION_PAGE_MAX_LIMIT",
    "ConversationCursor",
    "ConversationCursorCodec",
    "ConversationCursorError",
    "ConversationPage",
    "ConversationPageRequest",
    "ConversationRowPage",
    "ConversationSnapshot",
    "ConversationSummary",
    "ConversationSubmissionConflict",
    "LinkedTurn",
    "WebConversationSchemaError",
    "WebConversationStore",
    "WebConversationUnavailableError",
]
