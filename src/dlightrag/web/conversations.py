# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Principal-scoped service adapter for durable Web conversations."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, TypeVar

import asyncpg

from dlightrag.api.auth import UserContext
from dlightrag.api.principal import owner_id_from_user
from dlightrag.core.answer.media import answer_images_from_sources
from dlightrag.core.answer_runs.snapshots import load_answer_snapshot
from dlightrag.core.client_payloads import project_source_payloads
from dlightrag.core.resources.models import ResourceInput
from dlightrag.core.retrieval.source_links import SourceDownloadLinkBuilder
from dlightrag.storage.pool import POSTGRES_UNAVAILABLE_EXCEPTIONS
from dlightrag.storage.web_conversations import (
    CommitTurnResult,
    ConversationSnapshot,
    PendingConversationAttachment,
    PGWebConversationStore,
    StoredConversationAttachment,
)
from dlightrag.utils.images import thumbnail_bytes
from dlightrag.web.attachment_models import ValidatedWebAttachment
from dlightrag.web.conversation_models import (
    ConversationAttachmentReference,
    ConversationHistory,
    ConversationSummary,
    ConversationTurn,
)
from dlightrag.web.safe_html import safe_answer_done

logger = logging.getLogger(__name__)
T = TypeVar("T")
_COMMIT_ATTEMPT_TIMEOUT_SECONDS = 45.0
_RECONCILE_ATTEMPT_TIMEOUT_SECONDS = 10.0
_RECONCILE_ATTEMPTS = 2
_RECONCILE_BACKOFF_SECONDS = 0.25
_WEB_STORAGE_UNAVAILABLE_EXCEPTIONS = (
    *POSTGRES_UNAVAILABLE_EXCEPTIONS,
    asyncpg.InterfaceError,
)
_HISTORY_THUMBNAIL_MAX_PX = 320
_HISTORY_THUMBNAIL_MAX_BYTES = 128 * 1024
_HISTORY_THUMBNAIL_QUALITY = 82
_HISTORY_THUMBNAIL_MIN_QUALITY = 50
_HISTORY_THUMBNAIL_MIN_PX = 64
_PRUNE_INTERVAL_SECONDS = 60 * 60
_PRUNE_BATCH_SIZE = 500


def _is_image_mime(mime_type: str | None) -> bool:
    return bool(mime_type) and mime_type.lower().startswith("image/")


class WebConversationUnavailableError(RuntimeError):
    """Raised when durable Web conversation storage cannot be reached."""

    detail = "Web conversation storage is unavailable"


@dataclass(frozen=True, slots=True)
class PreparedWebConversation:
    principal_id: str
    conversation_id: str
    content_revision: int
    text_history: tuple[dict[str, Any], ...]
    committed_submission: CommitTurnResult | None = None
    # Compact manifest of prior attachments (id, ordinal, filename, mime,
    # byte_size). Registered as lazy authorized resources each answer; bytes load
    # only when the model reads/inspects one by id.
    attachment_manifest: tuple[dict[str, Any], ...] = ()


class WebConversationService:
    """Map authenticated browser operations onto the scoped persistence store."""

    def __init__(
        self,
        *,
        store: PGWebConversationStore,
        max_turns: int,
        ttl_days: int,
        max_attachments: int,
        validate_schema_only: bool = False,
    ) -> None:
        self._store = store
        self._max_turns = max_turns
        self._ttl_days = ttl_days
        self._max_attachments = max_attachments
        self._validate_schema_only = validate_schema_only
        self._prune_task: asyncio.Task[None] | None = None

    async def initialize(self) -> None:
        """Establish the schema and start bounded global retention.

        Readers write conversations but own no schema, so they validate the
        migrated schema instead of applying it.
        """
        await self._store.initialize(validate_only=self._validate_schema_only)
        await self._prune_expired_batch()
        if self._prune_task is None:
            self._prune_task = asyncio.create_task(self._prune_expired_loop())

    async def _prune_expired_batch(self) -> int:
        try:
            return await self._store.prune_expired(
                ttl_days=self._ttl_days,
                batch_size=_PRUNE_BATCH_SIZE,
            )
        except Exception:
            logger.exception("Failed to prune expired Web conversations")
            return 0

    async def _prune_expired_loop(self) -> None:
        while True:
            await asyncio.sleep(_PRUNE_INTERVAL_SECONDS)
            while await self._prune_expired_batch() >= _PRUNE_BATCH_SIZE:
                await asyncio.sleep(0)

    async def aclose(self) -> None:
        """Stop periodic retention. Safe to call more than once."""
        task = self._prune_task
        self._prune_task = None
        if task is None:
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    async def create(self, user: UserContext | None) -> ConversationSummary:
        principal_id = owner_id_from_user(user)
        row = await self._store_call(self._store.create_conversation(principal_id))
        return _conversation_summary(row)

    async def list(self, user: UserContext | None) -> list[ConversationSummary]:
        principal_id = owner_id_from_user(user)
        rows = await self._store_call(
            self._store.list_conversations(
                principal_id,
                ttl_days=self._ttl_days,
            )
        )
        return [_conversation_summary(row) for row in rows]

    async def history(
        self,
        user: UserContext | None,
        conversation_id: str,
        *,
        downloadable_workspaces: set[str] | None = None,
        visual_workspaces: set[str] | None = None,
    ) -> ConversationHistory | None:
        principal_id = owner_id_from_user(user)
        snapshot = await self._snapshot(principal_id, conversation_id)
        if snapshot is None:
            return None

        return ConversationHistory(
            conversation=_snapshot_summary(snapshot),
            turns=[
                _conversation_turn(
                    conversation_id,
                    row,
                    downloadable_workspaces=downloadable_workspaces,
                    visual_workspaces=visual_workspaces,
                )
                for row in snapshot.history
            ],
        )

    async def rename(
        self,
        user: UserContext | None,
        conversation_id: str,
        title: str,
    ) -> ConversationSummary | None:
        principal_id = owner_id_from_user(user)
        row = await self._store_call(
            self._store.rename_conversation(
                principal_id,
                conversation_id,
                title=title,
                ttl_days=self._ttl_days,
            )
        )
        return _conversation_summary(row) if row is not None else None

    async def delete(
        self,
        user: UserContext | None,
        conversation_id: str,
    ) -> bool:
        principal_id = owner_id_from_user(user)
        return await self._store_call(
            self._store.delete_conversation(
                principal_id,
                conversation_id,
                ttl_days=self._ttl_days,
            )
        )

    async def delete_all(self, user: UserContext | None) -> int:
        principal_id = owner_id_from_user(user)
        return await self._store_call(self._store.delete_all_conversations(principal_id))

    async def prepare_answer(
        self,
        user: UserContext | None,
        conversation_id: str,
        submission_id: str | None = None,
    ) -> PreparedWebConversation | None:
        principal_id = owner_id_from_user(user)
        snapshot = await self._snapshot(principal_id, conversation_id)
        if snapshot is None:
            return None
        text_history: list[dict[str, Any]] = []
        attachment_manifest: list[dict[str, Any]] = []
        for turn in snapshot.history:
            text_history.extend(
                (
                    {"role": "user", "content": str(turn["user_text"])},
                    {"role": "assistant", "content": str(turn["assistant_text"])},
                )
            )
            for attachment in turn.get("attachments") or []:
                attachment_manifest.append(
                    {
                        "attachment_id": str(attachment["attachment_id"]),
                        "turn_number": turn.get("turn_number"),
                        "ordinal": attachment.get("ordinal"),
                        "filename": attachment.get("filename"),
                        "mime_type": attachment.get("mime_type"),
                        "byte_size": attachment.get("byte_size"),
                    }
                )
        committed_submission = None
        if submission_id is not None:
            try:
                async with asyncio.timeout(_RECONCILE_ATTEMPT_TIMEOUT_SECONDS):
                    committed_submission = await self._store.find_committed_turn(
                        principal_id,
                        conversation_id,
                        submission_id,
                        ttl_days=self._ttl_days,
                        retry=False,
                    )
            except _WEB_STORAGE_UNAVAILABLE_EXCEPTIONS as exc:
                raise WebConversationUnavailableError from exc
        return PreparedWebConversation(
            principal_id=principal_id,
            conversation_id=snapshot.conversation_id,
            content_revision=snapshot.content_revision,
            text_history=tuple(text_history),
            committed_submission=committed_submission,
            attachment_manifest=tuple(attachment_manifest),
        )

    def build_answer_resources(
        self,
        prepared: PreparedWebConversation,
        current_attachments: tuple[ValidatedWebAttachment, ...],
    ) -> list[ResourceInput]:
        """Build the ordered request resources for one answer.

        Current-turn attachments carry inline bytes (the manager extracts verified
        images into current-image blocks and registers documents). Prior
        attachments are compact manifest entries registered as lazy authorized
        resources whose bytes load only when the model reads/inspects them.
        """
        resources: list[ResourceInput] = [
            ResourceInput(
                filename=attachment.filename,
                content=attachment.attachment_bytes,
                declared_mime=attachment.mime_type,
            )
            for attachment in current_attachments
        ]
        remaining = max(0, self._max_attachments - len(resources))
        history = list(prepared.attachment_manifest)[-remaining:] if remaining else []
        for entry in history:
            resources.append(
                ResourceInput(
                    filename=entry.get("filename"),
                    declared_mime=entry.get("mime_type"),
                    loader=self._history_loader(
                        prepared.principal_id,
                        prepared.conversation_id,
                        str(entry["attachment_id"]),
                    ),
                )
            )
        return resources

    def _history_loader(
        self,
        principal_id: str,
        conversation_id: str,
        attachment_id: str,
    ) -> Callable[[], Awaitable[bytes]]:
        async def _load() -> bytes:
            stored = await self._store.get_attachment(
                principal_id,
                conversation_id,
                attachment_id,
                ttl_days=self._ttl_days,
            )
            if stored is None:
                raise FileNotFoundError(f"attachment {attachment_id} is unavailable")
            return stored.attachment_bytes

        return _load

    async def attachment(
        self,
        user: UserContext | None,
        conversation_id: str,
        attachment_id: str,
    ) -> StoredConversationAttachment | None:
        principal_id = owner_id_from_user(user)
        return await self._store_call(
            self._store.get_attachment(
                principal_id,
                conversation_id,
                attachment_id,
                ttl_days=self._ttl_days,
            )
        )

    async def thumbnail(
        self,
        user: UserContext | None,
        conversation_id: str,
        attachment_id: str,
    ) -> tuple[bytes, str] | None:
        """Derive one bounded UI thumbnail for an image attachment."""
        stored = await self.attachment(user, conversation_id, attachment_id)
        if stored is None or not _is_image_mime(stored.mime_type):
            return None
        try:
            payload, mime_type = await asyncio.to_thread(
                thumbnail_bytes,
                stored.attachment_bytes,
                max_px=_HISTORY_THUMBNAIL_MAX_PX,
                max_bytes=_HISTORY_THUMBNAIL_MAX_BYTES,
                quality=_HISTORY_THUMBNAIL_QUALITY,
                min_quality=_HISTORY_THUMBNAIL_MIN_QUALITY,
                min_px=_HISTORY_THUMBNAIL_MIN_PX,
            )
        except Exception:
            logger.warning("Failed to derive Web conversation thumbnail", exc_info=True)
            return None
        return payload, mime_type

    async def commit_answer(
        self,
        prepared: PreparedWebConversation,
        *,
        submission_id: str,
        user_text: str,
        assistant_text: str,
        answer_sources: dict[str, Any],
        queried_workspaces: list[str],
        attachments: tuple[ValidatedWebAttachment, ...] = (),
    ) -> CommitTurnResult:
        """Atomically append a completed answer against its captured revision."""
        pending_attachments = [
            PendingConversationAttachment(
                attachment_id=attachment.attachment_id,
                ordinal=attachment.ordinal,
                filename=attachment.filename,
                mime_type=attachment.mime_type,
                suffix=attachment.suffix,
                attachment_bytes=attachment.attachment_bytes,
                content_sha256=attachment.content_sha256,
            )
            for attachment in attachments
        ]
        try:
            async with asyncio.timeout(_COMMIT_ATTEMPT_TIMEOUT_SECONDS):
                return await self._store.commit_turn(
                    principal_id=prepared.principal_id,
                    conversation_id=prepared.conversation_id,
                    submission_id=submission_id,
                    expected_revision=prepared.content_revision,
                    user_text=user_text,
                    assistant_text=assistant_text,
                    answer_sources=answer_sources,
                    queried_workspaces=queried_workspaces,
                    attachments=pending_attachments,
                    max_turns=self._max_turns,
                    ttl_days=self._ttl_days,
                )
        except _WEB_STORAGE_UNAVAILABLE_EXCEPTIONS:
            return await self._reconcile_commit(prepared, submission_id)

    async def update_answer_highlights(
        self,
        prepared: PreparedWebConversation,
        *,
        submission_id: str,
        answer_sources: dict[str, Any],
    ) -> None:
        """Persist semantic highlights into a committed turn's stored sources.

        Best-effort: highlights are a display enhancement computed after the
        turn is committed, so a failure here must never affect the answer.
        """
        try:
            async with asyncio.timeout(_COMMIT_ATTEMPT_TIMEOUT_SECONDS):
                await self._store.update_turn_sources(
                    principal_id=prepared.principal_id,
                    conversation_id=prepared.conversation_id,
                    submission_id=submission_id,
                    answer_sources=answer_sources,
                )
        except Exception:
            logger.warning("Failed to persist semantic highlights", exc_info=True)

    async def _reconcile_commit(
        self,
        prepared: PreparedWebConversation,
        submission_id: str,
    ) -> CommitTurnResult:
        """Resolve an ambiguous mutation through a short, one-shot lookup budget."""
        for attempt in range(_RECONCILE_ATTEMPTS):
            try:
                async with asyncio.timeout(_RECONCILE_ATTEMPT_TIMEOUT_SECONDS):
                    committed = await self._store.find_committed_turn(
                        prepared.principal_id,
                        prepared.conversation_id,
                        submission_id,
                        ttl_days=self._ttl_days,
                        retry=False,
                    )
            except _WEB_STORAGE_UNAVAILABLE_EXCEPTIONS:
                if attempt + 1 < _RECONCILE_ATTEMPTS:
                    await asyncio.sleep(_RECONCILE_BACKOFF_SECONDS)
                continue
            if committed is not None:
                return committed
            return CommitTurnResult(False, "commit_not_found", None, None)
        return CommitTurnResult(False, "commit_outcome_unknown", None, None)

    async def _snapshot(
        self,
        principal_id: str,
        conversation_id: str,
    ) -> ConversationSnapshot | None:
        return await self._store_call(
            self._store.snapshot(
                principal_id,
                conversation_id,
                ttl_days=self._ttl_days,
                max_turns=self._max_turns,
            )
        )

    async def _store_call(self, operation: Awaitable[T]) -> T:
        try:
            return await operation
        except _WEB_STORAGE_UNAVAILABLE_EXCEPTIONS as exc:
            raise WebConversationUnavailableError from exc


def _conversation_summary(row: dict[str, Any]) -> ConversationSummary:
    return ConversationSummary(
        conversation_id=str(row["conversation_id"]),
        title=row.get("title"),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _snapshot_summary(snapshot: ConversationSnapshot) -> ConversationSummary:
    return ConversationSummary(
        conversation_id=snapshot.conversation_id,
        title=snapshot.title,
        created_at=snapshot.created_at,
        updated_at=snapshot.updated_at,
    )


def _conversation_turn(
    conversation_id: str,
    row: dict[str, Any],
    *,
    downloadable_workspaces: set[str] | None = None,
    visual_workspaces: set[str] | None = None,
) -> ConversationTurn:
    turn_number = int(row["turn_number"])
    attachments = [
        _attachment_reference(conversation_id, turn_number, attachment)
        for attachment in row.get("attachments", [])
    ]
    internal_sources = load_answer_snapshot(row.get("answer_sources") or {"sources": []})
    sources = project_source_payloads(
        internal_sources,
        resolver=SourceDownloadLinkBuilder(base_url="/web/files/raw"),
        downloadable_workspaces=downloadable_workspaces,
        visual_workspaces=visual_workspaces,
    )
    answer_images = answer_images_from_sources(
        internal_sources,
        visual_workspaces=visual_workspaces,
    )
    assistant_text = str(row["assistant_text"])
    return ConversationTurn(
        turn_id=str(row["turn_id"]),
        turn_number=turn_number,
        user_text=str(row["user_text"]),
        assistant_text=assistant_text,
        user_attachments=attachments,
        answer_html=safe_answer_done(
            answer=assistant_text,
            sources=sources,
            answer_images=answer_images,
        ),
        created_at=row["created_at"],
    )


def _attachment_reference(
    conversation_id: str,
    turn_number: int,
    attachment: dict[str, Any],
) -> ConversationAttachmentReference:
    attachment_id = str(attachment["attachment_id"])
    ordinal = int(attachment["ordinal"])
    mime_type = str(attachment["mime_type"])
    is_image = _is_image_mime(mime_type)
    url = f"/web/conversations/{conversation_id}/attachments/{attachment_id}"
    return ConversationAttachmentReference(
        attachment_id=attachment_id,
        ordinal=ordinal,
        kind="image" if is_image else "document",
        filename=str(attachment["filename"]),
        mime_type=mime_type,
        byte_size=int(attachment["byte_size"]),
        url=url,
        thumbnail_url=(url + "/thumbnail") if is_image else None,
        label=f"Turn {turn_number}, attachment {ordinal}",
    )


__all__ = [
    "PreparedWebConversation",
    "WebConversationService",
    "WebConversationUnavailableError",
]
