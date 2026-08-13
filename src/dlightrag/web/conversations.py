# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Principal-scoped service adapter for durable Web conversations.

A conversation owns navigation and history only. One browser submission becomes
one durable Answer run plus the conversation entry that links to it, committed
together before the 202 response. Every read projects a turn from the run's
authoritative state, so no answer text, source snapshot, or uploaded byte is
stored a second time under the conversation.
"""

import asyncio
import logging
from collections.abc import Awaitable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, TypeVar

import asyncpg

from dlightrag.api.auth import UserContext
from dlightrag.api.principal import owner_id_from_user
from dlightrag.core.answer_runs.execution import AnswerRunInput, AttachmentReference
from dlightrag.core.answer_runs.results import project_answer_result
from dlightrag.core.retrieval.source_links import SourceDownloadLinkBuilder
from dlightrag.storage.answer_runs import (
    AnswerRunRecord,
    PendingArtifact,
    PendingArtifactReference,
    PGAnswerRunStore,
    artifact_digest,
    parse_run_id,
)
from dlightrag.storage.pool import POSTGRES_UNAVAILABLE_EXCEPTIONS
from dlightrag.storage.web_conversations import (
    AnswerTurnCreation,
    ConversationSnapshot,
    ConversationSubmissionConflict,
    LinkedTurn,
    PGWebConversationStore,
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

#: Browser answer sources are served through the Web-scoped download route.
WEB_SOURCE_DOWNLOAD_BASE = "/web/files/raw"


def _is_image_mime(mime_type: str | None) -> bool:
    return bool(mime_type) and mime_type.lower().startswith("image/")


class WebConversationUnavailableError(RuntimeError):
    """Raised when durable Web conversation storage cannot be reached."""

    detail = "Web conversation storage is unavailable"


@dataclass(frozen=True, slots=True)
class WebAnswerSubmission:
    """The run and conversation entry one accepted browser submission created."""

    run: AnswerRunRecord
    turn_id: str
    turn_number: int
    conversation: ConversationSummary
    replayed: bool


class WebConversationService:
    """Map authenticated browser operations onto the scoped persistence store."""

    def __init__(
        self,
        *,
        store: PGWebConversationStore,
        run_store: PGAnswerRunStore | None = None,
        max_turns: int,
        ttl_days: int,
        max_attachments: int,
        validate_schema_only: bool = False,
    ) -> None:
        self._store = store
        self._run_store = run_store or PGAnswerRunStore()
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

    # ------------------------------------------------------------------
    # Conversation lifecycle
    # ------------------------------------------------------------------

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

    async def delete(self, user: UserContext | None, conversation_id: str) -> bool:
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

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    async def history(
        self,
        user: UserContext | None,
        conversation_id: str,
        *,
        downloadable_workspaces: set[str] | None = None,
        visual_workspaces: set[str] | None = None,
    ) -> ConversationHistory | None:
        """Project every linked turn from its run's authoritative state."""
        principal_id = owner_id_from_user(user)
        snapshot = await self._snapshot(principal_id, conversation_id)
        if snapshot is None:
            return None
        return ConversationHistory(
            conversation=_snapshot_summary(snapshot),
            turns=[
                project_conversation_turn(
                    turn,
                    downloadable_workspaces=downloadable_workspaces,
                    visual_workspaces=visual_workspaces,
                )
                for turn in snapshot.turns
            ],
        )

    async def turn_for_run(self, user: UserContext | None, run_id: str) -> LinkedTurn | None:
        """Return the conversation entry an owned run belongs to, if any.

        An unparseable identifier is simply unknown here: every run route reads
        through this projection, so a malformed id is the same opaque miss as a
        pruned or foreign one and never reaches storage.
        """
        principal_id = owner_id_from_user(user)
        if parse_run_id(run_id) is None:
            return None
        return await self._store_call(self._store.find_turn_by_run(principal_id, run_id))

    async def attachment(
        self,
        user: UserContext | None,
        run_id: str,
        ordinal: int,
    ) -> tuple[AttachmentReference, bytes] | None:
        """Load one owned run's uploaded attachment bytes by its ordinal."""
        principal_id = owner_id_from_user(user)
        turn = await self.turn_for_run(user, run_id)
        if turn is None:
            return None
        reference = next(
            (
                item
                for item in AnswerRunInput.from_request(turn.run.request).attachments
                if item.ordinal == ordinal
            ),
            None,
        )
        if reference is None:
            return None
        content = await self._store_call(
            self._run_store.load_artifact(owner_id=principal_id, digest=reference.digest)
        )
        return None if content is None else (reference, content)

    async def thumbnail(
        self,
        user: UserContext | None,
        run_id: str,
        ordinal: int,
    ) -> tuple[bytes, str] | None:
        """Derive one bounded UI thumbnail for an image attachment."""
        stored = await self.attachment(user, run_id, ordinal)
        if stored is None or not _is_image_mime(stored[0].mime_type):
            return None
        try:
            payload, mime_type = await asyncio.to_thread(
                thumbnail_bytes,
                stored[1],
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

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    async def start_answer(
        self,
        user: UserContext | None,
        *,
        conversation_id: str,
        submission_id: str,
        query: str,
        workspaces: Sequence[str],
        attachments: Sequence[ValidatedWebAttachment] = (),
    ) -> WebAnswerSubmission | None:
        """Create or replay one submission's run and its conversation entry.

        The conversation is validated and locked, and the run, its uploaded
        bytes, and the linked turn are written in one transaction, so the
        descriptor returned here is already durable history.
        """
        principal_id = owner_id_from_user(user)
        snapshot = await self._snapshot(principal_id, conversation_id)
        if snapshot is None:
            return None
        run_input = _answer_run_input(
            query=query,
            workspaces=workspaces,
            snapshot=snapshot,
            attachments=attachments,
            max_attachments=self._max_attachments,
        )
        creation = await self._store_call(
            self._store.create_answer_turn(
                principal_id=principal_id,
                conversation_id=conversation_id,
                submission_id=submission_id,
                request=run_input.as_request(),
                artifacts=[
                    PendingArtifact(content=attachment.attachment_bytes)
                    for attachment in attachments
                ],
                references=_artifact_references(run_input),
                title_hint=_auto_title(query),
                max_turns=self._max_turns,
                ttl_days=self._ttl_days,
            )
        )
        return None if creation is None else _submission(creation)

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


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def _auto_title(query: str) -> str | None:
    return " ".join(query.split())[:120] or None


def _submission(creation: AnswerTurnCreation) -> WebAnswerSubmission:
    return WebAnswerSubmission(
        run=creation.turn.run,
        turn_id=creation.turn.turn_id,
        turn_number=creation.turn.turn_number,
        conversation=_conversation_summary(creation.summary),
        replayed=creation.replayed,
    )


def _artifact_references(run_input: AnswerRunInput) -> list[PendingArtifactReference]:
    """Order this run's current and carried-forward attachment references."""
    references = [
        PendingArtifactReference(
            resource_id=attachment.resource_id,
            reference_kind="current_attachment",
            ordinal=attachment.ordinal,
            digest=attachment.digest,
            filename=attachment.filename,
            mime_type=attachment.mime_type,
        )
        for attachment in run_input.attachments
    ]
    references.extend(
        PendingArtifactReference(
            resource_id=attachment.history_resource_id,
            reference_kind="history_attachment",
            ordinal=attachment.ordinal,
            digest=attachment.digest,
            filename=attachment.filename,
            mime_type=attachment.mime_type,
        )
        for attachment in run_input.history_attachments
    )
    return references


def _answer_run_input(
    *,
    query: str,
    workspaces: Sequence[str],
    snapshot: ConversationSnapshot,
    attachments: Sequence[ValidatedWebAttachment],
    max_attachments: int,
) -> AnswerRunInput:
    """Normalize one browser submission into the run's immutable input.

    Only succeeded turns become model history, and only their uploads stay
    readable as prior resources: a pending, failed, or cancelled turn remains
    visible in the browser but is never replayed to the model.
    """
    history: list[dict[str, Any]] = []
    prior: list[AttachmentReference] = []
    for turn in snapshot.turns:
        if turn.run.status != "succeeded":
            continue
        request = AnswerRunInput.from_request(turn.run.request)
        history.extend(
            (
                {"role": "user", "content": request.query},
                {"role": "assistant", "content": str((turn.run.result or {}).get("answer") or "")},
            )
        )
        prior.extend(request.attachments)
    remaining = max(0, max_attachments - len(attachments))
    carried = prior[-remaining:] if remaining else []
    return AnswerRunInput(
        query=query,
        workspaces=tuple(workspaces),
        history=tuple(history),
        semantic_highlights=True,
        attachments=tuple(
            AttachmentReference(
                digest=artifact_digest(attachment.attachment_bytes),
                filename=attachment.filename,
                mime_type=attachment.mime_type,
                ordinal=attachment.ordinal,
                byte_size=attachment.byte_size,
            )
            for attachment in attachments
        ),
        history_attachments=tuple(
            AttachmentReference(
                digest=item.digest,
                filename=item.filename,
                mime_type=item.mime_type,
                ordinal=ordinal,
                byte_size=item.byte_size,
            )
            for ordinal, item in enumerate(carried)
        ),
    )


def project_conversation_turn(
    turn: LinkedTurn,
    *,
    downloadable_workspaces: set[str] | None = None,
    visual_workspaces: set[str] | None = None,
) -> ConversationTurn:
    """Render one linked turn from the authoritative state of its run.

    A queued or running turn is a pending entry carrying its run id and
    cancellation state, so a reloaded browser resubscribes without remembering
    the original 202 response. A failed or cancelled turn stays visible with its
    public terminal state until its run prunes. Only a succeeded turn carries an
    answer, and that answer is projected from the run's canonical result.
    """
    run = turn.run
    request = AnswerRunInput.from_request(run.request)
    answer = ""
    answer_html = ""
    if run.status == "succeeded" and run.result is not None:
        projected = project_answer_result(
            run.result,
            source_link_builder=SourceDownloadLinkBuilder(base_url=WEB_SOURCE_DOWNLOAD_BASE),
            downloadable_workspaces=downloadable_workspaces,
            visual_workspaces=visual_workspaces,
        )
        answer = str(projected["answer"])
        answer_html = safe_answer_done(
            answer=answer,
            sources=projected["sources"],
            answer_images=projected["answer_images"],
        )
    return ConversationTurn(
        turn_id=turn.turn_id,
        turn_number=turn.turn_number,
        answer_run_id=run.run_id,
        submission_id=turn.submission_id,
        status=run.status,
        cancel_requested=run.cancel_requested,
        user_text=request.query,
        assistant_text=answer,
        user_attachments=[
            _attachment_reference(run.run_id, turn.turn_number, attachment)
            for attachment in request.attachments
        ],
        answer_html=answer_html,
        error_kind=run.error_kind,
        error_message=run.error_message,
        created_at=turn.created_at,
    )


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


def _attachment_reference(
    run_id: str,
    turn_number: int,
    attachment: AttachmentReference,
) -> ConversationAttachmentReference:
    is_image = _is_image_mime(attachment.mime_type)
    url = f"/web/runs/{run_id}/attachments/{attachment.ordinal}"
    return ConversationAttachmentReference(
        attachment_id=f"{run_id}:{attachment.ordinal}",
        ordinal=attachment.ordinal,
        kind="image" if is_image else "document",
        filename=attachment.filename,
        mime_type=attachment.mime_type,
        byte_size=attachment.byte_size,
        url=url,
        thumbnail_url=(url + "/thumbnail") if is_image else None,
        label=f"Turn {turn_number}, attachment {attachment.ordinal}",
    )


__all__ = [
    "WEB_SOURCE_DOWNLOAD_BASE",
    "ConversationSubmissionConflict",
    "WebAnswerSubmission",
    "WebConversationService",
    "WebConversationUnavailableError",
    "project_conversation_turn",
]
