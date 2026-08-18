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
from collections.abc import Awaitable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from dlightrag_ai.media import thumbnail_bytes

from dlightrag.access import UserContext, owner_id_from_user
from dlightrag.answer.resources.models import ResourceInput
from dlightrag.answer.runs.execution import (
    AnswerRunRequest,
    AttachmentReference,
)
from dlightrag.answer.runs.results import project_answer_result
from dlightrag.answer.sources import SourceDownloadLinkBuilder
from dlightrag.runtime import (
    AnswerRunRecord,
    PendingArtifact,
    PendingArtifactReference,
    answer_run_request_fingerprint,
    parse_run_id,
)
from dlightrag.services.answers import (
    AnswerHistoryResource,
    AnswerInputArtifact,
    AnswerRequest,
    AnswerRunAcceptor,
    AnswerService,
)
from dlightrag.web.attachment_models import ValidatedWebAttachment
from dlightrag.web.conversation_models import (
    AnswerTurnCreation,
    ConversationAttachmentReference,
    ConversationHistory,
    ConversationSnapshot,
    ConversationSubmissionConflict,
    ConversationSummary,
    ConversationTurn,
    LinkedTurn,
    WebConversationUnavailableError,
)
from dlightrag.web.safe_html import safe_answer_done

logger = logging.getLogger(__name__)
T = TypeVar("T")


class WebConversationStore(Protocol):
    async def prune_expired(self, *, ttl_days: int, batch_size: int = 500) -> int: ...

    async def create_conversation(self, principal_id: str) -> dict[str, Any]: ...

    async def list_conversations(
        self, principal_id: str, *, ttl_days: int
    ) -> list[dict[str, Any]]: ...

    async def rename_conversation(
        self,
        principal_id: str,
        conversation_id: str,
        *,
        title: str,
        ttl_days: int,
    ) -> dict[str, Any] | None: ...

    async def delete_conversation(
        self,
        principal_id: str,
        conversation_id: str,
        *,
        ttl_days: int,
    ) -> bool: ...

    async def delete_all_conversations(self, principal_id: str) -> int: ...

    async def snapshot(
        self,
        principal_id: str,
        conversation_id: str,
        *,
        ttl_days: int,
        max_turns: int = 100,
    ) -> ConversationSnapshot | None: ...

    async def find_turn_by_run(self, principal_id: str, run_id: str) -> LinkedTurn | None: ...

    async def replay_answer_turn(
        self,
        *,
        principal_id: str,
        conversation_id: str,
        submission_id: str,
        idempotency_fingerprint: str,
    ) -> LinkedTurn | None: ...

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
        max_turns: int,
        ttl_days: int,
    ) -> AnswerTurnCreation | None: ...


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


@dataclass(frozen=True, slots=True)
class WebAnswerSubmission:
    """The run and conversation entry one accepted browser submission created."""

    run: AnswerRunRecord
    turn_id: str
    turn_number: int
    conversation: ConversationSummary


@dataclass(frozen=True, slots=True)
class _PreparedSubmission:
    """One browser submission projected into the Answer service contract."""

    request: AnswerRequest


@dataclass(frozen=True, slots=True)
class _WebAnswerAcceptor(AnswerRunAcceptor[WebAnswerSubmission]):
    """Atomically link AnswerService acceptance to one browser conversation."""

    store: WebConversationStore
    snapshot: ConversationSnapshot
    conversation_id: str
    title_hint: str | None
    max_turns: int
    ttl_days: int

    async def replay_run(
        self,
        *,
        owner_id: str,
        idempotency_key: str,
        idempotency_fingerprint: str,
    ) -> WebAnswerSubmission | None:
        turn = await self.store.replay_answer_turn(
            principal_id=owner_id,
            conversation_id=self.conversation_id,
            submission_id=idempotency_key,
            idempotency_fingerprint=idempotency_fingerprint,
        )
        if turn is None:
            return None
        return WebAnswerSubmission(
            run=turn.run,
            turn_id=turn.turn_id,
            turn_number=turn.turn_number,
            conversation=_snapshot_summary(self.snapshot),
        )

    async def create_run(
        self,
        *,
        owner_id: str,
        prepared_input: Mapping[str, Any],
        idempotency_fingerprint: str,
        idempotency_key: str | None = None,
        resources: Sequence[Mapping[str, Any]] = (),
        artifacts: Sequence[PendingArtifact] = (),
        references: Sequence[PendingArtifactReference] = (),
    ) -> WebAnswerSubmission | None:
        if idempotency_key is None:
            raise ValueError("Web Answer acceptance requires a submission id")
        creation = await self.store.create_answer_turn(
            principal_id=owner_id,
            conversation_id=self.conversation_id,
            submission_id=idempotency_key,
            request=prepared_input,
            idempotency_fingerprint=idempotency_fingerprint,
            artifacts=artifacts,
            references=references,
            title_hint=self.title_hint,
            max_turns=self.max_turns,
            ttl_days=self.ttl_days,
        )
        return None if creation is None else _submission(creation)


class WebConversationService:
    """Map authenticated browser operations onto the scoped persistence store."""

    def __init__(
        self,
        *,
        store: WebConversationStore,
        answers: AnswerService,
        max_turns: int,
        ttl_days: int,
        max_attachments: int,
    ) -> None:
        self._store = store
        self._answers = answers
        self._max_turns = max_turns
        self._ttl_days = ttl_days
        self._max_attachments = max_attachments
        self._prune_task: asyncio.Task[None] | None = None

    async def start_retention(self) -> None:
        """Start retention after composition has already established the schema."""
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
    ) -> AnswerInputArtifact | None:
        """Load one owned run's uploaded attachment bytes by its ordinal."""
        principal_id = owner_id_from_user(user)
        turn = await self.turn_for_run(user, run_id)
        if turn is None:
            return None
        return await self._store_call(
            self._answers.read_input_artifact(
                owner_id=principal_id,
                run_id=run_id,
                ordinal=ordinal,
            )
        )

    async def thumbnail(
        self,
        user: UserContext | None,
        run_id: str,
        ordinal: int,
    ) -> tuple[bytes, str] | None:
        """Derive one bounded UI thumbnail for an image attachment."""
        stored = await self.attachment(user, run_id, ordinal)
        if stored is None or not _is_image_mime(stored.mime_type):
            return None
        try:
            payload, mime_type = await asyncio.to_thread(
                thumbnail_bytes,
                stored.content,
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
        idempotency_fingerprint = _web_answer_request_fingerprint(
            conversation_id=conversation_id,
            query=query,
            workspaces=workspaces,
            attachments=attachments,
        )
        prepared = _prepare_submission(
            query=query,
            workspaces=workspaces,
            snapshot=snapshot,
            attachments=attachments,
            max_attachments=self._max_attachments,
        )
        return await self._answers.accept(
            request=prepared.request,
            owner_id=principal_id,
            idempotency_key=submission_id,
            idempotency_fingerprint=idempotency_fingerprint,
            acceptor=_WebAnswerAcceptor(
                store=self._store,
                snapshot=snapshot,
                conversation_id=conversation_id,
                title_hint=_auto_title(query),
                max_turns=self._max_turns,
                ttl_days=self._ttl_days,
            ),
        )

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
        return await operation


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
    )


def _prepare_submission(
    *,
    query: str,
    workspaces: Sequence[str],
    snapshot: ConversationSnapshot,
    attachments: Sequence[ValidatedWebAttachment],
    max_attachments: int,
) -> _PreparedSubmission:
    """Normalize one browser submission into the run's immutable input.

    Only succeeded turns become model history, and only their uploads stay
    readable as prior resources: a pending, failed, or cancelled turn remains
    visible in the browser but is never replayed to the model.
    """
    history: list[dict[str, Any]] = []
    prior: list[tuple[str, AttachmentReference]] = []
    for turn in snapshot.turns:
        if turn.run.status != "succeeded":
            continue
        turn_request = AnswerRunRequest.from_request(turn.run.prepared_input or {})
        history.extend(
            (
                {"role": "user", "content": turn_request.query},
                {"role": "assistant", "content": str((turn.run.result or {}).get("answer") or "")},
            )
        )
        prior.extend((turn.run.run_id, attachment) for attachment in turn_request.attachments)
    remaining = max(0, max_attachments - len(attachments))
    carried = prior[-remaining:] if remaining else []
    request = AnswerRequest(
        query=query,
        workspaces=tuple(workspaces),
        history=tuple(history),
        semantic_highlights=True,
        resources=tuple(
            ResourceInput(
                filename=attachment.filename,
                declared_mime=attachment.mime_type,
                content=attachment.attachment_bytes,
            )
            for attachment in attachments
        ),
        history_resources=tuple(
            AnswerHistoryResource(
                run_id=run_id,
                source_ordinal=item.ordinal,
                digest=item.digest,
                filename=item.filename,
                mime_type=item.mime_type,
                byte_size=item.byte_size,
            )
            for run_id, item in carried
        ),
    )
    return _PreparedSubmission(request=request)


def _web_answer_request_fingerprint(
    *,
    conversation_id: str,
    query: str,
    workspaces: Sequence[str],
    attachments: Sequence[ValidatedWebAttachment],
) -> str:
    """Hash only the stable browser submission, before conversation enrichment."""
    return answer_run_request_fingerprint(
        {
            "conversation_id": conversation_id,
            "query": query,
            "workspaces": list(workspaces),
            "attachments": [
                {
                    "digest": attachment.content_sha256,
                    "filename": attachment.filename,
                    "mime_type": attachment.mime_type,
                    "ordinal": attachment.ordinal,
                    "byte_size": attachment.byte_size,
                }
                for attachment in attachments
            ],
        }
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
    request = AnswerRunRequest.from_request(run.prepared_input or {})
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
