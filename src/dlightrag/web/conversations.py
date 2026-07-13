# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Principal-scoped service adapter for durable Web conversations."""

import asyncio
import logging
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, TypeVar

import asyncpg

from dlightrag.api.auth import UserContext
from dlightrag.citations.schemas import SourceReferencePayload
from dlightrag.core.answer_turn import PreparedAnswerTurn
from dlightrag.storage.pool import POSTGRES_UNAVAILABLE_EXCEPTIONS
from dlightrag.storage.web_conversations import (
    CommitTurnResult,
    ConversationSnapshot,
    PendingConversationImage,
    PGWebConversationStore,
    StoredConversationImage,
)
from dlightrag.utils.images import (
    ValidatedWebImage,
    image_bytes_to_data_uri,
    thumbnail_bytes,
)
from dlightrag.web.conversation_models import (
    ConversationHistory,
    ConversationImageReference,
    ConversationSummary,
    ConversationTurn,
)
from dlightrag.web.principal import principal_id_from_user
from dlightrag.web.safe_html import safe_answer_done

logger = logging.getLogger(__name__)
T = TypeVar("T")
_COMMIT_ATTEMPT_TIMEOUT_SECONDS = 45.0
_RECONCILE_ATTEMPT_TIMEOUT_SECONDS = 10.0
_RECONCILE_ATTEMPTS = 2
_RECONCILE_BACKOFF_SECONDS = 0.25
_AMBIGUOUS_COMMIT_EXCEPTIONS = (
    *POSTGRES_UNAVAILABLE_EXCEPTIONS,
    asyncpg.exceptions.InterfaceError,
)
_HISTORY_THUMBNAIL_MAX_PX = 320
_HISTORY_THUMBNAIL_MAX_BYTES = 128 * 1024
_HISTORY_THUMBNAIL_QUALITY = 82
_HISTORY_THUMBNAIL_MIN_QUALITY = 50
_HISTORY_THUMBNAIL_MIN_PX = 64


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


class WebConversationService:
    """Map authenticated browser operations onto the scoped persistence store."""

    def __init__(
        self,
        *,
        store: PGWebConversationStore,
        max_turns: int,
        ttl_days: int,
    ) -> None:
        self._store = store
        self._max_turns = max_turns
        self._ttl_days = ttl_days

    async def initialize(self) -> None:
        """Apply schema migrations and best-effort global startup retention."""
        await self._store.initialize()
        try:
            await self._store.prune_expired(ttl_days=self._ttl_days)
        except Exception:
            logger.exception("Failed to prune expired Web conversations at startup")

    async def create(self, user: UserContext | None) -> ConversationSummary:
        principal_id = principal_id_from_user(user)
        row = await self._store_call(self._store.create_conversation(principal_id))
        return _conversation_summary(row)

    async def list(self, user: UserContext | None) -> list[ConversationSummary]:
        principal_id = principal_id_from_user(user)
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
    ) -> ConversationHistory | None:
        principal_id = principal_id_from_user(user)
        snapshot = await self._snapshot(principal_id, conversation_id)
        if snapshot is None:
            return None

        return ConversationHistory(
            conversation=_snapshot_summary(snapshot),
            turns=[_conversation_turn(conversation_id, row) for row in snapshot.history],
        )

    async def rename(
        self,
        user: UserContext | None,
        conversation_id: str,
        title: str,
    ) -> ConversationSummary | None:
        principal_id = principal_id_from_user(user)
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
        principal_id = principal_id_from_user(user)
        return await self._store_call(
            self._store.delete_conversation(
                principal_id,
                conversation_id,
                ttl_days=self._ttl_days,
            )
        )

    async def prepare_answer(
        self,
        user: UserContext | None,
        conversation_id: str,
        submission_id: str | None = None,
    ) -> PreparedWebConversation | None:
        principal_id = principal_id_from_user(user)
        snapshot = await self._snapshot(principal_id, conversation_id)
        if snapshot is None:
            return None
        text_history: list[dict[str, Any]] = []
        for turn in snapshot.history:
            text_history.extend(
                (
                    {"role": "user", "content": str(turn["user_text"])},
                    {"role": "assistant", "content": str(turn["assistant_text"])},
                )
            )
        committed_submission = None
        if submission_id is not None:
            try:
                async with asyncio.timeout(_RECONCILE_ATTEMPT_TIMEOUT_SECONDS):
                    committed_submission = await self._store.find_committed_turn_once(
                        principal_id,
                        conversation_id,
                        submission_id,
                        ttl_days=self._ttl_days,
                    )
            except _AMBIGUOUS_COMMIT_EXCEPTIONS as exc:
                raise WebConversationUnavailableError from exc
        return PreparedWebConversation(
            principal_id=principal_id,
            conversation_id=snapshot.conversation_id,
            content_revision=snapshot.content_revision,
            text_history=tuple(text_history),
            committed_submission=committed_submission,
        )

    async def prepare_answer_turn(
        self,
        *,
        manager: Any,
        prepared: PreparedWebConversation,
        query: str,
        current_images: list[dict[str, Any]],
        workspaces: list[str] | None = None,
    ) -> PreparedAnswerTurn:
        """Plan the turn and materialize any referenced history images.

        The web-variant planner rewrites the query and selects scoped history
        images in one call; the selected ids are re-validated and materialized
        here (web owns the store), and the finished plan is injected so core
        skips re-planning. Current images always come first and are never
        displaced by history selection.
        """
        capability = getattr(manager, "answer_image_capability", None)
        effective = capability.effective_max_images if capability is not None else 0
        remaining = max(0, effective - len(current_images))

        catalog: list[dict[str, Any]] = []
        if remaining > 0:
            catalog = await self._store_call(
                self._store.list_image_catalog(
                    prepared.principal_id,
                    prepared.conversation_id,
                    max_turns=self._max_turns,
                    ttl_days=self._ttl_days,
                )
            )

        plan = await manager.aplan_query(
            query,
            text_history=list(prepared.text_history),
            image_catalog=catalog or None,
            allowed_history_image_count=remaining,
            workspaces=workspaces,
        )

        history_blocks: list[dict[str, Any]] = []
        if plan.selected_history_image_ids:
            owned = await self._store_call(
                self._store.fetch_images_by_ids(
                    prepared.principal_id,
                    prepared.conversation_id,
                    list(plan.selected_history_image_ids),
                    ttl_days=self._ttl_days,
                )
            )
            by_id = {image.image_id: image for image in owned}
            captions: list[str] = []
            for image_id in plan.selected_history_image_ids:  # preserve relevance order
                image = by_id.get(image_id)
                if image is None:
                    continue
                history_blocks.append(_history_image_block(image))
                if image.vlm_description:
                    captions.append(image.vlm_description)
            if captions:
                # Fold persisted captions into the retrieval text; no re-VLM.
                plan.standalone_query = (
                    f"{plan.standalone_query}\n\nReferenced prior images:\n" + "\n".join(captions)
                )

        return PreparedAnswerTurn(
            current_query=query,
            retrieval_query=query,
            text_history=tuple(prepared.text_history),
            materialized_query_images=tuple([*current_images, *history_blocks]),
            plan=plan,
        )

    async def image(
        self,
        user: UserContext | None,
        conversation_id: str,
        image_id: str,
    ) -> StoredConversationImage | None:
        principal_id = principal_id_from_user(user)
        return await self._store_call(
            self._store.get_image(
                principal_id,
                conversation_id,
                image_id,
                ttl_days=self._ttl_days,
            )
        )

    async def thumbnail(
        self,
        user: UserContext | None,
        conversation_id: str,
        image_id: str,
    ) -> StoredConversationImage | None:
        """Derive one bounded UI thumbnail after the scoped original lookup."""
        image = await self.image(user, conversation_id, image_id)
        if image is None:
            return None
        try:
            payload, mime_type = await asyncio.to_thread(
                thumbnail_bytes,
                image.image_bytes,
                max_px=_HISTORY_THUMBNAIL_MAX_PX,
                max_bytes=_HISTORY_THUMBNAIL_MAX_BYTES,
                quality=_HISTORY_THUMBNAIL_QUALITY,
                min_quality=_HISTORY_THUMBNAIL_MIN_QUALITY,
                min_px=_HISTORY_THUMBNAIL_MIN_PX,
            )
        except Exception:
            logger.warning("Failed to derive Web conversation thumbnail", exc_info=True)
            return None
        return StoredConversationImage(
            image_id=image.image_id,
            mime_type=mime_type,
            image_bytes=payload,
        )

    async def commit_answer(
        self,
        prepared: PreparedWebConversation,
        *,
        submission_id: str,
        user_text: str,
        assistant_text: str,
        answer_sources: dict[str, Any],
        queried_workspaces: list[str],
        images: tuple[ValidatedWebImage, ...],
        image_descriptions: dict[str, str],
    ) -> CommitTurnResult:
        """Atomically append a completed answer against its captured revision."""

        def description_for(image: ValidatedWebImage) -> str | None:
            return image_descriptions.get(image.image_id) or image_descriptions.get(
                str(image.ordinal)
            )

        pending = [
            PendingConversationImage(
                image_id=image.image_id,
                ordinal=image.ordinal,
                mime_type=image.mime_type,
                image_bytes=image.image_bytes,
                content_sha256=image.content_sha256,
                vlm_description=description_for(image),
            )
            for image in images
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
                    images=pending,
                    max_turns=self._max_turns,
                    ttl_days=self._ttl_days,
                )
        except _AMBIGUOUS_COMMIT_EXCEPTIONS:
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
                    committed = await self._store.find_committed_turn_once(
                        prepared.principal_id,
                        prepared.conversation_id,
                        submission_id,
                        ttl_days=self._ttl_days,
                    )
            except _AMBIGUOUS_COMMIT_EXCEPTIONS:
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
        except POSTGRES_UNAVAILABLE_EXCEPTIONS as exc:
            raise WebConversationUnavailableError from exc


def _history_image_block(image: StoredConversationImage) -> dict[str, Any]:
    data_uri = image_bytes_to_data_uri(image.image_bytes, fallback_mime=image.mime_type)
    return {"type": "image_url", "image_url": {"url": data_uri}}


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


def _answer_snapshot(value: Any) -> tuple[dict[str, Any], list[SourceReferencePayload], list[Any]]:
    raw = dict(value) if isinstance(value, dict) else {}
    sources_value = raw.get("sources", [])
    sources = []
    for source_value in sources_value:
        source = SourceReferencePayload.model_validate(source_value)
        if source.chunks is None:
            source = source.model_copy(update={"chunks": []})
        sources.append(source)
    answer_images = raw.get("answer_images", [])
    if not isinstance(answer_images, list):
        answer_images = []
    raw["sources"] = [source.model_dump(mode="json") for source in sources]
    raw["answer_images"] = answer_images
    return raw, sources, answer_images


def _conversation_turn(conversation_id: str, row: dict[str, Any]) -> ConversationTurn:
    turn_number = int(row["turn_number"])
    images = [
        _image_reference(conversation_id, turn_number, image) for image in row.get("images", [])
    ]
    answer_sources, sources, answer_images = _answer_snapshot(row.get("answer_sources"))
    assistant_text = str(row["assistant_text"])
    return ConversationTurn(
        turn_id=str(row["turn_id"]),
        turn_number=turn_number,
        user_text=str(row["user_text"]),
        assistant_text=assistant_text,
        user_images=images,
        answer_sources=answer_sources,
        answer_html=safe_answer_done(
            answer=assistant_text,
            sources=sources,
            answer_images=answer_images,
        ),
        queried_workspaces=[str(workspace) for workspace in row.get("queried_workspaces", [])],
        created_at=row["created_at"],
    )


def _image_reference(
    conversation_id: str,
    turn_number: int,
    image: dict[str, Any],
) -> ConversationImageReference:
    image_id = str(image["image_id"])
    ordinal = int(image["ordinal"])
    url = f"/web/conversations/{conversation_id}/images/{image_id}"
    return ConversationImageReference(
        image_id=image_id,
        ordinal=ordinal,
        mime_type=str(image["mime_type"]),
        url=url,
        thumbnail_url=url + "/thumbnail",
        label=f"Turn {turn_number}, image {ordinal}",
    )


__all__ = [
    "PreparedWebConversation",
    "WebConversationService",
    "WebConversationUnavailableError",
]
