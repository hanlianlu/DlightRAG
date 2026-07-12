# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Principal-scoped service adapter for durable Web conversations."""

import logging
from dataclasses import dataclass
from typing import Any

from dlightrag.api.auth import UserContext
from dlightrag.citations.schemas import SourceReferencePayload
from dlightrag.storage.web_conversations import (
    ConversationSnapshot,
    PGWebConversationStore,
    StoredConversationImage,
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


@dataclass(frozen=True, slots=True)
class PreparedWebConversation:
    principal_id: str
    conversation_id: str
    content_revision: int
    text_history: tuple[dict[str, Any], ...]


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
        row = await self._store.create_conversation(principal_id)
        return _conversation_summary(row)

    async def list(self, user: UserContext | None) -> list[ConversationSummary]:
        principal_id = principal_id_from_user(user)
        rows = await self._store.list_conversations(
            principal_id,
            ttl_days=self._ttl_days,
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

        summary = await self._summary(principal_id, conversation_id)
        if summary is None:
            return None
        return ConversationHistory(
            conversation=summary,
            turns=[_conversation_turn(conversation_id, row) for row in snapshot.history],
        )

    async def rename(
        self,
        user: UserContext | None,
        conversation_id: str,
        title: str,
    ) -> ConversationSummary | None:
        principal_id = principal_id_from_user(user)
        row = await self._store.rename_conversation(
            principal_id,
            conversation_id,
            title=title,
            ttl_days=self._ttl_days,
        )
        return _conversation_summary(row) if row is not None else None

    async def delete(
        self,
        user: UserContext | None,
        conversation_id: str,
    ) -> bool:
        principal_id = principal_id_from_user(user)
        return await self._store.delete_conversation(
            principal_id,
            conversation_id,
            ttl_days=self._ttl_days,
        )

    async def prepare_answer(
        self,
        user: UserContext | None,
        conversation_id: str,
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
        return PreparedWebConversation(
            principal_id=principal_id,
            conversation_id=snapshot.conversation_id,
            content_revision=snapshot.content_revision,
            text_history=tuple(text_history),
        )

    async def image(
        self,
        user: UserContext | None,
        conversation_id: str,
        image_id: str,
    ) -> StoredConversationImage | None:
        principal_id = principal_id_from_user(user)
        return await self._store.get_image(
            principal_id,
            conversation_id,
            image_id,
            ttl_days=self._ttl_days,
        )

    async def _snapshot(
        self,
        principal_id: str,
        conversation_id: str,
    ) -> ConversationSnapshot | None:
        return await self._store.snapshot(
            principal_id,
            conversation_id,
            ttl_days=self._ttl_days,
            max_turns=self._max_turns,
        )

    async def _summary(
        self,
        principal_id: str,
        conversation_id: str,
    ) -> ConversationSummary | None:
        rows = await self._store.list_conversations(
            principal_id,
            ttl_days=self._ttl_days,
        )
        for row in rows:
            if str(row["conversation_id"]) == conversation_id:
                return _conversation_summary(row)
        return None


def _conversation_summary(row: dict[str, Any]) -> ConversationSummary:
    return ConversationSummary(
        conversation_id=str(row["conversation_id"]),
        title=row.get("title"),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
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
        thumbnail_url=url,
        label=f"Turn {turn_number}, image {ordinal}",
    )


__all__ = ["PreparedWebConversation", "WebConversationService"]
