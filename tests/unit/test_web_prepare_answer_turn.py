# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unit tests for WebConversationService.prepare_answer_turn (merge wiring)."""

from typing import TYPE_CHECKING, Any, cast

from dlightrag.core.answer_capability import AnswerImageCapability
from dlightrag.core.query_planner import QueryPlan
from dlightrag.storage.web_conversations import StoredConversationImage
from dlightrag.web.conversations import (
    PreparedWebConversation,
    WebConversationService,
)

if TYPE_CHECKING:
    from dlightrag.core.servicemanager import RAGServiceManager

_ID = "11111111-1111-1111-1111-111111111111"


class _FakeStore:
    def __init__(self) -> None:
        self.fetched: list[list[str]] = []

    async def list_image_catalog(
        self, principal_id: str, conversation_id: str, *, max_turns: int, ttl_days: int
    ) -> list[dict[str, Any]]:
        return [
            {
                "image_id": _ID,
                "turn_number": 1,
                "ordinal": 0,
                "vlm_description": "2023 revenue chart",
            }
        ]

    async def fetch_images_by_ids(
        self, principal_id: str, conversation_id: str, image_ids: list[str], *, ttl_days: int
    ) -> list[StoredConversationImage]:
        self.fetched.append(list(image_ids))
        return [
            StoredConversationImage(
                image_id=_ID,
                mime_type="image/png",
                image_bytes=b"PNGDATA",
                vlm_description="2023 revenue chart",
            )
        ]


class _FakeManager:
    def __init__(self, effective: int = 6) -> None:
        self.answer_image_capability = AnswerImageCapability(
            status="supported" if effective else "unsupported",
            configured_ceiling=effective,
            effective_max_images=effective,
            provider="openai",
            base_url=None,
            model="gpt-4o",
            failure_kind=None,
        )
        self.plan_kwargs: dict[str, Any] = {}
        self.described_images: list[dict[str, Any]] = []

    async def adescribe_query_images(self, images: list[dict[str, Any]]) -> dict[str, str]:
        self.described_images = list(images)
        return {"1": "Image 1: revenue chart"} if images else {}

    async def aplan_query(
        self,
        query: str,
        *,
        text_history: Any = None,
        image_catalog: Any = None,
        allowed_history_image_count: int = 0,
        current_image_descriptions: Any = None,
        workspaces: Any = None,
    ) -> QueryPlan:
        self.plan_kwargs = {
            "image_catalog": image_catalog,
            "allowed_history_image_count": allowed_history_image_count,
            "current_image_descriptions": current_image_descriptions,
            "workspaces": workspaces,
        }
        # Mirror real planner: no catalog -> no selection.
        selected = (_ID,) if image_catalog else ()
        return QueryPlan(
            original_query=query,
            standalone_query="2023 revenue trend",
            selected_history_image_ids=selected,
        )


def _service(store: _FakeStore) -> WebConversationService:
    return WebConversationService(store=store, max_turns=100, ttl_days=30)  # type: ignore[arg-type]


def _prepared() -> PreparedWebConversation:
    return PreparedWebConversation(
        principal_id="alice", conversation_id="c1", content_revision=0, text_history=()
    )


async def test_prepare_answer_turn_injects_plan_and_orders_current_first() -> None:
    store = _FakeStore()
    manager = _FakeManager(effective=6)
    service = _service(store)
    current = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}]

    turn = await service.prepare_answer_turn(
        manager=cast("RAGServiceManager", manager),
        prepared=_prepared(),
        query="explain that chart",
        current_images=current,
        workspaces=["default"],
    )

    urls = [block["image_url"]["url"] for block in turn.materialized_query_images]
    assert urls[0] == "data:image/png;base64,AAA"  # current image is always first
    assert len(urls) == 2  # current + one materialized history image
    assert turn.plan is not None
    assert turn.plan.standalone_query.startswith("2023 revenue trend")
    assert manager.plan_kwargs["allowed_history_image_count"] == 5  # 6 effective - 1 current
    assert store.fetched == [[_ID]]  # only the selected id is materialized
    # Current images are described before planning; descriptions feed the planner
    # and are carried on the turn so retrieval never re-describes them.
    assert manager.described_images == current
    assert manager.plan_kwargs["current_image_descriptions"] == ["Image 1: revenue chart"]
    assert turn.current_image_descriptions == {"1": "Image 1: revenue chart"}


async def test_prepare_answer_turn_skips_history_when_no_capacity() -> None:
    store = _FakeStore()
    manager = _FakeManager(effective=0)  # answer model does not support images
    service = _service(store)

    turn = await service.prepare_answer_turn(
        manager=cast("RAGServiceManager", manager),
        prepared=_prepared(),
        query="q",
        current_images=[],
        workspaces=None,
    )

    assert manager.plan_kwargs["image_catalog"] is None  # no room -> no catalog fetched
    assert manager.plan_kwargs["allowed_history_image_count"] == 0
    assert manager.plan_kwargs["current_image_descriptions"] is None  # no images -> no descriptions
    assert turn.current_image_descriptions == {}
    assert store.fetched == []  # nothing materialized
    assert turn.materialized_query_images == ()
