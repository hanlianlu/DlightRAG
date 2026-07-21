# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unit tests for WebConversationService.prepare_answer_turn (merge wiring)."""

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from dlightrag.core.answer.capability import AnswerImageCapability
from dlightrag.core.request.planner import QueryPlan
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
        self.processing_resources = object()

    async def aget_composer_processing_resources(self, workspaces: Any = None) -> object:
        return self.processing_resources

    async def adescribe_query_images(self, images: list[dict[str, Any]]) -> dict[str, str]:
        self.described_images = list(images)
        return {"1": "Image 1: revenue chart"} if images else {}

    async def aplan_web_conversation_query(
        self,
        query: str,
        *,
        text_history: Any = None,
        image_catalog: Any = None,
        attachment_catalog: Any = None,
        current_attachment_catalog: Any = None,
        allowed_history_image_count: int = 0,
        allowed_history_attachment_count: int = 0,
        current_image_descriptions: Any = None,
        workspaces: Any = None,
    ) -> QueryPlan:
        self.plan_kwargs = {
            "image_catalog": image_catalog,
            "attachment_catalog": attachment_catalog,
            "current_attachment_catalog": current_attachment_catalog,
            "allowed_history_image_count": allowed_history_image_count,
            "allowed_history_attachment_count": allowed_history_attachment_count,
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

    assert turn.current_query_images == tuple(current)
    assert len(turn.history_query_images) == 1
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
    assert turn.current_query_images == ()
    assert turn.history_query_images == ()


async def test_prepare_resolves_processing_resources_once_and_reuses_identity() -> None:
    store = _FakeStore()
    history_id = "22222222-2222-2222-2222-222222222222"

    async def fetch_documents_by_ids(
        principal_id: str,
        conversation_id: str,
        attachment_ids: list[str],
        *,
        ttl_days: int,
    ) -> list[Any]:
        assert attachment_ids == [history_id]
        return [
            SimpleNamespace(
                attachment_id=history_id,
                filename="history.pdf",
                document_bytes=b"history",
                content_sha256="history-sha",
            )
        ]

    store.fetch_documents_by_ids = fetch_documents_by_ids  # type: ignore[attr-defined]
    manager = _FakeManager(effective=0)
    resources = object()
    resolutions: list[Any] = []

    async def resolve_processing_resources(workspaces: Any = None) -> object:
        resolutions.append(workspaces)
        return resources

    manager.aget_composer_processing_resources = resolve_processing_resources  # type: ignore[attr-defined]

    async def plan_with_history(*args: Any, **kwargs: Any) -> QueryPlan:
        return QueryPlan(
            original_query=str(args[0]),
            standalone_query="standalone",
            selected_history_attachment_ids=(history_id,),
        )

    manager.aplan_web_conversation_query = plan_with_history  # type: ignore[method-assign]
    service = _service(store)
    seen_resources: list[object] = []

    async def parse_documents(
        *,
        resources: object,
        prepared: PreparedWebConversation,
        documents: list[Any],
    ) -> tuple[list[Any], list[dict[str, str]], list[Any]]:
        seen_resources.append(resources)
        return [], [], []

    service._parse_attachment_documents = parse_documents  # type: ignore[method-assign]
    current = SimpleNamespace(
        attachment_id=_ID,
        filename="current.pdf",
        document_bytes=b"current",
        content_sha256="current-sha",
    )

    turn = await service.prepare_answer_turn(
        manager=cast("RAGServiceManager", manager),
        prepared=_prepared(),
        query="compare documents",
        current_images=[],
        current_documents=[current],  # type: ignore[list-item]
        workspaces=[" Research Notes ", "ignored"],
    )

    assert resolutions == [[" Research Notes ", "ignored"]]
    assert seen_resources == [resources, resources]
    assert all(item is resources for item in seen_resources)
    assert turn.composer_processing_resources is resources
