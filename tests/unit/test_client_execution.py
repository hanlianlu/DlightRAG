# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for transport-neutral client execution helpers."""

from unittest.mock import AsyncMock

from dlightrag.api.models import MetadataFilterRequest, RetrieveRequest
from dlightrag.core.client_contracts import ConversationMessage, ImageURLContentBlock
from dlightrag.core.client_execution import execute_answer, execute_retrieve
from dlightrag.core.retrieval.protocols import RetrievalResult
from dlightrag.core.scope import RequestScope
from dlightrag.mcp.contracts import AnswerInput


async def test_execute_retrieve_forwards_shared_query_kwargs_and_scope() -> None:
    result = RetrievalResult(contexts={"chunks": []})
    manager = AsyncMock()
    manager.aretrieve = AsyncMock(return_value=result)
    payload = RetrieveRequest(
        query="report",
        workspaces=["finance"],
        top_k=8,
        chunk_top_k=5,
        bm25_query="quarterly report",
        filters=MetadataFilterRequest(doc_author="Ada"),
        query_images=[
            ImageURLContentBlock.model_validate(
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
            )
        ],
    )
    scope = RequestScope(user_id="alice", auth_mode="jwt").for_workspaces(["finance"])

    executed = await execute_retrieve(
        manager=manager,
        payload=payload,
        resolved_workspaces=["finance"],
        scope=scope,
    )

    assert executed is result
    call_kwargs = manager.aretrieve.await_args.kwargs
    assert manager.aretrieve.await_args.args == ("report",)
    assert call_kwargs["workspaces"] == ["finance"]
    assert call_kwargs["top_k"] == 8
    assert call_kwargs["chunk_top_k"] == 5
    assert call_kwargs["scope"] == scope
    assert call_kwargs["bm25_query"] == "quarterly report"
    assert call_kwargs["filters"].doc_author == "Ada"
    assert call_kwargs["query_images"] == [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
    ]


async def test_execute_answer_projects_history_and_answer_kwargs() -> None:
    result = RetrievalResult(answer="done", contexts={"chunks": []})
    manager = AsyncMock()
    manager.aanswer = AsyncMock(return_value=result)
    payload = AnswerInput(
        query="follow up",
        workspaces=["finance"],
        top_k=6,
        chunk_top_k=4,
        answer_context_top_k=2,
        filters={"doc_title": "Runbook"},
        query_images=[
            ImageURLContentBlock.model_validate(
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
            )
        ],
        semantic_highlights=True,
        history=[
            ConversationMessage(role="user", content="Earlier question"),
            ConversationMessage(role="assistant", content="Earlier answer"),
        ],
    )
    scope = RequestScope(user_id="alice", auth_mode="jwt").for_workspaces(["finance"])

    executed = await execute_answer(
        manager=manager,
        payload=payload,
        resolved_workspaces=["finance"],
        scope=scope,
    )

    assert executed is result
    call_kwargs = manager.aanswer.await_args.kwargs
    assert manager.aanswer.await_args.args == ("follow up",)
    assert call_kwargs["workspaces"] == ["finance"]
    assert call_kwargs["top_k"] == 6
    assert call_kwargs["chunk_top_k"] == 4
    assert call_kwargs["answer_context_top_k"] == 2
    assert call_kwargs["semantic_highlights"] is True
    assert call_kwargs["history"] == [
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": "Earlier answer"},
    ]
    assert call_kwargs["scope"] == scope
    assert call_kwargs["filters"].doc_title == "Runbook"
    assert call_kwargs["query_images"] == [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
    ]
