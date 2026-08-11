# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for transport-neutral client execution helpers."""

from unittest.mock import AsyncMock

from dlightrag.api.models import MetadataFilterRequest, RetrieveRequest
from dlightrag.core.client_contracts import AnswerAttachmentLink, ConversationMessage, QueryImage
from dlightrag.core.client_execution import execute_answer, execute_retrieve
from dlightrag.core.retrieval.protocols import RetrievalResult
from dlightrag.mcp.contracts import AnswerInput


async def test_execute_retrieve_forwards_shared_query_kwargs() -> None:
    result = RetrievalResult(contexts={"chunks": []})
    manager = AsyncMock()
    manager.aretrieve = AsyncMock(return_value=result)
    payload = RetrieveRequest(
        query="report",
        workspaces=["finance"],
        top_k=8,
        chunk_top_k=5,
        bm25_query="quarterly report",
        filters=MetadataFilterRequest(author="Ada"),
        query_images=[
            QueryImage.model_validate(
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
            )
        ],
    )
    executed = await execute_retrieve(
        manager=manager,
        payload=payload,
        resolved_workspaces=["finance"],
    )

    assert executed is result
    call_kwargs = manager.aretrieve.await_args.kwargs
    assert manager.aretrieve.await_args.args == ("report",)
    assert call_kwargs["workspaces"] == ["finance"]
    assert call_kwargs["top_k"] == 8
    assert call_kwargs["chunk_top_k"] == 5
    assert call_kwargs["bm25_query"] == "quarterly report"
    assert call_kwargs["filters"].author == "Ada"
    assert call_kwargs["query_images"] == [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
    ]


async def test_execute_answer_projects_history_and_attachment_resources() -> None:
    result = RetrievalResult(answer="done", contexts={"chunks": []})
    manager = AsyncMock()
    manager.aanswer = AsyncMock(return_value=result)
    payload = AnswerInput(
        query="follow up",
        workspaces=["finance"],
        top_k=6,
        chunk_top_k=4,
        filters={"title": "Runbook"},
        attachments=[
            AnswerAttachmentLink(url="https://example.com/report.pdf", filename="report.pdf"),
        ],
        semantic_highlights=True,
        history=[
            ConversationMessage(role="user", content="Earlier question"),
            ConversationMessage(role="assistant", content="Earlier answer"),
        ],
    )
    executed = await execute_answer(
        manager=manager,
        payload=payload,
        resolved_workspaces=["finance"],
    )

    assert executed is result
    call_kwargs = manager.aanswer.await_args.kwargs
    assert manager.aanswer.await_args.args == ("follow up",)
    assert call_kwargs["workspaces"] == ["finance"]
    assert call_kwargs["top_k"] == 6
    assert call_kwargs["chunk_top_k"] == 4
    assert call_kwargs["semantic_highlights"] is True
    assert call_kwargs["history"] == [
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": "Earlier answer"},
    ]
    assert call_kwargs["filters"].title == "Runbook"
    assert "query_images" not in call_kwargs
    resources = call_kwargs["resources"]
    assert [resource.url for resource in resources] == ["https://example.com/report.pdf"]
    assert resources[0].filename == "report.pdf"
    assert resources[0].content is None
