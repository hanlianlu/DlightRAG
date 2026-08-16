# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for transport-neutral client execution helpers."""

from unittest.mock import AsyncMock

from dlightrag_rag.retrieval import RetrievalResult

from dlightrag.api.models import MetadataFilterRequest, RetrieveRequest
from dlightrag.core.client_contracts import QueryImage
from dlightrag.core.client_execution import execute_retrieve


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
