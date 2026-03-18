# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for unified mode structural context propagation."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from dlightrag.unifiedrepresent.extractor import EntityExtractor


def _make_lightrag() -> MagicMock:
    """Create a fake LightRAG instance with all attributes EntityExtractor needs."""
    lr = MagicMock()
    lr.llm_response_cache = MagicMock()
    lr.text_chunks = MagicMock()
    lr.chunk_entity_relation_graph = MagicMock()
    lr.entities_vdb = MagicMock()
    lr.relationships_vdb = MagicMock()
    lr.full_entities = MagicMock()
    lr.full_relations = MagicMock()
    lr.entity_chunks = MagicMock()
    lr.relation_chunks = MagicMock()
    return lr


class TestStructuralContextPrompt:
    """Verify the prompt constant exists and has expected content."""

    def test_prompt_exists_and_mentions_json(self) -> None:
        from dlightrag.unifiedrepresent.prompts import STRUCTURAL_CONTEXT_PROMPT

        assert "update" in STRUCTURAL_CONTEXT_PROMPT.lower()
        assert "keep" in STRUCTURAL_CONTEXT_PROMPT.lower()
        assert "json" in STRUCTURAL_CONTEXT_PROMPT.lower()


class TestUpdateStructuralContext:
    """Test _update_structural_context LLM call and JSON parsing."""

    async def test_update_action_returns_new_context(self) -> None:
        llm_response = json.dumps({"action": "update", "context": "Section: Finance\nTable: Q, Rev"})
        context_fn = AsyncMock(return_value=llm_response)
        ext = EntityExtractor(
            _make_lightrag(), ["person"], AsyncMock(),
            context_model_func=context_fn,
        )
        result = await ext._update_structural_context("old ctx", "page text with new headers")
        assert result == "Section: Finance\nTable: Q, Rev"
        context_fn.assert_called_once()

    async def test_keep_action_returns_existing_context(self) -> None:
        llm_response = json.dumps({"action": "keep"})
        context_fn = AsyncMock(return_value=llm_response)
        ext = EntityExtractor(
            _make_lightrag(), ["person"], AsyncMock(),
            context_model_func=context_fn,
        )
        result = await ext._update_structural_context("existing ctx", "more data rows")
        assert result == "existing ctx"

    async def test_llm_failure_returns_existing_context(self) -> None:
        context_fn = AsyncMock(side_effect=Exception("API error"))
        ext = EntityExtractor(
            _make_lightrag(), ["person"], AsyncMock(),
            context_model_func=context_fn,
        )
        result = await ext._update_structural_context("safe ctx", "page text")
        assert result == "safe ctx"

    async def test_malformed_json_returns_existing_context(self) -> None:
        context_fn = AsyncMock(return_value="I don't understand the request")
        ext = EntityExtractor(
            _make_lightrag(), ["person"], AsyncMock(),
            context_model_func=context_fn,
        )
        result = await ext._update_structural_context("prev ctx", "page text")
        assert result == "prev ctx"

    async def test_json_embedded_in_markdown_fences(self) -> None:
        llm_response = '```json\n{"action": "update", "context": "New heading"}\n```'
        context_fn = AsyncMock(return_value=llm_response)
        ext = EntityExtractor(
            _make_lightrag(), ["person"], AsyncMock(),
            context_model_func=context_fn,
        )
        result = await ext._update_structural_context(None, "# New heading\nsome text")
        assert result == "New heading"

    async def test_none_context_passed_as_empty(self) -> None:
        llm_response = json.dumps({"action": "update", "context": "First page ctx"})
        context_fn = AsyncMock(return_value=llm_response)
        ext = EntityExtractor(
            _make_lightrag(), ["person"], AsyncMock(),
            context_model_func=context_fn,
        )
        result = await ext._update_structural_context(None, "# Title\nContent")
        assert result == "First page ctx"
        call_args = context_fn.call_args
        messages = call_args[1]["messages"]
        user_msg = [m for m in messages if m["role"] == "user"][0]
        assert "(empty" in user_msg["content"].lower() or "none" in user_msg["content"].lower()

    async def test_no_context_model_func_returns_existing(self) -> None:
        ext = EntityExtractor(
            _make_lightrag(), ["person"], AsyncMock(),
            context_model_func=None,
        )
        result = await ext._update_structural_context("existing", "page text")
        assert result == "existing"


class TestDescribePageWithContext:
    """Test that _describe_page injects structural_ctx into VLM prompt."""

    async def test_structural_ctx_injected_before_image(self) -> None:
        """When structural_ctx is provided, it appears as a text block before the image."""
        vision_fn = AsyncMock(return_value="extracted content")
        ext = EntityExtractor(_make_lightrag(), ["person"], vision_fn)

        await ext._describe_page(MagicMock(), page_index=1, structural_ctx="Table: Name, Role")

        messages = vision_fn.call_args[1]["messages"]
        user_msg = [m for m in messages if m["role"] == "user"][0]
        content_parts = user_msg["content"]

        # First part should be text with structural context
        assert content_parts[0]["type"] == "text"
        assert "Table: Name, Role" in content_parts[0]["text"]
        # Second part should be the image
        assert content_parts[1]["type"] == "image_url"

    async def test_no_structural_ctx_no_extra_text_block(self) -> None:
        """When structural_ctx is None, no extra text block is added."""
        vision_fn = AsyncMock(return_value="extracted content")
        ext = EntityExtractor(_make_lightrag(), ["person"], vision_fn)

        await ext._describe_page(MagicMock(), page_index=0, structural_ctx=None)

        messages = vision_fn.call_args[1]["messages"]
        user_msg = [m for m in messages if m["role"] == "user"][0]
        content_parts = user_msg["content"]

        # First part should be the image (no context text block)
        assert content_parts[0]["type"] == "image_url"
