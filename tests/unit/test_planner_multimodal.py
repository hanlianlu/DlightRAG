# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for multimodal conversation history in QueryPlanner."""

import pytest

from dlightrag.core.query_planner import QueryPlanner, _convert_history_to_text


class TestConvertHistoryToText:
    def test_pure_text_history_stays_unchanged(self) -> None:
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        result = _convert_history_to_text(history)
        assert "user: hello" in result
        assert "assistant: hi there" in result

    def test_multimodal_adds_image_placeholder(self) -> None:
        history = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            },
            {"role": "assistant", "content": "That is a revenue chart."},
        ]
        result = _convert_history_to_text(history)
        assert "[user shared 1 image]" in result
        assert "what is this?" in result
        assert "assistant: That is a revenue chart." in result

    def test_multiple_images_counted(self) -> None:
        history = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:a"}},
                    {"type": "image_url", "image_url": {"url": "data:b"}},
                    {"type": "text", "text": "compare these"},
                ],
            },
        ]
        result = _convert_history_to_text(history)
        assert "[user shared 2 images]" in result

    def test_mixed_string_and_list_content(self) -> None:
        history = [
            {"role": "user", "content": "first message"},
            {"role": "assistant", "content": "ok"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "now look at this"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            },
        ]
        result = _convert_history_to_text(history)
        assert "user: first message" in result
        assert "user: now look at this" in result
        assert "[user shared 1 image]" in result

    def test_empty_history_returns_empty_string(self) -> None:
        assert _convert_history_to_text([]) == ""

    def test_none_history_returns_empty_string(self) -> None:
        assert _convert_history_to_text(None) == ""


@pytest.mark.asyncio
async def test_web_planner_selects_history_documents_and_images() -> None:
    async def llm_func(**_kwargs):
        return """
        {
          "standalone_query": "compare termination clause against attached report",
          "bm25_query": "termination clause attached report",
          "filters": {},
          "filter_confidence": "low",
          "filter_evidence": [],
          "selected_history_image_ids": ["img-1"],
          "selected_history_attachment_ids": ["att-1"],
          "attachment_query": "termination clause",
          "attachment_directives": [{"attachment_id": "att-1", "hint": "termination clause"}]
        }
        """

    planner = QueryPlanner(llm_func=llm_func)
    plan = await planner.plan_web_conversation(
        "compare with the prior PDF",
        conversation_history=[{"role": "user", "content": "see prior PDF"}],
        image_catalog=[
            {"image_id": "img-1", "turn_number": 1, "ordinal": 1, "vlm_description": "chart"}
        ],
        attachment_catalog=[
            {
                "attachment_id": "att-1",
                "turn_number": 1,
                "ordinal": 1,
                "filename": "contract.pdf",
                "parse_summary": "contract",
            }
        ],
        allowed_history_image_count=3,
        allowed_history_attachment_count=3,
        current_image_descriptions=[],
        current_attachment_catalog=[],
        schema={},
    )

    assert plan.selected_history_image_ids == ("img-1",)
    assert plan.selected_history_attachment_ids == ("att-1",)
    assert plan.attachment_query == "termination clause"
    assert plan.attachment_directives[0]["attachment_id"] == "att-1"
