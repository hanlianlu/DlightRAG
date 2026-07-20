# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for multimodal conversation history in QueryPlanner."""

import pytest

from dlightrag.core.request.planner import QueryPlanner, _convert_history_to_text
from dlightrag.utils.tokens import estimate_tokens


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
          "selected_history_attachment_ids": ["att-1"]
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


@pytest.mark.asyncio
async def test_web_planner_preserves_current_attachment_by_truncating_old_history() -> None:
    captured_system_prompt = ""

    async def llm_func(**kwargs):
        nonlocal captured_system_prompt
        captured_system_prompt = kwargs["messages"][0]["content"]
        return """
        {
          "standalone_query": "current document topic",
          "bm25_query": "current document topic",
          "filters": {},
          "filter_confidence": "low",
          "filter_evidence": [],
          "selected_history_image_ids": [],
          "selected_history_attachment_ids": []
        }
        """

    planner = QueryPlanner(llm_func=llm_func)
    await planner.plan_web_conversation(
        "what does this say?",
        conversation_history=[
            {"role": "user", "content": "OLD-HISTORY-MARKER " + ("old " * 50_000)},
            {"role": "assistant", "content": "NEW-HISTORY-MARKER " + ("new " * 50_000)},
        ],
        current_attachment_catalog=[
            {
                "attachment_id": "current-doc",
                "filename": "report.pdf",
                "parse_summary": "CURRENT-DOCUMENT-MARKER " + ("document " * 3_000),
            }
        ],
        max_turns=10,
        max_tokens=81_920,
        schema={},
    )

    assert "CURRENT-DOCUMENT-MARKER" in captured_system_prompt
    assert "NEW-HISTORY-MARKER" in captured_system_prompt
    assert "OLD-HISTORY-MARKER" not in captured_system_prompt


@pytest.mark.asyncio
async def test_web_planner_bounds_large_prior_attachment_catalog() -> None:
    captured_messages = []

    async def llm_func(**kwargs):
        nonlocal captured_messages
        captured_messages = kwargs["messages"]
        return """
        {
          "standalone_query": "current report",
          "bm25_query": "current report",
          "filters": {},
          "filter_confidence": "low",
          "filter_evidence": [],
          "selected_history_image_ids": [],
          "selected_history_attachment_ids": []
        }
        """

    planner = QueryPlanner(llm_func=llm_func)
    await planner.plan_web_conversation(
        "CURRENT-QUERY-MARKER what does this say?",
        conversation_history=[
            {"role": "user", "content": "OLD-HISTORY-MARKER " + ("old " * 50_000)},
            {"role": "assistant", "content": "NEW-HISTORY-MARKER " + ("new " * 50_000)},
        ],
        attachment_catalog=[
            {
                "attachment_id": f"att-{turn}",
                "turn_number": turn,
                "ordinal": 0,
                "filename": f"report-{turn}.pdf",
                "parse_summary": f"ATTACHMENT-{turn}-MARKER " + ("digest " * 6_000),
            }
            for turn in range(1, 21)
        ],
        image_catalog=[
            {
                "image_id": f"img-{turn}",
                "turn_number": turn,
                "ordinal": 1,
                "vlm_description": f"IMAGE-{turn}-MARKER " + ("visual " * 3_000),
            }
            for turn in range(1, 21)
        ],
        current_attachment_catalog=[
            {
                "attachment_id": "current-doc",
                "filename": "current.pdf",
                "parse_summary": "CURRENT-DOCUMENT-MARKER " + ("current " * 3_000),
            }
        ],
        max_turns=50,
        max_tokens=81_920,
        allowed_history_image_count=3,
        allowed_history_attachment_count=3,
        schema={},
    )

    system_prompt = captured_messages[0]["content"]
    user_query = captured_messages[1]["content"]
    assert "CURRENT-DOCUMENT-MARKER" in system_prompt
    assert "ATTACHMENT-20-MARKER" in system_prompt
    assert "ATTACHMENT-1-MARKER" not in system_prompt
    assert "IMAGE-20-MARKER" in system_prompt
    assert "IMAGE-1-MARKER" not in system_prompt
    assert "NEW-HISTORY-MARKER" in system_prompt
    assert "OLD-HISTORY-MARKER" not in system_prompt
    assert estimate_tokens(system_prompt) + estimate_tokens(user_query) <= 102_400
