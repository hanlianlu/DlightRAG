# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for multimodal conversation history in QueryPlanner."""

from __future__ import annotations

from dlightrag.core.query_planner import _convert_history_to_text


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
