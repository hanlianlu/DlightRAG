# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for vision capability guard."""

from __future__ import annotations

from typing import Any, cast

import pytest

from dlightrag.core.servicemanager import (
    _check_vision_support,
    _history_has_images,
    set_vision_supported,
)


class TestHistoryHasImages:
    def test_pure_text_history_returns_false(self) -> None:
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        assert _history_has_images(history) is False

    def test_multimodal_history_returns_true(self) -> None:
        history = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            },
        ]
        assert _history_has_images(history) is True

    def test_mixed_history_returns_true(self) -> None:
        history = [
            {"role": "user", "content": "hello"},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "http://..."}},
                ],
            },
        ]
        assert _history_has_images(history) is True

    def test_empty_history_returns_false(self) -> None:
        assert _history_has_images([]) is False

    def test_none_history_returns_false(self) -> None:
        assert _history_has_images(None) is False


class TestVisionGuard:
    def teardown_method(self) -> None:
        """Reset module-level flag after each test."""
        set_vision_supported(True)  # restore to safe default

    def test_raises_when_query_images_and_no_vision(self) -> None:
        set_vision_supported(False)
        with pytest.raises(ValueError) as exc:
            _check_vision_support(
                query_images=cast(Any, ["data:..."]),
                conversation_history=None,
            )
        assert "does not support image input" in str(exc.value)
        assert "[IMAGES_NOT_SUPPORTED_BY_MODEL]" in str(exc.value)

    def test_passes_when_query_images_and_vision_supported(self) -> None:
        set_vision_supported(True)
        # Should not raise
        _check_vision_support(
            query_images=cast(Any, ["data:..."]),
            conversation_history=None,
        )

    def test_raises_when_history_has_images_but_no_vision(self) -> None:
        set_vision_supported(False)
        with pytest.raises(ValueError) as exc:
            _check_vision_support(
                query_images=None,
                conversation_history=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": "data:..."}},
                        ],
                    },
                ],
            )
        assert "does not support image input" in str(exc.value)

    def test_passes_when_no_images_at_all(self) -> None:
        set_vision_supported(False)
        # No images involved — should not raise
        _check_vision_support(
            query_images=None,
            conversation_history=[{"role": "user", "content": "hello"}],
        )

    def test_passes_when_vision_none_unknown(self) -> None:
        set_vision_supported(None)  # type: ignore[arg-type]
        # unprobed — allow through
        _check_vision_support(
            query_images=cast(Any, ["data:..."]),
            conversation_history=None,
        )
