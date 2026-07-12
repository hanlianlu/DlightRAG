# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for vision capability guard."""

from typing import Any, cast

import pytest

from dlightrag.core.servicemanager import _check_vision_support


class TestVisionGuard:
    def test_raises_when_query_images_and_no_vision(self) -> None:
        with pytest.raises(ValueError) as exc:
            _check_vision_support(
                query_images=cast(Any, ["data:..."]),
                supports_vision=False,
            )
        assert "does not support image input" in str(exc.value)
        assert "[IMAGES_NOT_SUPPORTED_BY_MODEL]" in str(exc.value)

    def test_passes_when_query_images_and_vision_supported(self) -> None:
        # Should not raise
        _check_vision_support(
            query_images=cast(Any, ["data:..."]),
            supports_vision=True,
        )

    def test_passes_when_no_images_at_all(self) -> None:
        # No images involved — should not raise
        _check_vision_support(
            query_images=None,
            supports_vision=False,
        )

    def test_passes_when_vision_none_unknown(self) -> None:
        # unprobed — allow through
        _check_vision_support(
            query_images=cast(Any, ["data:..."]),
            supports_vision=None,
        )
