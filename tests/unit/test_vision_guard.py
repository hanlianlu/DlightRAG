# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the answer-image capability guard."""

from typing import Any, cast

import pytest

from dlightrag.core.answer_capability import AnswerImageCapability, CapabilityStatus
from dlightrag.core.servicemanager import _check_answer_image_capability


def _capability(status: CapabilityStatus) -> AnswerImageCapability:
    return AnswerImageCapability(
        status=status,
        configured_ceiling=8,
        effective_max_images=8 if status == "supported" else 0,
        provider="test",
        base_url=None,
        model="m",
        failure_kind=None,
    )


class TestAnswerImageCapabilityGuard:
    def test_raises_when_query_images_and_unsupported(self) -> None:
        with pytest.raises(ValueError) as exc:
            _check_answer_image_capability(
                query_images=cast(Any, ["data:..."]),
                capability=_capability("unsupported"),
            )
        assert "does not support image input" in str(exc.value)
        assert "[IMAGES_NOT_SUPPORTED_BY_MODEL]" in str(exc.value)

    def test_passes_when_query_images_and_supported(self) -> None:
        _check_answer_image_capability(
            query_images=cast(Any, ["data:..."]),
            capability=_capability("supported"),
        )

    def test_passes_when_no_images_at_all(self) -> None:
        _check_answer_image_capability(
            query_images=None,
            capability=_capability("unsupported"),
        )

    def test_passes_when_capability_unknown(self) -> None:
        # Unknown allows through; the transport budget / provider surface any
        # deeper failure rather than a false boundary rejection.
        _check_answer_image_capability(
            query_images=cast(Any, ["data:..."]),
            capability=_capability("unknown"),
        )

    def test_passes_when_capability_unprobed_none(self) -> None:
        _check_answer_image_capability(
            query_images=cast(Any, ["data:..."]),
            capability=None,
        )
