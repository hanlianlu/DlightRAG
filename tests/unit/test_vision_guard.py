# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the answer-image capability guard."""

from typing import Any, cast

import pytest

from dlightrag.core.answer.capability import AnswerImageCapability, CapabilityStatus
from dlightrag.core.answer.errors import (
    ANSWER_IMAGE_CAPABILITY_UNKNOWN,
    CURRENT_DOCUMENT_PARSE_FAILED,
    CURRENT_IMAGES_UNSUPPORTED,
    AnswerImageError,
    AnswerInputError,
    classify_answer_error,
)
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


def test_answer_image_error_is_an_answer_input_error() -> None:
    error = AnswerImageError("images unavailable", error_kind=CURRENT_IMAGES_UNSUPPORTED)

    assert isinstance(error, AnswerInputError)


def test_classify_answer_error_preserves_generic_input_kind() -> None:
    error = AnswerInputError(
        "Could not parse current document.",
        error_kind=CURRENT_DOCUMENT_PARSE_FAILED,
    )

    assert classify_answer_error(error) == CURRENT_DOCUMENT_PARSE_FAILED

    import dlightrag

    assert dlightrag.AnswerInputError is AnswerInputError
    assert dlightrag.CURRENT_DOCUMENT_PARSE_FAILED == CURRENT_DOCUMENT_PARSE_FAILED


class TestAnswerImageCapabilityGuard:
    def test_raises_when_query_images_and_unsupported(self) -> None:
        with pytest.raises(AnswerImageError) as exc:
            _check_answer_image_capability(
                query_images=cast(Any, ["data:..."]),
                capability=_capability("unsupported"),
            )
        assert exc.value.error_kind == CURRENT_IMAGES_UNSUPPORTED
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

    def test_unknown_fails_closed(self) -> None:
        # Fail-closed: an unconfirmed capability rejects with a clear kind rather
        # than a late provider or transport-budget failure.
        with pytest.raises(AnswerImageError) as exc:
            _check_answer_image_capability(
                query_images=cast(Any, ["data:..."]),
                capability=_capability("unknown"),
            )
        assert exc.value.error_kind == ANSWER_IMAGE_CAPABILITY_UNKNOWN

    def test_unprobed_none_fails_closed(self) -> None:
        with pytest.raises(AnswerImageError) as exc:
            _check_answer_image_capability(
                query_images=cast(Any, ["data:..."]),
                capability=None,
            )
        assert exc.value.error_kind == ANSWER_IMAGE_CAPABILITY_UNKNOWN
