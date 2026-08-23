# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Public Answer Mode, Valid Mode Set, and request-fingerprint canonicalization."""

import pytest

from dlightrag.answer.client_contracts import AnswerRequestContract
from dlightrag.answer.errors import UnsupportedAnswerModeError, UnsupportedResourceCapabilityError
from dlightrag.answer.mode import (
    ModeCapability,
    ModeResource,
    canonical_answer_mode,
    require_supported_mode,
    valid_modes,
)
from dlightrag.answer.runs.execution import AnswerRunRequest
from dlightrag.runtime import answer_run_request_fingerprint


def test_omitted_mode_canonicalizes_to_auto_and_matches_explicit_auto_fingerprint() -> None:
    omitted = AnswerRunRequest(query="q", workspaces=("ws",))
    explicit = AnswerRunRequest(query="q", workspaces=("ws",), mode="auto")
    assert canonical_answer_mode(None) == "auto"
    assert omitted.as_request()["mode"] == "auto"
    assert answer_run_request_fingerprint(omitted.as_request()) == answer_run_request_fingerprint(
        explicit.as_request()
    )
    assert AnswerRequestContract(query="q").mode is None
    assert AnswerRequestContract(query="q", mode="research").mode == "research"


def test_explicit_fast_with_pdf_is_unsupported_answer_mode() -> None:
    valid = valid_modes(
        resources=(ModeResource(role="document"),),
        capability=ModeCapability(query_supports_images=False),
    )
    assert valid == frozenset({"research"})
    with pytest.raises(UnsupportedAnswerModeError) as caught:
        require_supported_mode(requested="fast", valid=valid)
    assert caught.value.error_kind == "unsupported_answer_mode"


def test_image_without_vision_or_inspect_is_unsupported_resource_capability() -> None:
    valid = valid_modes(
        resources=(ModeResource(role="image"),),
        capability=ModeCapability(
            query_supports_images=False,
            inspect_available=False,
        ),
    )
    assert valid == frozenset()
    with pytest.raises(UnsupportedResourceCapabilityError) as caught:
        require_supported_mode(requested="auto", valid=valid)
    assert caught.value.error_kind == "unsupported_resource_capability"


def test_web_search_does_not_remove_fast_from_a_text_only_request() -> None:
    valid = valid_modes(
        resources=(),
        capability=ModeCapability(
            query_supports_images=False,
            inspect_available=False,
            web_search_available=True,
        ),
    )
    assert valid == frozenset({"fast", "research"})
