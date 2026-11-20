# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Residual-capacity fitting of one model-visible Tool result."""

import pytest

from dlightrag.engine.agent.tool_content import ToolResourceAttachmentPart
from dlightrag.engine.agent.tools.capacity import fit_tool_result
from dlightrag.engine.agent.tools.contracts import (
    OBSERVED_TRUNCATED_DETAIL,
    ToolResult,
    ToolResultCapacityError,
)
from dlightrag.engine.ai.tokens import estimate_tokens


def test_a_result_that_already_fits_is_returned_unchanged() -> None:
    result = ToolResult.text("small answer")

    assert fit_tool_result(result, max_tokens=100) is result


def test_a_large_result_is_truncated_under_budget_and_marked() -> None:
    result = ToolResult.text("x" * 4000)

    fitted = fit_tool_result(result, max_tokens=80)

    assert estimate_tokens(fitted.text_content) <= 80
    assert "tool result truncated" in fitted.text_content
    assert fitted.text_content


def test_typed_attachments_survive_fitting() -> None:
    attachment = ToolResourceAttachmentPart(
        resource_id="image-1",
        safe_name="chart.png",
        media_type="image/png",
        content_digest="0" * 64,
        size_bytes=3,
        data=b"png",
    )
    result = ToolResult(parts=(*ToolResult.text("x" * 4000).parts, attachment))

    fitted = fit_tool_result(result, max_tokens=80)

    assert [part for part in fitted.parts if isinstance(part, ToolResourceAttachmentPart)] == [
        attachment
    ]


def test_a_required_continuation_suffix_is_preserved() -> None:
    suffix = "[more text available; cursor=opaque-cursor]"
    result = ToolResult.text(f"{'x' * 4000}\n{suffix}", protected_text=suffix)

    fitted = fit_tool_result(result, max_tokens=80)

    assert fitted.text_content.endswith(suffix)
    assert "tool result truncated" in fitted.text_content
    assert estimate_tokens(fitted.text_content) <= 80


def test_a_continuation_that_cannot_fit_fails_loudly() -> None:
    suffix = "[continuation " + "x" * 4000 + "]"
    result = ToolResult.text(f"body\n{suffix}", protected_text=suffix)

    with pytest.raises(ToolResultCapacityError, match="continuation"):
        fit_tool_result(result, max_tokens=5)


def test_no_residual_capacity_fails_loudly() -> None:
    with pytest.raises(ToolResultCapacityError, match="residual model input capacity"):
        fit_tool_result(ToolResult.text("x" * 4000), max_tokens=0)


def test_a_cut_is_recorded_on_the_result_and_not_only_described_in_the_text() -> None:
    """The Runtime's own truncation is a fact a reader can check, not a marker to parse.

    A result that describes work it did not finish is the defect this record exists to
    catch, so a viewer or an audit sees the cut without searching the model-visible text
    for a marker.
    """
    result = ToolResult.text("x" * 4000)

    fitted = fit_tool_result(result, max_tokens=80)

    assert fitted.observed_truncated is True
    assert fitted.details is not None
    assert fitted.details[OBSERVED_TRUNCATED_DETAIL] is True


def test_a_result_that_fits_records_nothing_and_is_returned_unchanged() -> None:
    result = ToolResult.text("small answer", details={"pages": 1})

    fitted = fit_tool_result(result, max_tokens=100)

    assert fitted is result
    assert fitted.observed_truncated is False
    assert fitted.details == {"pages": 1}


def test_a_cut_keeps_the_details_the_tool_reported() -> None:
    result = ToolResult.text("x" * 4000, details={"rows": 12, "truncated": False})

    fitted = fit_tool_result(result, max_tokens=80)

    assert fitted.details == {"rows": 12, "truncated": False, OBSERVED_TRUNCATED_DETAIL: True}
    # A tool-level claim is not overwritten by the Runtime's own: they are different facts.
    assert fitted.details is not None and fitted.details["truncated"] is False
