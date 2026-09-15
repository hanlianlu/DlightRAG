# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Residual-capacity fitting of one model-visible Tool result."""

import pytest

from dlightrag.engine.agent.tool_content import ToolResourceAttachmentPart
from dlightrag.engine.agent.tools import ToolResult, ToolResultCapacityError, fit_tool_result
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
