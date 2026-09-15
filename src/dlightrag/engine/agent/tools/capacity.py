# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Fit one model-visible Tool result into the model's remaining input capacity.

A Tool may answer with more text than the model can still read, so the residual
budget is spent here in one place: the typed attachments ride through untouched,
a continuation marker keeps a resumable result honest, and a required suffix the
residual cannot hold fails loudly instead of silently dropping it.
"""

from __future__ import annotations

from dataclasses import replace

from dlightrag.engine.agent.tool_content import ToolTextPart, tool_content_attachments
from dlightrag.engine.agent.tools.contracts import (
    ToolResult,
    ToolResultCapacityError,
)
from dlightrag.engine.ai.tokens import estimate_tokens, truncate_to_estimated_tokens

_TRUNCATION_MARKER = "[tool result truncated to the shared model residual]"


def fit_tool_result(result: ToolResult, *, max_tokens: int) -> ToolResult:
    """Fit result text while preserving typed attachments and continuation."""
    text = result.text_content
    if estimate_tokens(text) <= max_tokens:
        return result
    if max_tokens < 1:
        raise ToolResultCapacityError("tool result has no residual model input capacity")
    protected = result.protected_text.strip()
    protected_tokens = estimate_tokens(protected)
    if protected and protected_tokens > max_tokens:
        raise ToolResultCapacityError("tool result continuation does not fit the model residual")
    fixed = "\n".join(part for part in (_TRUNCATION_MARKER, protected) if part)
    fixed_tokens = estimate_tokens(fixed)
    if fixed_tokens >= max_tokens:
        fitted_text = protected or truncate_to_estimated_tokens(_TRUNCATION_MARKER, max_tokens)
    else:
        body = text
        if protected and body.endswith(result.protected_text):
            body = body[: -len(result.protected_text)].rstrip()
        body_tokens = max_tokens - fixed_tokens
        fitted_text = protected
        while body_tokens >= 0:
            truncated = truncate_to_estimated_tokens(body, body_tokens)
            candidate = "\n".join(part for part in (truncated, fixed) if part)
            if estimate_tokens(candidate) <= max_tokens:
                fitted_text = candidate
                break
            body_tokens -= 1
    attachments = tool_content_attachments(result.parts)
    return replace(
        result,
        parts=(ToolTextPart(fitted_text), *attachments),
    )


__all__ = ["fit_tool_result"]
