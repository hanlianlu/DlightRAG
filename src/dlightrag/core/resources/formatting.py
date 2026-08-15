# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Canonical model-visible formatting for bounded resource reads."""

from dlightrag.core.resources.models import (
    ResourceReadResult,
    TextWindowLocator,
    VisualHandle,
)


def format_resource_read(result: ResourceReadResult) -> str:
    parts: list[str] = []
    if result.locator is not None:
        parts.append(f"[{_describe_text_locator(result.locator)}]")
    parts.append(result.content)
    if result.visual_handles:
        parts.append(f"[visual handles: {_describe_handles(result.visual_handles)}]")
    if continuation := resource_read_continuation(result):
        parts.append(continuation)
    return "\n".join(parts)


def resource_read_continuation(result: ResourceReadResult) -> str:
    if result.has_more and result.next_cursor:
        return f"[more text available; cursor={result.next_cursor}]"
    return ""


def _describe_text_locator(locator: TextWindowLocator) -> str:
    if locator.char_start is not None:
        return f"lines {locator.start}-{locator.end}, chars {locator.char_start}-{locator.char_end}"
    return f"lines {locator.start}-{locator.end}"


def _describe_handles(handles: tuple[VisualHandle, ...]) -> str:
    return ", ".join(
        handle.handle_id + (f" ({handle.label})" if handle.label else "") for handle in handles
    )


__all__ = ["format_resource_read", "resource_read_continuation"]
