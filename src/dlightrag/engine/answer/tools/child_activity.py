# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""What a parent reads of a running child: its newest messages, bounded.

A tool result comes from the open web, so its excerpt is short on purpose and a part
that is not text is named, never shown. Every excerpt is one line, so no message can
pass for the start of another.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from dlightrag.engine.agent.session.effects import canonical_json

ASSISTANT_TEXT_LIMIT = 1200
CALL_ARGUMENTS_LIMIT = 160
TOOL_RESULT_LIMIT = 240
USER_TEXT_LIMIT = 400
RESULT_LIMIT = 8000


def render_child_activity(
    child_id: str,
    status: str,
    page: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    after: int | None,
) -> str:
    """Render one page of a child's transcript, oldest message first.

    ``page`` holds one message beyond ``limit`` when more exist: the oldest of a page
    that ends at the newest, the last of a page that follows ``after``. When the text
    would pass ``RESULT_LIMIT``, messages are left out until it fits: the oldest of the
    newest page, and the newest of a page that follows ``after``, whose cursor then stops
    at the last message shown, so the next call returns the rest and none is skipped.
    """
    more = len(page) > limit
    shown = page[:limit] if after is not None else page[-limit:]
    entries = [_entry(message) for message in shown]
    text = _page(child_id, status, shown, entries, more=more, after=after)
    for left_out in range(1, len(entries)):
        if len(text) <= RESULT_LIMIT:
            break
        keep = slice(None, -left_out) if after is not None else slice(left_out, None)
        text = _page(child_id, status, shown[keep], entries[keep], more=True, after=after)
    return text


def _page(
    child_id: str,
    status: str,
    kept: Sequence[Mapping[str, Any]],
    entries: Sequence[str],
    *,
    more: bool,
    after: int | None,
) -> str:
    if kept:
        oldest, newest = kept[0]["sequence"], kept[-1]["sequence"]
        span = f"#{oldest}" if oldest == newest else f"#{oldest}–#{newest}"
        summary = f"{len(kept)} message{'s' if len(kept) != 1 else ''}, {span}"
    else:
        summary = "no messages yet" if after is None else f"no messages after #{after}"
    lines = [f"Child {child_id} [{status}] — {summary}"]
    if more and after is None:
        lines.append("Earlier messages not shown.")
    if status != "running":
        lines.append("The child has settled; subagent_status returns its result.")
    lines.extend(entries)
    if more and after is not None:
        last = kept[-1]["sequence"]
        lines.append(f"More after #{last}: call again with after={last}.")
    return "\n".join(lines)


def _entry(message: Mapping[str, Any]) -> str:
    number = f"#{message['sequence']}"
    role = message.get("role")
    if role == "assistant":
        lines = [_line(f"{number} assistant", _cut(message.get("content"), ASSISTANT_TEXT_LIMIT))]
        lines.extend(_call(call) for call in message.get("tool_calls") or ())
        return "\n".join(lines)
    if role == "tool":
        tool = f"{message.get('name')} error" if message.get("is_error") else message.get("name")
        head = f"{number} tool {tool}"
        parts = [_cut(message.get("content"), TOOL_RESULT_LIMIT)]
        parts.extend(_attachment(item) for item in message.get("attachments") or ())
        return _line(head, " ".join(part for part in parts if part))
    return _line(f"{number} user", _cut(message.get("content"), USER_TEXT_LIMIT))


def _call(call: Mapping[str, Any]) -> str:
    arguments = _cut(canonical_json(call.get("arguments") or {}), CALL_ARGUMENTS_LIMIT)
    return f"  {call.get('name')}({arguments})"


def _attachment(attachment: Mapping[str, Any]) -> str:
    return "[image]" if str(attachment.get("media_type") or "").startswith("image/") else "[file]"


def _line(head: str, body: str) -> str:
    return f"{head}: {body}" if body else head


def _cut(value: object, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


__all__ = ["render_child_activity"]
