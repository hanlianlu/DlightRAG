# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""ChildContextSnapshot pins tool occurrences, never their transport pixels."""

import hashlib
import json
from dataclasses import replace
from typing import Any

import pytest

from dlightrag.engine.agent.session.ids import EntryId, SessionId
from dlightrag.engine.agent.tool_content import (
    ToolResourceAttachmentPart,
    VisualSource,
    tool_content_message_fields,
)
from dlightrag.engine.answer.attachment_replay import AttachmentOccurrence
from dlightrag.engine.answer.tools.subagents import ChildContextSnapshot


def snapshot_values() -> dict[str, Any]:
    pixels = b"synthetic-transport-only"
    part = ToolResourceAttachmentPart(
        resource_id="derivative",
        safe_name="page-2.png",
        media_type="image/png",
        content_digest=hashlib.sha256(pixels).hexdigest(),
        size_bytes=len(pixels),
        data=pixels,
        source=VisualSource(resource_id="original-pdf", kind="pdf_page", page=2),
    )
    occurrence = AttachmentOccurrence(EntryId.new().value, 1, replace(part, data=b""))
    return dict(
        parent_session_id=SessionId.new(),
        parent_entry_id=EntryId.new(),
        depth=0,
        messages=[{"role": "tool", "tool_call_id": "view", **tool_content_message_fields((part,))}],
        attachment_occurrences=(occurrence,),
    )


def test_child_context_snapshot_removes_tool_pixels_and_preserves_exact_occurrence():
    values = snapshot_values()
    snapshot = ChildContextSnapshot.from_values(**values)
    assert "data_url" not in snapshot.messages_json and "base64" not in snapshot.messages_json
    assert "data_url" in values["messages"][0]["attachments"][0]  # no caller mutation
    assert snapshot.messages[0]["attachments"][0]["source"] == {
        "resource_id": "original-pdf",
        "kind": "pdf_page",
        "page": 2,
        "handle_id": None,
        "anchor": None,
        "path": None,
        "overview": False,
        "origin_part": None,
    }
    occurrence = snapshot.attachment_occurrences[0]
    assert AttachmentOccurrence.from_payload(occurrence.canonical_payload()) == occurrence
    with pytest.raises(ValueError, match="byte-free"):
        replace(snapshot, messages_json=json.dumps(values["messages"]))


@pytest.mark.parametrize(
    "fault", ["unpinned", "source", "digest", "inline", "unknown", "duplicate"]
)
def test_child_context_snapshot_rejects_unpinned_or_mismatched_tool_attachments(fault):
    values = snapshot_values()
    occurrence = values["attachment_occurrences"][0]
    if fault == "unpinned":
        values["attachment_occurrences"] = ()
    elif fault == "source":
        values["attachment_occurrences"] = (
            replace(
                occurrence,
                attachment=replace(
                    occurrence.attachment,
                    source=replace(occurrence.attachment.source, page=3),
                ),
            ),
        )
    elif fault == "digest":
        values["attachment_occurrences"] = (
            replace(
                occurrence,
                attachment=replace(
                    occurrence.attachment,
                    content_digest="0" * 64,
                ),
            ),
        )
    elif fault == "inline":
        values["attachment_occurrences"] = (
            replace(
                occurrence,
                attachment=replace(
                    occurrence.attachment,
                    data=b"forbidden",
                ),
            ),
        )
    elif fault == "unknown":
        values["messages"][0]["attachments"][0]["base64"] = "forbidden"
    else:
        values["attachment_occurrences"] = (occurrence, occurrence)
        values["messages"] *= 2
    with pytest.raises(ValueError, match="occurrence"):
        ChildContextSnapshot.from_values(**values)


@pytest.mark.parametrize("fault", ["extra", "negative", "bool", "inline"])
def test_attachment_occurrence_decoder_is_closed(fault):
    payload = snapshot_values()["attachment_occurrences"][0].canonical_payload()
    if fault == "extra":
        payload["other_run"] = "forged"
    elif fault == "negative":
        payload["part_index"] = -1
    elif fault == "bool":
        payload["part_index"] = True
    else:
        payload["attachment"]["data_url"] = "data:image/png;base64,forbidden"
    with pytest.raises(ValueError):
        AttachmentOccurrence.from_payload(payload)
