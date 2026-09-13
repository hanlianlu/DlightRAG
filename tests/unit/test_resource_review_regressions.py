# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Resource cursor domain separation and manifest classification after local cleanup."""

import base64
import hashlib
import hmac
import struct

import pytest

from dlightrag.engine.answer.research.context import _resource_manifest_context
from dlightrag.engine.answer.resources.converters import conversion_format
from dlightrag.engine.answer.resources.models import ResourceCursorError, ResourceManifestEntry
from dlightrag.engine.answer.resources.registry import ResourceRegistry, _CursorState


def test_shared_cursor_codec_preserves_distinct_wire_domains_and_payloads():
    registry = ResourceRegistry(cursor_secret=b"generated-cursor-secret")
    state = _CursorState("source", 1000, 2, 300, 200)
    read = registry._mint_cursor(state)
    assert registry._resolve_cursor(read, resource_id="source") == state
    payload = struct.pack(">BIIII", 1, 1000, 2, 300, 200)
    # Read payload version is independent of the visual cursor's four-byte index.
    from dlightrag.engine.answer.resources.registry import _CURSOR_VERSION

    payload = bytes([_CURSOR_VERSION]) + payload[1:]
    signature = hmac.new(b"generated-cursor-secret", b"source|" + payload, hashlib.sha256).digest()[
        :8
    ]
    assert read == base64.urlsafe_b64encode(payload + signature).rstrip(b"=").decode()
    visual = registry.visual_cursor("source", 2, "overview")
    assert registry.resolve_visual_cursor(visual, "source", "overview") == 2
    for other_kind in ("inventory", "text"):
        with pytest.raises(ResourceCursorError):
            registry.resolve_visual_cursor(visual, "source", other_kind)
    with pytest.raises(ResourceCursorError):
        registry._resolve_cursor(visual, resource_id="source")
    with pytest.raises(ResourceCursorError):
        registry.resolve_visual_cursor(read, "source", "overview")
    with pytest.raises(ResourceCursorError):
        registry._resolve_cursor(read, resource_id="other")
    with pytest.raises(ResourceCursorError):
        registry.resolve_visual_cursor(visual, "other", "overview")


@pytest.mark.parametrize("damage", ["truncate", "append", "signature", "unicode", "padding"])
def test_shared_cursor_codec_rejects_malformed_or_tampered_tokens(damage):
    registry = ResourceRegistry(cursor_secret=b"generated-cursor-secret")
    read = registry._mint_cursor(_CursorState("source", 1000, 2, 300, 200))
    visual = registry.visual_cursor("source", 2, "overview")

    def damage_token(token):
        if damage == "truncate":
            return token[:-2]
        if damage == "append":
            return token + "AA"
        if damage == "signature":
            return token[:-3] + ("A" if token[-3] != "A" else "B") + token[-2:]
        if damage == "unicode":
            return token[:-2] + "非"
        return token + "="

    with pytest.raises(ResourceCursorError):
        registry._resolve_cursor(damage_token(read), resource_id="source")
    with pytest.raises(ResourceCursorError):
        registry.resolve_visual_cursor(damage_token(visual), "source", "overview")


@pytest.mark.parametrize(
    "filename,mime,format_name,guidance",
    [
        ("photo.JPG", None, None, "image; view"),
        ("download", "image/png", None, "image; view"),
        ("report.PDF", None, "pdf", "PDF; read text or view physical pages"),
        (
            "download",
            "application/pdf; charset=binary",
            "pdf",
            "PDF; read text or view physical pages",
        ),
        (
            "download",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "docx",
            "DOCX; read extracted text",
        ),
        ("data.xlsx", "application/pdf", "xlsx", "XLSX; read extracted text"),
        ("slides.pptx", None, "pptx", "PPTX; read extracted text"),
        ("unknown", None, None, "type verified on acquisition"),
    ],
)
def test_resource_manifest_uses_existing_image_and_conversion_classification(
    filename, mime, format_name, guidance
):
    assert conversion_format(filename, mime) == format_name
    entry = ResourceManifestEntry("source", filename, mime, "bytes", 20)
    assert guidance in _resource_manifest_context((entry,))
