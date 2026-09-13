# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""DOCX structured occurrences, independent of Markdown placement or engine selection."""

from __future__ import annotations

import hashlib
import io
import posixpath
import zipfile
from collections import Counter
from typing import TYPE_CHECKING
from xml.etree.ElementTree import ParseError

from defusedxml.common import DefusedXmlException
from defusedxml.ElementTree import fromstring

from dlightrag.engine.ai.media import verify_web_image_bytes

if TYPE_CHECKING:
    from anydoc import Block, Document, Inline

_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_PACKAGE_REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_IMAGE_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
_INCOMPLETE_VISUALS = (
    "Known incomplete visual extraction: external, unavailable, unsupported or unmapped source "
    "visuals remain; available handles are not whole-document coverage."
)


class AssetBindingError(ValueError):
    """A candidate asset cannot be bound to exact admitted package bytes."""


def docx_asset_occurrences(
    document: Document, content: bytes
) -> tuple[list[tuple[str, str, bytes]], str | None]:
    """Return verified (package part, MIME, bytes) per occurrence, plus omissions.

    Notes, nested links/lists and table origin cells are real typed containers.
    Covered table cells do not introduce a second occurrence. No package path is
    opened on disk and no external relationship is fetched.
    """
    from anydoc import Inline

    occurrences: list[tuple[str, str, bytes]] = []
    missing = False
    assets = {asset.id: asset for asset in document.assets}
    if len(assets) != len(document.assets):
        raise AssetBindingError("duplicate candidate asset identity")
    used: set[int] = set()
    validated: dict[int, tuple[str, str, bytes]] = {}
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise AssetBindingError("ambiguous package membership")
        for identity, asset in assets.items():
            part = asset.origin_part
            if (
                not part
                or part.startswith("/")
                or any(c in part for c in "\\%:#?")
                or posixpath.normpath(part) != part
                or part.startswith("../")
                or part not in names
            ):
                raise AssetBindingError("candidate asset has no exact package membership")
            data = asset.data
            if hashlib.sha256(archive.read(part)).digest() != hashlib.sha256(data).digest():
                raise AssetBindingError("candidate asset bytes differ from admitted source")
            try:
                media = verify_web_image_bytes(data)
            except ValueError as exc:
                raise AssetBindingError("asset image verification refused") from exc
            if media != asset.media_type:
                raise AssetBindingError("candidate asset media differs from verified bytes")
            validated[identity] = (part, media, data)

        stack: list[Block | Inline] = list(reversed(document.blocks))
        for note in reversed(document.notes):
            stack.extend(reversed(note.blocks))
        while stack:
            node = stack.pop()
            if isinstance(node, Inline) and node.kind == "image":
                source = node.source
                if source is None or source.kind != "asset":
                    missing = True
                    continue
                if source.asset_id not in assets:
                    missing = True
                    continue
                used.add(source.asset_id)
                visual = validated[source.asset_id]
                occurrences.append(visual)
            elif node.kind in {"paragraph", "heading", "link"}:
                stack.extend(reversed(node.content or ()))
            elif not isinstance(node, Inline) and node.kind == "block_quote":
                stack.extend(reversed(node.blocks or ()))
            elif not isinstance(node, Inline) and node.kind == "list":
                listing = node.list
                if listing is not None:
                    for item in reversed(listing.items):
                        stack.extend(reversed(item.blocks))
            elif not isinstance(node, Inline) and node.kind == "table":
                table = node.table
                if table is not None:
                    for row in reversed(table.grid):
                        for slot in reversed(row):
                            if slot.kind == "origin" and slot.cell is not None:
                                stack.extend(reversed(slot.cell.blocks))
            elif node.kind not in {
                "text",
                "anchor",
                "note_ref",
                "line_break",
                "math",
                "checkbox",
                "code_block",
                "rule",
            }:
                missing = True
        # Structured success is not proof of full image coverage. This bounded
        # package audit detects dropped source references (including header/VML
        # images). It never selects an engine or manufactures an occurrence.
        expected, source_missing = _source_image_references(archive)
        missing |= source_missing or expected != Counter(part for part, _, _ in occurrences)
        missing |= used != set(assets)
    return occurrences, (_INCOMPLETE_VISUALS if missing else None)


def bind_docx_fallback_images(
    content: bytes, images: list[tuple[str, bytes]]
) -> tuple[list[str | None], str | None]:
    """Validate incumbent occurrences too; fallback is not a coverage exemption.

    Markdown supplies occurrence order but not package parts. Bind only a unique
    matching part; byte-identical assets in multiple parts remain explicitly
    unmapped rather than assigning invented provenance.
    """
    parts: list[str | None] = []
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        expected, missing = _source_image_references(archive)
        remaining = expected.copy()
        digests = {part: hashlib.sha256(archive.read(part)).digest() for part in expected}
        for media, data in images:
            try:
                verified = verify_web_image_bytes(data)
            except ValueError as exc:
                raise AssetBindingError("fallback image verification refused") from exc
            if verified != media:
                raise AssetBindingError("fallback image media mismatch")
            matches = [
                part for part, digest in digests.items() if digest == hashlib.sha256(data).digest()
            ]
            if not matches:
                raise AssetBindingError("fallback image has no source membership")
            part = matches[0] if len(matches) == 1 else None
            parts.append(part)
            if part is None or remaining[part] <= 0:
                missing = True
            else:
                remaining[part] -= 1
        missing |= any(remaining.values())
    return parts, _INCOMPLETE_VISUALS if missing else None


def _source_image_references(archive: zipfile.ZipFile) -> tuple[Counter[str], bool]:
    expected: Counter[str] = Counter()
    missing = False
    names = set(archive.namelist())
    try:
        for name in sorted(names):
            if not name.startswith("word/") or not name.endswith(".xml"):
                continue
            directory, filename = posixpath.split(name)
            relname = f"{directory}/_rels/{filename}.rels"
            relationships: dict[str, tuple[str, str, str]] = {}
            if relname in names:
                for rel in fromstring(archive.read(relname), forbid_dtd=True):
                    if rel.tag == _PACKAGE_REL + "Relationship":
                        relationships[rel.get("Id", "")] = (
                            rel.get("Type", ""),
                            rel.get("Target", ""),
                            rel.get("TargetMode", "Internal"),
                        )
            for node in fromstring(archive.read(name), forbid_dtd=True).iter():
                local = node.tag.rsplit("}", 1)[-1]
                if local in {"OLEObject", "chart", "altChunk", "contentPart"}:
                    missing = True
                if local not in {"blip", "imagedata"}:
                    continue
                refs = [node.get(_REL + key) for key in ("embed", "link", "id")]
                refs = [ref for ref in refs if ref]
                if not refs:
                    missing = True
                for ref in refs:
                    kind, target, mode = relationships.get(ref, ("", "", ""))
                    if (
                        kind != _IMAGE_REL
                        or mode != "Internal"
                        or any(c in target for c in "\\%:#?")
                    ):
                        missing = True
                        continue
                    part = posixpath.normpath(posixpath.join(directory, target))
                    if part not in names:
                        missing = True
                    else:
                        expected[part] += 1
    except ParseError, DefusedXmlException:
        missing = True
    return expected, missing
