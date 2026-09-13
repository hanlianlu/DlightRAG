# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Adopted conversion views stored through existing owner-scoped resource effects."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from dlightrag.engine.agent.tool_content import VisualSource
from dlightrag.engine.agent.tools import ResourceAttachmentBytes
from dlightrag.engine.answer.resources.converters import ExtractedVisual


@dataclass(frozen=True)
class ConversionSnapshot:
    resource_id: str
    input_digest: str
    text: str
    visuals: tuple[ExtractedVisual, ...]
    extraction_status: str
    converter: str
    converter_version: str
    fallback_reason: str | None = None
    known_ocr_pages: tuple[int, ...] = ()
    known_page_count: int | None = None
    note: str | None = None

    def effects(self) -> tuple[ResourceAttachmentBytes, ...]:
        assets = []
        effects = []
        for asset in self.visuals:
            digest = hashlib.sha256(asset.data).hexdigest()
            source = VisualSource(
                resource_id=self.resource_id,
                kind="embedded_image",
                handle_id=asset.handle_id,
                anchor=asset.anchor,
                origin_part=asset.origin_part,
            )
            assets.append(
                {
                    "resource_id": asset.handle_id,
                    "anchor": asset.anchor,
                    "origin_part": asset.origin_part,
                    "media_type": asset.media_type,
                    "digest": digest,
                }
            )
            effects.append(
                ResourceAttachmentBytes(
                    resource_id=asset.handle_id,
                    filename=asset.handle_id,
                    mime_type=asset.media_type,
                    source_locator=self.resource_id,
                    content=asset.data,
                    resource_kind="conversion_asset",
                    source=source,
                )
            )
        payload = {
            "resource_id": self.resource_id,
            "input_digest": self.input_digest,
            "text": self.text,
            "output_digest": hashlib.sha256(self.text.encode()).hexdigest(),
            "assets": assets,
            "extraction_status": self.extraction_status,
            "converter": self.converter,
            "converter_version": self.converter_version,
            "fallback_reason": self.fallback_reason,
            "known_ocr_pages": self.known_ocr_pages,
            "known_page_count": self.known_page_count,
            "note": self.note,
        }
        encoded = json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
        effects.append(
            ResourceAttachmentBytes(
                resource_id=f"{self.resource_id}-conversion",
                filename="conversion.json",
                mime_type="application/json",
                source_locator=self.resource_id,
                content=encoded,
                resource_kind="conversion_snapshot",
            )
        )
        return tuple(effects)

    @classmethod
    def restore(cls, data: bytes, assets: dict[str, bytes]) -> ConversionSnapshot:
        raw = json.loads(data)
        text = raw["text"]
        if hashlib.sha256(text.encode()).hexdigest() != raw["output_digest"]:
            raise ValueError("conversion text digest mismatch")
        visuals = []
        for asset in raw["assets"]:
            content = assets[asset["resource_id"]]
            if hashlib.sha256(content).hexdigest() != asset["digest"]:
                raise ValueError("conversion asset digest mismatch")
            visuals.append(
                ExtractedVisual(
                    handle_id=asset["resource_id"],
                    anchor=asset["anchor"],
                    origin_part=asset["origin_part"],
                    media_type=asset["media_type"],
                    data=content,
                )
            )
        return cls(
            resource_id=raw["resource_id"],
            input_digest=raw["input_digest"],
            text=text,
            visuals=tuple(visuals),
            extraction_status=raw["extraction_status"],
            converter=raw["converter"],
            converter_version=raw["converter_version"],
            fallback_reason=raw["fallback_reason"],
            known_ocr_pages=tuple(raw["known_ocr_pages"]),
            known_page_count=raw["known_page_count"],
            note=raw["note"],
        )
