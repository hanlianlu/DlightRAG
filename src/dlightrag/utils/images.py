# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Image helpers for model payloads and visual asset serving."""

from __future__ import annotations

import base64
import io
import mimetypes
from pathlib import Path
from typing import Any

from PIL import Image

_PILLOW_TO_MIME: dict[str, str] = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "GIF": "image/gif",
    "BMP": "image/bmp",
    "TIFF": "image/tiff",
}
_PASSTHROUGH_FORMATS = {"JPEG", "PNG", "WEBP"}
_DATA_URI_PREFIX = "data:"


def split_data_uri(value: str) -> tuple[str | None, str]:
    """Return ``(mime, base64_payload)`` for a data URI or raw base64 string."""
    text = value.strip()
    if not text.startswith(_DATA_URI_PREFIX):
        return None, text
    header, sep, payload = text.partition(",")
    if not sep:
        return None, text
    mime = header.removeprefix(_DATA_URI_PREFIX).split(";", 1)[0] or None
    return mime, payload


def decode_image_base64(value: str) -> tuple[bytes, str | None]:
    """Decode raw base64 or a base64 data URI."""
    declared_mime, payload = split_data_uri(value)
    return base64.b64decode(payload), declared_mime


def detect_image_mime(raw: bytes, *, fallback: str | None = None) -> str:
    """Detect image MIME from bytes, falling back to a safe image type."""
    try:
        with Image.open(io.BytesIO(raw)) as image:
            mime = _PILLOW_TO_MIME.get(image.format or "")
            if mime:
                return mime
    except Exception:
        pass
    if fallback and fallback.startswith("image/"):
        return fallback
    return "image/png"


def detect_image_mime_type(path: str | Path) -> str:
    """Detect MIME using Pillow first, then filename suffixes."""
    file_path = Path(path)
    try:
        with Image.open(file_path) as image:
            mime = _PILLOW_TO_MIME.get(image.format or "")
            if mime:
                return mime
    except Exception:
        pass
    guessed = mimetypes.guess_type(str(file_path))[0]
    return guessed if guessed and guessed.startswith("image/") else "image/png"


def image_bytes_to_data_uri(raw: bytes, *, fallback_mime: str | None = None) -> str:
    """Return a model-safe data URI from image bytes."""
    try:
        with Image.open(io.BytesIO(raw)) as image:
            fmt = image.format or ""
            mime = _PILLOW_TO_MIME.get(fmt)
            if fmt in _PASSTHROUGH_FORMATS and mime:
                return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"

            converted = image.convert("RGB")
            buf = io.BytesIO()
            converted.save(buf, format="JPEG", quality=95, optimize=True)
            return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
    except Exception:
        mime = (
            fallback_mime if fallback_mime and fallback_mime.startswith("image/") else "image/png"
        )
        return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def image_data_uri(value: str) -> str:
    """Build a model-safe image data URI from raw base64 or an existing data URI."""
    try:
        raw, declared_mime = decode_image_base64(value)
    except Exception:
        return value if value.startswith(_DATA_URI_PREFIX) else f"data:image/png;base64,{value}"
    return image_bytes_to_data_uri(raw, fallback_mime=declared_mime)


def bounded_image_data_uri(
    value: str,
    *,
    max_bytes: int,
    max_px: int,
    min_px: int,
    quality: int,
    min_quality: int,
) -> tuple[str, int] | None:
    """Return a model image data URI bounded for answer-model payloads.

    ``value`` can be raw base64 or a base64 data URI. Non-image or undecodable
    payloads return ``None``. Already-budgeted JPEG/PNG/WebP images are passed
    through unchanged. Images that cannot fit without dropping below the quality
    or long-edge floor are skipped instead of being over-compressed.
    """
    try:
        raw, _ = decode_image_base64(value)
    except Exception:
        return None

    max_bytes = max(1, int(max_bytes))
    max_px = max(1, int(max_px))
    quality = min(95, max(1, int(quality)))
    min_px = min(max_px, max(1, int(min_px)))
    min_quality = min(quality, max(1, int(min_quality)))

    try:
        with Image.open(io.BytesIO(raw)) as original:
            original_format = original.format or ""
            original_mime = _PILLOW_TO_MIME.get(original_format)
            if (
                original_format in _PASSTHROUGH_FORMATS
                and original_mime
                and len(raw) <= max_bytes
                and max(original.size) <= max_px
            ):
                uri = f"data:{original_mime};base64,{base64.b64encode(raw).decode('ascii')}"
                return uri, len(raw)

            image = original.convert("RGB")
            image.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)

            current = image
            effective_min_px = min(min_px, max(current.size))
            qualities = _quality_steps(quality, min_quality)
            for _ in range(10):
                for current_quality in qualities:
                    buf = io.BytesIO()
                    current.save(buf, format="JPEG", quality=current_quality, optimize=True)
                    payload = buf.getvalue()
                    if len(payload) <= max_bytes:
                        uri = f"data:image/jpeg;base64,{base64.b64encode(payload).decode('ascii')}"
                        return uri, len(payload)

                long_edge = max(current.size)
                if long_edge <= effective_min_px:
                    break
                next_long_edge = max(effective_min_px, int(long_edge * 0.85))
                scale = next_long_edge / long_edge
                next_size = (
                    max(1, int(current.width * scale)),
                    max(1, int(current.height * scale)),
                )
                if next_size == current.size:
                    break
                current = current.resize(next_size, Image.Resampling.LANCZOS)
    except Exception:
        return None
    return None


def _quality_steps(quality: int, min_quality: int) -> list[int]:
    """Return descending JPEG qualities without crossing the configured floor."""
    values = [quality]
    current = quality
    while current > min_quality:
        current = max(min_quality, current - 8)
        values.append(current)
    return values


def thumbnail_bytes(
    raw: bytes, *, max_px: int, output_mime: str | None = None
) -> tuple[bytes, str]:
    """Return thumbnail bytes and media type for browser serving."""
    max_px = max(1, int(max_px))
    with Image.open(io.BytesIO(raw)) as original:
        image = original.copy()
        image.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
        mime = (
            output_mime
            if output_mime and output_mime.startswith("image/")
            else detect_image_mime(raw)
        )
        fmt = "JPEG" if mime == "image/jpeg" else "PNG"
        if fmt == "JPEG" and image.mode != "RGB":
            image = image.convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format=fmt)
        return buf.getvalue(), "image/jpeg" if fmt == "JPEG" else "image/png"


def image_url_block(value: str | dict[str, Any]) -> dict[str, Any] | None:
    """Normalize a URL/data/base64 payload to an OpenAI-style image block."""
    if isinstance(value, dict):
        return value if value.get("type") == "image_url" else None
    text = value.strip()
    if not text:
        return None
    if text.startswith(("http://", "https://", _DATA_URI_PREFIX)):
        url = text if text.startswith(("http://", "https://")) else image_data_uri(text)
    else:
        url = image_data_uri(text)
    return {"type": "image_url", "image_url": {"url": url}}


__all__ = [
    "bounded_image_data_uri",
    "decode_image_base64",
    "detect_image_mime",
    "detect_image_mime_type",
    "image_bytes_to_data_uri",
    "image_data_uri",
    "image_url_block",
    "split_data_uri",
    "thumbnail_bytes",
]
