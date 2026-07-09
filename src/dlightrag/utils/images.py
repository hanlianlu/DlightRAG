# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Image helpers for model payloads and visual asset serving."""

import base64
import io
import logging
import mimetypes
from pathlib import Path
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

# Pillow's decompression-bomb guard defaults to ~89.5MP (warn) / ~179MP (raise).
# DlightRAG ingests large document scans, so raise the decode ceiling: images up
# to 250MP decode cleanly, 250-500MP still decode (with a warning), and only truly
# absurd gigapixel bombs are rejected — and any over-limit image is caught and
# skipped per-image at ingest rather than failing the whole document. A 250MP RGB
# decode is ~750MB (RGBA ~1GB), comfortable on DlightRAG's 32GB+ target hosts.
MAX_DECODE_IMAGE_PIXELS = 250_000_000
Image.MAX_IMAGE_PIXELS = MAX_DECODE_IMAGE_PIXELS

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
_IMAGE_DETAIL_VALUES = frozenset({"auto", "low", "high"})

# Embedding image bounds — conservative and provider-agnostic. Voyage multimodal
# caps each image at 16M pixels / 20MB; other multimodal embedders are similar or
# looser. Staying under a 15M-pixel ceiling keeps a JPEG q90 payload to a few MB,
# so the byte budget is rarely the binding constraint — the pixel cap does the work.
EMBED_IMAGE_MAX_PX = 4096
EMBED_IMAGE_MAX_PIXELS = 15_000_000
EMBED_IMAGE_MAX_BYTES = 14 * 1024 * 1024
EMBED_IMAGE_QUALITY = 90
EMBED_IMAGE_MIN_QUALITY = 70
EMBED_IMAGE_MIN_PX = 256


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
        logger.debug("Could not detect image MIME from bytes", exc_info=True)
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
        logger.debug("Could not detect image MIME for %s", file_path, exc_info=True)
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


def flatten_image_to_rgb(image: Image.Image) -> Image.Image:
    """Flatten any image mode to RGB, compositing alpha over a white background.

    Naive ``Image.convert("RGB")`` strips alpha *without* compositing, which turns
    transparent regions black (or leftover garbage RGB). Document figures/logos are
    authored to sit on white, so composite over white to preserve their intended
    appearance before dropping the alpha channel — which the embedding model does
    not use and which JPEG (the large-image encoding) cannot carry anyway.
    """
    if image.mode == "RGB":
        return image
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


def _fit_pixel_budget(image: Image.Image, *, max_px: int, max_pixels: int) -> Image.Image:
    """Downscale so the long edge <= ``max_px`` and total pixels <= ``max_pixels``."""
    width, height = image.size
    long_edge = max(width, height)
    total = width * height
    scale = 1.0
    if long_edge > max_px:
        scale = min(scale, max_px / long_edge)
    if total > max_pixels:
        scale = min(scale, (max_pixels / total) ** 0.5)
    if scale >= 1.0:
        return image
    next_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(next_size, Image.Resampling.LANCZOS)


def _encode_image(image: Image.Image, image_format: str, **params: Any) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format, **params)
    return buffer.getvalue()


def bounded_embedding_image_data_uri(
    image: Image.Image,
    *,
    max_px: int = EMBED_IMAGE_MAX_PX,
    max_pixels: int = EMBED_IMAGE_MAX_PIXELS,
    max_bytes: int = EMBED_IMAGE_MAX_BYTES,
    quality: int = EMBED_IMAGE_QUALITY,
    min_quality: int = EMBED_IMAGE_MIN_QUALITY,
    min_px: int = EMBED_IMAGE_MIN_PX,
) -> str:
    """Return a fidelity-first, provider-safe image data URI for embedding.

    Unlike :func:`bounded_image_data_uri` (answer payloads, which *skip* images
    they cannot fit cleanly), this always returns a payload — for indexing a
    slightly degraded vector beats dropping the image. Fidelity is preserved in
    decreasing order:

    1. Fit the pixel budget (long edge ``<= max_px`` and total ``<= max_pixels``).
    2. Keep it lossless (PNG) when that already fits ``max_bytes``.
    3. Step JPEG quality down from ``quality`` to ``min_quality``.
    4. Progressively downscale (x0.85) and retry the JPEG ladder down to ``min_px``.

    JPEG size scales roughly linearly with pixel count, so the ladder is
    monotonic and converges within a few steps for any input.
    """
    max_px = max(1, int(max_px))
    max_pixels = max(1, int(max_pixels))
    max_bytes = max(1, int(max_bytes))
    quality = min(95, max(1, int(quality)))
    min_quality = min(quality, max(1, int(min_quality)))
    min_px = max(1, min(max_px, int(min_px)))

    working = flatten_image_to_rgb(image)
    working = _fit_pixel_budget(working, max_px=max_px, max_pixels=max_pixels)

    lossless = _encode_image(working, "PNG")
    if len(lossless) <= max_bytes:
        return f"data:image/png;base64,{base64.b64encode(lossless).decode('ascii')}"

    qualities = _quality_steps(quality, min_quality)
    current = working
    while True:
        for current_quality in qualities:
            payload = _encode_image(current, "JPEG", quality=current_quality, optimize=True)
            if len(payload) <= max_bytes:
                return f"data:image/jpeg;base64,{base64.b64encode(payload).decode('ascii')}"
        long_edge = max(current.size)
        if long_edge <= min_px:
            break
        next_long_edge = max(min_px, int(long_edge * 0.85))
        scale = next_long_edge / long_edge
        next_size = (max(1, int(current.width * scale)), max(1, int(current.height * scale)))
        if next_size == current.size:
            break
        current = current.resize(next_size, Image.Resampling.LANCZOS)

    payload = _encode_image(current, "JPEG", quality=min_quality, optimize=True)
    return f"data:image/jpeg;base64,{base64.b64encode(payload).decode('ascii')}"


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
        if value.get("type") != "image_url":
            return None
        image_url = value.get("image_url")
        if not isinstance(image_url, dict):
            return None
        url = image_url.get("url")
        if not isinstance(url, str) or not url.strip():
            return None
        normalized_image_url: dict[str, str] = {"url": url.strip()}
        detail = image_url.get("detail")
        if isinstance(detail, str) and detail in _IMAGE_DETAIL_VALUES:
            normalized_image_url["detail"] = detail
        return {"type": "image_url", "image_url": normalized_image_url}
    text = value.strip()
    if not text:
        return None
    if text.startswith(("http://", "https://", _DATA_URI_PREFIX)):
        url = text if text.startswith(("http://", "https://")) else image_data_uri(text)
    else:
        url = image_data_uri(text)
    return {"type": "image_url", "image_url": {"url": url}}


__all__ = [
    "bounded_embedding_image_data_uri",
    "bounded_image_data_uri",
    "decode_image_base64",
    "detect_image_mime",
    "detect_image_mime_type",
    "flatten_image_to_rgb",
    "image_bytes_to_data_uri",
    "image_data_uri",
    "image_url_block",
    "split_data_uri",
    "thumbnail_bytes",
]
