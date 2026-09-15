# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Page-context normalization for image sources before a parser sees them.

Both external parsers decide "is this a figure?" from page-level layout. A
full-bleed image gives their layout models no page context, so a whole class of
legitimate image documents produces nothing at all:

- docling returns an empty document, which LightRAG rejects as a zero-block IR
  ("Docling IR builder produced zero blocks") and the document fails;
- MinerU returns an empty ``content_list.json``, which LightRAG reports as a
  missing bundle file, or misclassifies the page as an empty table.

Compositing the image onto a white canvas with a page margin restores that
context. Measured on real artwork photographs (20 samples, widths 600-2000 px):

============================  ==========  =========
margin                        docling     MinerU
============================  ==========  =========
0% (raw source)               8/20 fail   2/10 fail
3% (this default)             0/20 fail   1/10 fail
8%                            -           0/10 fail
============================  ==========  =========

The threshold is a fraction of the page, not a pixel count: a 613 px image
needed 18 px (3%) and a 2000 px image needed 60 px (3%), while 2.5% (50 px)
still failed. One margin is shared by both engines — deliberately, since a
single deployment runs exactly one external parser. MinerU's response is not
monotonic in the margin (the same image fails at 3% and 12% but succeeds at
8%), so its residual cases stay visible as failures rather than being papered
over by a larger border.

The padded file is a **derived parser input**, never the source of record: it
is written beside the staging copy under :data:`PADDED_INPUT_DIR_NAME`, keeps
the original basename (LightRAG derives ``doc_id`` from the canonical basename,
so a renamed input would change document identity) and is deleted once the
batch settles. The original bytes stay untouched, so source downloads, content
hashes and metadata keep referring to what the user supplied.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

PADDED_INPUT_DIR_NAME = ".parse-input"

# Default page margin as a fraction of each dimension, per side.
DEFAULT_IMAGE_MARGIN = 0.03
MAX_IMAGE_MARGIN = 0.5

# Suffixes whose content is a raster image the parsers can lay out. Mirrors the
# union of the docling and MinerU image capabilities.
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp", ".gif"})

_PIL_FORMAT_BY_SUFFIX: dict[str, str] = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".webp": "WEBP",
    ".bmp": "BMP",
    ".gif": "GIF",
}

# JPEG is lossy, so re-encoding at PIL's default quality would degrade the
# artwork that this normalization exists to preserve.
_JPEG_QUALITY = 95


def is_image_source(path: Path | str) -> bool:
    """Whether ``path`` carries a raster image suffix."""
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def normalize_image_margin(margin: float) -> float:
    """Clamp a configured margin into ``[0, MAX_IMAGE_MARGIN]``."""
    if margin <= 0:
        return 0.0
    return min(float(margin), MAX_IMAGE_MARGIN)


def padded_parser_path(source: Path, *, margin: float) -> Path | None:
    """Write a white-bordered copy of ``source`` beside it; return its path.

    Returns ``None`` when normalization does not apply (no margin, not an image
    source, an unreadable/undecodable image, or a format Pillow cannot write),
    in which case the caller keeps parsing the source unchanged. Losing the
    layout improvement is recoverable; failing the ingest is not.
    """
    margin = normalize_image_margin(margin)
    if margin <= 0:
        return None
    source = Path(source)
    image_format = _PIL_FORMAT_BY_SUFFIX.get(source.suffix.lower())
    if image_format is None or not source.is_file():
        return None

    canvas = _padded_canvas(source, margin=margin)
    if canvas is None:
        return None

    target_dir = source.parent / PADDED_INPUT_DIR_NAME
    # The basename is load-bearing: LightRAG derives doc_id from the canonical
    # basename, so a renamed parser input would change document identity.
    target = target_dir / source.name
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=".pad-", dir=target_dir)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                _save_canvas(canvas, handle, image_format=image_format)
            temporary.replace(target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    except OSError as exc:
        logger.warning("Image page-margin normalization skipped for %s: %s", source.name, exc)
        return None
    logger.info(
        "Image page-margin normalization: %s -> %s (margin %.1f%%)",
        source.name,
        target.name,
        margin * 100,
    )
    return target


def discard_padded_images(paths: list[Path]) -> None:
    """Remove derived parser inputs written by :func:`padded_parser_path`."""
    directories: set[Path] = set()
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:  # pragma: no cover - best effort cleanup
            logger.warning("Could not remove normalized parser input %s: %s", path, exc)
            continue
        directories.add(path.parent)
    for directory in directories:
        try:
            directory.rmdir()
        except OSError:
            # Another batch still owns a file in this staging directory.
            continue


def _padded_canvas(source: Path, *, margin: float):
    """Return the composited RGB canvas, or ``None`` when the image is unusable."""
    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        with Image.open(source) as opened:
            # Honour EXIF orientation so a rotated photo stays upright on the
            # canvas instead of being laid out sideways.
            transposed = ImageOps.exif_transpose(opened)
            width, height = transposed.size
            if width < 1 or height < 1:
                return None
            pad_x = max(1, round(width * margin))
            pad_y = max(1, round(height * margin))
            canvas = Image.new("RGB", (width + 2 * pad_x, height + 2 * pad_y), "white")
            rgba = transposed.convert("RGBA")
            # Alpha compositing over white: transparent backgrounds (PNG art,
            # cut-outs) must not turn black.
            canvas.paste(rgba, (pad_x, pad_y), rgba)
            return canvas
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        logger.warning("Image page-margin normalization skipped for %s: %s", source.name, exc)
        return None
    except Image.DecompressionBombError as exc:  # pragma: no cover - defensive
        logger.warning("Image page-margin normalization skipped for %s: %s", source.name, exc)
        return None


def _save_canvas(canvas, handle, *, image_format: str) -> None:
    if image_format == "JPEG":
        canvas.save(handle, format=image_format, quality=_JPEG_QUALITY)
        return
    canvas.save(handle, format=image_format)


__all__ = [
    "DEFAULT_IMAGE_MARGIN",
    "IMAGE_SUFFIXES",
    "MAX_IMAGE_MARGIN",
    "PADDED_INPUT_DIR_NAME",
    "discard_padded_images",
    "is_image_source",
    "normalize_image_margin",
    "padded_parser_path",
]
