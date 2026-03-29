# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared utilities."""

import re

# Pillow format → MIME type mapping for LLM-safe formats
_PILLOW_TO_MIME: dict[str, str] = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


def image_data_uri(b64: str) -> str:
    """Build a data URI from base64 image data.

    Uses Pillow to detect the actual format. If the format is LLM-safe
    (JPEG, PNG, WebP), uses the original data. Otherwise converts to
    JPEG quality 95 (handles TIFF, JPEG2000, BMP, GIF, etc.).
    """
    import base64 as _b64
    import io

    from PIL import Image

    try:
        raw = _b64.b64decode(b64)
        img = Image.open(io.BytesIO(raw))
        mime = _PILLOW_TO_MIME.get(img.format or "")

        if mime:
            return f"data:{mime};base64,{b64}"

        # Unsupported format → convert to JPEG
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        jpeg_b64 = _b64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{jpeg_b64}"
    except Exception:
        return f"data:image/png;base64,{b64}"


def normalize_workspace(name: str) -> str:
    """Normalize workspace name to a safe, lowercase PG identifier.

    Replaces non-alphanumeric/underscore characters with ``_``, lowercases,
    and prepends ``_`` if the name starts with a digit (invalid PG schema).

    Examples::

        normalize_workspace("Ian Davis")   -> "ian_davis"
        normalize_workspace("project-alpha") -> "project_alpha"
        normalize_workspace("123abc")      -> "_123abc"
        normalize_workspace("café-art")    -> "caf__art"
    """
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", name.strip()).lower()
    if safe and safe[0].isdigit():
        safe = f"_{safe}"
    return safe
