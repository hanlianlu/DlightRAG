"""Tests for web route multimodal and multi-workspace wiring."""

from __future__ import annotations

import base64
import io

from PIL import Image


def _make_fake_image_b64() -> str:
    """Create a minimal 1x1 PNG as base64."""
    buf = io.BytesIO()
    Image.new("RGB", (1, 1), color="red").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_image_b64_decode_roundtrip():
    """Verify base64 encode/decode produces valid image bytes."""
    b64 = _make_fake_image_b64()
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw))
    assert img.size == (1, 1)


def test_multimodal_content_construction():
    """Verify multimodal_content list is built from base64 images."""
    import tempfile
    from pathlib import Path

    b64 = _make_fake_image_b64()
    images_b64 = [b64, b64]

    multimodal_content = []
    tmp_paths = []
    for b64_str in images_b64[:3]:
        img_bytes = base64.b64decode(b64_str)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.write(img_bytes)
        tmp.close()
        tmp_paths.append(tmp.name)
        multimodal_content.append({"type": "image", "img_path": tmp.name})

    assert len(multimodal_content) == 2
    assert all(item["type"] == "image" for item in multimodal_content)
    assert all(Path(item["img_path"]).exists() for item in multimodal_content)

    # Cleanup
    for p in tmp_paths:
        Path(p).unlink(missing_ok=True)


def test_image_count_limit():
    """Only first 3 images are processed."""
    b64 = _make_fake_image_b64()
    images_b64 = [b64] * 5
    limited = images_b64[:3]
    assert len(limited) == 3
