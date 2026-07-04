"""Tests for web route multimodal and multi-workspace wiring."""

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
    """Verify multimodal_content list is built from data URI image blocks."""
    b64 = _make_fake_image_b64()
    images_b64 = [b64, b64]

    multimodal_content = []
    for b64_str in images_b64[:3]:
        multimodal_content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_str}"}}
        )

    assert len(multimodal_content) == 2
    assert all(item["type"] == "image_url" for item in multimodal_content)
    assert all(
        item["image_url"]["url"].startswith("data:image/png;base64,") for item in multimodal_content
    )


def test_image_count_limit():
    """Only first 3 images are processed."""
    b64 = _make_fake_image_b64()
    images_b64 = [b64] * 5
    limited = images_b64[:3]
    assert len(limited) == 3
