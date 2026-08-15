# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for answer-model image budgeting."""

import base64
import io
from dataclasses import FrozenInstanceError

import pytest
from dlightrag_ai.media import (
    ImagePayloadBudget,
    bounded_embedding_image_data_uri,
    bounded_image_data_uri,
    decode_image_base64,
    flatten_image_to_rgb,
)
from PIL import Image

from dlightrag.core.answer.images import AnswerImageBudget, AnswerImagePolicy

_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def test_answer_image_policy_is_frozen_and_creates_fresh_budgets() -> None:
    policy = AnswerImagePolicy(
        max_images=1,
        max_total_bytes=10_000,
        max_bytes_per_image=10_000,
        max_pixels=40_000_000,
        max_px=64,
        min_px=32,
        quality=85,
        min_quality=72,
    )

    first = policy.new_budget()
    second = policy.new_budget()

    assert first.add_base64(_PNG_B64, label="chunk:c1") is not None
    assert first.count == 1
    assert second.count == 0
    assert second.used_bytes == 0
    with pytest.raises(FrozenInstanceError):
        policy.max_images = 2  # type: ignore[misc]


def test_answer_image_budget_bounds_base64_images() -> None:
    budget = AnswerImageBudget(
        max_images=1,
        max_total_bytes=10_000,
        max_bytes_per_image=10_000,
        max_pixels=40_000_000,
        max_px=64,
        min_px=32,
        quality=85,
        min_quality=72,
    )

    block = budget.add_base64(_PNG_B64, label="chunk:c1")

    assert block is not None
    assert block["type"] == "image_url"
    assert block["image_url"]["url"].startswith("data:image/png;base64,")
    assert budget.count == 1
    assert budget.add_base64(_PNG_B64, label="chunk:c2") is None


def test_answer_image_budget_passes_user_url_without_bytes_count() -> None:
    budget = AnswerImageBudget(
        max_images=2,
        max_total_bytes=1,
        max_bytes_per_image=1,
        max_pixels=40_000_000,
        max_px=64,
        min_px=32,
        quality=85,
        min_quality=72,
    )

    block = budget.add_user_image("https://example.com/chart.png", label="query:1")

    assert block == {"type": "image_url", "image_url": {"url": "https://example.com/chart.png"}}
    assert budget.count == 1
    assert budget.used_bytes == 0


def test_answer_image_budget_bounds_dict_data_uri_blocks() -> None:
    raw = _png_bytes((16, 16))
    data_uri = f"data:image/png;base64,{base64.b64encode(raw).decode('ascii')}"
    budget = AnswerImageBudget(
        max_images=2,
        max_total_bytes=10_000,
        max_bytes_per_image=10_000,
        max_pixels=40_000_000,
        max_px=1536,
        min_px=1024,
        quality=88,
        min_quality=72,
    )

    block = budget.add_user_image(
        {"type": "image_url", "image_url": {"url": data_uri, "detail": "high"}},
        label="query:dict",
    )

    assert block == {
        "type": "image_url",
        "image_url": {"url": data_uri, "detail": "high"},
    }
    assert budget.count == 1
    assert budget.used_bytes == len(raw)


def test_answer_image_budget_rejects_invalid_base64() -> None:
    budget = AnswerImageBudget(
        max_images=2,
        max_total_bytes=10_000,
        max_bytes_per_image=10_000,
        max_pixels=40_000_000,
        max_px=64,
        min_px=32,
        quality=85,
        min_quality=72,
    )

    assert budget.add_base64("not image data", label="bad") is None
    assert budget.count == 0


def test_answer_image_budget_rejects_image_over_pixel_limit_without_consuming_budget() -> None:
    raw = _png_bytes((11, 10))
    data_uri = f"data:image/png;base64,{base64.b64encode(raw).decode('ascii')}"
    budget = AnswerImageBudget(
        max_images=2,
        max_total_bytes=10_000,
        max_bytes_per_image=10_000,
        max_pixels=100,
        max_px=64,
        min_px=32,
        quality=85,
        min_quality=72,
    )

    assert budget.add_base64(data_uri, label="over-pixel-limit") is None
    assert budget.count == 0
    assert budget.used_bytes == 0


def test_answer_image_budget_accepts_image_at_pixel_limit() -> None:
    raw = _png_bytes((10, 10))
    data_uri = f"data:image/png;base64,{base64.b64encode(raw).decode('ascii')}"
    budget = AnswerImageBudget(
        max_images=1,
        max_total_bytes=10_000,
        max_bytes_per_image=10_000,
        max_pixels=100,
        max_px=64,
        min_px=32,
        quality=85,
        min_quality=72,
    )

    assert budget.add_base64(data_uri, label="at-pixel-limit") is not None
    assert budget.count == 1
    assert budget.used_bytes == len(raw)


def test_bounded_image_data_uri_skips_instead_of_degrading_below_quality_floor() -> None:
    raw = _jpeg_bytes((768, 768), quality=95)
    data_uri = f"data:image/jpeg;base64,{base64.b64encode(raw).decode('ascii')}"

    bounded = bounded_image_data_uri(
        data_uri,
        max_bytes=1_000,
        max_px=768,
        min_px=768,
        quality=88,
        min_quality=72,
    )

    assert bounded is None


def test_bounded_image_data_uri_preserves_small_budgeted_png() -> None:
    raw = _png_bytes((32, 32))
    data_uri = f"data:image/png;base64,{base64.b64encode(raw).decode('ascii')}"

    bounded = bounded_image_data_uri(
        data_uri,
        max_bytes=10_000,
        max_px=1536,
        min_px=1024,
        quality=88,
        min_quality=72,
    )

    assert bounded is not None
    uri, byte_count = bounded
    assert uri == data_uri
    assert byte_count == len(raw)


def test_bounded_image_rejects_pixel_bomb_before_rgb_conversion(monkeypatch) -> None:
    converted = False

    class HeaderOnlyImage:
        format = "PNG"
        size = (100_000, 100_000)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def convert(self, _mode):
            nonlocal converted
            converted = True
            raise AssertionError("pixel-bomb image was converted")

    monkeypatch.setattr(Image, "open", lambda *_args, **_kwargs: HeaderOnlyImage())

    bounded = bounded_image_data_uri(
        _PNG_B64,
        max_bytes=10_000,
        max_px=1536,
        max_pixels=40_000_000,
        min_px=1024,
        quality=88,
        min_quality=72,
    )

    assert bounded is None
    assert converted is False


def test_bounded_embedding_image_keeps_small_image_lossless() -> None:
    uri = bounded_embedding_image_data_uri(Image.new("RGB", (100, 100), "white"))

    assert uri.startswith("data:image/png;base64,")
    raw, _ = decode_image_base64(uri)
    with Image.open(io.BytesIO(raw)) as decoded:
        assert decoded.size == (100, 100)


def test_bounded_embedding_image_caps_long_edge_and_total_pixels() -> None:
    uri = bounded_embedding_image_data_uri(Image.new("RGB", (5000, 4000), "white"))

    raw, _ = decode_image_base64(uri)
    with Image.open(io.BytesIO(raw)) as decoded:
        assert max(decoded.size) <= 4096
        assert decoded.width * decoded.height <= 15_000_000


def test_bounded_embedding_image_caps_total_pixels_when_long_edge_fits() -> None:
    # 4000x4000 = 16MP: long edge is under 4096 but the total exceeds the 15MP cap.
    uri = bounded_embedding_image_data_uri(Image.new("RGB", (4000, 4000), "white"))

    raw, _ = decode_image_base64(uri)
    with Image.open(io.BytesIO(raw)) as decoded:
        assert decoded.width * decoded.height <= 15_000_000


def test_bounded_embedding_image_converges_under_tight_byte_budget() -> None:
    # Noise resists compression, so PNG is skipped and the JPEG/downscale ladder
    # must run to satisfy the byte budget; it always returns a payload.
    noise = Image.effect_noise((2000, 2000), 200).convert("RGB")

    uri = bounded_embedding_image_data_uri(noise, max_bytes=200_000)

    assert uri.startswith("data:image/jpeg;base64,")
    raw, _ = decode_image_base64(uri)
    assert len(raw) <= 200_000
    with Image.open(io.BytesIO(raw)) as decoded:
        assert max(decoded.size) >= 1


def test_bounded_embedding_image_converts_non_rgb_modes() -> None:
    uri = bounded_embedding_image_data_uri(Image.new("RGBA", (128, 128), (255, 0, 0, 128)))

    raw, _ = decode_image_base64(uri)
    with Image.open(io.BytesIO(raw)) as decoded:
        assert decoded.size == (128, 128)


def test_flatten_image_composites_transparency_over_white() -> None:
    # A fully transparent pixel: naive convert("RGB") keeps its stored RGB (here
    # black); compositing over white must yield white instead.
    flat = flatten_image_to_rgb(Image.new("RGBA", (8, 8), (0, 0, 0, 0)))

    assert flat.mode == "RGB"
    assert flat.getpixel((0, 0)) == (255, 255, 255)


def test_bounded_embedding_image_flattens_transparency_over_white() -> None:
    uri = bounded_embedding_image_data_uri(Image.new("RGBA", (64, 64), (0, 0, 0, 0)))

    raw, _ = decode_image_base64(uri)
    with Image.open(io.BytesIO(raw)) as decoded:
        assert decoded.convert("RGB").getpixel((0, 0)) == (255, 255, 255)


def _jpeg_bytes(size: tuple[int, int], *, quality: int) -> bytes:
    image = Image.new("RGB", size, "white")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _png_bytes(size: tuple[int, int]) -> bytes:
    image = Image.new("RGB", size, "white")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_with_orientation(size: tuple[int, int], orientation: int) -> bytes:
    image = Image.new("RGB", size, (30, 60, 90))
    exif = image.getexif()
    exif[0x0112] = orientation
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95, exif=exif)
    return buf.getvalue()


def _rgba_bytes(size: tuple[int, int], color: tuple[int, int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


# ── Canonical image normalization path (dlightrag_ai.media) ──


def test_bounded_image_applies_exif_orientation() -> None:
    # EXIF orientation 6 means "rotate 90deg CW to display"; the canonical path
    # must physically transpose a landscape 40x20 source into a 20x40 payload.
    raw = _jpeg_with_orientation((40, 20), orientation=6)

    bounded = bounded_image_data_uri(
        _b64(raw),
        max_bytes=5_000_000,
        max_px=1536,
        min_px=16,
        quality=90,
        min_quality=72,
    )

    assert bounded is not None
    uri, _ = bounded
    decoded, _ = decode_image_base64(uri)
    with Image.open(io.BytesIO(decoded)) as image:
        assert image.size == (20, 40)


def test_bounded_image_composites_alpha_over_white() -> None:
    # A fully transparent source must flatten to white, never black. Forcing a
    # re-encode (max_px below the source) routes it through the canonical path.
    raw = _rgba_bytes((64, 64), (0, 0, 0, 0))

    bounded = bounded_image_data_uri(
        _b64(raw),
        max_bytes=5_000_000,
        max_px=32,
        min_px=16,
        quality=90,
        min_quality=72,
    )

    assert bounded is not None
    uri, _ = bounded
    decoded, _ = decode_image_base64(uri)
    with Image.open(io.BytesIO(decoded)) as image:
        pixel = image.convert("RGB").getpixel((0, 0))
    assert isinstance(pixel, tuple)
    assert all(channel > 250 for channel in pixel)


def test_bounded_image_rejects_source_over_pixel_ceiling() -> None:
    raw = _png_bytes((11, 10))  # 110 source pixels

    assert (
        bounded_image_data_uri(
            _b64(raw),
            max_bytes=5_000_000,
            max_px=1536,
            max_pixels=100,
            min_px=16,
            quality=90,
            min_quality=72,
        )
        is None
    )


def test_bounded_image_descends_long_edge_and_quality_to_fit_bytes() -> None:
    # Incompressible noise forces the resize/quality ladder to run; the result
    # must satisfy both the byte budget and the long-edge cap.
    noise = Image.effect_noise((800, 800), 220).convert("RGB")
    buf = io.BytesIO()
    noise.save(buf, format="PNG")

    bounded = bounded_image_data_uri(
        _b64(buf.getvalue()),
        max_bytes=20_000,
        max_px=512,
        min_px=64,
        quality=90,
        min_quality=60,
    )

    assert bounded is not None
    uri, byte_count = bounded
    assert byte_count <= 20_000
    decoded, _ = decode_image_base64(uri)
    with Image.open(io.BytesIO(decoded)) as image:
        assert max(image.size) <= 512


def test_bounded_image_reflects_true_format_over_declared_mime() -> None:
    raw = _png_bytes((32, 32))
    lying_uri = f"data:image/jpeg;base64,{_b64(raw)}"

    bounded = bounded_image_data_uri(
        lying_uri,
        max_bytes=5_000_000,
        max_px=1536,
        min_px=16,
        quality=90,
        min_quality=72,
    )

    assert bounded is not None
    uri, _ = bounded
    assert uri.startswith("data:image/png;base64,")


def test_image_payload_budget_enforces_per_image_and_total_bounds() -> None:
    small = _b64(_png_bytes((16, 16)))
    budget = ImagePayloadBudget(
        max_total_bytes=len(_png_bytes((16, 16))) + 10,
        max_bytes_per_image=5_000_000,
        max_pixels=40_000_000,
        max_px=1536,
        min_px=16,
        quality=90,
        min_quality=72,
    )

    first = budget.add_base64(small, label="one")
    assert first is not None
    # The aggregate byte budget is now nearly exhausted; the second image cannot
    # fit and is rejected without corrupting the running totals.
    assert budget.add_base64(small, label="two") is None
    assert budget.count == 1
