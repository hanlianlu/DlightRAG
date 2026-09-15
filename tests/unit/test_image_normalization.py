# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for image page-margin normalization before external parsers."""

from pathlib import Path

import pytest
from PIL import Image

from dlightrag.engine.rag.corpus.ingestion.image_normalization import (
    DEFAULT_IMAGE_MARGIN,
    PADDED_INPUT_DIR_NAME,
    discard_padded_images,
    is_image_source,
    normalize_image_margin,
    padded_parser_path,
)


def _write_solid(path: Path, size: tuple[int, int], color: tuple[int, ...]) -> Path:
    mode = "RGBA" if len(color) == 4 else "RGB"
    Image.new(mode, size, color).save(path)
    return path


def test_pads_each_side_by_the_configured_fraction(tmp_path: Path) -> None:
    source = _write_solid(tmp_path / "art.png", (1000, 500), (10, 20, 30))

    padded = padded_parser_path(source, margin=0.03)

    assert padded is not None
    assert padded.name == source.name
    assert padded.parent == tmp_path / PADDED_INPUT_DIR_NAME
    with Image.open(padded) as canvas:
        assert canvas.size == (1000 + 2 * 30, 500 + 2 * 15)


def test_keeps_the_original_basename_so_document_identity_is_stable(
    tmp_path: Path,
) -> None:
    from lightrag.utils_pipeline import normalize_document_file_path

    source = _write_solid(tmp_path / "artwork.png", (200, 100), (0, 0, 0))

    padded = padded_parser_path(source, margin=0.1)

    assert padded is not None
    assert normalize_document_file_path(padded) == normalize_document_file_path(source)


def test_canvas_is_white_and_artwork_is_untouched(tmp_path: Path) -> None:
    source = _write_solid(tmp_path / "art.png", (20, 10), (255, 0, 0))

    padded = padded_parser_path(source, margin=0.5)

    assert padded is not None
    with Image.open(padded) as canvas:
        assert canvas.getpixel((0, 0)) == (255, 255, 255)
        assert canvas.getpixel((canvas.width - 1, canvas.height - 1)) == (255, 255, 255)
        assert canvas.getpixel((10, 5)) == (255, 0, 0)
        assert canvas.size == (40, 20)


def test_transparent_pixels_composite_over_white(tmp_path: Path) -> None:
    source = tmp_path / "cutout.png"
    Image.new("RGBA", (10, 10), (0, 0, 0, 0)).save(source)

    padded = padded_parser_path(source, margin=0.2)

    assert padded is not None
    with Image.open(padded) as canvas:
        assert canvas.getpixel((canvas.width // 2, canvas.height // 2)) == (255, 255, 255)


def test_exif_orientation_is_applied(tmp_path: Path) -> None:
    source = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (40, 20), (0, 128, 0))
    exif = image.getexif()
    exif[274] = 6  # rotate 90° clockwise on display
    image.save(source, exif=exif)

    padded = padded_parser_path(source, margin=0.0)
    assert padded is None  # margin 0 is a no-op by contract

    padded = padded_parser_path(source, margin=0.05)
    assert padded is not None
    with Image.open(padded) as canvas:
        # 40x20 becomes 20x40, plus 5% of each side (1px, 2px).
        assert canvas.size == (20 + 2 * 1, 40 + 2 * 2)


def test_non_image_and_zero_margin_sources_are_left_alone(tmp_path: Path) -> None:
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    image = _write_solid(tmp_path / "art.png", (10, 10), (0, 0, 0))

    assert padded_parser_path(pdf, margin=0.03) is None
    assert padded_parser_path(image, margin=0.0) is None
    assert not (tmp_path / PADDED_INPUT_DIR_NAME).exists()


def test_undecodable_image_is_skipped_instead_of_failing(tmp_path: Path) -> None:
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")

    assert padded_parser_path(broken, margin=0.03) is None


def test_missing_source_is_skipped(tmp_path: Path) -> None:
    assert padded_parser_path(tmp_path / "absent.png", margin=0.03) is None


@pytest.mark.parametrize(
    "suffix", [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp", ".gif"]
)
def test_supported_image_suffixes_round_trip(tmp_path: Path, suffix: str) -> None:
    source = _write_solid(tmp_path / f"art{suffix}", (40, 30), (12, 34, 56))

    padded = padded_parser_path(source, margin=DEFAULT_IMAGE_MARGIN)

    assert padded is not None
    assert padded.suffix == suffix
    with Image.open(padded) as canvas:
        assert canvas.size[0] > 40 and canvas.size[1] > 30


def test_image_source_detection() -> None:
    assert is_image_source("a.PNG")
    assert is_image_source(Path("deep/art.tiff"))
    assert not is_image_source("a.pdf")
    assert not is_image_source("a")


def test_margin_is_clamped_into_a_sane_range() -> None:
    assert normalize_image_margin(-1) == 0.0
    assert normalize_image_margin(0) == 0.0
    assert normalize_image_margin(0.03) == pytest.approx(0.03)
    assert normalize_image_margin(5) == 0.5


def test_discard_removes_files_and_the_staging_directory(tmp_path: Path) -> None:
    source = _write_solid(tmp_path / "art.png", (10, 10), (0, 0, 0))
    padded = padded_parser_path(source, margin=0.1)
    assert padded is not None

    discard_padded_images([padded])

    assert not padded.exists()
    assert not padded.parent.exists()
    assert source.exists()
