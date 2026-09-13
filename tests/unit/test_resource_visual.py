# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Physical PDF rendering is bounded and independent of text extraction."""

import io

import pytest
from PIL import Image

from dlightrag.engine.answer.resources.visual import (
    ResourceViewError,
    pdf_page_count,
    render_pdf_page,
)


def pdf_bytes(count: int = 3, size: tuple[int, int] = (300, 400)) -> bytes:
    images = [Image.new("RGB", size, (index * 20, 10, 10)) for index in range(count)]
    buffer = io.BytesIO()
    images[0].save(buffer, "PDF", save_all=True, append_images=images[1:])
    for image in images:
        image.close()
    return buffer.getvalue()


def test_physical_page_count_and_detail_geometry():
    data = pdf_bytes()
    assert pdf_page_count(data) == 3
    with Image.open(io.BytesIO(render_pdf_page(data, 2, overview=True))) as overview:
        assert max(overview.size) <= 900
        with Image.open(io.BytesIO(render_pdf_page(data, 2, overview=False))) as detail:
            assert detail.width > overview.width


@pytest.mark.parametrize("page", [0, 4])
def test_page_range_is_explicit(page):
    with pytest.raises(ResourceViewError, match="out of range"):
        render_pdf_page(pdf_bytes(), page, overview=False)


def test_malformed_pdf_is_not_blank():
    with pytest.raises(ResourceViewError, match="inventory unavailable"):
        pdf_page_count(b"%PDF-broken")
