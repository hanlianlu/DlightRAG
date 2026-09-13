# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Pilot evidence checks; no candidate installation or network needed by unit CI."""

import hashlib
import io
import json
import zipfile
from types import SimpleNamespace

import pypdfium2
import pytest

from scripts.anydoc_pilot import candidate_assets, deny_network, evaluate
from scripts.anydoc_pilot_fixtures import generate


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory):
    root = tmp_path_factory.mktemp("anydoc-pilot-gold")
    return root, generate(root)


def test_anydoc_pilot_has_twenty_source_authored_hashed_fixtures(fixtures):
    root, gold = fixtures
    assert len(gold) == 20
    assert json.loads((root / "gold.json").read_text()) == gold
    for entry in gold:
        data = (root / entry["name"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        assert len(data) == entry["bytes"]
    assert {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".csv"} == {
        (root / g["name"]).suffix for g in gold
    }


def test_anydoc_pilot_generated_bytes_repeat_with_fixed_archive_metadata(fixtures, tmp_path):
    _, gold = fixtures
    assert generate(tmp_path) == gold


def test_anydoc_pilot_pdf_gold_is_physical_not_parser_derived(fixtures):
    root, _ = fixtures
    with pypdfium2.PdfDocument(root / "pdf-text.pdf") as document:
        assert len(document) == 2
        for index, fact in enumerate(["ALPHA revenue 127.25", "BETA count 42"]):
            page = document[index]
            text = page.get_textpage()
            try:
                assert fact in text.get_text_range()
            finally:
                text.close()
                page.close()
    with pypdfium2.PdfDocument(root / "pdf-scan.pdf") as document:
        page = document[0]
        text = page.get_textpage()
        try:
            assert not text.get_text_range().strip()
            assert len(list(page.get_objects())) == 1
        finally:
            text.close()
            page.close()
    with pypdfium2.PdfDocument(root / "pdf-repeat-image.pdf") as document:
        page = document[0]
        try:
            assert sum(isinstance(obj, pypdfium2.PdfImage) for obj in page.get_objects()) == 2
        finally:
            page.close()


def test_anydoc_pilot_gold_proves_missing_slide_and_footnote_package(fixtures):
    root, _ = fixtures
    with zipfile.ZipFile(root / "pptx-missing-slide.pptx") as archive:
        assert "ppt/slides/slide2.xml" not in archive.namelist()
        assert b"slides/slide2.xml" in archive.read("ppt/_rels/presentation.xml.rels")
    with zipfile.ZipFile(root / "docx-footnote.docx") as archive:
        assert b"Warranty 36 months" in archive.read("word/footnotes.xml")
        assert b"footnotes.xml" in archive.read("word/_rels/document.xml.rels")


def test_anydoc_pilot_does_not_use_nonempty_as_completeness(fixtures):
    _, gold = fixtures
    fixture = next(g for g in gold if g["name"] == "pptx-missing-slide.pptx")
    assert not evaluate(fixture, {"text": "North slide 1"})["passed"]
    fixture = next(g for g in gold if g["name"] == "pdf-mixed.pdf")
    assert not evaluate(fixture, {"text": "COVER total 52"})["passed"]
    assert evaluate(fixture, {"error": "NeedsOcrError", "pages": [2], "page_count": 2})["passed"]
    assert not evaluate(fixture, {"error": "NeedsOcrError", "pages": [1], "page_count": 2})[
        "passed"
    ]


def test_anydoc_pilot_requires_numbering_not_just_tokens(fixtures):
    _, gold = fixtures
    fixture = next(g for g in gold if g["name"] == "docx-number-table.docx")
    result = evaluate(fixture, {"text": "Install bolt 14 Apply torque 27 Nm AXLE 62.90"})
    assert not result["missing_facts"]
    assert result["missing_patterns"]
    assert not result["passed"]


def test_anydoc_pilot_rich_assets_alone_do_not_prove_view_handles(fixtures):
    _, gold = fixtures
    fixture = next(g for g in gold if g["name"] == "docx-repeat-image.docx")
    result = {"text": fixture["facts"][0], "assets": [{"sha256": fixture["asset_digest"]}] * 2}
    quality = evaluate(fixture, result)
    assert quality["asset_occurrences_match"]
    assert not quality["visual_references_match"]
    assert not quality["passed"]
    assert not evaluate(fixture, {**result, "assets": result["assets"][:1]})[
        "asset_occurrences_match"
    ]


def test_anydoc_pilot_asset_occurrences_preserve_part_not_fabricated_page():
    image = SimpleNamespace(
        kind="image", source=SimpleNamespace(kind="asset", asset_id=0), anchor=None
    )
    asset = SimpleNamespace(
        id=0, origin_part="word/media/image1.png", media_type="image/png", data=b"pixels"
    )
    document = SimpleNamespace(assets=[asset], blocks=[SimpleNamespace(content=[image, image])])
    occurrences = candidate_assets(document)
    assert len(occurrences) == 2
    assert occurrences[0]["origin_part"] == "word/media/image1.png"
    assert occurrences[0]["anchor"] is None
    assert "page" not in occurrences[0]


def test_anydoc_pilot_safety_requires_zero_converter_calls(fixtures):
    root, gold = fixtures
    fixture = next(g for g in gold if g["name"] == "docx-unsafe.docx")
    with zipfile.ZipFile(io.BytesIO((root / fixture["name"]).read_bytes())) as archive:
        info = archive.infolist()[0]
        assert info.file_size / info.compress_size > 100
        assert info.file_size == 4096
    assert evaluate(fixture, {"error": "UnsafeArchiveError", "converter_calls": 0})["passed"]
    assert not evaluate(fixture, {"error": "UnsafeArchiveError", "converter_calls": 1})["passed"]


@pytest.mark.parametrize(
    "event", ["socket.connect", "socket.connect_ex", "socket.getaddrinfo", "socket.sendto"]
)
def test_anydoc_pilot_denies_python_network(event):
    with pytest.raises(RuntimeError, match="offline pilot denied"):
        deny_network(event, ())
