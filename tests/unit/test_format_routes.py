# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Approved PDF/XLSX integrated text, independent pixels and adopted snapshots."""

import hashlib
import io
import os
import zipfile
from pathlib import Path

import anydoc
import pypdfium2
import pytest

from dlightrag.engine.agent.tool_content import tool_content_attachments
from dlightrag.engine.answer.resources import converters
from dlightrag.engine.answer.resources.models import ResourceInput
from dlightrag.engine.answer.resources.registry import ResourceRegistry
from dlightrag.engine.answer.resources.snapshots import ConversionSnapshot
from scripts.format_route_fixtures import (
    PDF_FACTS,
    XLSX_ANCHORS,
    XLSX_FACTS,
    pdf_fixture,
    xlsx_fixture,
)
from tests.unit.test_resource_tools import call, tools

NONLATIN_FACTS = ("Привет мир 42.5", "Москва столица", "中文測試 7.5", "東京駅")


def route_source(kind):
    if kind == "xlsx":
        return ResourceInput(filename="gold.xlsx", content=xlsx_fixture()[0])
    if kind == "nonlatin":
        path = os.environ.get("ANYDOC_TEST_NONLATIN_PDF")
        if not path:
            pytest.skip("optional local generated embedded-font PDF; no proprietary font bundled")
        return ResourceInput(filename="gold.pdf", content=Path(path).read_bytes())
    return ResourceInput(filename="gold.pdf", content=pdf_fixture(kind))


def assert_route_gold(kind, snapshot):
    assert snapshot.converter == "firecrawl-anydoc"
    assert snapshot.converter_version == "0.2.4" and snapshot.fallback_reason is None
    if kind in {"scan", "mixed"}:
        assert snapshot.extraction_status == "known_incomplete" and snapshot.text == ""
        assert snapshot.known_ocr_pages == ((1, 2) if kind == "scan" else ())
        assert snapshot.known_page_count == (2 if kind == "scan" else None)
        assert snapshot.visuals == ()
    else:
        assert snapshot.extraction_status == "usable_text_unverified_coverage"
        facts = (
            XLSX_FACTS if kind == "xlsx" else NONLATIN_FACTS if kind == "nonlatin" else PDF_FACTS
        )
        assert all(fact in snapshot.text for fact in facts)
    if kind == "xlsx":
        assert "$1,234.50" in snapshot.text and "0007" in snapshot.text
        assert "| 28.5 |  | $1,234.50 | 0007 |" in snapshot.text
        assert all(noise not in snapshot.text for noise in ("NaN", "NaT", "585987", "=314159"))
        assert tuple(v.anchor for v in snapshot.visuals) == XLSX_ANCHORS
        assert len({v.handle_id for v in snapshot.visuals}) == 3
        assert all(v.data == xlsx_fixture()[1] for v in snapshot.visuals)


@pytest.mark.parametrize("kind", ["multi", "scan", "mixed", "nonlatin", "xlsx"])
async def test_format_routes_tool_snapshot_view_and_recovery(kind, monkeypatch):
    source = route_source(kind)
    assert source.content is not None
    candidate = anydoc.to_markdown_bytes
    calls = []

    def text(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs == {"format": "xlsx" if kind == "xlsx" else "pdf", "ocr": "reject"}
        return candidate(*args, **kwargs)

    monkeypatch.setattr(anydoc, "to_markdown_bytes", text)
    monkeypatch.setattr(
        anydoc, "to_document", lambda *a, **k: pytest.fail("unneeded structured parse")
    )
    monkeypatch.setattr(converters, "_convert_markitdown", lambda *a: pytest.fail("no fallback"))
    async with ResourceRegistry(resource_secret=b"format", cursor_secret=b"cursor") as registry:
        resource = registry.register(source)
        read, view = tools(registry)
        result = await call(read, resource_id=resource)
        assert not result.is_error and not tool_content_attachments(result.parts)
        effects = result.effects.attached_resources
        snapshot = ConversionSnapshot.restore(
            effects[-1].content, {e.resource_id: e.content for e in effects[:-1]}
        )
        assert_route_gold(kind, snapshot)
        assert snapshot.input_digest == hashlib.sha256(source.content).hexdigest()
        if kind == "xlsx":
            with zipfile.ZipFile(io.BytesIO(source.content)) as archive:
                source_images = [
                    archive.read(p) for p in archive.namelist() if p.startswith("xl/media/")
                ]
            for visual in snapshot.visuals:
                assert visual.data in source_images
                pixel_result = await call(view, resource_id=resource, locator=visual.handle_id)
                (pixel,) = tool_content_attachments(pixel_result.parts)
                assert pixel.source is not None
                assert pixel.source.anchor == visual.anchor and pixel.source.page is None
        else:
            assert "Physical PDF page count:" in result.text_content
            assert "view(resource_id=" in result.text_content
            pixel_result = await call(view, resource_id=resource)
            pixels = tool_content_attachments(pixel_result.parts)
            assert pixels
            for index, pixel in enumerate(pixels, 1):
                assert pixel.source is not None
                assert pixel.source.overview and pixel.source.page == index
        first = await registry.read(resource, max_window_tokens=1000)
    monkeypatch.setattr(anydoc, "to_markdown_bytes", lambda *a, **k: pytest.fail("reparsed"))
    async with ResourceRegistry(resource_secret=b"format", cursor_secret=b"cursor") as restored:
        restored.register(source)
        restored.adopt_conversion_snapshot(snapshot)
        assert (await restored.read(resource, max_window_tokens=1000)) == first
        for visual in snapshot.visuals:
            assert (await restored.visual_asset(resource, visual.handle_id)).data == visual.data
    assert len(calls) == 1


@pytest.mark.parametrize("kind", ["multi", "scan", "mixed", "nonlatin"])
def test_pdf_fixture_independent_page_and_text_layer_gold(kind):
    source = route_source(kind)
    with pypdfium2.PdfDocument(source.content) as pdf:
        assert len(pdf) == {"multi": 3, "scan": 2, "mixed": 1, "nonlatin": 2}[kind]
        texts = []
        for page in pdf:
            text = page.get_textpage()
            try:
                texts.append(text.get_text_range())
            finally:
                text.close()
                page.close()
    joined = "\n".join(texts)
    if kind == "scan":
        assert not joined.strip()
    else:
        facts = (
            PDF_FACTS
            if kind == "multi"
            else NONLATIN_FACTS
            if kind == "nonlatin"
            else ("PLAN total 52", "Q1 12", "Q2 40")
        )
        assert all(fact in joined for fact in facts)


@pytest.mark.parametrize("suffix", ["pdf", "xlsx"])
@pytest.mark.parametrize("error", [anydoc.MalformedError, anydoc.MissingPartError, ImportError])
async def test_new_routes_single_ordinary_fallback(suffix, error, monkeypatch):
    calls = []
    incumbent = converters._convert_markitdown

    def fail(*args, **kwargs):
        calls.append("ended")
        raise error("synthetic ordinary failure")

    def fallback(*args):
        assert calls == ["ended"]
        calls.append("fallback")
        return incumbent(*args)

    monkeypatch.setattr(
        converters if error is ImportError else anydoc,
        "_load_anydoc" if error is ImportError else "to_markdown_bytes",
        fail,
    )
    monkeypatch.setattr(converters, "_convert_markitdown", fallback)
    data = pdf_fixture("multi") if suffix == "pdf" else xlsx_fixture()[0]
    result = await converters.convert_resource(data, filename=f"a.{suffix}", declared_mime=None)
    assert result.converter == "markitdown" and result.fallback_reason
    assert error.__name__ in result.fallback_reason
    assert calls == ["ended", "fallback"]
    if suffix == "xlsx":
        assert tuple(v.anchor for v in result.visuals) == XLSX_ANCHORS


@pytest.mark.parametrize("suffix", ["pdf", "xlsx"])
@pytest.mark.parametrize("error", [anydoc.UnsupportedError, anydoc.ResourceLimitError, MemoryError])
async def test_new_routes_typed_terminals_never_fallback(suffix, error, monkeypatch):
    def fail(*args, **kwargs):
        raise error("synthetic terminal")

    monkeypatch.setattr(anydoc, "to_markdown_bytes", fail)
    monkeypatch.setattr(
        converters, "_convert_markitdown", lambda *a: pytest.fail("terminal fallback")
    )
    data = pdf_fixture("multi") if suffix == "pdf" else xlsx_fixture()[0]
    if error == anydoc.UnsupportedError:
        result = await converters.convert_resource(data, filename=f"a.{suffix}", declared_mime=None)
        assert result.extraction_status == "known_incomplete" and not result.text
        assert not result.known_ocr_pages and result.known_page_count is None
    else:
        with pytest.raises((converters.ConversionLimitError, MemoryError)):
            await converters.convert_resource(data, filename=f"a.{suffix}", declared_mime=None)


@pytest.mark.parametrize(
    "suffix,data", [("csv", b"name,value\nalpha,12\n"), ("html", b"<p>Alpha 12</p>")]
)
async def test_deferred_and_unsupported_routes_do_not_probe_candidate(suffix, data, monkeypatch):
    monkeypatch.setattr(converters, "_load_anydoc", lambda: pytest.fail("unapproved route"))
    result = await converters.convert_resource(data, filename=f"a.{suffix}", declared_mime=None)
    assert result.converter == "markitdown"


async def test_xlsx_asset_failure_adopts_no_speculative_text_or_fallback(monkeypatch):
    def fail(*args):
        raise ValueError("synthetic malformed drawing")

    monkeypatch.setattr(converters, "_extract_xlsx_visuals", fail)
    monkeypatch.setattr(
        converters,
        "_convert_markitdown",
        lambda *a: pytest.fail("same asset extractor cannot rescue itself"),
    )
    source = route_source("xlsx")
    async with ResourceRegistry(resource_secret=b"drawing") as registry:
        resource = registry.register(source)
        read, _ = tools(registry)
        result = await call(read, resource_id=resource)
        effects = result.effects.attached_resources
        snapshot = ConversionSnapshot.restore(effects[-1].content, {})
        assert snapshot.extraction_status == "conversion_failed"
        assert snapshot.text == "" and snapshot.visuals == ()
        assert snapshot.converter == "firecrawl-anydoc" and snapshot.fallback_reason is None
        first = await registry.read(resource, max_window_tokens=1000)
    async with ResourceRegistry(resource_secret=b"drawing") as restored:
        restored.register(source)
        restored.adopt_conversion_snapshot(snapshot)
        assert await restored.read(resource, max_window_tokens=1000) == first
