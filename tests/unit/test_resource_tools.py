# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Public read/view seams: text, pixels, inventories, identity, and budgets."""

import asyncio
from dataclasses import replace

import pytest
from pydantic import ValidationError

from dlightrag.engine.agent.environment import AccessScheduler
from dlightrag.engine.agent.environment.local import LocalExecutionEnvironment
from dlightrag.engine.agent.tool_content import (
    decode_tool_content,
    encode_tool_content,
    tool_content_attachments,
)
from dlightrag.engine.agent.tools import ToolResult
from dlightrag.engine.agent.tools.files import ViewArgs, view_tool
from dlightrag.engine.answer.resources.converters import ResourceConversionError
from dlightrag.engine.answer.resources.models import ResourceInput, ResourceRegistryError
from dlightrag.engine.answer.resources.registry import ResourceRegistry
from dlightrag.engine.answer.resources.snapshots import ConversionSnapshot
from dlightrag.engine.answer.tools.resources import make_resource_viewer
from dlightrag.engine.answer.web_sources import WebExtractResult
from dlightrag.engine.public_http import PublicHttpFetch
from tests.support.dns import public_dns
from tests.support.resources import call, docx_images, pdf_bytes, png, preparer, tools
from tests.tool_helpers import tool_runtime


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"path": "a", "url": "https://example.com"},
        {"path": "a", "locator": "1"},
        {"resource_id": "res-a", "focus": "x"},
        {"url": "https://example.com", "cursor": "x"},
        {"resource_id": "res-a", "http": {}},
        {"resource_id": "res-a", "locator": "1", "cursor": "x"},
    ],
)
def test_view_rejects_ambiguous_or_legacy_arguments(args):
    with pytest.raises(ValidationError):
        ViewArgs.model_validate(args)


async def test_read_image_returns_guidance_only_and_view_attaches_located_pixels():
    async with ResourceRegistry() as registry:
        resource = registry.register(ResourceInput(filename="plot.png", content=png()))
        read, view = tools(registry)
        text = await call(read, resource_id=resource)
        assert not tool_content_attachments(text.parts)
        assert "view(resource_id=" in text.text_content
        pixels = await call(view, resource_id=resource)
        (attachment,) = tool_content_attachments(pixels.parts)
        assert attachment.data == png()
        assert attachment.source is not None
        assert attachment.source.resource_id == resource
        assert attachment.source is not None
        assert attachment.source.kind == "image"
        restored = decode_tool_content(encode_tool_content(pixels.parts))
        (restored_attachment,) = tool_content_attachments(restored)
        assert restored_attachment.source == attachment.source
        assert not restored_attachment.data


async def test_resource_view_spends_the_image_budget_only_in_source_order():
    """Preparing pixels spends a budget every call of the batch shares."""
    budget = preparer()
    prepared: list[str] = []

    def prepare(data, label):
        prepared.append(label)
        return budget(data, label)

    waiting = asyncio.Event()
    earlier_returned = asyncio.Event()

    async def in_source_order() -> None:
        waiting.set()
        await earlier_returned.wait()

    async with ResourceRegistry() as registry:
        resource = registry.register(ResourceInput(filename="plot.png", content=png()))
        view = view_tool(
            None,
            AccessScheduler(),
            resource_viewer=make_resource_viewer(registry),
            image_preparer=prepare,
        )
        runtime = replace(tool_runtime(tool_name=view.name), _in_source_order=in_source_order)

        async def look() -> ToolResult:
            return await view.execute(ViewArgs(resource_id=resource), runtime)

        viewing = asyncio.create_task(look())
        await waiting.wait()
        assert prepared == []

        earlier_returned.set()
        result = await viewing

    assert tool_content_attachments(result.parts)
    # The label names the document: a later turn has no manifest to map the id.
    assert prepared == ["plot.png"]


async def test_workspace_read_image_is_text_and_view_rejects_escape_and_documents(tmp_path):
    (tmp_path / "a.png").write_bytes(png())
    (tmp_path / "doc.pdf").write_bytes(pdf_bytes())
    async with ResourceRegistry() as registry:
        read, view = tools(registry, environment=LocalExecutionEnvironment(tmp_path))
        assert not tool_content_attachments((await call(read, path="a.png")).parts)
        assert tool_content_attachments((await call(view, path="a.png")).parts)
        assert (await call(view, path="../escape.png")).is_error
        assert (await call(view, path="doc.pdf")).is_error


async def test_pdf_view_bypasses_failed_text_extraction(monkeypatch):
    async def fail(*args, **kwargs):
        raise ResourceConversionError("ordinary parser failure")

    monkeypatch.setattr("dlightrag.engine.answer.resources.registry.convert_resource", fail)
    async with ResourceRegistry() as registry:
        resource = registry.register(ResourceInput(filename="paper.pdf", content=pdf_bytes()))
        read, view = tools(registry)
        text = await call(read, resource_id=resource)
        assert "conversion_failed" in text.text_content
        assert "Physical PDF page count: 3" in text.text_content
        result = await call(view, resource_id=resource, locator="2")
        (attachment,) = tool_content_attachments(result.parts)
        assert attachment.source is not None
        assert attachment.source.page == 2
        assert attachment.source is not None
        assert not attachment.source.overview


async def test_pdf_overview_actual_coverage_aggregate_budget_and_signed_recovery_cursor():
    data = pdf_bytes(3)
    async with ResourceRegistry(resource_secret=b"r", cursor_secret=b"c") as registry:
        resource = registry.register(ResourceInput(filename="paper.pdf", content=data))
        _, view = tools(registry, max_images=1)
        result = await call(view, resource_id=resource)
        assert "physical pages 1-1 of 3 only" in result.text_content
        cursor = result.protected_text.split("cursor='")[1].split("'")[0]
        spent = await call(view, resource_id=resource, cursor=cursor)
        assert spent.is_error is True
        assert "remaining model image budget" in spent.text_content
    async with ResourceRegistry(resource_secret=b"r", cursor_secret=b"c") as recovered:
        assert recovered.register(ResourceInput(filename="paper.pdf", content=data)) == resource
        _, view = tools(recovered, max_images=1)
        second = await call(view, resource_id=resource, cursor=cursor)
        (attachment,) = tool_content_attachments(second.parts)
        assert attachment.source is not None
        assert attachment.source.page == 2
        tampered = await call(view, resource_id=resource, cursor=cursor + "x")
        assert tampered.is_error is True
        assert (
            "call read or view on the resource again for a current continuation"
            in tampered.text_content
        )


async def test_duplicate_occurrences_membership_inventory_and_snapshot_reuse(monkeypatch):
    data = docx_images(12)
    async with ResourceRegistry(resource_secret=b"r", cursor_secret=b"c") as registry:
        resource = registry.register(ResourceInput(filename="a.docx", content=data))
        other = registry.register(ResourceInput(filename="b.docx", content=docx_images(1)))
        read, view = tools(registry)
        result = await call(read, resource_id=resource)
        assert "more: read(" in result.text_content
        assets = [
            a for a in result.effects.attached_resources if a.resource_kind == "conversion_asset"
        ]
        assert len(assets) == 12
        assert len({a.resource_id for a in assets}) == 12
        assert len({a.content for a in assets}) == 1
        foreign = await call(view, resource_id=other, locator=assets[0].resource_id)
        assert foreign.is_error is True
        assert "unknown visual handle" in foreign.text_content
        cursor = result.text_content.split("more: read(")[1].split("cursor='")[1].split("'")[0]
        page = await call(read, resource_id=resource, cursor=cursor)
        assert "Visual inventory" in page.text_content
        stored = {a.resource_id: a.content for a in result.effects.attached_resources}
        snapshot = ConversionSnapshot.restore(stored[f"{resource}-conversion"], stored)

    async def forbidden(*args, **kwargs):
        raise AssertionError("adopted snapshots never reparse")

    monkeypatch.setattr("dlightrag.engine.answer.resources.registry.convert_resource", forbidden)
    async with ResourceRegistry(resource_secret=b"r", cursor_secret=b"c") as registry:
        assert registry.register(ResourceInput(filename="a.docx", content=data)) == resource
        registry.adopt_conversion_snapshot(snapshot)
        read, view = tools(registry)
        assert "Revenue was 123" in (await call(read, resource_id=resource)).text_content
        pixels = await call(view, resource_id=resource, locator=assets[-1].resource_id)
        source = tool_content_attachments(pixels.parts)[0].source
        assert source is not None
        assert source.handle_id == assets[-1].resource_id


async def test_extensionless_url_image_is_classified_after_acquisition_and_reuses_snapshot(
    monkeypatch,
):
    from types import SimpleNamespace

    calls = []

    async def fetch(url, **kwargs):
        calls.append(url)
        return SimpleNamespace(content=png(), media_type="image/png", final_url=url)

    monkeypatch.setattr("dlightrag.engine.answer.resources.registry.fetch_public_http", fetch)
    async with ResourceRegistry() as registry:
        read, view = tools(registry)
        result = await call(view, url="https://example.com/asset")
        (attachment,) = tool_content_attachments(result.parts)
        assert attachment.source is not None
        resource = attachment.source.resource_id
        assert attachment.source is not None
        assert attachment.source.kind == "image"
        assert not tool_content_attachments((await call(read, resource_id=resource)).parts)
        await call(view, resource_id=resource)
        with pytest.raises(ResourceRegistryError, match="cannot replace"):
            await call(view, url="https://example.com/asset", http={"accept": "image/webp"})
        assert len(calls) == 1


async def test_conversion_cancellation_does_not_overlap_native_work_or_cleanup(monkeypatch):
    import asyncio
    import threading

    started, release = threading.Event(), threading.Event()
    calls = []

    def native(*args, **kwargs):
        calls.append(1)
        started.set()
        assert release.wait(5)
        return "adopted"

    monkeypatch.setattr("anydoc.to_markdown_bytes", native)
    registry = ResourceRegistry()
    resource = registry.register(ResourceInput(filename="a.pdf", content=pdf_bytes(1)))
    first = asyncio.create_task(registry.read(resource, max_window_tokens=1000))
    assert await asyncio.to_thread(started.wait, 5)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    second = asyncio.create_task(registry.read(resource, max_window_tokens=1000))
    await asyncio.sleep(0)
    release.set()
    from dlightrag.engine.answer.resources.converters import ConversionLimitError

    with pytest.raises(ConversionLimitError):
        await second
    assert len(calls) == 1
    await registry.aclose()


_SCAN_URL = "https://example.com/scan.pdf"
_EXTRACT_TEXT = "Extracted text of the scanned report."


class _ScanRun:
    """A Run reading one scanned PDF its server labels HTML.

    The PDF has pages to view and no text layer, and its label makes it a textual Web
    resource, so a read takes its text from the Extract chain. The Run records what
    it makes durable as a Run does: fetched bytes through the sink, and everything a
    call attaches through that call's result.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.content = pdf_bytes(2)
        self.fetches: list[str] = []
        self.persisted: list = []
        content = self.content
        fetches = self.fetches

        async def fetch(url: str, **_kwargs: object) -> PublicHttpFetch:
            fetches.append(url)
            return PublicHttpFetch(content, url, "text/html", 200)

        monkeypatch.setattr("dlightrag.engine.answer.resources.registry.fetch_public_http", fetch)
        monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)

    async def persist(self, fetched, _owner) -> None:
        self.persisted.append(fetched)

    def registry(self) -> ResourceRegistry:
        async def extract(url: str) -> WebExtractResult:
            return WebExtractResult(
                url=url, text=_EXTRACT_TEXT, provider="exa", acquisition="exa_extract"
            )

        return ResourceRegistry(
            url_text_fallback=extract,
            fetched_bytes_sink=self.persist,
            resource_secret=b"scan-run",
            cursor_secret=b"scan-cursors",
        )

    def resumed(self, *results: ToolResult) -> ResourceRegistry:
        """A registry restored from what the Run made durable, as recovery restores one."""
        registry = ResourceRegistry(resource_secret=b"scan-run", cursor_secret=b"scan-cursors")
        for fetched in self.persisted:
            registry.restore_fetched_resource(
                resource_id=fetched.resource_id,
                ordinal=fetched.ordinal,
                filename=fetched.filename,
                mime_type=fetched.mime_type,
                url=fetched.url,
                content=fetched.content,
                admission_origin=fetched.admission_origin,
                acquisition=fetched.acquisition,
                aliases=fetched.aliases,
            )
        rows = [row for result in results for row in result.effects.attached_resources]
        stored = {row.resource_id: row.content for row in rows}
        for row in rows:
            if row.resource_kind == "conversion_snapshot":
                registry.adopt_conversion_snapshot(ConversionSnapshot.restore(row.content, stored))
        return registry


def _acquisition(result: ToolResult) -> str:
    (source,) = result.effects.evidence_sources
    return dict(source.attributes)["acquisition"]


def _assert_viewed_and_read(run: _ScanRun, pages: ToolResult, text: ToolResult) -> None:
    assert pages.is_error is False, pages.text_content
    assert len(tool_content_attachments(pages.parts)) == 2
    assert text.is_error is False, text.text_content
    assert _EXTRACT_TEXT in text.text_content
    assert "Physical PDF page count: 2." in text.text_content
    # One fetch and one admitted representation, the bytes, whichever call came first;
    # each call names where its own evidence came from.
    assert run.fetches == [_SCAN_URL]
    assert {fetched.content for fetched in run.persisted} == {run.content}
    assert (_acquisition(pages), _acquisition(text)) == ("direct_http", "exa_extract")


async def test_a_web_pdf_viewed_and_then_read_keeps_its_bytes_and_reads_extract_text(
    monkeypatch,
):
    run = _ScanRun(monkeypatch)
    async with run.registry() as registry:
        read, view = tools(registry)
        pages = await call(view, url=_SCAN_URL)
        text = await call(read, url=_SCAN_URL)
        again = await call(read, url=_SCAN_URL)

    _assert_viewed_and_read(run, pages, text)
    assert again.text_content == text.text_content


async def test_a_web_pdf_read_and_then_viewed_keeps_its_bytes_and_reads_extract_text(
    monkeypatch,
):
    run = _ScanRun(monkeypatch)
    async with run.registry() as registry:
        read, view = tools(registry)
        text = await call(read, url=_SCAN_URL)
        pages = await call(view, url=_SCAN_URL)

    _assert_viewed_and_read(run, pages, text)


async def test_a_web_pdf_viewed_and_read_in_one_batch_keeps_its_bytes_and_reads_extract_text(
    monkeypatch,
):
    run = _ScanRun(monkeypatch)
    async with run.registry() as registry:
        read, view = tools(registry)
        pages, text = await asyncio.gather(call(view, url=_SCAN_URL), call(read, url=_SCAN_URL))

    _assert_viewed_and_read(run, pages, text)


async def test_a_resumed_run_reads_and_views_a_web_pdf_as_before(monkeypatch):
    """Recovery restores the bytes and their Extract text view, and fetches nothing."""
    run = _ScanRun(monkeypatch)
    async with run.registry() as registry:
        read, view = tools(registry)
        pages = await call(view, url=_SCAN_URL)
        text = await call(read, url=_SCAN_URL)

    async with run.resumed(pages, text) as resumed:
        read, view = tools(resumed)
        assert (await call(read, url=_SCAN_URL)).text_content == text.text_content
        again = await call(view, url=_SCAN_URL)
    assert [a.content_digest for a in tool_content_attachments(again.parts)] == [
        a.content_digest for a in tool_content_attachments(pages.parts)
    ]
    assert run.fetches == [_SCAN_URL]
