# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Public read/view seams: text, pixels, inventories, identity, and budgets."""

import asyncio
import re
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
from dlightrag.engine.agent.tools import ToolEffects, ToolResult, fit_tool_result
from dlightrag.engine.agent.tools.files import (
    RenderedReadArgs,
    ViewArgs,
    read_declaration,
    read_tool,
    view_tool,
)
from dlightrag.engine.ai.capacity import CONTEXT_POLICY
from dlightrag.engine.ai.tokens import estimate_tokens
from dlightrag.engine.answer.agent_browser import browser_failure
from dlightrag.engine.answer.resource_settlement import attached_resource_update
from dlightrag.engine.answer.resources.converters import ResourceConversionError
from dlightrag.engine.answer.resources.models import ResourceInput, ResourceRegistryError
from dlightrag.engine.answer.resources.registry import HostedExtract, ResourceRegistry
from dlightrag.engine.answer.resources.snapshots import ConversionSnapshot
from dlightrag.engine.answer.tools.resources import make_resource_reader, make_resource_viewer
from dlightrag.engine.answer.web_sources import WebExtractResult
from dlightrag.engine.public_http import PublicHttpFetch
from tests.support.agent_browser import RecordingRenderer
from tests.support.dns import public_dns
from tests.support.resources import (
    call,
    docx_images,
    pdf_bytes,
    png,
    preparer,
    printed_handle,
    tools,
)
from tests.tool_helpers import tool_runtime
from tests.unit.conftest import answer_model_profile


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
            extract_chain=(HostedExtract(extract),),
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


# -- Rendered Reads (ADR 0032) --------------------------------------------------------------

_APP = "https://spa.example.com/app.html"
_APP_PAGE = "<html><body><h1>Quotes</h1><p>Albert Einstein</p><p>J.K. Rowling</p></body></html>"


def test_rendered_is_offered_only_with_an_agent_browser_and_only_for_urls() -> None:
    plain = read_declaration(public_url=True)
    offered = read_declaration(public_url=True, rendered=True)

    assert "rendered" not in plain.definition.parameters["properties"]
    assert "rendered" in offered.definition.parameters["properties"]
    assert "rendered=true" in offered.description and "rendered=true" not in plain.description
    # A host that cannot read a URL has no rendering of one to offer.
    assert read_declaration(public_url=False, rendered=True) == read_declaration(public_url=False)


@pytest.mark.parametrize(
    ("arguments", "complaint"),
    [
        ({"path": "notes.txt", "rendered": True}, "only for url or resource_id"),
        ({"url": _APP, "rendered": True, "http": {"accept": "text/html"}}, "direct acquisition"),
        ({"url": _APP, "rendered": True, "cursor": "r.x"}, "returned resource_id"),
        ({"rendered": True}, "exactly one of"),
    ],
)
def test_a_rendered_read_of_anything_but_a_url_or_a_web_resource_is_a_validation_error(
    arguments, complaint
) -> None:
    with pytest.raises(ValidationError, match=complaint):
        RenderedReadArgs.model_validate(arguments)


async def test_a_rendered_read_settles_its_rendering_its_view_and_where_it_came_from(
    monkeypatch,
) -> None:
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    renderer = RecordingRenderer({_APP: _APP_PAGE})
    async with ResourceRegistry(page_renderer=renderer) as registry:
        read, _ = tools(registry, rendered=True)

        result = await call(read, url=_APP, rendered=True)

        assert result.is_error is False, result.text_content
        resource_id = printed_handle(result)
        assert f"[resource: {resource_id} | rendered | lines 1-" in result.text_content
        assert "Albert Einstein" in result.text_content
        assert "[Rendered view from the Agent Browser (browser_render)." in result.text_content
        (source,) = result.effects.evidence_sources
        assert dict(source.attributes)["acquisition"] == "browser_render"
        assert source.source_uri == _APP
        rendering, snapshot = result.effects.attached_resources
        assert (rendering.resource_kind, snapshot.resource_kind) == (
            "web_render",
            "conversion_snapshot",
        )
        assert (rendering.resource_id, rendering.source_locator) == (
            f"{resource_id}-rendered",
            resource_id,
        )
        assert snapshot.filename == "conversion.json"
        assert dict(rendering.attributes) == {
            "acquisition": "browser_render",
            "admission_origin": "agent",
            "url": _APP,
            "final_url": _APP,
        }
        # The row records these beside its kind.
        row = attached_resource_update(rendering, session_id="s", intent_id="i")
        assert row.resource.capabilities["resource_kind"] == "web_render"
        assert row.resource.capabilities["url"] == _APP
        assert row.resource.source_locator == resource_id.encode()


async def test_a_render_the_browser_could_not_give_is_an_error_that_settles_nothing(
    monkeypatch,
) -> None:
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    renderer = RecordingRenderer({_APP: browser_failure("busy")})
    async with ResourceRegistry(page_renderer=renderer) as registry:
        read, _ = tools(registry, rendered=True)

        result = await call(read, url=_APP, rendered=True)

        assert result.is_error is True
        assert result.text_content == browser_failure("busy").public_message
        assert result.effects == ToolEffects()


async def test_a_rendered_read_of_uploaded_bytes_is_an_error_that_settles_nothing() -> None:
    async with ResourceRegistry(page_renderer=RecordingRenderer({})) as registry:
        resource = registry.register(ResourceInput(filename="a.txt", content=b"uploaded"))
        read, _ = tools(registry, rendered=True)

        result = await call(read, resource_id=resource, rendered=True)

        assert result.is_error is True
        assert f"{resource} is not a Web Resource" in result.text_content
        assert result.effects == ToolEffects()


async def test_viewing_an_image_of_a_rendering_fetches_nothing(monkeypatch) -> None:
    import base64

    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    fetches: list[str] = []

    async def fetch(url, **_kwargs):
        fetches.append(url)
        raise AssertionError("the image is in the rendering the Run holds")

    monkeypatch.setattr("dlightrag.engine.answer.resources.registry.fetch_public_http", fetch)
    image = base64.b64encode(png()).decode()
    page = f"<html><body><p>Chart</p><img alt='chart' src='data:image/png;base64,{image}'></body></html>"
    async with ResourceRegistry(page_renderer=RecordingRenderer({_APP: page})) as registry:
        read, view = tools(registry, rendered=True)
        text = await call(read, url=_APP, rendered=True)
        resource_id = printed_handle(text)
        handle = text.text_content.split("[visual handles: ")[1].split(" ")[0].rstrip("]")

        pixels = await call(view, resource_id=resource_id, locator=handle)

        assert pixels.is_error is False, pixels.text_content
        (attachment,) = tool_content_attachments(pixels.parts)
        assert attachment.data == png()
        assert attachment.source is not None and attachment.source.resource_id == resource_id
        (source,) = pixels.effects.evidence_sources
        assert dict(source.attributes)["acquisition"] == "browser_render"
        # What the view settles includes the rendering it came from, so recovery has it.
        assert "web_render" in {row.resource_kind for row in pixels.effects.attached_resources}
        assert fetches == []


async def test_a_text_longer_than_one_result_is_read_to_the_end_in_pages_the_runtime_keeps_whole():
    profile = answer_model_profile()
    capacity = CONTEXT_POLICY.observation_capacity(profile)
    line_count = 5_000
    text = "".join(
        f"{index}. Quarterly revenue grew while operating margin held across all segments.\n"
        for index in range(line_count)
    )
    assert estimate_tokens(text) > 2 * capacity
    async with ResourceRegistry(resource_secret=b"r", cursor_secret=b"c") as registry:
        resource = registry.register(
            ResourceInput(filename="long.txt", content=text.encode(), declared_mime="text/plain")
        )
        read = read_tool(
            None,
            AccessScheduler(),
            resource_reader=make_resource_reader(
                registry, CONTEXT_POLICY.read_window_tokens(profile)
            ),
        )
        seen: list[int] = []
        cursor: str | None = None
        while True:
            args = {"resource_id": resource} | ({"cursor": cursor} if cursor else {})
            page = await call(read, **args)
            # The runtime cuts a result at the observation capacity; a page it would cut
            # is text the model is shown without a cursor to continue from.
            assert fit_tool_result(page, max_tokens=capacity).text_content == page.text_content
            seen += [
                int(line.split(".", 1)[0])
                for line in page.text_content.splitlines()
                if line.split(".", 1)[0].isdigit() and line.endswith("segments.")
            ]
            continuation = re.search(r"cursor=([^\]\s']+)", page.protected_text)
            if continuation is None:
                break
            cursor = continuation.group(1)

    assert seen == list(range(line_count))
