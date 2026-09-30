# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Lineage adoption: an earlier Run's handle becomes this Run's Resource."""

import asyncio
import base64
import hashlib
import io
from collections.abc import Sequence
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from dlightrag.engine.agent.environment.access import AccessScheduler
from dlightrag.engine.agent.tool_content import decode_tool_content, encode_tool_content
from dlightrag.engine.agent.tools import ResourceAttachmentBytes, ToolResult
from dlightrag.engine.agent.tools.files import PreparedImageAttachment, read_tool, view_tool
from dlightrag.engine.ai.media import decode_image_base64
from dlightrag.engine.answer.research.context import _resource_manifest_context
from dlightrag.engine.answer.resources.converters import ConvertedResource, ExtractedVisual
from dlightrag.engine.answer.resources.lineage import (
    ASSET_KIND,
    LINEAGE_ADOPTION_KIND,
    SNAPSHOT_KIND,
    LineageAdoptionConflict,
    LineageResourceBytes,
)
from dlightrag.engine.answer.resources.models import (
    ResourceAdmissionError,
    ResourceInput,
    ResourceManifestEntry,
    ResourceNotFoundError,
    TextWindowBudget,
)
from dlightrag.engine.answer.resources.registry import ResourceEffectOwner, ResourceRegistry
from dlightrag.engine.answer.resources.snapshots import ConversionSnapshot
from dlightrag.engine.answer.tools.resources import make_resource_reader, make_resource_viewer
from dlightrag.engine.runtime.coordinator import LeaseLostError
from dlightrag.engine.runtime.records import RunFetchedResource
from tests.tool_helpers import tool_runtime
from tests.unit.conftest import answer_image_policy

EARLIER_HANDLE = "res-earlier-run-handle"
_ADOPTED_TEXT = "Extracted text the earlier Run already paid for."


def png(size: int = 24) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (size, size), (3, 40, 7)).save(buffer, "PNG")
    return buffer.getvalue()


def preparer(max_images: int = 4):
    budget = answer_image_policy(max_images=max_images).new_budget()

    def prepare(data, label):
        block = budget.add_base64(base64.b64encode(data).decode(), label=label)
        if block is None:
            return None
        content, media = decode_image_base64(block["image_url"]["url"])
        return PreparedImageAttachment(content, media or "image/png", content != data)

    return prepare


class Recorder:
    """What adoption asked the loader to make durable, and a write that may fail."""

    def __init__(self) -> None:
        self.reads = 0
        self.recorded: list[tuple[ResourceAttachmentBytes, ...]] = []
        self.fails: BaseException | None = None
        # A write that commits and then reports an error, once: its outcome is unknown.
        self.lands_then_fails: BaseException | None = None

    async def record(
        self, resources: tuple[ResourceAttachmentBytes, ...], owner: ResourceEffectOwner
    ) -> None:
        del owner
        # A durable write yields, so another call can run while it is in flight.
        await asyncio.sleep(0)
        if self.fails is not None:
            raise self.fails
        self.recorded.append(resources)
        landed, self.lands_then_fails = self.lands_then_fails, None
        if landed is not None:
            raise landed


class Loader(Recorder):
    """One authorized earlier Run Resource, counting how often it is read."""

    def __init__(self, loaded: LineageResourceBytes | None) -> None:
        super().__init__()
        self.loaded = loaded

    async def load(self, resource_id: str) -> LineageResourceBytes | None:
        self.reads += 1
        if self.loaded is None or resource_id != self.loaded.resource_id:
            return None
        return self.loaded


def tools(registry, *, lineage):
    access = AccessScheduler()
    return (
        read_tool(
            None,
            access,
            resource_reader=make_resource_reader(registry, TextWindowBudget(1000), lineage=lineage),
        ),
        view_tool(
            None,
            access,
            resource_viewer=make_resource_viewer(registry, lineage=lineage),
            image_preparer=preparer(),
        ),
    )


async def call(tool, **args) -> ToolResult:
    return await tool.execute(
        tool.input_model.model_validate(args), tool_runtime(tool_name=tool.name)
    )


def adopted_document(*, with_snapshot: bool) -> LineageResourceBytes:
    content = b"%PDF-1.7 lineage document"
    snapshot = ConversionSnapshot(
        resource_id=EARLIER_HANDLE,
        input_digest=hashlib.sha256(content).hexdigest(),
        text=_ADOPTED_TEXT,
        visuals=(
            ExtractedVisual(
                handle_id="embedded-1",
                anchor="page 2",
                origin_part=None,
                media_type="image/png",
                data=png(),
            ),
        ),
        extraction_status="complete",
        converter="fixture",
        converter_version="1",
    )
    encoded: bytes | None = None
    assets: dict[str, bytes] = {}
    if with_snapshot:
        effects = snapshot.effects()
        encoded = next(item.content for item in effects if item.resource_kind == SNAPSHOT_KIND)
        assets = {
            item.resource_id: item.content for item in effects if item.resource_kind == ASSET_KIND
        }
    return LineageResourceBytes(
        resource_id=EARLIER_HANDLE,
        origin_run_id="01a0a737-e1d3-7421-8e25-27ca8abd3dad",
        filename="lineage.pdf",
        media_type="application/pdf",
        content=content,
        conversion_snapshot=encoded,
        assets=assets,
        source_url="https://example.com/lineage.pdf",
    )


async def test_read_adopts_an_earlier_handle_and_reuses_its_stored_text(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("adopted snapshots never re-run parser selection")

    monkeypatch.setattr("dlightrag.engine.answer.resources.registry.convert_resource", forbidden)
    lineage = Loader(adopted_document(with_snapshot=True))
    async with ResourceRegistry() as registry:
        read, _ = tools(registry, lineage=lineage)
        result = await call(read, resource_id=EARLIER_HANDLE)

        assert result.is_error is False
        assert _ADOPTED_TEXT in result.text_content
        assert lineage.reads == 1

        canonical = registry.canonical_resource_id(EARLIER_HANDLE)
        assert canonical != EARLIER_HANDLE
        (recorded,) = lineage.recorded
        adoption, *view = recorded
        assert adoption.resource_kind == LINEAGE_ADOPTION_KIND
        assert (adoption.resource_id, adoption.source_locator) == (canonical, canonical)
        assert adoption.aliases == (EARLIER_HANDLE,), (
            "the earlier handle is recorded, not only bound"
        )
        assert sorted(effect.resource_kind for effect in view) == [ASSET_KIND, SNAPSHOT_KIND]
        assert {effect.source_locator for effect in view} == {canonical}, "the view is this Run's"
        assert f"{canonical}-conversion" in {effect.resource_id for effect in view}
        # A read settles exactly the view the adoption recorded, so the two never disagree.
        assert result.effects.attached_resources == tuple(view)

        second = await call(read, resource_id=EARLIER_HANDLE)
        assert second.is_error is False
        assert _ADOPTED_TEXT in second.text_content
        assert lineage.reads == 1, "the alias needs no second read"
        assert len(lineage.recorded) == 1


async def test_view_adopts_an_earlier_image_without_any_snapshot() -> None:
    lineage = Loader(
        LineageResourceBytes(
            resource_id="res-earlier-image",
            origin_run_id="01a0a737-e1d3-7421-8e25-27ca8abd3dad",
            filename="page.png",
            media_type="image/png",
            content=png(),
        )
    )
    async with ResourceRegistry() as registry:
        _, view = tools(registry, lineage=lineage)
        result = await call(view, resource_id="res-earlier-image")

        assert result.is_error is False
        (recorded,) = lineage.recorded
        assert [effect.resource_kind for effect in recorded] == [LINEAGE_ADOPTION_KIND], (
            "the adopted source document is recorded before the call returns"
        )
        kinds = [effect.resource_kind for effect in result.effects.attached_resources]
        assert kinds == ["tool_attachment"], "the viewed pixels stay a tool attachment"
        restored = decode_tool_content(encode_tool_content(result.parts))
        assert any(part for part in restored if part.type == "resource_attachment")


def adopted_product() -> LineageResourceBytes:
    """One published Markdown product: no conversion route, so no stored view."""
    content = b"# Analysis\n\nversion one\n"
    return LineageResourceBytes(
        resource_id="artifact-431b1900963e6cd2f4a1",
        origin_run_id="01a0a737-e1d3-7421-8e25-27ca8abd3dad",
        filename="analysis.md",
        media_type="text/markdown",
        content=content,
        conversion_snapshot=None,
        assets={},
    )


async def test_reading_a_published_product_decodes_it_without_a_stored_view() -> None:
    """The handle the Tool returns must actually read: a product is directly decodable.

    A convertible document needs the earlier Run's own view because reading it here
    would convert it again; a Markdown report has no conversion route, so demanding a
    view would refuse the one call ADR 0023 teaches.
    """
    lineage = Loader(adopted_product())
    async with ResourceRegistry() as registry:
        read, _ = tools(registry, lineage=lineage)

        result = await call(read, resource_id="artifact-431b1900963e6cd2f4a1")

        assert result.is_error is False
        assert "version one" in result.text_content
        assert lineage.reads == 1
        assert registry.canonical_resource_id("artifact-431b1900963e6cd2f4a1") is not None
        assert len(lineage.recorded) == 1


async def test_reading_a_document_the_earlier_run_never_converted_refuses(monkeypatch) -> None:
    """Text needs the earlier Run's own view; converting it here would invent one."""

    def forbidden(*_args, **_kwargs):
        raise AssertionError("adoption must not run parser selection")

    monkeypatch.setattr("dlightrag.engine.answer.resources.registry.convert_resource", forbidden)
    lineage = Loader(adopted_document(with_snapshot=False))
    async with ResourceRegistry() as registry:
        read, _ = tools(registry, lineage=lineage)

        refused = await call(read, resource_id=EARLIER_HANDLE)
        assert refused.is_error is True
        assert "never extracted text" in refused.text_content
        assert "View its pages" in refused.text_content
        assert lineage.reads == 1
        assert lineage.recorded == [], "a refused read adopts nothing"
        assert registry.manifest() == ()
        # Pixels need no stored view; that path is covered by the image adoption test,
        # which can only view a target the registry can actually render.


async def test_an_unauthorized_handle_keeps_the_typed_refusal() -> None:
    lineage = Loader(None)
    async with ResourceRegistry() as registry:
        read, view = tools(registry, lineage=lineage)
        for result in (
            await call(read, resource_id="res-foreign"),
            await call(view, resource_id="res-foreign"),
        ):
            assert result.is_error is True
            assert "neither holds that handle nor can adopt it" in result.text_content
        assert lineage.reads == 2
        assert lineage.recorded == []


def _stored_view(loaded: LineageResourceBytes, **changes) -> bytes:
    """Encode the fixture's stored view with some of its facts replaced."""
    snapshot = ConversionSnapshot.restore(loaded.conversion_snapshot or b"", dict(loaded.assets))
    effects = replace(snapshot, **changes).effects()
    return next(item.content for item in effects if item.resource_kind == SNAPSHOT_KIND)


@pytest.mark.parametrize(
    ("broken", "reason"),
    [
        (lambda loaded: b'{"text":"x"}', "is unusable"),
        (
            lambda loaded: _stored_view(loaded, input_digest=hashlib.sha256(b"other").hexdigest()),
            "does not belong to these bytes",
        ),
        (
            lambda loaded: _stored_view(loaded, resource_id="res-another-handle"),
            "does not belong to this resource",
        ),
    ],
    ids=["undecodable", "other-bytes", "other-resource"],
)
async def test_an_unusable_stored_snapshot_keeps_refusing_instead_of_repairing(
    monkeypatch, broken, reason
) -> None:
    """A refused adoption registers nothing, so asking again cannot convert the bytes."""

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a broken snapshot must not trigger conversion")

    monkeypatch.setattr("dlightrag.engine.answer.resources.registry.convert_resource", forbidden)
    loaded = adopted_document(with_snapshot=True)
    lineage = Loader(replace(loaded, conversion_snapshot=broken(loaded)))
    async with ResourceRegistry() as registry:
        read, _ = tools(registry, lineage=lineage)
        for _attempt in range(2):
            result = await call(read, resource_id=EARLIER_HANDLE)

            assert result.is_error is True
            assert reason in result.text_content
            assert "was not converted again" in result.text_content
        assert lineage.reads == 2, "each attempt asks the lineage rule again"
        assert lineage.recorded == []
        assert registry.manifest() == ()
        with pytest.raises(ResourceNotFoundError):
            registry.canonical_resource_id(EARLIER_HANDLE)


async def test_adoption_past_the_attachment_allowance_refuses_as_a_tool_error() -> None:
    """Adoption spends this Run's allowance, and a spent one refuses rather than raising."""
    async with ResourceRegistry(max_attachments=1) as registry:
        registry.register(
            ResourceInput(filename="own.txt", declared_mime="text/plain", content=b"own")
        )
        lineage = Loader(adopted_document(with_snapshot=True))
        read, view = tools(registry, lineage=lineage)
        for result in (
            await call(read, resource_id=EARLIER_HANDLE),
            await call(view, resource_id=EARLIER_HANDLE),
        ):
            assert result.is_error is True
            assert "too many attachments" in result.text_content
            assert "was not adopted" in result.text_content
        with pytest.raises(ResourceNotFoundError):
            registry.canonical_resource_id(EARLIER_HANDLE)
        assert lineage.recorded == []


@pytest.mark.parametrize("own_read_first", [True, False], ids=["own-view-first", "adopted-first"])
async def test_bytes_this_run_can_convert_keep_this_runs_view(
    monkeypatch, own_read_first: bool
) -> None:
    """The same document attached again is one Resource with one conversion history.

    Bytes this Run can convert itself never take the earlier view, whether this Run
    converted them already or not, so their adoption records the alias alone.
    """
    conversions: list[str] = []

    async def convert(_content, *, filename, declared_mime, deadline=None):
        del declared_mime, deadline
        conversions.append(filename)
        return ConvertedResource(
            text="This Run's own extraction.", visuals=(), converter="own", converter_version="1"
        )

    monkeypatch.setattr("dlightrag.engine.answer.resources.registry.convert_resource", convert)
    loaded = adopted_document(with_snapshot=True)
    async with ResourceRegistry() as registry:
        own = registry.register(
            ResourceInput(
                filename=loaded.filename, declared_mime=loaded.media_type, content=loaded.content
            )
        )
        lineage = Loader(loaded)
        read, _ = tools(registry, lineage=lineage)
        if own_read_first:
            assert (await call(read, resource_id=own)).is_error is False

        result = await call(read, resource_id=EARLIER_HANDLE)

        assert result.is_error is False
        assert "This Run's own extraction." in result.text_content
        assert _ADOPTED_TEXT not in result.text_content
        assert registry.canonical_resource_id(EARLIER_HANDLE) == own
        assert conversions == [loaded.filename], "nothing is converted a second time"
        (recorded,) = lineage.recorded
        assert [(row.resource_kind, row.resource_id, row.aliases) for row in recorded] == [
            (LINEAGE_ADOPTION_KIND, own, (EARLIER_HANDLE,))
        ]


def pdf(pages: int = 2) -> bytes:
    images = [Image.new("RGB", (120, 160), (index * 40, 10, 10)) for index in range(pages)]
    buffer = io.BytesIO()
    images[0].save(buffer, "PDF", save_all=True, append_images=images[1:])
    return buffer.getvalue()


async def test_viewing_an_unconverted_adoption_never_opens_it_to_conversion(monkeypatch) -> None:
    """``view`` adopts pixels only; a later ``read`` still refuses to build the text view."""

    def forbidden(*_args, **_kwargs):
        raise AssertionError("an adopted document without a stored view is never converted")

    monkeypatch.setattr("dlightrag.engine.answer.resources.registry.convert_resource", forbidden)
    lineage = Loader(
        LineageResourceBytes(
            resource_id=EARLIER_HANDLE,
            origin_run_id="01a0a737-e1d3-7421-8e25-27ca8abd3dad",
            filename="scan.pdf",
            media_type="application/pdf",
            content=pdf(),
        )
    )
    async with ResourceRegistry() as registry:
        read, view = tools(registry, lineage=lineage)
        viewed = await call(view, resource_id=EARLIER_HANDLE)
        assert viewed.is_error is False
        adopted = registry.canonical_resource_id(EARLIER_HANDLE)

        for handle in (EARLIER_HANDLE, adopted):
            refused = await call(read, resource_id=handle)

            assert refused.is_error is True
            assert "never extracted text from scan.pdf" in refused.text_content
        assert lineage.reads == 1, "the alias answers the later calls"
        (recorded,) = lineage.recorded
        assert [row.resource_kind for row in recorded] == [LINEAGE_ADOPTION_KIND]


class Loaders(Recorder):
    """Several earlier Runs' Resources, by the handle each was printed under."""

    def __init__(self, *loaded: LineageResourceBytes) -> None:
        super().__init__()
        self.by_id = {item.resource_id: item for item in loaded}

    async def load(self, resource_id: str) -> LineageResourceBytes | None:
        self.reads += 1
        return self.by_id.get(resource_id)


def viewed_document(handle: str, content: bytes, *, text: str) -> LineageResourceBytes:
    """An earlier Run's PDF together with the text view that Run stored for it."""
    effects = ConversionSnapshot(
        resource_id=handle,
        input_digest=hashlib.sha256(content).hexdigest(),
        text=text,
        visuals=(),
        extraction_status="complete",
        converter="fixture",
        converter_version="1",
    ).effects()
    return LineageResourceBytes(
        resource_id=handle,
        origin_run_id="01a0a737-e1d3-7421-8e25-27ca8abd3dad",
        filename="report.pdf",
        media_type="application/pdf",
        content=content,
        conversion_snapshot=next(e.content for e in effects if e.resource_kind == SNAPSHOT_KIND),
    )


def forbid_conversion(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("an adopted document is never converted here")

    monkeypatch.setattr("dlightrag.engine.answer.resources.registry.convert_resource", forbidden)


def durable_rows(
    *writes: Sequence[ResourceAttachmentBytes],
) -> tuple[tuple[RunFetchedResource, ...], dict[str, bytes]]:
    """The catalog these writes leave, merged the way the store merges one row.

    A row written again with the same bytes and locator merges its aliases; one
    naming other bytes is the conflict the store refuses, and fails here too.
    """
    rows: dict[str, RunFetchedResource] = {}
    blobs: dict[str, bytes] = {}
    for write in writes:
        for effect in write:
            digest = hashlib.sha256(effect.content).hexdigest()
            locator = effect.source_locator.encode()
            blobs[digest] = effect.content
            earlier = rows.get(effect.resource_id)
            assert earlier is None or (earlier.digest, earlier.source_locator) == (
                digest,
                locator,
            ), f"{effect.resource_id} would conflict with the row already stored"
            aliases = sorted(
                {*effect.aliases, *(earlier.capabilities["resource_aliases"] if earlier else ())}
            )
            rows[effect.resource_id] = RunFetchedResource(
                resource_id=effect.resource_id,
                ordinal=0,
                digest=digest,
                filename=effect.filename,
                mime_type=effect.mime_type,
                source_locator=locator,
                capabilities={"resource_kind": effect.resource_kind, "resource_aliases": aliases},
            )
    return tuple(rows.values()), blobs


def resuming_executor(rows: tuple[RunFetchedResource, ...], blobs: dict[str, bytes]):
    from tests.unit.test_answer_executor import _executor

    executor = _executor()
    executor._store.list_fetched_resources = AsyncMock(return_value=rows)

    async def stream(*, owner_id: str, digest: str, **kwargs: object):
        del owner_id, kwargs
        yield blobs[digest]

    executor._blob_store.stream = stream
    return executor


async def resumed_from(
    *writes: Sequence[ResourceAttachmentBytes], max_attachments: int = 6
) -> ResourceRegistry:
    """A fresh Run's registry restored from the rows these writes left, with no lineage."""
    rows, blobs = durable_rows(*writes)
    resumed = ResourceRegistry(max_attachments=max_attachments)
    try:
        await resuming_executor(rows, blobs)._restore_registry_fetches(
            resumed, owner_id="owner", run_id="run"
        )
    except BaseException:
        await resumed.aclose()
        raise
    return resumed


async def test_two_earlier_handles_for_the_same_bytes_record_one_resource(
    monkeypatch,
) -> None:
    """The same URL fetched in two turns gives two handles for one document.

    Both adoptions record the one Resource under one locator, so the store merges
    their aliases into one row, and only the first brings a view.
    """
    forbid_conversion(monkeypatch)
    content = pdf()
    lineage = Loaders(
        viewed_document("res-earlier-one", content, text="View one."),
        viewed_document("res-earlier-two", content, text="View two."),
    )
    async with ResourceRegistry() as registry:
        read, _ = tools(registry, lineage=lineage)
        one = await call(read, resource_id="res-earlier-one")
        two = await call(read, resource_id="res-earlier-two")

        assert (one.is_error, two.is_error) == (False, False)
        canonical = registry.canonical_resource_id("res-earlier-one")
        assert registry.canonical_resource_id("res-earlier-two") == canonical
        assert "View one." in two.text_content, "one Resource keeps the view it adopted first"
    first, second = lineage.recorded
    assert [row.resource_kind for row in second] == [LINEAGE_ADOPTION_KIND]
    rows, _ = durable_rows(first, second)
    adoptions = [row for row in rows if row.capabilities["resource_kind"] == LINEAGE_ADOPTION_KIND]
    assert [(row.resource_id, row.source_locator) for row in adoptions] == [
        (canonical, canonical.encode())
    ]
    assert adoptions[0].capabilities["resource_aliases"] == ["res-earlier-one", "res-earlier-two"]


async def test_an_adoption_is_durable_before_its_retried_call_fails(monkeypatch) -> None:
    """The adoption is recorded before its handle resolves, so a refused retry keeps it.

    The refusal carries nothing on the adoption's behalf, and a resume restores the
    Resource, both handles, and its view from the rows alone.
    """
    forbid_conversion(monkeypatch)
    lineage = Loader(viewed_document(EARLIER_HANDLE, pdf(), text="Stored text."))
    async with ResourceRegistry() as registry:
        read, view = tools(registry, lineage=lineage)
        refused = await call(view, resource_id=EARLIER_HANDLE, cursor="overview.not-a-cursor")

        assert refused.is_error is True
        assert "call read or view on the resource again" in refused.text_content
        assert refused.effects.attached_resources == ()
        (recorded,) = lineage.recorded
        later = await call(read, resource_id=EARLIER_HANDLE)
        assert later.is_error is False
        canonical = registry.canonical_resource_id(EARLIER_HANDLE)

    resumed = await resumed_from(recorded, later.effects.attached_resources)
    try:
        # The resumed registry mints with another secret; the recorded handle holds.
        assert resumed.canonical_resource_id(EARLIER_HANDLE) == canonical
        for handle in (EARLIER_HANDLE, canonical):
            result = await resumed.read(handle, max_window_tokens=1000)
            assert "Stored text." in result.content
    finally:
        await resumed.aclose()


async def test_an_adoption_holds_when_its_retried_call_is_cancelled(monkeypatch) -> None:
    """A cancelled call cannot leave a view whose Resource no durable row names."""
    forbid_conversion(monkeypatch)
    lineage = Loader(viewed_document(EARLIER_HANDLE, pdf(), text="Stored text."))
    async with ResourceRegistry() as registry:
        read, _ = tools(registry, lineage=lineage)
        original = registry.read
        calls = 0

        async def cancel_the_retry(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise asyncio.CancelledError
            return await original(*args, **kwargs)

        monkeypatch.setattr(registry, "read", cancel_the_retry)
        with pytest.raises(asyncio.CancelledError):
            await call(read, resource_id=EARLIER_HANDLE)
        monkeypatch.setattr(registry, "read", original)

        later = await call(read, resource_id=EARLIER_HANDLE)
        assert later.is_error is False
        assert lineage.reads == 1, "the cancelled call's alias answers the next one"
        (recorded,) = lineage.recorded

    resumed = await resumed_from(recorded, later.effects.attached_resources)
    try:
        result = await resumed.read(EARLIER_HANDLE, max_window_tokens=1000)
        assert "Stored text." in result.content
    finally:
        await resumed.aclose()


async def test_a_refused_stored_view_is_adopted_and_restored_as_refused(monkeypatch) -> None:
    """A view the earlier Run refused for safety stays that refusal, durably."""
    forbid_conversion(monkeypatch)
    loaded = viewed_document(EARLIER_HANDLE, pdf(), text="Stored text.")
    refused_view = _stored_view(loaded, text="", visuals=(), extraction_status="safety_refused")
    lineage = Loader(replace(loaded, conversion_snapshot=refused_view))
    async with ResourceRegistry() as registry:
        read, view = tools(registry, lineage=lineage)
        pixels = await call(view, resource_id=EARLIER_HANDLE, locator="1")
        text = await call(read, resource_id=EARLIER_HANDLE)

        assert pixels.is_error is True
        assert "refused by safety" in pixels.text_content
        assert text.is_error is True
        assert "extraction_status=safety_refused" in text.text_content
        (recorded,) = lineage.recorded

    resumed = await resumed_from(recorded, text.effects.attached_resources)
    try:
        with pytest.raises(ResourceAdmissionError):
            await resumed.read(EARLIER_HANDLE, max_window_tokens=1000)
    finally:
        await resumed.aclose()


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("storage unavailable"), LeaseLostError(), asyncio.CancelledError()],
    ids=["storage", "lease-lost", "cancelled"],
)
async def test_nothing_changes_here_until_the_adoption_is_recorded(failure) -> None:
    """A write that did not complete leaves this Run as it was, allowance included.

    A lost lease stops the call without a refusal of its own: this worker may
    persist nothing further, and the Run's next claimant adopts afresh.
    """
    document = adopted_document(with_snapshot=True)
    lineage = Loader(document)
    lineage.fails = failure
    # Room for exactly this one document, by count and by bytes.
    async with ResourceRegistry(
        max_attachments=1, max_total_attachment_bytes=len(document.content)
    ) as registry:
        read, _ = tools(registry, lineage=lineage)
        with pytest.raises(type(failure)):
            await call(read, resource_id=EARLIER_HANDLE)

        assert registry.manifest() == ()
        with pytest.raises(ResourceNotFoundError):
            registry.canonical_resource_id(EARLIER_HANDLE)

        lineage.fails = None
        adopted = await call(read, resource_id=EARLIER_HANDLE)

        assert adopted.is_error is False, "the slot and the bytes were both given back"
        assert _ADOPTED_TEXT in adopted.text_content
        assert len(lineage.recorded) == 1


async def test_a_failed_write_keeps_the_bytes_an_earlier_adoption_holds(monkeypatch) -> None:
    """Only bytes this adoption admitted are withdrawn when its write fails."""
    forbid_conversion(monkeypatch)
    content = pdf()
    lineage = Loaders(
        viewed_document("res-earlier-one", content, text="View one."),
        viewed_document("res-earlier-two", content, text="View two."),
    )
    async with ResourceRegistry() as registry:
        read, _ = tools(registry, lineage=lineage)
        assert (await call(read, resource_id="res-earlier-one")).is_error is False
        manifest = registry.manifest()
        lineage.fails = RuntimeError("storage unavailable")
        with pytest.raises(RuntimeError):
            await call(read, resource_id="res-earlier-two")
        lineage.fails = None

        assert registry.manifest() == manifest
        again = await call(read, resource_id="res-earlier-one")
        assert "View one." in again.text_content
        with pytest.raises(ResourceNotFoundError):
            registry.canonical_resource_id("res-earlier-two")


async def test_a_conflicting_adoption_is_refused_in_the_tools_own_words() -> None:
    """The store's refusal of a second view reaches the model as a typed refusal."""
    lineage = Loader(adopted_document(with_snapshot=True))
    lineage.fails = LineageAdoptionConflict("this run records another view for that document")
    async with ResourceRegistry() as registry:
        read, _ = tools(registry, lineage=lineage)
        refused = await call(read, resource_id=EARLIER_HANDLE)

        assert refused.is_error is True
        assert refused.text_content == (
            "this run records another view for that document; "
            "the earlier document was not adopted into this run."
        )
        assert registry.manifest() == ()


async def test_an_adoption_that_landed_despite_its_error_is_restored_without_refusal() -> None:
    """A write whose outcome was unknown may have landed after its slot went elsewhere.

    The live Run gave the slot back and spent it on another document; the resume
    restores both recorded adoptions as durable state under the handles they were
    recorded with, and refuses neither, although together they exceed the allowance.
    """
    first = adopted_product()
    second = replace(
        first, resource_id="artifact-5c1d2e3f4a5b6c7d8e9f", filename="notes.md", content=b"# b\n"
    )
    lineage = Loaders(first, second)
    lineage.lands_then_fails = ConnectionResetError("connection lost after COMMIT")
    async with ResourceRegistry(max_attachments=1) as registry:
        read, _ = tools(registry, lineage=lineage)
        with pytest.raises(ConnectionResetError):
            await call(read, resource_id=first.resource_id)
        assert registry.manifest() == ()
        assert (await call(read, resource_id=second.resource_id)).is_error is False
    handles = [write[0].resource_id for write in lineage.recorded]
    assert len(handles) == 2

    resumed = await resumed_from(*lineage.recorded, max_attachments=1)
    try:
        for loaded, handle in zip((first, second), handles, strict=True):
            assert resumed.canonical_resource_id(loaded.resource_id) == handle
            text = await resumed.read(loaded.resource_id, max_window_tokens=1000)
            assert loaded.content.decode().strip() in text.content
    finally:
        await resumed.aclose()


async def test_concurrent_adoptions_of_one_document_record_one_view(monkeypatch) -> None:
    """Adoptions run one at a time, so one Resource never records a second view."""
    forbid_conversion(monkeypatch)
    content = pdf()
    lineage = Loaders(
        viewed_document("res-earlier-one", content, text="View one."),
        viewed_document("res-earlier-three", content, text="View three."),
    )
    handles = ("res-earlier-one", "res-earlier-one", "res-earlier-three", "res-earlier-three")
    async with ResourceRegistry() as registry:
        read, _ = tools(registry, lineage=lineage)
        results = await asyncio.gather(*(call(read, resource_id=handle) for handle in handles))

        assert [result.is_error for result in results] == [False] * 4
        text = "View one." if "View one." in results[0].text_content else "View three."
        assert all(text in result.text_content for result in results)
    views = [
        row for write in lineage.recorded for row in write if row.resource_kind == SNAPSHOT_KIND
    ]
    assert len(views) == 1

    resumed = await resumed_from(
        *lineage.recorded, *(result.effects.attached_resources for result in results)
    )
    try:
        canonical = resumed.canonical_resource_id("res-earlier-one")
        assert resumed.canonical_resource_id("res-earlier-three") == canonical
        assert text in (await resumed.read("res-earlier-three", max_window_tokens=1000)).content
    finally:
        await resumed.aclose()


async def test_a_text_read_uses_the_view_held_through_another_handle(monkeypatch) -> None:
    """Text needs a stored view, and one this Run holds for the same bytes is one."""
    forbid_conversion(monkeypatch)
    content = pdf()
    lineage = Loaders(
        viewed_document("res-earlier-one", content, text="View one."),
        replace(viewed_document("res-earlier-two", content, text="-"), conversion_snapshot=None),
    )
    async with ResourceRegistry() as registry:
        read, _ = tools(registry, lineage=lineage)
        assert (await call(read, resource_id="res-earlier-one")).is_error is False

        two = await call(read, resource_id="res-earlier-two")

        assert two.is_error is False
        assert "View one." in two.text_content
        assert [row.resource_kind for row in lineage.recorded[-1]] == [LINEAGE_ADOPTION_KIND]


async def test_a_view_adopted_later_serves_the_handle_adopted_without_one(monkeypatch) -> None:
    """Bytes adopted for pixels take the first stored view that arrives for them."""
    forbid_conversion(monkeypatch)
    content = pdf()
    lineage = Loaders(
        replace(viewed_document("res-earlier-two", content, text="-"), conversion_snapshot=None),
        viewed_document("res-earlier-one", content, text="View one."),
    )
    async with ResourceRegistry() as registry:
        read, view = tools(registry, lineage=lineage)
        pixels = await call(view, resource_id="res-earlier-two", locator="1")
        text = await call(read, resource_id="res-earlier-one")
        again = await call(read, resource_id="res-earlier-two")

        assert pixels.is_error is False
        assert "View one." in text.text_content
        assert "View one." in again.text_content

    resumed = await resumed_from(
        *lineage.recorded, pixels.effects.attached_resources, text.effects.attached_resources
    )
    try:
        assert (
            "View one." in (await resumed.read("res-earlier-two", max_window_tokens=1000)).content
        )
    finally:
        await resumed.aclose()


async def test_an_unconverted_document_refuses_embedded_images_without_offering_pages(
    monkeypatch,
) -> None:
    """Only a PDF has pages to view; a document's embedded images need its text view."""
    forbid_conversion(monkeypatch)
    lineage = Loader(
        LineageResourceBytes(
            resource_id=EARLIER_HANDLE,
            origin_run_id="01a0a737-e1d3-7421-8e25-27ca8abd3dad",
            filename="notes.docx",
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            content=b"PK\x03\x04 an earlier run's document",
        )
    )
    async with ResourceRegistry() as registry:
        _, view = tools(registry, lineage=lineage)
        whole = await call(view, resource_id=EARLIER_HANDLE)
        embedded = await call(view, resource_id=EARLIER_HANDLE, locator="vis-0123456789abcdef")

        assert whole.is_error is True
        assert len(lineage.recorded) == 1, "the first call adopted it, refused or not"
        assert embedded.is_error is True
        assert "never extracted text from notes.docx" in embedded.text_content
        assert "View its pages" not in embedded.text_content


DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def earlier_docx(*, handle_id: str | None) -> LineageResourceBytes:
    """An earlier Run's DOCX, with a stored view holding one embedded image or none."""
    content = b"PK\x03\x04 an earlier run's document"
    snapshot: bytes | None = None
    assets: dict[str, bytes] = {}
    if handle_id is not None:
        effects = ConversionSnapshot(
            resource_id=EARLIER_HANDLE,
            input_digest=hashlib.sha256(content).hexdigest(),
            text="Stored text.",
            visuals=(
                ExtractedVisual(
                    handle_id=handle_id,
                    anchor="page 1",
                    origin_part=None,
                    media_type="image/png",
                    data=png(),
                ),
            ),
            extraction_status="complete",
            converter="fixture",
            converter_version="1",
        ).effects()
        snapshot = next(e.content for e in effects if e.resource_kind == SNAPSHOT_KIND)
        assets = {e.resource_id: e.content for e in effects if e.resource_kind == ASSET_KIND}
    return LineageResourceBytes(
        resource_id=EARLIER_HANDLE,
        origin_run_id="01a0a737-e1d3-7421-8e25-27ca8abd3dad",
        filename="notes.docx",
        media_type=DOCX,
        content=content,
        conversion_snapshot=snapshot,
        assets=assets,
    )


async def test_an_embedded_image_of_an_unconverted_adoption_refuses_on_the_first_call(
    monkeypatch,
) -> None:
    """The first call adopts and fails inside the retry; the adoption stays recorded."""
    forbid_conversion(monkeypatch)
    lineage = Loader(earlier_docx(handle_id=None))
    async with ResourceRegistry() as registry:
        _, view = tools(registry, lineage=lineage)
        refused = await call(view, resource_id=EARLIER_HANDLE, locator="vis-0123456789abcdef")

        assert refused.is_error is True
        assert "never extracted text from notes.docx" in refused.text_content
        assert "View its pages" not in refused.text_content
        assert len(lineage.recorded) == 1


async def test_an_unknown_embedded_image_of_an_adopted_document_names_what_is_missing() -> None:
    """Once adopted, the handle is held: only the image handle inside it is unknown."""
    lineage = Loader(earlier_docx(handle_id="vis-embedded"))
    async with ResourceRegistry() as registry:
        _, view = tools(registry, lineage=lineage)
        missing = await call(view, resource_id=EARLIER_HANDLE, locator="vis-not-there")
        found = await call(view, resource_id=EARLIER_HANDLE, locator="vis-embedded")

        assert missing.is_error is True
        assert "unknown visual handle: vis-not-there" in missing.text_content
        assert "neither holds" not in missing.text_content
        assert len(lineage.recorded) == 1
        assert found.is_error is False


async def test_a_pdf_named_without_a_suffix_is_still_offered_its_pages(monkeypatch) -> None:
    """The declared type decides it is a PDF, as a URL without an extension leaves it."""
    forbid_conversion(monkeypatch)
    loaded = replace(
        adopted_document(with_snapshot=False), filename="2401.12345", media_type="application/pdf"
    )
    async with ResourceRegistry() as registry:
        read, _ = tools(registry, lineage=Loader(loaded))
        refused = await call(read, resource_id=EARLIER_HANDLE)

        assert refused.is_error is True
        assert "View its pages for pixels" in refused.text_content


async def test_recovery_fails_on_a_view_no_restored_resource_owns() -> None:
    """A view whose Resource no row restores is a real inconsistency, never skipped.

    An adoption records its Resource and its view together, so this can only be a
    corrupt catalog, and recovery does not read the lineage to explain it away.
    """
    loaded = viewed_document(EARLIER_HANDLE, pdf(), text="Stored text.")
    view = ConversionSnapshot.restore(loaded.conversion_snapshot or b"", dict(loaded.assets))
    with pytest.raises(ResourceNotFoundError):
        await resumed_from(view.effects())


async def test_a_suffixless_pdf_viewed_then_read_is_offered_its_pages(monkeypatch) -> None:
    """The registry's own refusal carries the declared type, not only the precheck's."""
    forbid_conversion(monkeypatch)
    lineage = Loader(
        LineageResourceBytes(
            resource_id=EARLIER_HANDLE,
            origin_run_id="01a0a737-e1d3-7421-8e25-27ca8abd3dad",
            filename="2401.12345",
            media_type="application/pdf",
            content=pdf(),
        )
    )
    async with ResourceRegistry() as registry:
        read, view = tools(registry, lineage=lineage)
        assert (await call(view, resource_id=EARLIER_HANDLE)).is_error is False

        refused = await call(read, resource_id=EARLIER_HANDLE)

        assert refused.is_error is True
        assert "View its pages for pixels" in refused.text_content


def test_the_manifest_leaves_an_earlier_resource_id_to_adoption() -> None:
    """The manifest must not forbid the read lineage adoption exists to serve (ADR 0013)."""
    text = _resource_manifest_context(
        (ResourceManifestEntry("res-now", "report.pdf", "application/pdf", "bytes", 20),)
    )

    assert "resource id printed by an earlier turn may still resolve" in text
    assert "cursor printed by an earlier turn is historical" in text
