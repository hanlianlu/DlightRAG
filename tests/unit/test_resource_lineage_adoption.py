# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Lineage adoption: an earlier Run's handle becomes this Run's Resource."""

import base64
import hashlib
import io
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from dlightrag.engine.agent.environment.access import AccessScheduler
from dlightrag.engine.agent.tool_content import decode_tool_content, encode_tool_content
from dlightrag.engine.agent.tools import ToolEffects, ToolResult
from dlightrag.engine.agent.tools.files import PreparedImageAttachment, read_tool, view_tool
from dlightrag.engine.ai.media import decode_image_base64
from dlightrag.engine.answer.research.context import _resource_manifest_context
from dlightrag.engine.answer.resources.converters import ConvertedResource, ExtractedVisual
from dlightrag.engine.answer.resources.lineage import (
    ASSET_KIND,
    LINEAGE_ADOPTION_KIND,
    SNAPSHOT_KIND,
    LineageResourceBytes,
)
from dlightrag.engine.answer.resources.models import (
    ResourceInput,
    ResourceManifestEntry,
    ResourceNotFoundError,
    TextWindowBudget,
)
from dlightrag.engine.answer.resources.registry import ResourceRegistry
from dlightrag.engine.answer.resources.snapshots import ConversionSnapshot
from dlightrag.engine.answer.tools.resources import make_resource_reader, make_resource_viewer
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


class Loader:
    """One authorized earlier Run Resource, counting how often it is read."""

    def __init__(self, loaded: LineageResourceBytes | None) -> None:
        self.loaded = loaded
        self.reads = 0

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

        kinds = [effect.resource_kind for effect in result.effects.attached_resources]
        assert LINEAGE_ADOPTION_KIND in kinds
        assert SNAPSHOT_KIND in kinds
        assert ASSET_KIND in kinds
        ids = [effect.resource_id for effect in result.effects.attached_resources]
        assert len(ids) == len(set(ids)), "each adopted Resource settles once"
        source = next(
            effect
            for effect in result.effects.attached_resources
            if effect.resource_kind == LINEAGE_ADOPTION_KIND
        )
        assert source.aliases == (EARLIER_HANDLE,), "the earlier handle is recorded, not only bound"
        snapshot_effect = next(
            effect
            for effect in result.effects.attached_resources
            if effect.resource_kind == SNAPSHOT_KIND
        )
        assert snapshot_effect.resource_id == f"{EARLIER_HANDLE}-conversion"
        assert snapshot_effect.source_locator == EARLIER_HANDLE
        assert registry.canonical_resource_id(EARLIER_HANDLE) != EARLIER_HANDLE

        second = await call(read, resource_id=EARLIER_HANDLE)
        assert second.is_error is False
        assert _ADOPTED_TEXT in second.text_content
        assert lineage.reads == 1, "the alias needs no second read"


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
        kinds = [effect.resource_kind for effect in result.effects.attached_resources]
        assert LINEAGE_ADOPTION_KIND in kinds, "the adopted source document is pinned"
        assert "tool_attachment" in kinds, "the viewed pixels stay a tool attachment"
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
        assert registry.manifest() == ()
        with pytest.raises(ResourceNotFoundError):
            registry.canonical_resource_id(EARLIER_HANDLE)


async def test_adoption_past_the_attachment_allowance_refuses_as_a_tool_error() -> None:
    """Adoption spends this Run's allowance, and a spent one refuses rather than raising."""
    async with ResourceRegistry(max_attachments=1) as registry:
        registry.register(
            ResourceInput(filename="own.txt", declared_mime="text/plain", content=b"own")
        )
        read, view = tools(registry, lineage=Loader(adopted_document(with_snapshot=True)))
        for result in (
            await call(read, resource_id=EARLIER_HANDLE),
            await call(view, resource_id=EARLIER_HANDLE),
        ):
            assert result.is_error is True
            assert "too many attachments" in result.text_content
            assert "was not adopted" in result.text_content
        with pytest.raises(ResourceNotFoundError):
            registry.canonical_resource_id(EARLIER_HANDLE)


async def test_bytes_this_run_already_reads_keep_this_runs_view(monkeypatch) -> None:
    """The same document attached again is one Resource with one conversion history."""
    conversions: list[str] = []

    async def convert(_content, *, filename, declared_mime):
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
        read, _ = tools(registry, lineage=Loader(loaded))
        assert (await call(read, resource_id=own)).is_error is False

        result = await call(read, resource_id=EARLIER_HANDLE)

        assert result.is_error is False
        assert "This Run's own extraction." in result.text_content
        assert _ADOPTED_TEXT not in result.text_content
        assert registry.canonical_resource_id(EARLIER_HANDLE) == own
        assert conversions == [loaded.filename], "nothing is converted a second time"


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


class Loaders:
    """Several earlier Runs' Resources, by the handle each was printed under."""

    def __init__(self, *loaded: LineageResourceBytes) -> None:
        self.by_id = {item.resource_id: item for item in loaded}
        self.reads = 0

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


async def test_two_earlier_handles_for_the_same_bytes_settle_as_one_resource(
    monkeypatch,
) -> None:
    """The same URL fetched in two turns gives two handles for one document.

    Both adoptions settle the one Resource under one locator, so settlement merges
    their aliases rather than refusing a second row for the same Resource.
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
        adoptions = [
            effect
            for result in (one, two)
            for effect in result.effects.attached_resources
            if effect.resource_kind == LINEAGE_ADOPTION_KIND
        ]
        assert {effect.resource_id for effect in adoptions} == {canonical}
        assert {effect.source_locator for effect in adoptions} == {canonical}
        assert adoptions[-1].aliases == ("res-earlier-one", "res-earlier-two")


def settled_rows(*results: ToolResult) -> tuple[tuple[RunFetchedResource, ...], dict[str, bytes]]:
    """The catalog settlement leaves: one row per Resource, aliases merged."""
    rows: dict[str, RunFetchedResource] = {}
    blobs: dict[str, bytes] = {}
    for result in results:
        for effect in result.effects.attached_resources:
            digest = hashlib.sha256(effect.content).hexdigest()
            blobs[digest] = effect.content
            earlier = rows.get(effect.resource_id)
            aliases = sorted(
                {
                    *effect.aliases,
                    *(earlier.capabilities.get("resource_aliases", []) if earlier else []),
                }
            )
            rows[effect.resource_id] = RunFetchedResource(
                resource_id=effect.resource_id,
                ordinal=0,
                digest=digest,
                filename=effect.filename,
                mime_type=effect.mime_type,
                source_locator=effect.source_locator.encode(),
                capabilities={"resource_kind": effect.resource_kind, "resource_aliases": aliases},
            )
    return tuple(rows.values()), blobs


async def test_an_adoption_whose_retried_call_fails_still_settles(monkeypatch) -> None:
    """The alias is bound either way, so the adoption must settle with the refusal.

    Otherwise a later call through the alias settles a view whose parent handle no
    durable row names, and the Run cannot resume.
    """
    from tests.unit.test_answer_executor import _executor

    forbid_conversion(monkeypatch)
    loaded = viewed_document(EARLIER_HANDLE, pdf(), text="Stored text.")
    async with ResourceRegistry() as registry:
        read, view = tools(registry, lineage=Loader(loaded))
        refused = await call(view, resource_id=EARLIER_HANDLE, cursor="overview.not-a-cursor")

        assert refused.is_error is True
        assert "call read or view on the resource again" in refused.text_content
        kinds = {effect.resource_kind for effect in refused.effects.attached_resources}
        assert {LINEAGE_ADOPTION_KIND, SNAPSHOT_KIND} <= kinds
        later = await call(read, resource_id=EARLIER_HANDLE)
        assert later.is_error is False

    rows, blobs = settled_rows(refused, later)
    executor = _executor()
    executor._store.list_fetched_resources = AsyncMock(return_value=rows)

    async def stream(*, owner_id: str, digest: str, **kwargs: object):
        del owner_id, kwargs
        yield blobs[digest]

    executor._blob_store.stream = stream
    async with ResourceRegistry() as resumed:
        await executor._restore_registry_fetches(resumed, owner_id="owner", run_id="run")
        result = await resumed.read(EARLIER_HANDLE, max_window_tokens=1000)
        assert "Stored text." in result.content


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
        assert any(
            effect.resource_kind == LINEAGE_ADOPTION_KIND
            for effect in whole.effects.attached_resources
        )
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


def adoption_settles(result: ToolResult) -> bool:
    return any(
        effect.resource_kind == LINEAGE_ADOPTION_KIND
        for effect in result.effects.attached_resources
    )


async def test_an_embedded_image_of_an_unconverted_adoption_refuses_on_the_first_call(
    monkeypatch,
) -> None:
    """The first call adopts and fails inside the retry; the refusal still settles it."""
    forbid_conversion(monkeypatch)
    async with ResourceRegistry() as registry:
        _, view = tools(registry, lineage=Loader(earlier_docx(handle_id=None)))
        refused = await call(view, resource_id=EARLIER_HANDLE, locator="vis-0123456789abcdef")

        assert refused.is_error is True
        assert "never extracted text from notes.docx" in refused.text_content
        assert "View its pages" not in refused.text_content
        assert adoption_settles(refused)


async def test_an_unknown_embedded_image_of_an_adopted_document_names_what_is_missing() -> None:
    """Once adopted, the handle is held: only the image handle inside it is unknown."""
    async with ResourceRegistry() as registry:
        _, view = tools(registry, lineage=Loader(earlier_docx(handle_id="vis-embedded")))
        missing = await call(view, resource_id=EARLIER_HANDLE, locator="vis-not-there")
        found = await call(view, resource_id=EARLIER_HANDLE, locator="vis-embedded")

        assert missing.is_error is True
        assert "unknown visual handle: vis-not-there" in missing.text_content
        assert "neither holds" not in missing.text_content
        assert adoption_settles(missing)
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


def orphaned_view_rows(
    loaded: LineageResourceBytes,
) -> tuple[tuple[RunFetchedResource, ...], dict[str, bytes]]:
    """The rows a view settles under an alias whose adoption row never settled."""
    snapshot = next(
        effect
        for effect in ConversionSnapshot.restore(
            loaded.conversion_snapshot or b"", dict(loaded.assets)
        ).effects()
        if effect.resource_kind == SNAPSHOT_KIND
    )
    return settled_rows(ToolResult.text("x", effects=ToolEffects(attached_resources=(snapshot,))))


def resuming_executor(rows: tuple[RunFetchedResource, ...], blobs: dict[str, bytes]):
    from tests.unit.test_answer_executor import _executor

    executor = _executor()
    executor._store.list_fetched_resources = AsyncMock(return_value=rows)

    async def stream(*, owner_id: str, digest: str, **kwargs: object):
        del owner_id, kwargs
        yield blobs[digest]

    executor._blob_store.stream = stream
    return executor


async def test_recovery_adopts_again_the_resource_an_orphaned_view_belongs_to(
    monkeypatch,
) -> None:
    """The view's earlier handle is adopted again through the lineage, view and all."""
    forbid_conversion(monkeypatch)
    loaded = viewed_document(EARLIER_HANDLE, pdf(), text="Stored text.")
    rows, blobs = orphaned_view_rows(loaded)
    executor = resuming_executor(rows, blobs)
    async with ResourceRegistry() as resumed:
        await executor._restore_registry_fetches(
            resumed, owner_id="owner", run_id="run", lineage=Loader(loaded)
        )

        result = await resumed.read(EARLIER_HANDLE, max_window_tokens=1000)
        assert "Stored text." in result.content


@pytest.mark.parametrize("lineage", ["none", "absent", "other-bytes"])
async def test_recovery_still_fails_on_a_view_no_lineage_explains(lineage: str) -> None:
    """A view whose parent the lineage cannot supply is a real inconsistency, not noise."""
    loaded = viewed_document(EARLIER_HANDLE, pdf(), text="Stored text.")
    rows, blobs = orphaned_view_rows(loaded)
    loader = {
        "none": None,
        "absent": Loader(None),
        "other-bytes": Loader(replace(loaded, content=pdf(pages=3))),
    }[lineage]
    async with ResourceRegistry() as resumed:
        with pytest.raises(ResourceNotFoundError):
            await resuming_executor(rows, blobs)._restore_registry_fetches(
                resumed, owner_id="owner", run_id="run", lineage=loader
            )


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
