# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""A convertible product is published with the view a later turn reads it through."""

import asyncio
import base64
import hashlib
import io
import time
from collections.abc import AsyncIterator

import pytest
from PIL import Image

from dlightrag.engine.answer.execution import executor
from dlightrag.engine.answer.execution.executor import _publication_views, _stage_publications
from dlightrag.engine.answer.execution.lineage import RetainedResourceLoader
from dlightrag.engine.answer.publication import (
    PublicationPlan,
    StagedArtifact,
    artifact_resource_id,
)
from dlightrag.engine.answer.resources.converters import ConversionLimitError
from dlightrag.engine.answer.resources.lineage import ASSET_KIND, SNAPSHOT_KIND
from dlightrag.engine.answer.resources.models import ResourceInput
from dlightrag.engine.answer.resources.registry import ResourceRegistry
from dlightrag.engine.answer.resources.snapshots import ConversionSnapshot
from dlightrag.engine.runtime.records import RunFetchedResource

_SESSION = "01930000-0000-7000-8000-000000000001"
_SECRET = b"s" * 32


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (6, 6), (200, 30, 30)).save(buffer, "PNG")
    return buffer.getvalue()


def _html_report() -> bytes:
    image = base64.b64encode(_png()).decode()
    return (
        "<html><body><h1>Quarterly report</h1><p>Revenue grew eleven percent.</p>"
        f'<img alt="revenue chart" src="data:image/png;base64,{image}"></body></html>'
    ).encode()


def _staged(path: str, media_type: str, content: bytes) -> StagedArtifact:
    return StagedArtifact(
        relative_path=path,
        media_type=media_type,
        size_bytes=len(content),
        resource_id=artifact_resource_id(path),
        filename=path.rsplit("/", 1)[-1],
        digest=hashlib.sha256(content).hexdigest(),
        presentation="download",
        content=content,
    )


async def test_a_products_view_is_the_view_a_read_of_it_adopts() -> None:
    content = _html_report()
    async with ResourceRegistry(resource_secret=_SECRET) as reading:
        handle = reading.register(ResourceInput(filename="report.html", content=content))
        page = await reading.read(handle, max_window_tokens=2_000)
        assert "Revenue grew eleven percent." in page.content
        (read_view,) = (
            effect
            for effect in reading.conversion_effects(handle)
            if effect.resource_kind == SNAPSHOT_KIND
        )

    async with ResourceRegistry(resource_secret=_SECRET) as publishing:
        published = await publishing.conversion_view(
            handle, content, filename="report.html", declared_mime="text/html"
        )
        # Building a view registers nothing in the publishing Run.
        assert publishing.manifest() == ()

    (published_view,) = (
        effect for effect in published.effects() if effect.resource_kind == SNAPSHOT_KIND
    )
    assert published_view == read_view
    assert [visual.handle_id.startswith("vis-") for visual in published.visuals] == [True]
    assert f"visual://{published.visuals[0].handle_id}" in published.text


async def test_publication_converts_what_a_later_read_needs_a_view_for() -> None:
    report = _staged("reports/report.html", "text/html", _html_report())
    table = _staged("reports/data.csv", "text/csv", b"quarter,revenue\nQ1,11\n")
    notes = _staged("reports/notes.md", "text/markdown", b"# notes\n")
    broken = _staged(
        "reports/model.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        b"not a workbook",
    )

    async with ResourceRegistry(resource_secret=_SECRET) as registry:
        views = await _publication_views((report, broken, table, notes), registry=registry)

    # A Markdown product is decoded later and needs no view; a product that cannot
    # be converted keeps none, and the products after it still get theirs.
    assert set(views) == {report.resource_id, table.resource_id}
    assert all(view.resource_id == rid for rid, view in views.items())
    assert "Q1" in views[table.resource_id].text

    publications, _descriptors, _sources = _stage_publications(
        plan=PublicationPlan(answer="See the report.", artifacts=(report, broken, table, notes)),
        answer="See the report.",
        session_id=_SESSION,
        views=views,
    )
    by_id = {publication.resource_id: publication for publication in publications}
    assert by_id[broken.resource_id].view == ()
    assert by_id[notes.resource_id].view == ()
    rows = [update.resource for update in by_id[report.resource_id].view]
    assert sorted(row.capabilities["resource_kind"] for row in rows) == [ASSET_KIND, SNAPSHOT_KIND]
    # The rows a read settles: named by the product, stamped with its Session.
    assert {row.source_locator for row in rows} == {report.resource_id.encode()}
    assert {row.session_id for row in rows} == {_SESSION}
    assert {row.intent_id for row in rows} == {None}
    assert f"{report.resource_id}-conversion" in {row.resource_id for row in rows}


async def test_publication_waits_for_views_within_one_conversions_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _staged("reports/data.csv", "text/csv", b"quarter,revenue\nQ1,11\n")
    monkeypatch.setattr(executor, "MAX_CONVERSION_SECONDS", 0.0)

    async with ResourceRegistry(resource_secret=_SECRET) as registry:
        # A spent budget publishes the product without a view instead of waiting.
        assert await _publication_views((table,), registry=registry) == {}
        with pytest.raises(ConversionLimitError):
            await registry.conversion_view(
                table.resource_id,
                table.content,
                filename=table.filename,
                declared_mime=table.media_type,
                deadline=time.monotonic(),
            )


async def test_a_cancelled_publication_stops_waiting_but_the_registry_joins_its_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started, release = asyncio.Event(), asyncio.Event()
    finished: list[str] = []

    async def native(content: bytes, **_kwargs: object) -> object:
        # Native conversion cannot be interrupted: it runs on after a cancellation.
        started.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                current = asyncio.current_task()
                assert current is not None
                current.uncancel()
        finished.append("joined")
        raise ConversionLimitError("conversion total budget exhausted")

    monkeypatch.setattr("dlightrag.engine.answer.resources.registry.convert_resource", native)
    table = _staged("reports/data.csv", "text/csv", b"quarter,revenue\nQ1,11\n")
    registry = ResourceRegistry(resource_secret=_SECRET)
    publishing = asyncio.create_task(_publication_views((table,), registry=registry))
    await started.wait()

    publishing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await publishing
    assert finished == []

    asyncio.get_running_loop().call_soon(release.set)
    await registry.aclose()
    assert finished == ["joined"]


class _Rows:
    """Lineage rows of two Runs that published one path, and their Blobs."""

    def __init__(self, rows: tuple[RunFetchedResource, ...], blobs: dict[str, bytes]) -> None:
        self._rows = rows
        self._blobs = blobs

    async def lineage_resource_rows(self, **_query: str) -> tuple[RunFetchedResource, ...]:
        return self._rows

    async def record_lineage_adoption(self, **_write: object) -> None:
        raise AssertionError("loading records nothing")

    async def stream(
        self, *, owner_id: str, digest: str, offset: int = 0, length: int | None = None
    ) -> AsyncIterator[bytes]:
        del owner_id, offset, length
        yield self._blobs.get(digest, b"")


def _row(
    resource_id: str, run_id: str, kind: str, content: bytes, locator: str
) -> RunFetchedResource:
    return RunFetchedResource(
        resource_id=resource_id,
        ordinal=0,
        digest=hashlib.sha256(content).hexdigest(),
        filename=resource_id,
        mime_type="text/csv",
        source_locator=locator.encode(),
        capabilities={"resource_kind": kind, "origin_run_id": run_id},
    )


async def test_a_newer_version_without_a_view_never_borrows_an_older_versions() -> None:
    handle = artifact_resource_id("reports/data.csv")
    older, newer = b"quarter,revenue\nQ1,11\n", b"quarter,revenue\nQ1,12\n"
    view = ConversionSnapshot(
        resource_id=handle,
        input_digest=hashlib.sha256(older).hexdigest(),
        text="| quarter | revenue |",
        visuals=(),
        extraction_status="usable_text_unverified_coverage",
        converter="markitdown",
        converter_version="test",
    )
    (stored_view,) = view.effects()
    rows = (
        # Newest first, as the store orders them: the version published last wins.
        _row(handle, "run-newer", "published_artifact", newer, handle),
        _row(handle, "run-older", "published_artifact", older, handle),
        _row(stored_view.resource_id, "run-older", SNAPSHOT_KIND, stored_view.content, handle),
    )
    blobs = {
        row.digest: content
        for row, content in zip(rows, (newer, older, stored_view.content), strict=True)
    }
    store = _Rows(rows, blobs)
    loader = RetainedResourceLoader(
        store=store,
        blobs=store,
        owner_id="owner",
        session_id=_SESSION,
        run_id="run-reading",
        worker_id="worker",
        fencing_epoch=1,
    )

    loaded = await loader.load(handle)

    assert loaded is not None
    assert (loaded.content, loaded.origin_run_id) == (newer, "run-newer")
    assert loaded.conversion_snapshot is None
    assert loaded.assets == {}
