# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""One run-scoped read surface answers for every registry a run records bytes in."""

import pytest

from dlightrag.application.answer_runs import AnswerService
from tests.unit.test_answer_service import (
    _OWNER,
    _fetched_resource,
    _reference,
    _service,
    _Store,
)


def _service_with(store: _Store) -> AnswerService:
    return _service(store=store)


async def test_an_accepted_upload_is_reachable_by_its_resource_id() -> None:
    store = _Store(
        references=(
            _reference(kind="current_attachment", ordinal=0, digest="d0", filename="notes.md"),
        )
    )
    store._blobs["d0"] = b"content"
    service = _service_with(store)

    resolved = await service.run_resource(
        owner_id=_OWNER, run_id=store._run.run_id, resource_id="current_attachment-0"
    )
    assert resolved is not None
    assert resolved.registry == "artifact"
    assert resolved.reference_kind == "current_attachment"
    assert resolved.filename == "notes.md"
    assert resolved.digest == "d0"

    streamed = await service.open_run_resource(
        owner_id=_OWNER, run_id=store._run.run_id, resource_id="current_attachment-0"
    )
    assert streamed is not None
    assert b"".join([piece async for piece in streamed]) == b"content"
    assert (
        await service.run_resource_size(
            owner_id=_OWNER, run_id=store._run.run_id, resource_id="current_attachment-0"
        )
        == 7
    )


async def test_a_worker_resource_is_reachable_by_the_id_it_recorded() -> None:
    store = _Store()
    store._resources = (
        _fetched_resource(
            resource_id="res-view-1", digest="d1", source_url="https://example.com/a.png"
        ),
    )
    store._blobs["d1"] = b"png"
    service = _service_with(store)

    resolved = await service.run_resource(
        owner_id=_OWNER, run_id=store._run.run_id, resource_id="res-view-1"
    )
    assert resolved is not None
    assert resolved.registry == "resource"
    assert resolved.reference_kind is None
    assert resolved.mime_type == "image/jpeg"
    read = await service.read_run_resource(
        owner_id=_OWNER, run_id=store._run.run_id, resource_id="res-view-1"
    )
    assert read is not None
    assert read[1] == b"png"


async def test_an_adopted_entry_attachment_resolves_like_any_other_id() -> None:
    # Adopted rows carry a locator digest rather than a URL and are excluded from
    # the URL-bearing catalog, so only an unfiltered by-id read can reach them.
    adopted_id = "attachment-occurrence:019893f4-0000-7000-8000-0000000000ff:1"
    store = _Store()
    store._resources = (
        _fetched_resource(resource_id=adopted_id, digest="d9", filename="source image"),
    )
    store._blobs["d9"] = b"page"
    service = _service_with(store)

    resolved = await service.run_resource(
        owner_id=_OWNER, run_id=store._run.run_id, resource_id=adopted_id
    )

    assert resolved is not None
    assert resolved.registry == "resource"
    assert resolved.digest == "d9"
    read = await service.read_run_resource(
        owner_id=_OWNER, run_id=store._run.run_id, resource_id=adopted_id
    )
    assert read is not None and read[1] == b"page"
    # It has no URL, so it never claims an answer source.
    assert await service.run_external_source_map(owner_id=_OWNER, run_id=store._run.run_id) == {}


async def test_a_current_upload_wins_over_a_carried_forward_one_sharing_its_id() -> None:
    shared_id = "attachment-0"
    current = _reference(kind="current_attachment", ordinal=0, digest="current", filename="now.txt")
    carried = _reference(
        kind="history_attachment", ordinal=0, digest="carried", filename="before.txt"
    )
    object.__setattr__(current, "resource_id", shared_id)
    object.__setattr__(carried, "resource_id", shared_id)
    store = _Store(references=(carried, current))
    service = _service_with(store)

    resolved = await service.run_resource(
        owner_id=_OWNER, run_id=store._run.run_id, resource_id=shared_id
    )
    assert resolved is not None
    assert resolved.reference_kind == "current_attachment"
    assert resolved.digest == "current"


@pytest.mark.parametrize(
    ("owner_id", "run_id", "resource_id"),
    [
        (_OWNER, "another-run", "current_attachment-0"),
        ("another-owner", "run-1", "current_attachment-0"),
        (_OWNER, "run-1", "res-missing"),
        (_OWNER, "run-1", ""),
    ],
    ids=["other-run", "other-owner", "unknown-id", "empty-id"],
)
async def test_an_unreachable_resource_resolves_to_nothing(
    owner_id: str, run_id: str, resource_id: str
) -> None:
    store = _Store(
        references=(
            _reference(kind="current_attachment", ordinal=0, digest="d0", filename="a.txt"),
        )
    )
    store._blobs["d0"] = b"content"
    store._resources = (_fetched_resource(resource_id="res-view-1", digest="d1"),)
    service = _service_with(store)

    assert (
        await service.run_resource(owner_id=owner_id, run_id=run_id, resource_id=resource_id)
        is None
    )
    assert (
        await service.open_run_resource(owner_id=owner_id, run_id=run_id, resource_id=resource_id)
        is None
    )
    assert (
        await service.run_resource_size(owner_id=owner_id, run_id=run_id, resource_id=resource_id)
        is None
    )
    assert (
        await service.read_run_resource(owner_id=owner_id, run_id=run_id, resource_id=resource_id)
        is None
    )


async def test_the_external_source_map_normalizes_and_includes_aliases() -> None:
    store = _Store()
    store._resources = (
        _fetched_resource(
            resource_id="res-a",
            digest="d1",
            source_url="HTTPS://Example.COM/a.png#fragment",
        ),
        _fetched_resource(
            resource_id="res-b",
            digest="d2",
            source_url="https://example.com/b.png",
            aliases=("https://cdn.example.com/b.png",),
        ),
        _fetched_resource(resource_id="res-c", digest="d3", source_url=""),
    )
    service = _service_with(store)

    sources = await service.run_external_source_map(owner_id=_OWNER, run_id=store._run.run_id)

    assert sources == {
        "https://example.com/a.png": "res-a",
        "https://example.com/b.png": "res-b",
        "https://cdn.example.com/b.png": "res-b",
    }
    # An unknown run never leaks a registry read.
    assert await service.run_external_source_map(owner_id=_OWNER, run_id="another-run") == {}
