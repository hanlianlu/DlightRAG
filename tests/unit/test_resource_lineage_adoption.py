# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Lineage adoption: an earlier Run's handle becomes this Run's Resource."""

import base64
import hashlib
import io
from dataclasses import replace

from PIL import Image

from dlightrag.engine.agent.environment.access import AccessScheduler
from dlightrag.engine.agent.tool_content import decode_tool_content, encode_tool_content
from dlightrag.engine.agent.tools import ToolResult
from dlightrag.engine.agent.tools.files import PreparedImageAttachment, read_tool, view_tool
from dlightrag.engine.ai.media import decode_image_base64
from dlightrag.engine.answer.resources.converters import ExtractedVisual
from dlightrag.engine.answer.resources.lineage import (
    ASSET_KIND,
    LINEAGE_ADOPTION_KIND,
    SNAPSHOT_KIND,
    LineageResourceBytes,
)
from dlightrag.engine.answer.resources.models import TextWindowBudget
from dlightrag.engine.answer.resources.registry import ResourceRegistry
from dlightrag.engine.answer.resources.snapshots import ConversionSnapshot
from dlightrag.engine.answer.tools.resources import make_resource_reader, make_resource_viewer
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


async def test_an_unauthorized_handle_keeps_the_typed_refusal() -> None:
    lineage = Loader(None)
    async with ResourceRegistry() as registry:
        read, view = tools(registry, lineage=lineage)
        for result in (
            await call(read, resource_id="res-foreign"),
            await call(view, resource_id="res-foreign"),
        ):
            assert result.is_error is True
            assert "earlier turn is historical" in result.text_content
        assert lineage.reads == 2


async def test_an_unusable_stored_snapshot_refuses_instead_of_repairing(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("a broken snapshot must not trigger conversion")

    monkeypatch.setattr("dlightrag.engine.answer.resources.registry.convert_resource", forbidden)
    broken = adopted_document(with_snapshot=True)
    corrupt = replace(broken, conversion_snapshot=b'{"text":"x"}')
    async with ResourceRegistry() as registry:
        read, _ = tools(registry, lineage=Loader(corrupt))
        result = await call(read, resource_id=EARLIER_HANDLE)

        assert result.is_error is True
        assert "was not converted again" in result.text_content
